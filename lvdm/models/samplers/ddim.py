import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps, rescale_noise_cfg
from lvdm.common import noise_like
from lvdm.common import extract_into_tensor
import copy

from lvdm.modules.attention import MSATracker
from configs.v2v_config import MSAType


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

        self.all_sa_collect_cond = []
        self.all_sa_collect_uncond = []

        self.first_run = True

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        if self.model.use_dynamic_rescale:
            self.ddim_scale_arr = self.model.scale_arr[self.ddim_timesteps]
            # self.ddim_scale_arr_prev = torch.cat([self.ddim_scale_arr[0:1], self.ddim_scale_arr[:-1]])
            # fix a bug
            self.ddim_scale_arr_prev = torch.cat([self.model.scale_arr[0:1], self.ddim_scale_arr[:-1]])

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def ddim_inversion(self,
                       x0,
                       cond,
                       ddim_steps=50,
                       verbose=True,
                       ddim_eta=0.0,
                       unconditional_conditioning=None,
                       unconditional_guidance_scale=1.0,
                       **kwargs):
        """
        DDIM Inversion: converts generated latents x0 back to noise x_T

        Args:
            x0: Clean latent frames from run 0 [B, C, T, H, W]
            cond: Conditioning dict with c_crossattn and c_concat from run 0
            ddim_steps: Number of inversion steps (should match generation)
            verbose: Show progress bar
            ddim_eta: Should be 0.0 for deterministic inversion
            unconditional_conditioning: For CFG during inversion (if used)
            unconditional_guidance_scale: CFG scale during inversion

        Returns:
            x_T: Inverted noise at timestep T
            intermediates: Dict with intermediate latents
        """
        # Make schedule for inversion
        self.make_schedule(ddim_num_steps=ddim_steps, ddim_discretize='uniform',
                           ddim_eta=ddim_eta, verbose=False)

        device = x0.device
        b = x0.shape[0]
        is_video = (x0.dim() == 5)

        # Start from clean latents (t=0)
        x_t = x0.clone()

        # Inversion goes forward in time: t=0 -> t=T
        # So we need to reverse the timesteps
        timesteps = self.ddim_timesteps
        time_range = timesteps  # Already in ascending order for inversion
        total_steps = timesteps.shape[0]

        if verbose:
            iterator = tqdm(time_range, desc='DDIM Inversion', total=total_steps)
        else:
            iterator = time_range

        intermediates = {'x_inter': [x_t.clone()]}

        for i, step in enumerate(iterator):
            index = i  # For inversion, we go forward through indices
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Predict noise at current timestep
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x_t, ts, cond, **kwargs)
            else:
                # Apply CFG during inversion (if you used it during generation)
                e_t_cond = self.model.apply_model(x_t, ts, cond, **kwargs)
                e_t_uncond = self.model.apply_model(x_t, ts, unconditional_conditioning, **kwargs)
                model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

            # Convert to noise prediction if using v-parameterization
            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x_t, ts, model_output)
            else:
                e_t = model_output

            # Get alpha values for inversion step
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas

            if is_video:
                size = (b, 1, 1, 1, 1)
            else:
                size = (b, 1, 1, 1)

            # For inversion, we go from t to t+1
            # So we swap the roles of alpha_t and alpha_prev
            if i < total_steps - 1:
                # Current timestep (lower noise)
                a_t = torch.full(size, alphas[index], device=device)
                sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device)

                # Next timestep (higher noise)
                a_next = torch.full(size, alphas[index + 1], device=device)
                sqrt_one_minus_a_next = torch.full(size, sqrt_one_minus_alphas[index + 1], device=device)
            else:
                # Last step: go to pure noise
                a_t = torch.full(size, alphas[index], device=device)
                sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device)
                a_next = torch.full(size, 0.0, device=device)  # Pure noise
                sqrt_one_minus_a_next = torch.full(size, 1.0, device=device)

            # Predict x0 from current x_t
            pred_x0 = (x_t - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # Apply dynamic rescale if used
            if self.model.use_dynamic_rescale:
                scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
                if i < total_steps - 1:
                    next_scale_t = torch.full(size, self.ddim_scale_arr[index + 1], device=device)
                else:
                    next_scale_t = torch.full(size, 1.0, device=device)
                rescale = next_scale_t / scale_t
                pred_x0 *= rescale

            # DDIM inversion step: compute x_{t+1} (adding noise)
            # x_{t+1} = sqrt(alpha_{t+1}) * pred_x0 + sqrt(1 - alpha_{t+1}) * e_t
            if i < total_steps - 1:
                dir_xt = sqrt_one_minus_a_next * e_t
                x_t = a_next.sqrt() * pred_x0 + dir_xt
            else:
                # Last step: pure noise
                x_t = sqrt_one_minus_a_next * e_t

            if i % 1 == 0 or i == total_steps - 1:
                intermediates['x_inter'].append(x_t.clone())

        return x_t, intermediates

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask_move=None,
               mask_static=None,
               x0=None,
               first_conds_z0=None,
               prev_conds_z0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               precision=None,
               fs=None,
               timestep_spacing='uniform', #uniform_trailing for starting from last timestep
               guidance_rescale=0.0,
               msa=None,
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=timestep_spacing, ddim_eta=eta, verbose=schedule_verbose)
        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask_move=mask_move,
                                                    mask_static=mask_static,
                                                    x0=x0,
                                                    prev_conds_z0=prev_conds_z0,
                                                    first_conds_z0=first_conds_z0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    precision=precision,
                                                    fs=fs,
                                                    guidance_rescale=guidance_rescale,
                                                    msa=msa,
                                                    **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, first_conds_z0=None, prev_conds_z0=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask_move=None, mask_static=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
                      precision=None,fs=None,guidance_rescale=0.0, msa=None, **kwargs):
        device = self.model.betas.device        
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T # DDIM inversion noise
        if precision is not None:
            if precision == 16:
                img = img.to(dtype=torch.float16)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        if msa == MSAType.MASACTRL:
            msa_tracker = MSATracker(start_step=4, start_layer=10) # from MasaCtrl
        elif msa == MSAType.PIX_2_VIDEO:
            msa_tracker = MSATracker(start_step=0, start_layer=7) # from Pix2Video
        elif msa == MSAType.ALL:
            msa_tracker = MSATracker(start_step=0, start_layer=0) # all steps and layers
        else:
            msa_tracker = None

        # cond_copy, unconditional_conditioning_copy = copy.deepcopy(cond), copy.deepcopy(unconditional_conditioning)
        for i, step in enumerate(iterator):

            if msa_tracker is not None:
                msa_tracker.cur_step = i

            index = total_steps - i - 1

            ts = torch.full((b,), step, device=device, dtype=torch.long)

            ## use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask_move is not None and mask_static is not None:
                assert first_conds_z0 is not None and prev_conds_z0 is not None

                ms = mask_static.clamp(0, 1)
                md = 1.0 - ms

                bg = first_conds_z0[i]
                fg = prev_conds_z0[i]
                # mix = (bg + fg) / 2

                # background guidance
                if i < 0.8 * total_steps:
                    w_static = 1
                    alpha_bg = w_static * ms
                    img = img + alpha_bg * (bg - img)

                # dynamic guidance
                if i < 0.1 * total_steps:
                    w_dynamic = 1
                    alpha_dyn = w_dynamic * md
                    img = img + alpha_dyn * (fg - img)

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      mask_move=mask_move, mask_static=mask_static,x0=x0,fs=fs, guidance_rescale=guidance_rescale, msa_tracker=msa_tracker,
                                      **kwargs)
            

            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None,mask=None,x0=None,
                      guidance_rescale=0.0, msa_tracker=None, **kwargs):
        b, *_, device = *x.shape, x.device
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:

            if msa_tracker is None:
                model_output = self.model.apply_model(x, t, c, **kwargs)  # unet denoiser
            else:
                sa_collect = [] if self.first_run else None
                sa_inject = self.all_sa_collect_cond[msa_tracker.cur_step].copy() if not self.first_run else None
                assert sa_collect is None or sa_inject is None, "cant have both"
                msa_tracker.reset_att_layer()

                model_output = self.model.apply_model(x, t, c, sa_collect=sa_collect, sa_inject=sa_inject, msa_tracker=msa_tracker, **kwargs) # unet denoiser

                if self.first_run:
                    self.all_sa_collect_cond.append(sa_collect)

        else:

            ### do_classifier_free_guidance
            if isinstance(c, torch.Tensor) or isinstance(c, dict):

                if msa_tracker is None: # normal diffusion

                    e_t_cond = self.model.apply_model(x, t, c, **kwargs)
                    e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)

                else:
                    sa_collect = [] if self.first_run else None
                    sa_inject = self.all_sa_collect_cond[msa_tracker.cur_step].copy() if not self.first_run else None
                    assert sa_collect is None or sa_inject is None, "cant have both"

                    msa_tracker.reset_att_layer()
                    e_t_cond = self.model.apply_model(x, t, c, sa_collect=sa_collect, sa_inject=sa_inject, msa_tracker=msa_tracker, **kwargs)

                    if self.first_run:
                        self.all_sa_collect_cond.append(sa_collect)

                    sa_collect = [] if self.first_run else None
                    sa_inject = self.all_sa_collect_uncond[msa_tracker.cur_step].copy() if not self.first_run else None
                    assert sa_collect is None or sa_inject is None, "cant have both"

                    msa_tracker.reset_att_layer()
                    e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, sa_collect=sa_collect, sa_inject=sa_inject, msa_tracker=msa_tracker, **kwargs)

                    if self.first_run:
                        self.all_sa_collect_uncond.append(sa_collect)

            else:
                raise NotImplementedError

            model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

            if guidance_rescale > 0.0:
                model_output = rescale_noise_cfg(model_output, e_t_cond, guidance_rescale=guidance_rescale)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if self.model.use_dynamic_rescale:
            scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
            prev_scale_t = torch.full(size, self.ddim_scale_arr_prev[index], device=device)
            rescale = (prev_scale_t / scale_t)
            pred_x0 *= rescale

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)
