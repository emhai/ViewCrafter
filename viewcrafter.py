import sys

from local_utils.mask_utils import create_frame_diff_masks, clean_mask
from local_utils.metric_utils import run_metrics

sys.path.append('./extern/dust3r')
sys.path.append('./extern/mast3r')

from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from mast3r.model import AsymmetricMASt3R

import glob

from pytorch3d.structures import Pointclouds

from local_utils.pvd_utils import *
from local_utils.v2v_utils import *
from local_utils.gaussians4d_utils import *
from local_utils.easi3r_utils import *
from local_utils.visualization_utils import *

from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from local_utils.diffusion_utils import instantiate_from_config, load_model_checkpoint, image_guided_synthesis, guided_DDIM_inversion
from pathlib import Path
from torchvision.utils import save_image
import time
import pickle

class ViewCrafter:
    def __init__(self, opts, timer, gradio = False):
        self.opts = opts
        self.device = opts.device

        is_v2v = self.opts.mode in ['single_video_interp', 'multi_video_interp']
        if is_v2v:

            self.run_number = 0                 # which run currently
            self.timer = timer                  # timer for result_csv file
            self.predicted_poses = None         # pred poses for dust3r with multiple cams
            self.predicted_focals = None        # pred focals for dust3r with multiple cams
            self.guidance_image = None          # cross-attention guidance. Set to first picture in first run

            self.prev_image = None              # for mask creation -> not used when easi3r mask
            self.prev_latents = None            # for latent blending
            self.first_image = None             # for mask creation -> not used when easi3r mask
            self.first_latents = None           # for latent blending
            self.mask_type = MaskType[self.opts.mask_type.upper()]
            self.msa_type = MSAType[self.opts.msa.upper()] if self.opts.msa is not None else None
            self.radius = None                  # static radius to keep trajectory same even if point clouds differ
            self.DDIM_noise = None              # starting noise for subsequent runs - DDIM inversion

            assert os.path.isdir(self.opts.image_dir)

            self.base_dir = Path(self.opts.save_dir)
            setup_structure(self.base_dir, Path(self.opts.image_dir), Path(self.opts.gt_dir))

            with self.timer.time("easi3r"):
                run_easi3r_from_viewcrafter(self.base_dir, self.opts.n_frames) # stores masks and pickles to folders

        if self.opts.use_mast3r:
            self.setup_mast3r()
        else:
            self.setup_dust3r()

        self.setup_diffusion()
        self.ddim_sampler = DDIMSampler(self.diffusion) #if not multiple_cond_cfg else DDIMSampler_multicond(model)

        # initialize ref images, pcd
        if not gradio and not is_v2v:
            if os.path.isfile(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images)
            elif os.path.isdir(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_dir(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images, clean_pc = True)    
            else:
                print(f"{self.opts.image_dir} doesn't exist")           
        
    def run_dust3r(self, input_images,clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)

        if self.opts.set_position and self.predicted_poses is not None:
            print("Found predicted camera poses")
            scene.preset_pose(self.predicted_poses)
            scene.preset_focal(self.predicted_focals)
            init_string = "known_poses"
        else:
            init_string = "mst"

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init=init_string, niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        if self.opts.set_position and self.predicted_poses is None:
            print("Saving predicted camera poses")
            self.predicted_poses = scene.get_im_poses().detach().cpu()
            self.predicted_focals = scene.get_focals().detach().cpu()

        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

    def render_pcd(self, pts3d ,imgs, masks, views, renderer, device,nbv=False):
        
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)

        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)

        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else: 
            view_masks = None

        return images, view_masks
    
    def run_render(self, pcd, imgs, masks, H, W, camera_traj,num_views,nbv=False):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views,renderer, self.device,nbv=False)
        return render_results, viewmask

    def run_diffusion(self, renderings, masks=None):

        prompts = [self.opts.prompt]
        videos = (renderings * 2.0 - 1.0).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        condition_index = [0]

        latents = None

        if self.opts.use_latent_blending:
            if self.mask_type in [MaskType.COMP_WITH_PREV, MaskType.EASI3R_PREV]:
                latents = self.prev_latents
            elif self.mask_type in [MaskType.COMP_WITH_FIRST, MaskType.EASI3R_FIRST]:
                latents = self.first_latents

        guidance_image = self.guidance_image if self.opts.reuse_guidance_image else None
        if self.opts.use_latent_blending and latents is not None:
            print("run_number:", self.run_number)
            print("type(latents):", type(latents))
            if isinstance(latents, dict):
                print("latents keys:", latents.keys())
            else:
                print("len(latents):", len(latents))
                print("latents[0].shape:", latents[0].shape)
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_samples, current_x0, intermediates = image_guided_synthesis(
                                                                            self.diffusion,
                                                                            prompts,
                                                                            videos,
                                                                            self.noise_shape,
                                                                            self.opts.n_samples,
                                                                            self.opts.ddim_steps,
                                                                            self.opts.ddim_eta,
                                                                            self.opts.temperature,
                                                                            self.opts.unconditional_guidance_scale,
                                                                            self.opts.cfg_img,
                                                                            self.opts.frame_stride,
                                                                            self.opts.text_input,
                                                                            self.opts.multiple_cond_cfg,
                                                                            self.opts.timestep_spacing,
                                                                            self.opts.guidance_rescale,
                                                                            condition_index,
                                                                            guidance_image=guidance_image,
                                                                            latents=latents,
                                                                            only_x0=False,
                                                                            mask=masks,
                                                                            x_T=self.DDIM_noise,
                                                                            ddim_sampler=self.ddim_sampler,
                                                                            msa=self.msa_type
            )

            self.prev_latents = intermediates
            self.ddim_sampler.first_run = False

            if self.run_number == 0:
                if self.opts.visualize_latents:
                    with self.timer.time("visualize_latents"):
                        visualize_latents(self.base_dir, intermediates, self.diffusion)

                self.first_latents = intermediates

                if self.opts.use_ddim_inversion:
                    with self.timer.time("ddim"):
                        self.DDIM_noise, intermediates = guided_DDIM_inversion(
                            self.diffusion,
                            videos,
                            current_x0,
                            self.guidance_image,
                            prompts,
                            self.ddim_sampler,
                            self.opts.ddim_steps,
                            self.opts.ddim_eta,
                            self.opts.unconditional_guidance_scale,
                        )
                    if self.opts.visualize_latents:
                        with self.timer.time("visualize_latents"):
                            visualize_latents(self.base_dir, intermediates["x_inter"], self.diffusion, "ddim")

        return torch.clamp(batch_samples[0][0].permute(1, 2, 3, 0), -1.0, 1.0)

    def complete_mask_creation(self, point_cloud, images, height, width, trajectory, no_views):

        if self.run_number == 0:
            return None

        # binary_masks are either: difference between current and previous frame, difference between current and first
        # frame, loaded dynamic mask from easi3r - depends on self.mask_type. masks of shape (1, 1, H /2, W/2) which
        # is the same dim as point cloud created by dust3r
        binary_masks = self.create_binary_masks()
        mask_save_dir = Path(self.opts.save_dir) / MASKS_DIR
        # masked_render_results are the masks + point maps from duster, rendered to the calculated camera trajectory
        masked_render_results, viewmask = self.run_render(point_cloud, images, binary_masks, height, width, trajectory, no_views)
        masked_render_results = F.interpolate(masked_render_results.permute(0, 3, 1, 2), size=(self.opts.height, self.opts.width),
                                       mode='bilinear',
                                       align_corners=False).permute(0, 2, 3, 1)
        save_video(masked_render_results, str(mask_save_dir / 'masked_render.mp4'), str(mask_save_dir / "masked_render_results"))
        visualize_masks_horizontal(masked_render_results, mask_save_dir / "diff_masks_all.png")

        # boolean_masks are the masked_render_results, thresholded to [0, 1]
        boolean_masks = self.rendered_mask_to_binary(masked_render_results)
        visualize_masks_horizontal(boolean_masks, mask_save_dir / "bool_masks_all.png", cmap='Greys')

        cleaned = []
        for i in range(boolean_masks.shape[0]):
            single_mask = boolean_masks[i]
            mask_np = single_mask.detach().cpu().numpy().astype(np.uint8) * 255
            cleaned_mask = clean_mask(mask_np)
            cleaned.append(torch.from_numpy(cleaned_mask > 127))  # threshold back to bool

        cleaned_masks = torch.stack(cleaned, dim=0).to(self.device)
        visualize_masks_horizontal(cleaned_masks, mask_save_dir / "cleaned_masks.png", cmap='Greys')

        # float_cleaned_mask = cleaned_masks.float()
        # float_cleaned_mask = float_cleaned_mask * 0.9
        # visualize_masks_horizontal(float_cleaned_mask, mask_save_dir / "float_cleaned_masks.png", cmap='grey')

        # latent_masks are the boolean_masks downsampled to latent shape
        latent_masks = self.binary_mask_to_latent(cleaned_masks)
        visualize_masks_horizontal(latent_masks.squeeze(), mask_save_dir / "latent_masks_all.png", cmap='Greys')


        return latent_masks

    def create_binary_masks(self, easi3r_path=None):

        current_image = self.img_ori
        mask_save_path = Path(self.opts.save_dir) / MASKS_DIR

        if self.mask_type in [MaskType.EASI3R_PREV, MaskType.EASI3R_FIRST]:
            prev_mask_dir = self.base_dir / EASI3R_MASKS_DIR / str(self.run_number - 1)
            mask_dir = self.base_dir / EASI3R_MASKS_DIR / str(self.run_number)
            assert prev_mask_dir.exists() and prev_mask_dir.is_dir() and len(list(prev_mask_dir.iterdir())) != 0
            assert mask_dir.exists() and mask_dir.is_dir() and len(list(mask_dir.iterdir())) != 0
            prev_mask_folders = sorted(prev_mask_dir.iterdir())
            mask_folders = sorted(mask_dir.iterdir())
            return load_easi3r_masks(mask_folders, prev_mask_folders, current_image, H=self.opts.height // 2, W=self.opts.width // 2, output_dir=mask_save_path)

        if self.mask_type == MaskType.COMP_WITH_FIRST:
            return create_frame_diff_masks(self.first_image, current_image, output_dir=mask_save_path, threshold=0.01)

        if self.mask_type == MaskType.COMP_WITH_PREV:
            return create_frame_diff_masks(self.prev_image, current_image, output_dir=mask_save_path, threshold=0.01)

        return None

    def rendered_mask_to_binary(self, rendered_mask):

        threshold = 1e-6
        return (rendered_mask.abs() > threshold).any(dim=-1)

    def binary_mask_to_latent(self, binary_mask):

        _, _, n, h, w = self.noise_shape
        binary_mask = binary_mask.float()
        binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)

        mask_latent = F.interpolate(
            binary_mask,
            size=(n, h, w),
            mode='nearest'
        )

        return mask_latent

    def nvs_single_view(self, gradio=False):
        # 最后一个view为 0 pose
        c2ws = self.scene.get_im_poses().detach()[1:]
        principal_points = self.scene.get_principal_points().detach()[1:] #cx cy
        focals = self.scene.get_focals().detach()[1:]
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2] #以图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## change coordinate
        c2ws, pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)

        imgs = np.array(self.scene.imgs)

        masks = None

        if self.opts.mode == 'single_view_nbv':
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 2
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0],num_candidates, self.device)
            _, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, candidate_poses,num_candidates)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image( viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask0_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv
        elif self.opts.mode == 'single_view_target':
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0], self.opts.d_r[0],self.opts.d_x[0]*depth_avg/focals.item(),self.opts.d_y[0]*depth_avg/focals.item(),self.opts.video_length, self.device)
        elif self.opts.mode == 'single_view_txt':
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else:
                phi, theta, r = self.gradio_traj
            camera_traj,num_views = generate_traj_txt(c2ws, H, W, focals, principal_points, phi, theta, r,self.opts.video_length, self.device,viz_traj=True, save_dir = self.opts.save_dir)
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")

        render_results, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, camera_traj, num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1]==0. and theta[-1]==0. and r[-1]==0.:
                render_results[-1] = self.img_ori

        save_video(render_results, os.path.join(self.opts.save_dir, 'render0.mp4'))
        save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd0.ply') , mask_pc=False, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion0.mp4'))

        return diffusion_results

    def get_pickle_vals(self):

        t_shape = self.images[0]['true_shape']
        t_H, t_W = int(t_shape[0][0]), int(t_shape[0][1])

        pickle_dir = Path(self.base_dir) / PICKLES_DIR
        pickle_file = list(pickle_dir.rglob("*.pkl"))[0]
        with open(pickle_file, 'rb') as f:
            pickle_im_poses = pickle.load(f)
            pickle_im_poses = pickle_im_poses[self.run_number].unsqueeze(0)

            pickle_principal_points = pickle.load(f)
            pickle_principal_points = pickle_principal_points[self.run_number].unsqueeze(0)  # wrong pp from easi3r since resolution is different
            pickle_principal_points = torch.tensor([[t_W // 2., t_H // 2.]], dtype=torch.float32,
                                                   device=self.device)  # todo, always right?

            pickle_focals = pickle.load(f)
            pickle_focals = pickle_focals[self.run_number].unsqueeze(0)

            crop = CenterCrop((t_H, t_W))

            pickle_pts3d = pickle.load(f)
            pickle_pts3d = pickle_pts3d[self.run_number]  # H, W, C
            pickle_pts3d = pickle_pts3d.permute(2, 0, 1)  # C, H, W
            pickle_pts3d = crop(pickle_pts3d)
            pickle_pts3d = pickle_pts3d.permute(1, 2, 0)  # H, W, C

            pickle_depths = pickle.load(f)
            pickle_depths = pickle_depths[self.run_number]
            pickle_depths = crop(pickle_depths)

            pickle_imgs = pickle.load(f)
            pickle_imgs = pickle_imgs[self.run_number]
            pickle_imgs_tensor = torch.from_numpy(pickle_imgs)
            pickle_imgs_tensor = pickle_imgs_tensor.permute(2, 0, 1)
            pickle_imgs_tensor = crop(pickle_imgs_tensor)
            pickle_imgs_tensor = pickle_imgs_tensor.permute(1, 2, 0)
            pickle_imgs = pickle_imgs_tensor.numpy()

        return pickle_imgs, pickle_im_poses, pickle_principal_points, pickle_focals, pickle_pts3d, pickle_depths

    def nvs_single_view_v2v(self, gradio=False):
        # 最后一个view为 0 pose
        # todo cleanup

        if self.opts.use_easi3r:
            print("Using Easi3r for PC")
            pickle_imgs, c2ws, principal_points, focals, pickle_pts3d, pickle_depths = self.get_pickle_vals()

            shape = pickle_imgs.shape
            H, W = int(shape[0]), int(shape[1])

            pcd = [pickle_pts3d, pickle_pts3d]  # emulate result from original VC-run
            depth = [pickle_depths, pickle_depths]

            depth_avg = depth[-1][H // 2, W // 2]  # 以图像中心处的depth(z)为球心旋转
            if self.radius is None:
                self.radius = depth_avg * self.opts.center_scale  # 缩放调整

            ## change coordinate
            c2ws, pcd = world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=0, r=self.radius,
                                           elevation=self.opts.elevation, device=self.device)
            imgs = np.array([pickle_imgs, pickle_imgs])
        else:
            print("Using dust3r for PC")
            c2ws = self.scene.get_im_poses().detach()[1:]
            principal_points = self.scene.get_principal_points().detach()[1:]  # cx cy
            focals = self.scene.get_focals().detach()[1:]
            shape = self.images[0]['true_shape']
            H, W = int(shape[0][0]), int(shape[0][1])
            pcd = [i.detach() for i in
                   self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)]  # a list of points of size whc
            depth = [i.detach() for i in self.scene.get_depthmaps()]
            depth_avg = depth[-1][H // 2, W // 2]  # 以图像中心处的depth(z)为球心旋转
            radius = depth_avg * self.opts.center_scale  # 缩放调整

            ## change coordinate
            c2ws, pcd = world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius,
                                           elevation=self.opts.elevation, device=self.device)
            imgs = np.array(self.scene.imgs)


        masks = None

        if self.opts.mode == 'single_view_nbv':
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 2
            candidate_poses, thetas, phis = generate_candidate_poses(c2ws, H, W, focals, principal_points,
                                                                     self.opts.d_theta[0], self.opts.d_phi[0],
                                                                     num_candidates, self.device)
            _, viewmask = self.run_render([pcd[-1]], [imgs[-1]], masks, H, W, candidate_poses, num_candidates)
            nbv_id = torch.argmin(viewmask.sum(dim=[1, 2, 3])).item()
            save_image(viewmask.permute(0, 3, 1, 2),
                       os.path.join(self.opts.save_dir, f"candidate_mask0_nbv{nbv_id}.png"), normalize=True,
                       value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]
            # generate camera trajectory from T_curr to T_nbv
            camera_traj, num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, theta_nbv, phi_nbv,
                                                             self.opts.d_r[0], self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv
        elif self.opts.mode == 'single_view_target':
            camera_traj, num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta[0],
                                                             self.opts.d_phi[0], self.opts.d_r[0],
                                                             self.opts.d_x[0] * depth_avg / focals.item(),
                                                             self.opts.d_y[0] * depth_avg / focals.item(),
                                                             self.opts.video_length, self.device)
        elif self.opts.mode == 'single_view_txt':
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else:
                phi, theta, r = self.gradio_traj
            camera_traj, num_views = generate_traj_txt(c2ws, H, W, focals, principal_points, phi, theta, r,
                                                       self.opts.video_length, self.device, viz_traj=True,
                                                       save_dir=self.opts.save_dir)
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")

        render_results, viewmask = self.run_render([pcd[-1]], [imgs[-1]], masks, H, W, camera_traj, num_views)
        render_results = F.interpolate(render_results.permute(0, 3, 1, 2), size=(self.opts.height, self.opts.width), mode='bilinear',
                                       align_corners=False).permute(0, 2, 3, 1)


        render_results[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1] == 0. and theta[-1] == 0. and r[-1] == 0.:
                render_results[-1] = self.img_ori

        save_video(render_results, os.path.join(self.opts.save_dir, 'render.mp4'),
                   os.path.join(self.opts.save_dir, RENDER_FRAMES))
        save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None,
                                     save_path=os.path.join(self.opts.save_dir, 'pcd.ply'), mask_pc=False,
                                     reduce_pc=False)

        latent_masks = self.complete_mask_creation([pcd[-1]], [imgs[-1]], H, W, camera_traj, num_views)

        with self.timer.time("diffusion"):
            diffusion_results = self.run_diffusion(render_results, latent_masks)

        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion.mp4'),
                   os.path.join(self.opts.save_dir, DIFFUSION_FRAMES))
        return diffusion_results

    def nvs_sparse_view(self,iter):

        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[0][H//2,W//2] #以ref图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## masks for cleaner point cloud
        self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
        masks = self.scene.get_masks()
        depth = self.scene.get_depthmaps()
        bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
        masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
        masks = to_numpy(masks_new)

        ## render, 从c2ws[0]即ref image对应的相机开始
        imgs = np.array(self.scene.imgs)

        if self.opts.mode == 'single_view_ref_iterative':
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=0, r=radius, elevation=self.opts.elevation, device=self.device)
            camera_traj,num_views = generate_traj_specified(c2ws[0:1], H, W, focals[0:1], principal_points[0:1], self.opts.d_theta[iter], self.opts.d_phi[iter], self.opts.d_r[iter],self.opts.video_length, self.device)
            render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = self.img_ori
        elif self.opts.mode == 'single_view_1drc_iterative':
            self.opts.elevation -= self.opts.d_theta[iter-1]
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
            camera_traj,num_views = generate_traj_specified(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], self.opts.d_theta[iter], self.opts.d_phi[iter], self.opts.d_r[iter],self.opts.video_length, self.device)
            render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = (self.images[-1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.
        elif self.opts.mode == 'single_view_nbv':
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 3
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], self.opts.d_theta[0], self.opts.d_phi[0], num_candidates, self.device)
            _, viewmask = self.run_render(pcd, imgs,masks, H, W, candidate_poses,num_candidates,nbv=True)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image(viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask{iter}_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]   
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv    
            render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = (self.images[-1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. 
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")

        save_video(render_results, os.path.join(self.opts.save_dir, f'render{iter}.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd{iter}.ply') , mask_pc=True, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion{iter}.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results
    
    def nvs_sparse_view_interp(self):

        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]

        if len(self.images) == 2:
            masks = None
            mask_pc = False
        else:
            ## masks for cleaner point cloud
            self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
            masks = self.scene.get_masks()
            depth = self.scene.get_depthmaps()
            bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
            masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
            masks = to_numpy(masks_new)
            mask_pc = True

        imgs = np.array(self.scene.imgs)

        camera_traj,num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
        render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        for i in range(len(self.img_ori)):
            render_results[i*(self.opts.video_length - 1)] = self.img_ori[i]
        save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd.ply') , mask_pc=mask_pc, reduce_pc=False)

        diffusion_results = []
        print(f'Generating {len(self.img_ori)-1} clips\n')
        for i in range(len(self.img_ori)-1 ):
            print(f'Generating clip {i} ...\n')
            diffusion_results.append(self.run_diffusion(render_results[i*(self.opts.video_length - 1):self.opts.video_length+i*(self.opts.video_length - 1)]))
        print(f'Finish!\n')
        diffusion_results = torch.cat(diffusion_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results

    def nvs_sparse_view_interp_v2v(self):

        # todo cleanup, add easi3r pointclouds?
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)]  # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]

        if len(self.images) == 2:
            masks = None
            mask_pc = False
        else:
            ## masks for cleaner point cloud
            self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
            masks = self.scene.get_masks()
            depth = self.scene.get_depthmaps()
            bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
            masks_new = [m+mb for m, mb in zip(masks,bgs_mask)]
            masks = to_numpy(masks_new)
            mask_pc = True

        imgs = np.array(self.scene.imgs)
        no_cameras = len(self.img_ori)

        if no_cameras > 2:
            camera_traj, num_views = generate_traj_interp_closed(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
            print("SAME?", c2ws, H, W, focals, principal_points)
        else:
            camera_traj, num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
            print("SAME?", c2ws, H, W, focals, principal_points)


        render_results, viewmask = self.run_render(pcd, imgs, masks, H, W, camera_traj, num_views)
        render_results = F.interpolate(render_results.permute(0, 3, 1, 2), size=(self.opts.height, self.opts.width), mode='bilinear',   align_corners=False).permute(0, 2, 3, 1)

        if no_cameras > 2:
            indices = [i * (self.opts.video_length // no_cameras) for i in range(no_cameras + 1)]
            cams = list(range(no_cameras)) + [0]
        else:
            indices = [0, self.opts.video_length - 1]
            cams = [0, 1]

        for i, j in zip(indices, cams):
            render_results[i] = self.img_ori[j]

        save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'), os.path.join(self.opts.save_dir, RENDER_FRAMES))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd.ply'), mask_pc=mask_pc, reduce_pc=False)

        latent_masks = self.complete_mask_creation(pcd, imgs, H, W, camera_traj, num_views)
        print(latent_masks)
        with self.timer.time("diffusion"):
            diffusion_results = self.run_diffusion(render_results, latent_masks)

        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'),
                   os.path.join(self.opts.save_dir, DIFFUSION_FRAMES))
        torch.Size([25, 576, 1024, 3])
        return diffusion_results

    def nvs_single_view_eval(self):

        # get camera trajectory of the input frames
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        c2ws,pcd =  world_point_to_kth(poses=c2ws, points=torch.stack(pcd), k=0, device=self.device)
        camera_traj,num_views = generate_traj(c2ws, H, W, focals, principal_points, self.device)
        
        # estimate pcd again using only one ref image
        images_ref = [self.images[0], copy.deepcopy(self.images[0])]
        images_ref[1]['idx'] = 1
        self.run_dust3r(input_images=images_ref)
        pcd_ref = self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)[0].detach()
        img_ref = np.array(self.scene.imgs)[0]
        masks = None

        render_results, viewmask = self.run_render([pcd_ref], [img_ref],masks, H, W, camera_traj,num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori[0]
        save_video(render_results, os.path.join(self.opts.save_dir, f'render_ref0.mp4'))
        diffusion_results = self.run_diffusion(render_results)

        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion_ref0.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results

    def nvs_single_view_ref_iterative(self):

        all_results = []
        sample_rate = 6
        idx = 1 #初始包含1张ref image
        for itr in range(0, len(self.opts.d_phi)):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def nvs_single_view_1drc_iterative(self):

        all_results = []
        sample_rate = 6
        idx = 1 #初始包含1张ref image
        for itr in range(0, len(self.opts.d_phi)):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def nvs_single_view_nbv(self):
        # lef and right
        # d_theta and a_phi 是搜索空间的顶点间隔
        all_results = []
        ## FIXME: hard coded
        sample_rate = 6
        max_itr = 3

        idx = 1 #初始包含1张ref image
        for itr in range(0, max_itr):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def run_video_interp(self, mode):

        input_dir = self.base_dir / INPUTS_DIR  # all inputs
        results_dir = self.base_dir / RESULTS_DIR  # all results

        all_frames = [x.name for x in sorted(input_dir.iterdir(), key=lambda x: int(x.stem))]
        all_frames = all_frames[:self.opts.n_frames] # todo assert that n_frames < input_vid_frames

        if mode == "single":
            self.opts.mode = 'single_view_txt'  # necessary for inner functions - txt needs to be provided todo also different kinds possible, maybe with self.opts.mode.contains()

        print(all_frames)
        for frame in all_frames:
            print("running frame", int(frame) + 1, "/", len(all_frames), "run_no: ", self.run_number)
            start = time.time()

            current_input_dir = input_dir / frame
            current_result_dir = results_dir / frame

            if mode == "single":
                current_input_dir, = current_input_dir.glob("*") # only one file in directory, choose that

            self.opts.image_dir = str(current_input_dir)
            self.opts.save_dir = str(current_result_dir)

            os.mkdir(self.opts.save_dir)

            if mode == "single":
                self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
                if not self.opts.use_easi3r or dir_empty(self.base_dir / PICKLES_DIR):
                    with self.timer.time("dust3r"):
                        self.run_dust3r(input_images=self.images)

            else: # mode == "multi"
                self.images, self.img_ori = self.load_initial_dir(image_dir=self.opts.image_dir)
                with self.timer.time("dust3r"):
                    self.run_dust3r(input_images=self.images, clean_pc=True) # if single, pc is from easi3r

            if self.run_number == 0:
                self.first_image = self.img_ori
                self.setup_guidance()

            if mode == "single":
                self.nvs_single_view_v2v()
            else: # mode == "multi"
                self.nvs_sparse_view_interp_v2v()

            self.prev_image = self.img_ori

            end = time.time()
            time_per_frame = (end - start) / 60
            remaining_time = time_per_frame * (len(all_frames) - int(frame) - 1)
            print("elapsed time: {:.2f}min, est.remaining time: {:.2f}min, {:.2f}h\n".format(time_per_frame,
                                                                            remaining_time, remaining_time / 60))
            self.run_number += 1

        separate_cameras(self.base_dir, DIFFUSION_FRAMES)
        separate_cameras(self.base_dir, RENDER_FRAMES)

        print("Cleaning GPU up")
        torch.cuda.synchronize()  # finish kernels
        torch.cuda.empty_cache()  # release cached blocks to the driver
        torch.cuda.ipc_collect()  # clean IPC memory
        del self.diffusion

        if self.opts.gt_dir is not None:
            with self.timer.time("metrics"):
                run_metrics(self.base_dir)


    def setup_diffusion(self):
        seed_everything(self.opts.seed)

        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        print("instantiating: ", model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        #print_diffusion_model(model, max_depth=7)

        h, w = self.opts.height // 8, self.opts.width // 8 # latent size
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def setup_dust3r(self):
        assert "DUSt3R" in self.opts.model_path, "wrong checkpoint"
        self.dust3r = load_model(self.opts.model_path, self.device)

    def setup_mast3r(self):
        assert "MASt3R" in self.opts.model_path, "wrong checkpoint"
        self.dust3r = AsymmetricMASt3R.from_pretrained(self.opts.model_path).to(self.device)

    def setup_guidance(self):
        if isinstance(self.img_ori, list):
            guidance_image = self.img_ori[0]
        else:
            guidance_image = self.img_ori
        self.guidance_image = (guidance_image * 2. -1.).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def load_initial_images(self, image_dir):
        ## load images
        ## dict_keys(['img', 'true_shape', 'idx', 'instance', 'img_ori']),张量形式

        images = load_images([image_dir], size=512, force_1024 = True, force_height = self.opts.height, force_width = self.opts.width)
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1

        return images, img_ori

    def load_initial_dir(self, image_dir):

        image_files = glob.glob(os.path.join(image_dir, "*"))

        if len(image_files) < 2:
            raise ValueError("Input views should not less than 2.")
        image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images = load_images(image_files, size=512, force_1024 = True, force_height = self.opts.height, force_width = self.opts.width)

        img_gts = []
        for i in range(len(image_files)):
            img_gts.append((images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.) 

        return images, img_gts

    def run_gradio(self,i2v_input_image, i2v_elevation, i2v_center_scale, i2v_d_phi, i2v_d_theta, i2v_d_r, i2v_steps, i2v_seed):
        self.opts.elevation = float(i2v_elevation)
        self.opts.center_scale = float(i2v_center_scale)
        self.opts.ddim_steps = i2v_steps
        self.gradio_traj = [float(i) for i in i2v_d_phi.split()],[float(i) for i in i2v_d_theta.split()],[float(i) for i in i2v_d_r.split()]
        seed_everything(i2v_seed)
        torch.cuda.empty_cache()
        img_tensor = torch.from_numpy(i2v_input_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        img_tensor = (img_tensor / 255. - 0.5) * 2

        image_tensor_resized = center_crop_image(img_tensor) #1,3,h,w
        images = get_input_dict(image_tensor_resized,idx = 0,dtype = torch.float32)
        images = [images, copy.deepcopy(images)]
        images[1]['idx'] = 1
        self.images = images
        self.img_ori = (image_tensor_resized.squeeze(0).permute(1,2,0) + 1.)/2.

        # self.images: torch.Size([1, 3, 288, 512]), [-1,1]
        # self.img_ori:  torch.Size([576, 1024, 3]), [0,1]
        # self.images, self.img_ori = self.load_initial_images(image_dir=i2v_input_image)
        self.run_dust3r(input_images=self.images)
        self.nvs_single_view(gradio=True)

        traj_dir = os.path.join(self.opts.save_dir, "viz_traj.mp4")
        gen_dir = os.path.join(self.opts.save_dir, "diffusion0.mp4")
        
        return traj_dir, gen_dir