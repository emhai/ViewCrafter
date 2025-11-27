import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.isfile(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out + '\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.10f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'real':
            if isinstance(value, np.ndarray):
                value = value.item()
            self._write('{}: {:.10f}'.format(key, value))  # as accurate as possible
        else:
            raise NotImplementedError

    def read(self, key, dt='mat'):
        node = self.fs.getNode(key)
        if node.empty():
            # Helpful error so you know exactly which key is missing/mistyped
            raise KeyError(f"Key '{key}' not found in file.")

        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'real':
            output = self.fs.getNode(key).real()
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def peek_keys(yml_path, limit=50):
    print(f"Top-level keys in {yml_path}:")
    cnt = 0
    with open(yml_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip()
            if ln.startswith(("%YAML", "---", "..." )):
                continue
            # capture lines like: K_0001:, R_0001:, Rot_0001:, T_0001:, names:
            if ":" in ln and not ln.lstrip().startswith("-"):
                key = ln.split(":", 1)[0].strip()
                if key:
                    print("  ", key)
                    cnt += 1
                    if cnt >= limit:
                        print("  ...")
                        return

def read_camera(intri_path: str, extri_path: str = None, cam_names=[]):

    # rewritten from SelfCap repo
    assert os.path.isfile(intri_path), f"{intri_path} doesnt exist"
    assert os.path.isfile(extri_path), f"{extri_path} doesnt exist"

    intri = FileStorage(intri_path)
    extri = FileStorage(extri_path)
    cam_names = intri.read('names', dt='list')

    peek_keys(intri_path)
    centers, labels = [], []

    for cam in cam_names:
        # Intrinsics
        K = intri.read('K_{}'.format(cam))
        H = int(intri.read('H_{}'.format(cam), dt='real')) or -1
        W = int(intri.read('W_{}'.format(cam), dt='real')) or -1
        invK = np.linalg.inv(K)

        # Extrinsics
        Tvec = extri.read('T_{}'.format(cam))
        Rvec = extri.read('R_{}'.format(cam))
        if Rvec is not None:
            R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read('Rot_{}'.format(cam))
            Rvec = cv2.Rodrigues(R)[0]
        RT = np.hstack((R, Tvec))

        R = R
        T = Tvec
        C = - R.T @ Tvec  # camera center
        RT = RT
        Rvec = Rvec
        P = K @ RT

        # --- collect for plotting ---
        centers.append(C.reshape(-1))
        labels.append(cam)

    # --- plot all centers with names ---
    centers = np.array(centers)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=40)
    for c, lab in zip(centers, labels):
        ax.text(c[0], c[1], c[2], lab, fontsize=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("/media/emmahaidacher/Volume/DATASETS/INTERNET/SelfCap/corgi-release/optimized/cameras.png")
    plt.show()


def main():
    path_to_dataset = '/media/emmahaidacher/Volume/DATASETS/INTERNET/SelfCap/corgi-release/optimized/'
    intrinsics_path = os.path.join(path_to_dataset, 'intri.yml')
    extrinsics_path = os.path.join(path_to_dataset, 'extri.yml')
    read_camera(intrinsics_path, extrinsics_path)

if __name__ == '__main__':
    main()