import fire
from det3d.datasets.nuscenes.nusc_common import create_nuscenes_infos 
# from det3d.datasets.waymo.waymo_convert import create_waymo_infos
from create_gt_database import create_groundtruth_database
import torch.multiprocessing as mp


def nuscenes_data_prep(root_path, painted_path, version="v1.0-trainval", nsweeps=10, fuse_camera=False, cam_name="CAM_FRONT", padding=True):
    print("here")
    create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps)
    create_groundtruth_database('NUSC', 
                                root_path, 
                                'infos_train_10sweeps_withvelo_filterZero.pkl',
                                painted_path=painted_path,
                                nsweeps=nsweeps,
                                version=version,
                                fuse_camera=fuse_camera,
                                cam_name=cam_name,
                                padding=padding)


# def waymo_data_prep(root_path, save_path, nsweeps=3):
#     create_waymo_infos(root_path, save_path)
#     create_groundtruth_database('WAYMO', 
#                                 save_path, 
#                                 'waymo_infos_train.pkl',
#                                 nsweeps=nsweeps)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    fire.Fire()