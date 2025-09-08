import json
import operator
from time import time
import numpy as np
import os
import itertools
from pathlib import Path
from pyquaternion import Quaternion
import PIL.Image as pil
from os import path
import time
from functools import lru_cache
import cv2
from det3d.datasets.nuscenes.clustering import paint_by_DBSCAN_per_instance

@lru_cache(maxsize=4096)
def _load_paint_npz(npz_path):
    return np.load(npz_path)

from nuscenes import NuScenes

from det3d.datasets.base import BaseDataset

from det3d.datasets.nuscenes.nusc_common import (
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)

from det3d.datasets.nuscenes.fusion_utils import project_points
from nuscenes.utils.data_classes import LidarPointCloud, PointCloud

class NuScenesDataset(BaseDataset):

    def __init__(self,
                 info_path,
                 root_path,
                 nsweeps,
                 sampler=None,
                 loading_pipelines=None,
                 augmentation=None,
                 prepare_label=None,
                 class_names=[],
                 resampling=False,
                 evaluations=None,
                 create_database=False,
                 use_gt_sampling=True,
                 version="v1.0-trainval",
                 fuse_camera=False,
                 cam_name="CAM_FRONT",
                 padding=True,
                 painted_path=None):

        super(NuScenesDataset, self).__init__(
            root_path, info_path, sampler, loading_pipelines, augmentation, prepare_label, evaluations, create_database,
            use_gt_sampling=use_gt_sampling)

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"

        self._class_names = list(itertools.chain(*[t for t in class_names]))
        self.version = version
        self.fuse_camera = fuse_camera
        self.cam_name = cam_name
        self.padding = padding
        self.nusc = NuScenes(version=self.version, dataroot=str(
            self._root_path), verbose=False)
        self.painted_path = painted_path
        self.paint_K = 10

        if resampling:
            self.cbgs()
        print(f"[DEBUG] NuScenesDataset initialized with version {self.version}")

    def cbgs(self):
        _cls_infos = {name: [] for name in self._class_names}
        for info in self.infos:
            for name in set(info["gt_names"]):
                if name in self._class_names:
                    _cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
        _cls_dist = {k: len(v) / duplicated_samples for k,
                     v in _cls_infos.items()}

        _nusc_infos = []

        frac = 1.0 / len(self._class_names)
        ratios = [frac / v for v in _cls_dist.values()]

        for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
            _nusc_infos += np.random.choice(cls_infos,
                                            int(len(cls_infos) * ratio)).tolist()

        self.infos = _nusc_infos

    def read_file(self, path, num_point_feature=4):
        points = np.fromfile(os.path.join(self._root_path, path),
                             dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        return points

    def read_sweep(self, sweep, min_distance=2.0):
        points_sweep = self.read_file(str(sweep["lidar_path"])).T

        nbr_points = points_sweep.shape[1]
        if sweep["transform_matrix"] is not None:
            # print(f"[DEBUG] Applying transform matrix to sweep points")
            points_sweep[:3, :] = sweep["transform_matrix"].dot(
                np.vstack((points_sweep[:3, :], np.ones(nbr_points))))[:3, :]
        points_sweep = self.remove_close(points_sweep, min_distance)
        curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

        return points_sweep.T, curr_times.T

    @staticmethod
    def remove_close(points, radius: float):
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """
        x_filt = np.abs(points[0, :]) < radius
        y_filt = np.abs(points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        points = points[:, not_close]
        x_bulk = np.abs(points[0, :]) < 1.0
        y_bulk = np.abs(points[1, :]) < 6.0
        z_bulk = np.logical_and((points[2, :]) < 0.0, (points[2, :]) > -1.0)
        not_bulk = np.logical_not(np.logical_and(np.logical_and(x_bulk, y_bulk), z_bulk))
        points = points[:, not_bulk]
        return points

    @staticmethod
    def to_struct(arr):
        return arr.view([('', arr.dtype)] * arr.shape[1]).squeeze()

    def read_sweep_from_info(self, info):
        lidar_path = info["lidar_path"]

        points = self.read_file(str(lidar_path))

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]
        # stores the time lag for each point relative to the reference frame
        
        for i in range(len(info["sweeps"])):
            sweep = info["sweeps"][i]
            points_sweep, times_sweep = self.read_sweep(sweep)
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
        # print(f"[DEBUG] Loaded pointcloud with shape {points.shape} and times shape {times.shape}\n")
        return points, times
    
    def load_and_transform_lidar_to_cam(self, nusc: NuScenes, sample, info,
                                    cam_name='CAM_FRONT', pc_full=None, time_lags_full=None,
                                    nsweeps=10):
        """
        Load LIDAR points and transform to camera coordinates
        Args:
            nusc: Nuscenss instance
            sample: Nuscenes sample
            cam_name: Camera name to transform into
            nsweeps: Number of LIDAR sweeps to use

        Returns: LidarPointCloud in the camera coordinates system

        """
        # t0 = time.time()
        pointsensor_token = sample['data']['LIDAR_TOP']
        pointsensor = nusc.get('sample_data', pointsensor_token)
        camera_token = sample['data'][cam_name]
        cam = nusc.get('sample_data', camera_token)
        # t1 = time.time()
        # print(f"[TIME][{cam_name}] image + calibration load: {t1 - t0:.4f}s")
        # pcl_path = path.join(nusc.dataroot, pointsensor['filename'])

        # chan = pointsensor['channel']
        # ref_chan = 'LIDAR_TOP'

        # t2 = time.time()
        # pc, time_lags = self.read_sweep_from_info(info)

        pc_lidar = pc_full.copy()
        time_lags_lidar = time_lags_full.copy()
        # t25 = time.time()
        # print(f"[TIME][{cam_name}] read_sweep_from_info: {t25 - t2:.4f}s")
        
        pc_cam = np.array(pc_full).T
        # print(f"[DEBUG] Loaded pointcloud with shape {pc.shape}\n")
        # pc_cam_old = LidarPointCloud(pc_cam.copy())
        pc_cam = LidarPointCloud(pc_cam)
        
        # t3 = time.time()
        # print(f"[TIME][{cam_name}] pointcloud creation: {t3 - t25:.4f}s")
        
        

        # First step: transform the point-cloud to the ego vehicle frame for the
        # timestamp of the sweep.
        # t4 = time.time()
        cs_record = nusc.get('calibrated_sensor',
                            pointsensor['calibrated_sensor_token'])
        R1 = Quaternion(cs_record['rotation']).rotation_matrix
        T1 = np.array(cs_record['translation'])
        # pc_cam_old.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        # pc_cam_old.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        R2 = Quaternion(poserecord['rotation']).rotation_matrix
        T2 = np.array(poserecord['translation'])
        # pc_cam_old.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        # pc_cam_old.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of
        # the image.
        poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        R3 = Quaternion(poserecord['rotation']).rotation_matrix.T
        T3 = -np.array(poserecord['translation'])
        T3 = R3.dot(T3)
        # pc_cam_old.translate(-np.array(poserecord['translation']))
        # pc_cam_old.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        R4 = Quaternion(cs_record['rotation']).rotation_matrix.T
        T4 = -np.array(cs_record['translation'])
        T4 = R4.dot(T4)
        # pc_cam_old.translate(-np.array(cs_record['translation']))
        # pc_cam_old.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        Atotal = R4.dot(R3.dot(R2.dot(R1)))
        Ttotal = R4.dot(R3.dot(R2.dot(T1) + T2) + T3) + T4
        
        pc_cam.rotate(Atotal)
        pc_cam.translate(Ttotal) 
        

        return pc_cam, pc_lidar, time_lags_lidar

    def get_camera_fused_pointcloud(self, nusc: NuScenes, sample, info,
                                cam_name='CAM_FRONT', pc_full=None, time_lags_full=None,
                                min_dist=1.0,
                                nsweeps=10,
                                fuse_camera=True):
        """
        Loads points from lidar pointcloud, finds the points that are within a
        camera's FOV, and the color of the corresponding points in the camera's
        image. Optionally will estimate depths for more camera pixels based on the
        lidar points and will add them to the returned pointcloud.
        Args:
            nusc: Nuscenes instance
            sample: Nuscenes sample
            cam_name: Name of the camera
            min_dist: Distance in meters below which points will be ignored
            nsweeps: Number of lidar sweeps to use
            fuse_camera: Weather to add color dimension to the pointcloud
            fill_method:
                - None: No depth completion performed
                - 'ipbasic': depth completion based on ip-basic
                - 'knn': depth completion based on KNN regression.
                - 'maskconv': depth completion with masked convolutions.
                This parameter will be ignored if fuse_camera is false

        Returns: PointCloud containing only the points that are within the
        camera's field of view transformed to ego vehicle reference frame.

        Some of the logic in this function is taken from map_pointcloud_to_image
        function in the Nuscenes dev-kit:
        https://github.com/nutonomy/nuscenes-devkit/python-sdk/nuscenes/
            nuscenes.py#L532
        """

        # t0 = time.time()
        pc_cam, pc_lidar, time_lags_cam = self.load_and_transform_lidar_to_cam(nusc, sample, info, cam_name=cam_name, 
                                                                       pc_full=pc_full, time_lags_full=time_lags_full,
                                                                       nsweeps=nsweeps)
        # t1 = time.time()
        # print(f"[TIME][{cam_name}] load_and_transform_lidar_to_cam: {t1 - t0:.4f}s")
        # pc is already in the type of PointCloud
        # pc = PointCloud(pc)
        # pc now is in the camera reference frame

        camera_token = sample['data'][cam_name]
        cam_sd = nusc.get('sample_data', camera_token)
        cs_record = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        
        W_img = cam_sd.get('width', None)
        H_img = cam_sd.get('height', None)
        if W_img is None or H_img is None:
            print("fall back to image fusion")
            im = pil.open(path.join(nusc.dataroot, cam_sd['filename']))
            W_img, H_img = im.size

        p_points, mask = project_points(pc_cam.points[:3, :],
                                        np.array(cs_record['camera_intrinsic']),
                                        (W_img, H_img), min_dist)
        # depths = depths[mask]
        p_points = p_points[:, mask]    # projected points
        # pc.points = pc.points[:, mask]  # masked points in camera FOV
        # time_lags = time_lags[:, mask]
        # print(f"[FUSION DEBUG] shape of pc: {pc.points.shape}, shape of time_lags: {time_lags.shape}\n")
        
        pc_lidar = pc_lidar[mask, :]
        time_lags_cam = time_lags_cam[mask, :]

        # pc = PointCloud(pc.points)
        if not fuse_camera:
            return np.hstack([pc_lidar, time_lags_cam])
        # if fuse_camera:
            # Get colors of the projected points from the RGB image
        npz_path = os.path.join(self.painted_path, f"{camera_token}.npz")
        paint_feats = None
        inst_ids = None
        
        try:
            data = _load_paint_npz(npz_path) if '_load_paint_npz' in globals() else np.load(npz_path)
            S = data['scores']
            H_s, W_s, K = S.shape
            if hasattr(self, 'paint_K') and K != self.paint_K:
                print(f"[WARNING] Inconsistent paint_K: {K} vs {self.paint_K}") 
                pass
            inst_map = data['inst_id'] if 'inst_id' in data else None
            if (H_s != H_img) or (W_s != W_img):
                S = np.stack([
                    cv2.resize(S[..., c].astype(np.float32), (W_img, H_img), interpolation=cv2.INTER_LINEAR)
                    for c in range(S.shape[-1])
                ], axis=-1)
                if inst_map is not None:
                    inst_map = cv2.resize(inst_map.astype(np.int32), (W_img, H_img), interpolation=cv2.INTER_NEAREST)

            xs = p_points[0].astype(np.int32)   # floor all values
            ys = p_points[1].astype(np.int32)
            paint_feats = S[ys, xs, :].astype(np.float32, copy=False)
            inst_ids = inst_map[ys, xs].astype(np.int32, copy=False)
        # im_arr = np.asarray(im)  # shape (H, W, C)

        except FileNotFoundError:
            # Missing .npz: fall back to zeros so pipeline can continue
            print(f"[WARN] paint file missing for {camera_token}: {npz_path}")
            K = getattr(self, 'paint_K', 10)
            Nf = pc_lidar.shape[0]
            paint_feats = np.zeros((pc_lidar.shape[0], K), dtype=np.float32)
            inst_ids = np.zeros((Nf,), dtype=np.int32)

        # 3. Keep idxs within [0..W-1] and [0..H-1]
        # xs = np.clip(xs, 0, im_arr.shape[1] - 1)
        # ys = np.clip(ys, 0, im_arr.shape[0] - 1)

        # print(f"[FUSION DEBUG] colors shape after transpose: {colors.shape}")
        # back to lidar reference frame
        # pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        # pc.translate(np.array(cs_record['translation']))

        # fused_pc = np.vstack([pc.points, time_lags, colors]).T
        
        
        uniq_iids, inv = np.unique(inst_ids, return_inverse=True)   # sorted array of unique instance ids
        # inv: integer array of length N where each entry tells you which unique id bucket that point belongs to
        inst_to_indices = {}
        for u_i, iid in enumerate(uniq_iids):
            idxs = np.nonzero(inv == u_i)[0]
            inst_to_indices[int(iid)] = idxs    # build lists of grouped indices
            
        inst_cluster_results = {}
        for iid, idxs in inst_to_indices.items():
            if idxs.size == 0:
                continue
            if iid == 0:
                continue    # ignore background
            
            _ = paint_by_DBSCAN_per_instance(pc_lidar=pc_lidar, inst_ids=inst_ids, paint_feats=paint_feats,
                                                          inst_to_indices=inst_to_indices, eps=0.3, min_samples=5, 
                                                          selection_mode="largest", cluster_dims="xy")

        # print(f"[FUSION DEBUG] fused_pc shape: {fused_pc.shape}\n")
        fused_pc = np.hstack([pc_lidar, time_lags_cam, paint_feats]).astype(np.float32, copy=False)
        return fused_pc

    def load_pointcloud(self, res, info):
        # t_load_start = time.time()
        
        if not self.fuse_camera:

            points, times = self.read_sweep_from_info(info)

            res["points"] = np.hstack([points, times])
            # print(f"[DEBUG] points.shape={res['points'].shape}")

            return res
        
        else:
            # print(f"[DEBUG] loading pointcloud with camera fusion\n")
            # t0 = time.time()
            all_cam_names = [
                "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
            ]
            
            sample = self.nusc.get('sample', info['token'])
            # t1 = time.time()
            # print(f"[TIME] Loading sample {info['token']} from NuScenes: {t1 - t0:.4f}s")
            pc_full, time_lags_full = self.read_sweep_from_info(info)
            fused_list = []
            for cam in all_cam_names:
                # t_get_fused_pc_start = time.time()
                fused_pts = self.get_camera_fused_pointcloud(
                    nusc= self.nusc, sample= sample, info= info,
                    cam_name= cam, pc_full= pc_full, time_lags_full= time_lags_full,
                    min_dist= 1.0, nsweeps= self.nsweeps,
                    fuse_camera= self.fuse_camera
                )
                fused_list.append(fused_pts)
                # t_get_fused_pc_end = time.time()
                # print(f"[TIME] get_camera_fused_pointcloud({cam}): {t_get_fused_pc_end - t_get_fused_pc_start:.4f}s")
            # t2 = time.time()
            # print(f"[TIME] Total get_camera_fused_pointcloud: {t2 - t1:.4f}s")
            fused_pts = np.concatenate(fused_list, axis=0)
            
            scale = 1000  # same as rounding to 3 decimals for faster unique operation
            xyz_int = np.rint(fused_pts[:, :3] * scale).astype(np.int32)

            # 2) view each triple as a single void-dtype item
            dtype_void = np.dtype((np.void, xyz_int.dtype.itemsize * 3))
            xyz_void = xyz_int.view(dtype_void).ravel()

            # 3) unique on the 1-D void array
            _, unique_indices = np.unique(xyz_void, return_index=True)

            # 4) pick out the deduped points
            fused_pts = fused_pts[unique_indices]
            # t3 = time.time()
            # print(f"[TIME] Camera fusion + dedup: {t3 - t2:.4f}s")
            
            if self.padding:
                # t4 = time.time()
                
                # full_points, time_lags = self.read_sweep_from_info(info)
                full_points = pc_full
                time_lags = time_lags_full
                
                fused_xyz = np.round(fused_pts[:, :3].astype(np.float64), 3)
                full_xyz = np.round(full_points[:, :3].astype(np.float64), 3)
                

                fused_struct = self.to_struct(fused_xyz)
                full_struct = self.to_struct(full_xyz)

                
                mask_not_in_fused = ~np.isin(full_struct, fused_struct)
                
                unseen_points = full_points[mask_not_in_fused]
                unseen_time_lags = time_lags[mask_not_in_fused]
                black_rgb = np.zeros((unseen_points.shape[0], 3), dtype=np.float32)
                
                padded_pts = np.hstack([unseen_points, unseen_time_lags, black_rgb])
                
                all_pts = np.concatenate([fused_pts, padded_pts], axis=0)
                res["points"] = all_pts.astype(np.float32)
                # t5 = time.time()
                # print(f"[TIME] Padding process: {t5 - t4:.4f}s")
            else:
                # save_dir = f"/home/betty/CMU-intern/pillarnext/visualize_pointcloud/clustered"
                # os.makedirs(save_dir, exist_ok=True)
                # filename = os.path.join(save_dir, f"{info['token']}_fused_pts.npz")
                # np.savez_compressed(filename, fused_pts.astype(np.float32))
                # print(f"Saved {info['token']}_fused_pts.npz with shape {fused_pts.shape}")
                res["points"] = fused_pts.astype(np.float32)

            # print(f"[DEBUG] points.shape={res['points'].shape}")
            return res

    def evaluation(self, detections, output_dir=None, testset=False):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        dets = [v for _, v in detections.items()]
        # assert len(dets) == 6019
        print(f"[DEBUG] Got {len(dets)} detections for {len(self._nusc_infos)} samples")



        nusc_annos = {
            "results": {},
            "meta": None,
        }

        nusc = NuScenes(version=version, dataroot=str(
            self._root_path), verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            mapped_class_names.append(n)

        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes, det["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.parked"
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token": det["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({det["token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,
                "detection_cvpr_2019",
                res_path,
                eval_set_map[self.version],
                output_dir,
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"], },
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"], },
            }
            return res['results']
        else:
            return None
