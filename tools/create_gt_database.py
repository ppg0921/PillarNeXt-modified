import pickle
from pathlib import Path
import os
import numpy as np
from det3d.core import box_np_ops
from tqdm import tqdm
import fire
import time

def create_groundtruth_database(
    dataset_name,
    data_path,
    info_path,
    painted_path,
    used_classes=None,
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    nsweeps=1,
    version="v1.0-trainval",
    fuse_camera=False,
    cam_name="CAM_FRONT",
    padding=True
):
    if dataset_name == "WAYMO":
        from det3d.datasets.waymo.waymo import WaymoDataset
        Dataset = WaymoDataset
    elif dataset_name == "NUSC":
        from det3d.datasets.nuscenes import NuScenesDataset
        Dataset = NuScenesDataset
    pipeline = ['load_pointcloud', 'load_box3d']
    if nsweeps > 1:
        dataset = Dataset(
            loading_pipelines=pipeline,
            nsweeps=nsweeps,
            root_path=data_path,
            info_path=info_path,
            version=version,
            create_database=True,
            fuse_camera=fuse_camera,
            cam_name=cam_name,
            padding=padding,
            painted_path=painted_path
        )
        nsweeps = dataset.nsweeps
    else:
        dataset = Dataset(
            info_path=info_path,
            root_path=data_path,
            loading_pipelines=pipeline,
            nsweeps=1,
            version=version,
            create_database=True,
            fuse_camera=fuse_camera,
            cam_name=cam_name,
            padding=padding,
            painted_path=painted_path
        )
        nsweeps = 1

    root_path = Path(data_path)

    if dataset_name in ["WAYMO", "NUSC"]:
        if db_path is None:
            db_path = root_path / f"gt_database_{nsweeps}sweeps_withvelo_xyzclustered"
        if dbinfo_path is None:
            dbinfo_path = root_path / \
                f"dbinfos_train_{nsweeps}sweeps_withvelo_xyzclustered.pkl"
    else:
        raise NotImplementedError()

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    for index in tqdm(range(len(dataset))):
        t_sample_start = time.time()

        image_idx = index

        # ───── Data loading ─────
        # t0 = time.time()
        sensor_data = dataset[index]
        # t1 = time.time()

        if 'token' in sensor_data:
            image_idx = sensor_data['token']
        points = sensor_data['points']

        annos = sensor_data['annotations']
        gt_boxes = annos['gt_boxes']
        names = annos['gt_names']

        # ───── GT filtering ─────
        # t2 = time.time()
        if gt_boxes.shape[0] == 0:
            continue
        if dataset_name == 'WAYMO':
            if index % 4 != 0:
                mask = (names == 'vehicle')
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]
            if index % 2 != 0:
                mask = (names == 'pedestrian')
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]
        # t3 = time.time()

        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)

        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue

        # ───── Point-in-box test ─────
        # t4 = time.time()
        point_indices = box_np_ops.points_in_rbbox(
            points, gt_boxes[:, [0, 1, 2, 3, 4, 5, -1]]
        )
        # t5 = time.time()

        for i in range(num_obj):
            # t_obj_start = time.time()

            if (used_classes is None) or names[i] in used_classes:
                # ───── File path setup ─────
                # t_path_start = time.time()
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                dirpath = os.path.join(str(db_path), names[i])
                os.makedirs(dirpath, exist_ok=True)
                filepath = os.path.join(str(db_path), names[i], filename)
                # t_path_end = time.time()

                # ───── Extract & center points ─────
                # t_extract_start = time.time()
                gt_points = points[point_indices[:, i]]  # points in box
                gt_points[:, :3] -= gt_boxes[i, :3]
                # t_extract_end = time.time()

                # ───── Write file ─────
                # t_write_start = time.time()
                with open(filepath, "w") as f:
                    try:
                        gt_points.tofile(f)
                    except:
                        print(f"process {index} files")
                        break
                # t_write_end = time.time()

            if (used_classes is None) or names[i] in used_classes:
                # ───── DB Info Dict ─────
                # t_info_start = time.time()
                if relative_path:
                    db_dump_path = os.path.join(db_path.stem, names[i], filename)
                else:
                    db_dump_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }

                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]

                if "score" in annos:
                    db_info["score"] = annos["score"][i]

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
                # t_info_end = time.time()

            # t_obj_end = time.time()
            # print(
            #     f"[TIME][Sample {index}][Obj {i}] Path: {t_path_end - t_path_start:.4f}s, "
            #     f"Extract: {t_extract_end - t_extract_start:.4f}s, "
            #     f"Write: {t_write_end - t_write_start:.4f}s, "
            #     f"Info: {t_info_end - t_info_start:.4f}s, "
            #     f"Total obj: {t_obj_end - t_obj_start:.4f}s"
            # )

        # t_sample_end = time.time()
        # print(
        #     f"[TIME][Sample {index}] Load: {t1 - t0:.4f}s, "
        #     f"Filter: {t3 - t2:.4f}s, "
        #     f"InBox: {t5 - t4:.4f}s, "
        #     f"Total sample: {t_sample_end - t_sample_start:.4f}s"
        # )
    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


if __name__ == "__main__":
    fire.Fire()
    '''
    create_groundtruth_database(
        dataset_name="WAYMO",
        data_path='/mnt/sda/jinyu/datasets/waymo_det/',
        info_path='waymo_infos_train.pkl',
        nsweeps=1
    )
    create_groundtruth_database(
        dataset_name="NUSC",
        data_path='/mnt/sda/jinyu/datasets/nuscenes/',
        info_path='infos_train_10sweeps_withvelo_filterZero.pkl',
        nsweeps=10
    )
    '''