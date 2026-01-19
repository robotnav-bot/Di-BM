import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]
        image_dict = dict()
        # 增加容错：检查 observation 是否存在
        if "observation" in root:
            for cam_name in root[f"/observation/"].keys():
                image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]
        
    return left_gripper, left_arm, right_gripper, right_arm, vector, image_dict


def main():
    parser = argparse.ArgumentParser(description="Merge multiple tasks into a single Zarr dataset.")
    
    parser.add_argument(
        "--task_names",  
        nargs='+',       
        type=str,
        required=True,   
        help="List of task names (e.g. adjust_bottle click_bell)",
    )

    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)

    args = parser.parse_args()

    task_names = args.task_names
    task_config = args.task_config
    num = args.expert_data_num

    output_filename = f"multi_task-{task_config}-{num}.zarr"
    
    save_dir = os.path.join("./data/", output_filename)
    # ===========================================

    print(f"Tasks to merge: {task_names}")
    print(f"Config: {task_config}")
    print(f"Episodes per task: {num}")
    print(f"Output file: {save_dir}")

    if os.path.exists(save_dir):
        print(f"Removing existing zarr at {save_dir}")
        shutil.rmtree(save_dir)
    else:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays = []
    state_arrays = []
    joint_action_arrays = []
    episode_ends_arrays = []
    
    total_count = 0

    for task in task_names:
        # 拼接加载路径：data_root / task_name / task_config
        load_dir = os.path.join("../../data/", task, task_config)
        
        if not os.path.exists(load_dir):
            print(f"[Warning] Directory not found, skipping: {load_dir}")
            continue

        data_subdir = os.path.join(load_dir, "data")
        if not os.path.exists(data_subdir):
            print(f"[Warning] 'data' folder not found in: {load_dir}")
            continue

        existing_files = [f for f in os.listdir(data_subdir) if f.endswith(".hdf5")]
        if len(existing_files) < num:
            print(f"[Warning] Task {task} has only {len(existing_files)} episodes, fewer than requested {num}.")
        
        print(f"Processing Task: {task}")

        current_ep = 0
        while current_ep < num:
            print(f"  > Episode: {current_ep + 1} / {num}", end="\r")

            load_path = os.path.join(data_subdir, f"episode{current_ep}.hdf5")
            
            if not os.path.exists(load_path):
                print(f"\n[Error] File missing: {load_path}. Skipping episode.")
                current_ep += 1
                continue

            (
                left_gripper_all,
                left_arm_all,
                right_gripper_all,
                right_arm_all,
                vector_all,
                image_dict_all,
            ) = load_hdf5(load_path)

            for j in range(0, left_gripper_all.shape[0]):
                head_img_bit = image_dict_all["head_camera"][j]
                joint_state = vector_all[j]

                if j != left_gripper_all.shape[0] - 1:
                    head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                    head_camera_arrays.append(head_img)
                    state_arrays.append(joint_state)
                
                if j != 0:
                    joint_action_arrays.append(joint_state)

            current_ep += 1
            total_count += left_gripper_all.shape[0] - 1
            episode_ends_arrays.append(total_count)
        print("") 

    print(f"Merging complete. Total frames: {total_count}")
    
    if total_count == 0:
        print("Error: No data found. Exiting.")
        return

    print("Converting lists to numpy arrays (this may take memory)...")
    episode_ends_arrays = np.array(episode_ends_arrays)
    state_arrays = np.array(state_arrays, dtype="float32")
    joint_action_arrays = np.array(joint_action_arrays, dtype="float32")
    
    head_camera_arrays = np.array(head_camera_arrays)
    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1) # NHWC -> NCHW

    print(f"Writing to Zarr at {save_dir}...")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])

    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=head_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=joint_action_arrays,
        chunks=joint_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    print("Done!")

if __name__ == "__main__":
    main()