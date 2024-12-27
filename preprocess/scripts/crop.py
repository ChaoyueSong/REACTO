# Modified from https://github.com/lab4d-org/lab4d

import glob
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(
    0,
    "%s/../" % os.path.join(os.path.dirname(__file__)),
)

from libs.io import flow_process, read_raw


def extract_crop(seqname, crop_size, vidname, use_full, obj_idx):
    if use_full:
        save_prefix = "full"
    else:
        save_prefix = "crop"
    save_prefix = "%s-%d" % (save_prefix, crop_size)

    delta_list = [1, 2, 4, 8]

    flowfw_list = {delta: [] for delta in delta_list}
    flowbw_list = {delta: [] for delta in delta_list}
    rgb_list = []
    mask_list = []
    depth_list = []
    crop2raw_list = []
    is_detected_list = []

    imglist = sorted(
        glob.glob("database/processed_%s/JPEGImages/Full-Resolution/%s/*.jpg" % (vidname, seqname))
    )
    for im0idx in tqdm(range(len(imglist))):
        for delta in delta_list:
            if im0idx % delta != 0:
                continue
            if im0idx + delta >= len(imglist):
                continue
            # print("%s %d %d" % (seqname, frameid0, frameid1))
            data_dict0 = read_raw(imglist[im0idx], delta, crop_size, use_full, obj_idx)
            data_dict1 = read_raw(imglist[im0idx + delta], -delta, crop_size, use_full, obj_idx)
            flow_process(data_dict0, data_dict1)

            # save img, mask, vis2d
            if delta == 1:
                rgb_list.append(data_dict0["img"])
                mask_list.append(data_dict0["mask"])
                depth_list.append(data_dict0["depth"])
                crop2raw_list.append(data_dict0["crop2raw"])
                is_detected_list.append(data_dict0["is_detected"])

                if im0idx == len(imglist) - 2:
                    rgb_list.append(data_dict1["img"])
                    mask_list.append(data_dict1["mask"])
                    depth_list.append(data_dict1["depth"])
                    crop2raw_list.append(data_dict1["crop2raw"])
                    is_detected_list.append(data_dict1["is_detected"])

            flowfw_list[delta].append(data_dict0["flow"])
            flowbw_list[delta].append(data_dict1["flow"])

    # save cropped data
    
    if use_full: ## use_full, only save once for flow/rgb/depth
        if obj_idx == 0:  ## only save once, remove obj_idx in name
            for delta in delta_list:
                if len(flowfw_list[delta]) == 0:
                    continue
                np.save(
                    "database/processed_%s/FlowFW_%d/Full-Resolution/%s/%s.npy"
                    % (vidname, delta, seqname, save_prefix),
                    np.stack(flowfw_list[delta], 0),
                )
                np.save(
                    "database/processed_%s/FlowBW_%d/Full-Resolution/%s/%s.npy"
                    % (vidname, delta, seqname, save_prefix),
                    np.stack(flowbw_list[delta], 0),
                )
            
            np.save(
                "database/processed_%s/JPEGImages/Full-Resolution/%s/%s.npy"
                % (vidname, seqname, save_prefix),
                np.stack(rgb_list, 0),
            )
            
            np.save(
                "database/processed_%s/Depth/Full-Resolution/%s/%s.npy" 
                % (vidname, seqname, save_prefix),
                np.stack(depth_list, 0),
            )
    else:
        for delta in delta_list:
            if len(flowfw_list[delta]) == 0:
                continue
            np.save(
                "database/processed_%s/FlowFW_%d/Full-Resolution/%s/%s-%02d.npy"
                % (vidname, delta, seqname, save_prefix, obj_idx),
                np.stack(flowfw_list[delta], 0),
            )
            np.save(
                "database/processed_%s/FlowBW_%d/Full-Resolution/%s/%s-%02d.npy"
                % (vidname, delta, seqname, save_prefix, obj_idx),
                np.stack(flowbw_list[delta], 0),
            )
        
        np.save(
            "database/processed_%s/JPEGImages/Full-Resolution/%s/%s-%02d.npy"
            % (vidname, seqname, save_prefix, obj_idx),
            np.stack(rgb_list, 0),
        )
        
        np.save(
            "database/processed_%s/Depth/Full-Resolution/%s/%s-%02d.npy" 
            % (vidname, seqname, save_prefix, obj_idx),
            np.stack(depth_list, 0),
        )
    np.save(
        "database/processed_%s/Annotations/Full-Resolution/%s/%s-%02d.npy"
        % (vidname, seqname, save_prefix, obj_idx),
        np.stack(mask_list, 0),
    )

    np.save(
        "database/processed_%s/Annotations/Full-Resolution/%s/%s-crop2raw-%02d.npy"
        % (vidname, seqname, save_prefix, obj_idx),
        np.stack(crop2raw_list, 0),
    )

    np.save(
        "database/processed_%s/Annotations/Full-Resolution/%s/%s-is_detected-%02d.npy"
        % (vidname, seqname, save_prefix, obj_idx),
        np.stack(is_detected_list, 0),
    )

    print("crop (size: %d, full: %d) done: %s" % (crop_size, use_full, seqname))


if __name__ == "__main__":
    seqname = sys.argv[1]
    crop_size = int(sys.argv[2])
    use_full = bool(int(sys.argv[3]))

    extract_crop(seqname, crop_size, use_full)
