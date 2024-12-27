# Modified from https://github.com/lab4d-org/lab4d

import glob
import os
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(
    0,
    "%s/../../" % os.path.join(os.path.dirname(__file__)),
)

sys.path.insert(
    0,
    "%s/../" % os.path.join(os.path.dirname(__file__)),
)

from libs.geometry import two_frame_registration
from libs.io import flow_process, read_raw
from libs.utils import reduce_component

from utils.geom_utils import K2inv, K2mat
from utils.vis_utils import draw_cams

### rewrite here, if bg, do not input obj_idx
def camera_registration(seqname, crop_size, vidname, obj_idx=None, num_obj=None):
    imgdir = "database/processed_%s/JPEGImages/Full-Resolution/%s" % (vidname, seqname)
    imglist = sorted(glob.glob("%s/*.jpg" % imgdir))
    delta = 1
    use_full = True
    registration_type = "procrustes"

    # get camera intrinsics
    raw_shape = cv2.imread(imglist[0]).shape[:2]
    max_l = max(raw_shape)
    Kraw = np.array([max_l, max_l, raw_shape[1] / 2, raw_shape[0] / 2])
    Kraw = K2mat(Kraw)

    cam_current = np.eye(4)  # scene to camera: I, R01 I, R12 R01 I, ...
    cams = [cam_current]
    for im0idx in range(len(imglist)):
        if im0idx + delta >= len(imglist):
            continue
        # TODO: load croped images directly
        frameid0 = int(imglist[im0idx].split("/")[-1].split(".")[0])
        frameid1 = int(imglist[im0idx + delta].split("/")[-1].split(".")[0])
        # print("%s %d %d" % (seqname, frameid0, frameid1))
        data_dict0 = read_raw(imglist[im0idx], delta, crop_size, use_full, obj_idx, num_obj)
        data_dict1 = read_raw(imglist[im0idx + delta], -delta, crop_size, use_full, obj_idx, num_obj)
        flow_process(data_dict0, data_dict1)

        # compute intrincs for the cropped images
        K0 = K2inv(data_dict0["crop2raw"]) @ Kraw
        K1 = K2inv(data_dict1["crop2raw"]) @ Kraw

        # get mask
        # mask = data_dict0["mask"][..., 0].astype(int) == 1 if obj_idx is not None else 0
        if obj_idx is not None:#component_id > 0:
            mask = data_dict0["mask"][..., 0].astype(int) == 1
            # reduce the mask to the largest connected component
            mask = reduce_component(mask)
        else:
            mask = data_dict0["mask"][..., 0].astype(int) == 0
            # for background, additionally remove flow with low confidence
            mask = np.logical_and(mask, data_dict0["flow"][..., 2] > 0).flatten()
        cam_0_to_1 = two_frame_registration(
            data_dict0["depth"],
            data_dict1["depth"],
            data_dict0["flow"],
            K0,
            K1,
            mask,
            registration_type,
        )
        cam_current = cam_0_to_1 @ cam_current
        cams.append(cam_current)

    os.makedirs(imgdir.replace("JPEGImages", "Cameras"), exist_ok=True)
    save_path = imgdir.replace("JPEGImages", "Cameras")
    # for idx, img_path in enumerate(sorted(glob.glob("%s/*.jpg" % imgdir))):
    #     frameid = int(img_path.split("/")[-1].split(".")[0])
    #     campath = "%s/%05d-%02d.txt" % (save_path, frameid, component_id)
    #     np.savetxt(campath, cams[idx])
    if obj_idx is not None:
        np.save("%s/%02d.npy" % (save_path, obj_idx), cams)
        mesh_cam = draw_cams(cams)
        mesh_cam.export("%s/cameras-%02d.obj" % (save_path, obj_idx))

        print("camera registration done: %s, %d" % (seqname, obj_idx))
    else:
        np.save("%s/bg.npy" % (save_path), cams)
        mesh_cam = draw_cams(cams)
        mesh_cam.export("%s/cameras-bg.obj" % (save_path))

        print("camera registration done: %s, bg" % (seqname))

if __name__ == "__main__":
    seqname = sys.argv[1]
    obj_idx = int(sys.argv[2])
    num_obj = int(sys.argv[2])

    camera_registration(seqname, obj_idx, num_obj)
