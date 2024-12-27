# Modified from https://github.com/lab4d-org/lab4d
import glob
import os
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(
    0,
    "%s/../third_party" % os.path.join(os.path.dirname(__file__)),
)


sys.path.insert(
    0,
    "%s/../" % os.path.join(os.path.dirname(__file__)),
)

sys.path.insert(
    0,
    "%s/../../" % os.path.join(os.path.dirname(__file__)),
)

import fusion
from libs.io import read_frame_data

from utils.geom_utils import K2inv, K2mat
from utils.vis_utils import draw_cams

# def read_cam(imgpath, component_id):
#     campath = imgpath.replace("JPEGImages", "Cameras").replace(
#         ".jpg", "-%02d.txt" % component_id
#     )
#     scene2cam = np.loadtxt(campath)
#     cam2scene = np.linalg.inv(scene2cam)
#     return cam2scene


def tsdf_fusion(seqname, vidname, obj_idx=None, num_obj=None, crop_size=256, use_full=True):
    # load rgb/depth
    imgdir = "database/processed_%s/JPEGImages/Full-Resolution/%s" % (vidname, seqname)
    imglist = sorted(glob.glob("%s/*.jpg" % imgdir))

    # camera path
    save_path = imgdir.replace("JPEGImages", "Cameras")
    if obj_idx is not None:
        save_path = "%s/%02d.npy" % (save_path, obj_idx)
    else:
        save_path = "%s/bg.npy" % (save_path)
    cams_prev = np.load(save_path)

    # get camera intrinsics
    raw_shape = cv2.imread(imglist[0]).shape[:2]
    max_l = max(raw_shape)
    Kraw = np.array([max_l, max_l, raw_shape[1] / 2, raw_shape[0] / 2])
    Kraw = K2mat(Kraw)

    # initialize volume
    vol_bnds = np.zeros((3, 2))
    for it, imgpath in enumerate(imglist[:-1]):
        rgb, depth, mask, crop2raw = read_frame_data(
            imgpath, crop_size, use_full, obj_idx, num_obj
        )
        K0 = K2inv(crop2raw) @ Kraw
        # cam2scene = read_cam(imgpath, component_id)
        cam2scene = np.linalg.inv(cams_prev[it])
        depth[~mask] = 0
        depth[depth > 10] = 0
        view_frust_pts = fusion.get_view_frustum(depth, K0, cam2scene)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.2, use_gpu=False)

    # fusion
    for it, imgpath in enumerate(imglist[:-1]):
        # print(imgpath)
        rgb, depth, mask, crop2raw = read_frame_data(
            imgpath, crop_size, use_full, obj_idx, num_obj
        )
        K0 = K2inv(crop2raw) @ Kraw
        depth[~mask] = 0
        # cam2scene = read_cam(imgpath, component_id)
        cam2scene = np.linalg.inv(cams_prev[it])
        tsdf_vol.integrate(rgb, depth, K0, cam2scene, obs_weight=1.0)

    save_path = imgdir.replace("JPEGImages", "Cameras")
    # get mesh, compute center
    rt = tsdf_vol.get_mesh()
    verts, faces = rt[0], rt[1]
    mesh = trimesh.Trimesh(verts, faces)
    aabb = mesh.bounds
    center = aabb.mean(0)
    mesh.vertices = mesh.vertices - center[None]
    mesh.export("%s/mesh-test-centered.obj" % (save_path))

    # save cameras
    cams = []
    for it, imgpath in enumerate(imglist):
        # campath = imgpath.replace("JPEGImages", "Cameras").replace(
        #     ".jpg", "-%02d.txt" % component_id
        # )
        # cam = np.loadtxt(campath)
        # shift the camera in the scene space
        cam = np.linalg.inv(cams_prev[it])
        cam[:3, 3] -= center
        cam = np.linalg.inv(cam)
        # np.savetxt(campath, cam)
        cams.append(cam)
    if obj_idx is not None:
        np.save("%s/%02d.npy" % (save_path, obj_idx), cams)
        mesh_cam = draw_cams(cams)
        mesh_cam.export("%s/cameras-%02d-centered.obj" % (save_path, obj_idx))

        print("tsdf fusion done: %s, %d" % (seqname, obj_idx))
    else:
        np.save("%s/bg.npy" % (save_path), cams)
        mesh_cam = draw_cams(cams)
        mesh_cam.export("%s/cameras-bg-centered.obj" % (save_path))

        print("tsdf fusion done: %s, bg" % (seqname))


if __name__ == "__main__":
    seqname = sys.argv[1]
    vidname = sys.argv[2]
    obj_idx = int(sys.argv[3])
    num_obj = int(sys.argv[4])

    # tsdf_fusion(seqname, vidname, obj_idx, num_obj, use_full=False)
