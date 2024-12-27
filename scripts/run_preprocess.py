# Modified from https://github.com/lab4d-org/lab4d
import configparser
import glob
import importlib
import os
import pdb
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.gpu_utils import gpu_map
from preprocess.libs.io import run_bash_command
from preprocess.scripts.download import download_seq
from preprocess.scripts.camera_registration import camera_registration
from preprocess.scripts.canonical_registration import canonical_registration
from preprocess.scripts.crop import extract_crop
from preprocess.scripts.depth import extract_depth
from preprocess.scripts.extract_dinov2 import extract_dinov2
from preprocess.scripts.extract_frames import extract_frames
from preprocess.scripts.tsdf_fusion import tsdf_fusion
from preprocess.scripts.write_config import write_config
from preprocess.third_party.vcnplus.compute_flow import compute_flow
from preprocess.third_party.vcnplus.frame_filter import frame_filter

track_anything_module = importlib.import_module(
    "preprocess.third_party.Track-Anything.app"
)
track_anything_gui = track_anything_module.track_anything_interface
track_anything_cli = importlib.import_module(
    "preprocess.third_party.Track-Anything.track_anything_cli"
)
track_anything_cli = track_anything_cli.track_anything_cli
crop_size = 512

def track_anything_lab4d(seqname, outdir, obj_idx, text_prompt):
    input_folder = "%s/JPEGImages/Full-Resolution/%s" % (outdir, seqname)
    output_folder = "%s/Annotations/Full-Resolution/%s" % (outdir, seqname)
    track_anything_cli(input_folder, text_prompt, obj_idx, output_folder)


def remove_exist_dir(seqname, outdir):
    run_bash_command(f"rm -rf {outdir}/JPEGImages/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Annotations/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Cameras/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Features/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Depth/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Flow*/Full-Resolution/{seqname}")


def run_extract_frames(seqname, outdir, infile, use_filter_frames):
    # extract frames
    imgpath = f"{outdir}/JPEGImagesRaw/Full-Resolution/{seqname}"
    run_bash_command(f"rm -rf {imgpath}")
    os.makedirs(imgpath, exist_ok=True)
    extract_frames(infile, imgpath, desired_fps=10)

    # remove existing dirs for preprocessing
    remove_exist_dir(seqname, outdir)

    # filter frames without motion: frame id is the time stamp
    if use_filter_frames:
        frame_filter(seqname, outdir)
    else:
        outpath = f"{outdir}/JPEGImages/Full-Resolution/{seqname}"
        run_bash_command(f"rm -rf {outpath}")
        os.makedirs(outpath, exist_ok=True)
        run_bash_command(f"cp {imgpath}/* {outpath}/")

def run_extract_priors_single(seqname, outdir, num_obj, vidname):
    print("extracting priors (flow/depth): ", seqname)
    # flow
    for dframe in [1, 2, 4, 8]:
        compute_flow(seqname, outdir, dframe)
    # depth
    extract_depth(seqname, vidname) 

def run_extract_priors_bg(seqname, outdir, num_obj, vidname):
    print("extracting priors (bg_camera): ", seqname)

    # compute bg cameras
    camera_registration(seqname, crop_size, vidname, num_obj=num_obj)
    tsdf_fusion(seqname, vidname, num_obj=num_obj)
    
def run_extract_priors_multi(seqname, vidname, obj_idx, num_obj, obj_class_cam):
    print("extracting priors(crop, camera_fg, obj): ", seqname, obj_idx)

    # crop around object and process flow
    extract_crop(seqname, crop_size, vidname, 0, obj_idx) #0,1 crop/full
    extract_crop(seqname, crop_size, vidname, 1, obj_idx)

    # compute fg cameras
    camera_registration(seqname, crop_size, vidname, obj_idx) 
    canonical_registration(seqname, crop_size, vidname, obj_idx, obj_class_cam)


if __name__ == "__main__":
    if len(sys.argv) != 6: ### need to change follow the input
        print(
            f"Usage: python {sys.argv[0]} <vidname> <num_obj> <text_prompt_seg> <obj_class_cam> <gpulist>"
        )
        print(
            f"  Example: python {sys.argv[0]} 1 cat-pikachu-0 cat quad '0,1,2,3,4,5,6,7' "
        )
        exit()
    vidname = sys.argv[1]
    ### add arg num_obj; and re-design test_prompt and obj_class as lists
    num_obj = int(sys.argv[2])
    text_prompt_seg = [str(n) for n in sys.argv[3].split(",")]  ## if prompt is other, use track_anything. otherwise, auto
    obj_class_cam = [str(n) for n in sys.argv[4].split(",")]
    
    valid_values = ["human", "quad", "other", "arti"]
    assert all(item in valid_values for item in obj_class_cam), "Invalid values in obj_class_cam"
    gpulist = [int(n) for n in sys.argv[5].split(",")]
    # frame_rate = int(sys.argv[6])
    
    # True: filter frame based on motion magnitude | False: use all frames
    use_filter_frames = False

    outdir = "database/processed_%s/" % vidname
    viddir = "database/raw/%s" % vidname
    print("using gpus: ", gpulist)
    os.makedirs("tmp", exist_ok=True)

    # download the videos
    download_seq(vidname)

    # set up parallel extraction
    frame_args = []
    for counter, infile in enumerate(sorted(glob.glob("%s/*" % viddir))):
        seqname = "%s-%04d" % (vidname, counter)
        frame_args.append((seqname, outdir, infile, use_filter_frames))

    # extract frames and filter frames without motion: frame id is the time stamp
    gpu_map(run_extract_frames, frame_args, gpus=gpulist)

    # write config
    write_config(vidname, num_obj, obj_class_cam, vidname)

    # read config
    config = configparser.RawConfigParser()
    config.read("database/configs/%s.config" % vidname)
    seg_args = []
    prior_args_single = []
    prior_args_multi = []
    for vidid in range(len(config.sections()) - 1):
        seqname = config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
        
        prior_args_single.append((seqname, outdir, num_obj, vidname))
        
        for obj_idx in range(num_obj):
            ### the anno needs obj_idx: mask, camera, feature
            seg_args.append((seqname, outdir, obj_idx, text_prompt_seg[obj_idx]))
            prior_args_multi.append((seqname, vidname, obj_idx, num_obj, obj_class_cam[obj_idx]))
            # True: manually annotate object masks | False: use detect object based on text prompt
            use_manual_segment = True #if text_prompt_seg[obj_idx] == "other" else False
            # True: manually annotate camera for key frames
            use_manual_cameras = True if ((obj_class_cam[obj_idx] == "other") or (obj_class_cam[obj_idx] == "arti")) else False
            
            if obj_idx == 0:
                # extract flow/depth only once
                gpu_map(run_extract_priors_single, prior_args_single, gpus=gpulist)
                
            # let the user specify the segmentation mask
            if use_manual_segment:
                track_anything_gui(vidname, obj_idx)
            else:
                gpu_map(track_anything_lab4d, seg_args, gpus=gpulist)
        
            # Manually adjust camera positions
            if use_manual_cameras:
                from preprocess.scripts.manual_cameras import manual_camera_interface

                mesh_path = "database/mesh-templates/cat-pikachu-remeshed.obj"
                manual_camera_interface(vidname, obj_idx, mesh_path)

            # extract camera/crop
            gpu_map(run_extract_priors_multi, prior_args_multi, gpus=gpulist)

            # extract dinov2 features
            extract_dinov2(vidname, crop_size, vidname, obj_idx, gpulist=gpulist)
        
        # extract bg para
        gpu_map(run_extract_priors_bg, prior_args_single, gpus=gpulist)
        