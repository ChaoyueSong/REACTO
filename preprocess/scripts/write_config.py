# Modified from https://github.com/lab4d-org/lab4d
import configparser
import glob
import os
import sys

import cv2


def write_config(collection_name, num_obj, obj_class, vidname):
    min_nframe = 8
    imgroot = "database/processed_%s/JPEGImages/Full-Resolution/" % vidname

    config = configparser.ConfigParser()
    config["data"] = {
        "init_frame": "0",
        "end_frame": "-1",
    }

    seqname_all = sorted(
        glob.glob("%s/%s-[0-9][0-9][0-9][0-9]*" % (imgroot, collection_name))
    )
    total_vid = 0
    for i, seqname in enumerate(seqname_all):
        seqname = seqname.split("/")[-1]
        img = cv2.imread("%s/%s/00000.jpg" % (imgroot, seqname), 0)
        num_fr = len(glob.glob("%s/%s/*.jpg" % (imgroot, seqname)))
        if num_fr < min_nframe:
            continue

        fl = max(img.shape)
        px = img.shape[1] // 2
        py = img.shape[0] // 2
        camtxt = [fl, fl, px, py]
        config["data_%d" % total_vid] = {
            "num_obj": num_obj,
            "obj_class": " ".join([str(i) for i in obj_class]),
            "ks": " ".join([str(i) for i in camtxt]),
            "shape": " ".join([str(img.shape[0]), str(img.shape[1])]),
            "img_path": "database/processed_%s/JPEGImages/Full-Resolution/%s/" % (vidname, seqname),
        }   ### add num_obj and obj_class to config
        total_vid += 1

    os.makedirs("database/configs", exist_ok=True)
    with open("database/configs/%s.config" % collection_name, "w") as configfile:
        config.write(configfile)


if __name__ == "__main__":
    collection_name = sys.argv[1]

    write_config(collection_name)
