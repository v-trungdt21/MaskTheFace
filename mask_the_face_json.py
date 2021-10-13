# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
from collections import defaultdict

from tqdm import trange
import dlib
import json
import copy
from pathlib import Path
from utils.aux_functions import *
from utils.aux_functions import my_mask_image as mask_image

# Command-line input setup
parser = argparse.ArgumentParser(
    description="MaskTheFace - Python code to mask faces dataset"
)
parser.add_argument(
    "--anno_path", type=Path, default="/home/termanteus/workspace/face/data/pose/WFLW/WFLW/annotations/face_landmarks_wflw_train.json", help="Path to either the annotation json",
)
parser.add_argument(
    "--mask_type",
    type=str,
    default="surgical",
    choices=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"],
    help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
)

parser.add_argument(
    "--pattern",
    type=str,
    default="",
    help="Type of the pattern. Available options in masks/textures",
)

parser.add_argument(
    "--pattern_weight",
    type=float,
    default=0.5,
    help="Weight of the pattern. Must be between 0 and 1",
)

parser.add_argument(
    "--color",
    type=str,
    default="#0473e2",
    help="Hex color value that need to be overlayed to the mask",
)

parser.add_argument(
    "--color_weight",
    type=float,
    default=0.5,
    help="Weight of the color intensity. Must be between 0 and 1",
)

parser.add_argument(
    "--code",
    type=str,
    # default="cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.png, cloth-masks/textures/fruits/pineapple.png, N95, surgical_blue, surgical_green",
    default="",
    help="Generate specific formats",
)


parser.add_argument(
    "--verbose", dest="verbose", action="store_true", help="Turn verbosity on"
)
parser.add_argument(
    "--write_original_image",
    dest="write_original_image",
    action="store_true",
    help="If true, original image is also stored in the masked folder",
)
parser.set_defaults(feature=False)

args = parser.parse_args()

# Extract data from code
mask_code = "".join(args.code.split()).split(",")
args.code_count = np.zeros(len(mask_code))
args.mask_dict_of_dict = {}


for i, entry in enumerate(mask_code):
    mask_dict = {}
    mask_color = ""
    mask_texture = ""
    mask_type = entry.split("-")[0]
    if len(entry.split("-")) == 2:
        mask_variation = entry.split("-")[1]
        if "#" in mask_variation:
            mask_color = mask_variation
        else:
            mask_texture = mask_variation
    mask_dict["type"] = mask_type
    mask_dict["color"] = mask_color
    mask_dict["texture"] = mask_texture
    args.mask_dict_of_dict[i] = mask_dict


with open(args.anno_path.as_posix(), "r") as js:
    annotation_obj = json.load(js)

image_root = args.anno_path.parent.parent / "images"
image_out_root = args.anno_path.parent.parent / "images_masked"
image_out_root.mkdir(parents=True, exist_ok=True)

lookup_dict = defaultdict(list)

if not (args.anno_path.parent.parent/"lookup_dict.json").exists():
    print("Building look up dict from image_id -> list of its annotations....")
    for i in trange(len(annotation_obj["images"])):
        image_obj = annotation_obj["images"][i]
        for ann in annotation_obj["annotations"]:
            if ann["image_id"] == image_obj["id"]:

                kpts = np.array(ann["keypoints"], dtype=int)
                kpts = kpts.reshape(-1, 3)
                kpts = kpts[:, :2]
                ann["keypoints"] = kpts.flatten().tolist()
                ann["bbox"][2] += ann["bbox"][0] 
                ann["bbox"][3] += ann["bbox"][1] 
                lookup_dict[str(image_obj["id"])].append(ann)

    with open(args.anno_path.parent.parent/"lookup_dict.json", 'w') as f:
        json.dump(lookup_dict, f)
    print("Built look up dict done.")
else:
    print("Reading pre-made lookup dict...")
    with open(args.anno_path.parent.parent/"lookup_dict.json", 'r') as f:
        lookup_dict = json.load(f)
    print("Loaded pre-made lookup dict.")

res_dict = copy.deepcopy(annotation_obj)
res_dict["annotations"] = []
for i in trange(len(annotation_obj["images"])):
    # if i > 100:
    #     break
    image_obj = annotation_obj["images"][i]
    image_path = image_root / image_obj["file_name"]
    image_out_path = image_out_root / image_obj["file_name"]

    anns = lookup_dict[str(image_obj["id"])]
    if len(anns) == 0:
        print(f"Skipping image: {image_obj['file_name']} since it doesn't have any annotations.")
        continue
    masked_image, mask, mask_binary_array, original_image = mask_image(
        image_path, args, anns
    )
    res_dict["annotations"] += anns
    for i in range(len(mask)):
        image_out_path.parent.mkdir(parents=True,exist_ok=True)
        path, ext = image_out_path.as_posix().rsplit(".")
        # w_path = path + "_" + mask[i] + "." + ext
        w_path = image_out_path.as_posix()
        # print(w_path)
        img = masked_image[i]
        cv2.imwrite(w_path, img)

with open((args.anno_path.parent.parent/"masked_annotations.json").as_posix(), 'w') as js:
    json.dump(res_dict, js)
print("Processing Done")
