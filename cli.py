import argparse
import numpy as np
import imageio.v3 as iio
import module_color_to_normals
import module_normals_to_curvature
import module_normals_to_height
import os

# Parse CLI args
parser = argparse.ArgumentParser(description="DeepBump CLI")
parser.add_argument("input", help="path to the input image or folder", type=str)
parser.add_argument("output", help="path to the output image or folder", type=str)
parser.add_argument(
    "module",
    help="processing to be applied",
    choices=["color_to_normals", "normals_to_curvature", "normals_to_height"],
)
parser.add_argument(
    "--verbose",
    action=argparse.BooleanOptionalAction,
    help="prints progress to the console",
)
parser.add_argument(
    "--batch",
    action=argparse.BooleanOptionalAction,
    help="process all images in the input folder",
)
parser.add_argument(
    "--color_to_normals-overlap",
    choices=["SMALL", "MEDIUM", "LARGE"],
    required=False,
    default="LARGE",
)
parser.add_argument(
    "--normals_to_curvature-blur_radius",
    choices=["SMALLEST", "SMALLER", "SMALL", "MEDIUM", "LARGE", "LARGER", "LARGEST"],
    required=False,
    default="MEDIUM",
)
parser.add_argument(
    "--normals_to_height-seamless",
    choices=["TRUE", "FALSE"],
    required=False,
    default="FALSE",
)
args = parser.parse_args()

def processs_image(input, output, file_number=None, total_files=None):
    # Read input image
    in_img = iio.imread(input)
    # Convert from H,W,C in [0,  256] to C,H,W in [0,1]
    in_img = np.transpose(in_img, (2,  0,  1)) /  255

    # Apply processing
    if args.module == "color_to_normals":
        out_img = module_color_to_normals.apply(in_img, args.color_to_normals_overlap, None)
    if args.module == "normals_to_curvature":
        out_img = module_normals_to_curvature.apply(
            in_img, args.normals_to_curvature_blur_radius, None
        )
    if args.module == "normals_to_height":
        out_img = module_normals_to_height.apply(
            in_img, args.normals_to_height_seamless == "TRUE", None
        )

    # Convert from C,H,W in [0,1] to H,W,C in [0,  256]
    out_img = (np.transpose(out_img, (1,  2,  0)) *  255).astype(np.uint8)
    # Add _normal suffix to the output filename
    output_filename = os.path.splitext(os.path.basename(input))[0] + '_normal' + os.path.splitext(input)[1]
    output_file_path = os.path.join(output, output_filename)
    iio.imwrite(output_file_path, out_img)

    # Log the current file being processed
    if file_number is not None and total_files is not None:
        print(f"Processed file {file_number} of {total_files}: {output_filename}")
    else:
        print(f"Processed file: {output_filename}")

input_path = args.input
output_path = args.output

if os.path.isdir(args.input):
    # Batch processing
    if not args.batch:
        raise ValueError("Batch processing requires the --batch flag.")
    # Process all images in the folder
    files = os.listdir(args.input)
    total_files = len(files)
    for file_number, filename in enumerate(files, start=1):
        if not os.path.exists(args.output):
            os.makedirs(args.output)
    
        file_path = os.path.join(args.input, filename)
        processs_image(file_path, output_path, file_number, total_files)
else:
    # Single image processing
    processs_image(input_path, output_path)