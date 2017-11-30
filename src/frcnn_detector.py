FRCNN_DIM = 850.
import sys
from os import path
import json
from cntk import load_model

def get_classes_description(model_file_path, classes_count):
    model_dir = path.dirname(model_file_path)
    classes_names = {}
    model_desc_file_path = path.join(model_dir, 'model.json')
    if not path.exists(model_desc_file_path):
        # use default parameter names:
        for i in range(classes_count):
            classes_names["class_%d"%i] = i
        return classes_names

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='FRCNN Detector')
    
    parser.add_argument('--input', type=str, metavar='<path>',
                        help='Path to image file or to a directory containing image in jpg format', required=True)
    
    parser.add_argument('--output', type=str, metavar='<directory path>',
                        help='Path to output directory', required=False)
    
    parser.add_argument('--model', type=str, metavar='<file path>',
                        help='Path to model file',
                        required=True)

    parser.add_argument('--cntk-path', type=str, metavar='<dir path>',
                        help='Path to the directory in which CNTK is installed, e.g. c:\\local\\cntk',
                        required=False)

    parser.add_argument('--json-output', type=str, metavar='<file path>',
                        help='Path to output JSON file', required=False)

    args = parser.parse_args()

    if args.cntk_path:
        cntk_path = args.cntk_path
    else:
        cntk_path = "C:\\local\\cntk"
    cntk_scripts_path = path.join(cntk_path, r"Examples/Image/Detection/")
    sys.path.append(cntk_scripts_path)
    from ObjectDetector import predict

    input_path = args.input
    output_path = args.output
    json_output_path = args.json_output
    model_path =  args.model
    model = load_model(model_path)
    labels_count = model.cls_pred.shape[1]
    model_classes = get_classes_description(model_path, labels_count)
    classes = list(model_classes.keys())

    if (output_path is None and json_output_path is None):
        parser.error("No directory output path or json output path specified")

    if (output_path is not None) and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if os.path.isdir(input_path):
        import glob
        file_paths = glob.glob(os.path.join(input_path, '*.jpg'))
    else:
        file_paths = [input_path]


    if json_output_path is not None:
        json_output_obj = {"classes": model_classes,
                           "frames" : {}}

    print("Number of images to process: %d"%len(file_paths))

    for file_path, counter in zip(file_paths, range(len(file_paths))):
        from PIL import Image
        with Image.open(file_path) as img:
            width, height = img.size
        w, h = (width/FRCNN_DIM, height/FRCNN_DIM)

        print("Read file in path:", file_path)
        rectangles = predict(file_path, model, classes)
        print(rectangles)
        for rect in rectangles:
            image_base_name = path.basename(file_path)
            regions_list = []
            json_output_obj["frames"][image_base_name] = {"regions": regions_list}
            x1, y1, x2, y2 = rect["box"]
            regions_list.append({
                "x1" : int(x1 * w),
                "y1" : int(y1 * h),
                "x2" : int(x2 * w),
                "y2" : int(y2 * h),
                "class" : model_classes[rect["label"]]
            })

    if json_output_path is not None:
        with open(json_output_path, "wt") as handle:
            json_dump = json.dumps(json_output_obj, indent=2)
            handle.write(json_dump)

