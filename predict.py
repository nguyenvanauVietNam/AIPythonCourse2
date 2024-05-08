import argparse
import json
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F

# Define function to parse command line arguments
def parse_arguments_value():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('img_image_path_value', metavar='image_path', type=str, default='flowers/test/58/image_02663.jpg')
    parser.add_argument('img_checkpoint_value', metavar='checkpoint', type=str, default='train_checkpoint.pth')
    parser.add_argument('--img_top_k_value', action='store', dest="local_top_k", type=int, default=5)
    parser.add_argument('--img_category_names_value', action='store', dest='local_category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--img_gpu_value', action='store_true', default=False)
    return parser.parse_args()

# Define function to load model checkpoint
def load_checkpoint_value(filepath):
    """Load model checkpoint from file."""
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    local_model = getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    local_model.input_size = checkpoint['input_size']
    local_model.output_size = checkpoint['output_size']
    local_model.learning_rate = checkpoint['learning_rate']
    local_model.hidden_units = checkpoint['hidden_units']
    local_model.learning_rate = checkpoint['learning_rate']
    local_model.classifier = checkpoint['classifier']
    local_model.epochs = checkpoint['epochs']
    local_model.load_state_dict(checkpoint['state_dict'])
    local_model.class_to_idx = checkpoint['class_to_idx']
    local_model.optimizer = checkpoint['optimizer']
    return local_model

# Define function to process image
def process_img_image_value(local_image):
    """Process input image."""
    local_resize = 256
    local_crop_size = 224
    (local_width, local_height) = local_image.size

    if local_height > local_width:
        local_height = int(max(local_height * local_resize / local_width, 1))
        local_width = int(local_resize)
    else:
        local_width = int(max(local_width * local_resize / local_height, 1))
        local_height = int(local_resize)

    # Resize image
    local_im = local_image.resize((local_width, local_height))
    # Crop image
    local_left = (local_width - local_crop_size) / 2
    local_top = (local_height - local_crop_size) / 2
    local_right = local_left + local_crop_size
    local_bottom = local_top + local_crop_size
    local_im = local_im.crop((local_left, local_top, local_right, local_bottom))

    # Normalize and transpose color channels
    local_im = np.array(local_im)
    local_im = local_im / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    local_im = (local_im - mean) / std
    local_im = np.transpose(local_im, (2, 0, 1))
    return local_im

# Define function to make predictions
def predict_value(local_img_image_path, local_model, local_top_k, gpu):
    """Make predictions on input image."""
    if gpu:
        local_device = torch.device("cuda")
    else:
        local_device = torch.device("cpu")
    local_model.to(local_device)

    local_image = Image.open(local_img_image_path)
    local_image = process_img_image_value(local_image)
    local_image = torch.from_numpy(local_image)
    local_image = local_image.unsqueeze_(0)
    local_image = local_image.float()

    with torch.no_grad():
        local_output = local_model.forward(local_image.cuda())

    local_p = F.softmax(local_output.data, dim=1)

    local_top_p = np.array(local_p.topk(local_top_k)[0][0])

    local_index_to_class = {val: key for key, val in local_model.class_to_idx.items()}
    local_top_classes = [np.int(local_index_to_class[each]) for each in np.array(local_p.topk(local_top_k)[1][0])]

    return local_top_p, local_top_classes, local_device

# Define function to load category names
def load_category_names_value(local_category_names_file):
    """Load category names from file."""
    with open(local_category_names_file) as file:
        local_category_names = json.load(file)
    return local_category_names

# Main function to execute the prediction
def main_value():
    """Execute prediction."""
    args = parse_arguments_value()
    local_img_image_path = args.img_image_path_value
    local_checkpoint = args.img_checkpoint_value
    local_top_k = args.local_top_k
    local_category_names = args.local_category_names
    gpu = args.img_gpu_value

    local_model = load_checkpoint_value(local_checkpoint)

    top_p, classes, device = predict_value(local_img_image_path, local_model, local_top_k, gpu)

    local_category_names = load_category_names_value(local_category_names)

    labels = [local_category_names[str(index)] for index in classes]

    print(f"Results for your File: {local_img_image_path}")
    print(labels)
    print(top_p)
    print()

    for i in range(len(labels)):
        print("{} - {} with a probability of {}".format((i+1), labels[i], top_p[i]))

if __name__ == "__main__":
    main_value()
