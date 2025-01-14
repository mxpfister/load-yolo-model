import os
import torch
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import argparse
from model import Yolov1
from utils import cellboxes_to_boxes, non_max_suppression, plot_image


def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = Yolov1(split_size=7, num_boxes=2, num_classes=1)
    optimizer = optim.Adam(
        model.parameters()
    )
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])
    image = preprocess(image).unsqueeze(0)
    return image


def predict(model, image_tensor, image_name):
    with torch.no_grad():
        outputs = model(image_tensor)
    bboxes = cellboxes_to_boxes(outputs)
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.1, box_format="midpoint")
    print(f"Predictions for {image_name}: {bboxes}")
    plot_image(image_tensor[0].permute(1, 2, 0).to("cpu"), bboxes, image_name)
    return bboxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a model and make predictions on images in a directory.')
    parser.add_argument('image_dir', type=str, help='Path to the image directory')
    args = parser.parse_args()

    model_path = 'overfit.pth.tar'
    print('Loading model...')
    model = load_model(model_path)
    print('Model loaded.')

    image_files = [f for f in os.listdir(args.image_dir)]
    if not image_files:
        print(f"No image files found in directory: {args.image_dir}")
        exit()

    print(f"Found {len(image_files)} images. Processing...")

    for image_file in image_files:
        image_path = os.path.join(args.image_dir, image_file)
        print(f"Processing {image_file}...")
        try:
            image_tensor = preprocess_image(image_path)
            prediction = predict(model, image_tensor, image_file)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    print("Processing complete.")
    exit()
