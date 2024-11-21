import argparse
import os
import cv2
import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def get_args():
    parser = argparse.ArgumentParser(description="Test Quickdraw model")
    parser.add_argument("--image_size", "-i", type=int, default=28)
    parser.add_argument("--saved_checkpoint", "-s", type=str, default="trained_models/best.pt",
                        help="Continue from this checkpoint")
    args = parser.parse_args()
    return args


def inference(ori_image, categories):
    args = get_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # Modify the first convolution layer to accept 1-channel grayscale input
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(model.last_channel, len(categories))
    model = model.to(device)

    if args.saved_checkpoint is not None and os.path.isfile(args.saved_checkpoint):
        checkpoint = torch.load(args.saved_checkpoint)
        model.load_state_dict(checkpoint["model_params"])
    else:
        print("No checkpoint provided")
        exit(0)

    image = cv2.resize(ori_image, (args.image_size, args.image_size))
    image = torch.from_numpy(image).float()
    image = image[None, None, :, :] / 255.
    image = image.to(device)

    model.eval()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        prediction = model(image)
        prediction = softmax(prediction)
        conf_score, predicted_class = torch.max(prediction, dim=1)

    return conf_score.item(), categories[predicted_class]


if __name__ == '__main__':
    subdir = [os.path.join("Data", sub) for sub in os.listdir("Data")]
    categories = [os.path.basename(sub) for sub in subdir]
    categories = [cate.split("_")[-1].replace(".npy", "") for cate in categories]

    ori_image = cv2.imread("test.jpg")
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)

    conf_score, predicted_class = inference(ori_image, categories)
    print(f"Prediction: {predicted_class} with confidence score {conf_score:.4f}")
