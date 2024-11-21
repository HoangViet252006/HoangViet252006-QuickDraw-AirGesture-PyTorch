import argparse
import os
import shutil
import numpy as np
import torch
import torch.backends.mps
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from dataset import QuickDraw

def get_args():
    parser = argparse.ArgumentParser(description="Train Quickdraw model")
    parser.add_argument("--data_path", "-d", type=str, default="data", help="Path to Dataset")
    parser.add_argument("--ratio", "-r", type=float, default=0.8)
    parser.add_argument("--num_epochs", "-n", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=2048, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--tensorboard_dir", "-t", type=str, default="tensorboard", help="Path to generated tensorboard")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="trained_models", help="Path to save checkpoint")
    parser.add_argument("--saved_checkpoint", "-s", type=str, default=None, help="Continue from this checkpoint")
    args = parser.parse_args()
    return args

def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    if os.path.exists(args.tensorboard_dir):
        shutil.rmtree(args.tensorboard_dir)
    os.makedirs(args.tensorboard_dir)

    writer = SummaryWriter(args.tensorboard_dir)

    transform = Compose([
        ToTensor(),
    ])

    train_dataset = QuickDraw(args.data_path, True, args.ratio, transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    val_dataset = QuickDraw(args.data_path, False, args.ratio, transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # Modify the first convolutional layer to accept a single channel (grayscale)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    if args.saved_checkpoint is not None and os.path.isfile(args.saved_checkpoint):
        checkpoint = torch.load(args.saved_checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model_params"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        best_acc = -1
        start_epoch = 0

    num_iters = len(train_dataloader)

    for epoch in range(start_epoch, args.num_epochs):
        # Train
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        all_losses = []

        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            pred = model(images)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())
            progress_bar.set_description(f"Epoch: {epoch + 1}/{args.num_epochs}. Loss: {loss:.4f}")
            writer.add_scalar('Train/Loss', np.mean(all_losses), epoch * num_iters + iter)

        # Validation
        model.eval()
        all_losses = []
        all_labels, all_preds = [], []
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour="green")
            for iter, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(device), labels.to(device)

                pred = model(images)
                loss = criterion(pred, labels)
                all_losses.append(loss.item())

                predicts = torch.argmax(pred, dim=1)

                all_labels.extend(labels.tolist())
                all_preds.extend(predicts.tolist())

        acc_score = accuracy_score(all_labels, all_preds)
        loss_value = np.mean(all_losses)

        print(f"Epoch: {epoch + 1}/{args.num_epochs}. Accuracy: {acc_score:.4f}. Loss: {loss_value:.4f}")
        writer.add_scalar('Val/Loss', loss_value, epoch)
        writer.add_scalar('Val/Accuracy', acc_score, epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model_params": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_folder, "last.pt"))
        if best_acc < acc_score:
            torch.save(checkpoint, os.path.join(args.checkpoint_folder, "best.pt"))
            best_acc = acc_score

    writer.close()

if __name__ == '__main__':
    args = get_args()
    train(args)
