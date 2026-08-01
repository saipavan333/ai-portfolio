"""
transfer_learning.py - PRODUCTION computer vision: fine-tune a pretrained model (needs a GPU).

    python -m src.transfer_learning --data-dir /path/to/images   # run on Kaggle GPU

WHAT : fine-tunes a pretrained backbone (ResNet/ViT via `timm`) on YOUR image classes using the
       standard two-stage recipe (train the new head, then gently fine-tune the whole network).
WHY  : you'll never train a vision model from scratch in practice - you stand on a model already
       trained on millions of images and adapt it. This is how real CV gets done.
HOW  : data + augmentation -> pretrained backbone with a fresh head -> stage 1 (head only) ->
       stage 2 (unfreeze, small LR) -> save + push to the Hub.
WHERE: the production layer. convolution.py teaches the operation underneath; this uses learned
       filters at scale. Track runs with Weights & Biases; deploy the demo via app/app.py.

Requires: torch, torchvision, timm  (preinstalled on Kaggle GPU notebooks).
"""
from __future__ import annotations
import argparse


def build_dataloaders(data_dir: str, img_size: int = 224, batch: int = 32):
    # why: train transform AUGMENTS (random crop/flip) to fight overfitting; val does not.
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(f"{data_dir}/train", train_tf)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", val_tf)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2),
            DataLoader(val_ds, batch_size=batch),
            train_ds.classes)


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    import torch
    model.train() if train else model.eval()
    correct = total = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                loss.backward()
                optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with train/ and val/ subfolders")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    import timm, torch, torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, classes = build_dataloaders(args.data_dir)

    # a pretrained backbone with a NEW head sized to your number of classes
    model = timm.create_model(args.backbone, pretrained=True, num_classes=len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # STAGE 1: freeze the backbone, train only the head (fast, stable)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.get_classifier().parameters(), lr=1e-3)
    for _ in range(args.epochs):
        run_epoch(model, train_dl, loss_fn, opt, device, train=True)
        print("stage1 val acc:", run_epoch(model, val_dl, loss_fn, opt, device, train=False))

    # STAGE 2: unfreeze everything, fine-tune gently with a small learning rate
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    for _ in range(args.epochs):
        run_epoch(model, train_dl, loss_fn, opt, device, train=True)
        print("stage2 val acc:", run_epoch(model, val_dl, loss_fn, opt, device, train=False))

    torch.save(model.state_dict(), "model.pt")
    print("saved model.pt - push to the Hugging Face Hub and deploy app/app.py as a Space")


if __name__ == "__main__":
    main()
