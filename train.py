import argparse
import os
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
 
from dataset import SyntheticLesionDataset, HAM10000Dataset, get_train_transforms, get_eval_transforms
from model import build_model
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Train the skin lesion classifier.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="efficientnet_b4",
                         choices=["efficientnet_b4", "resnet50"])
    parser.add_argument("--real_data", action="store_true",
                         help="Use HAM10000Dataset (requires --data_dir) instead of synthetic demo data.")
    parser.add_argument("--data_dir", type=str, default="./ham10000_data",
                         help="Path to the downloaded HAM10000 dataset. Required if --real_data is set.")
    parser.add_argument("--num_synthetic_samples", type=int, default=160)
    parser.add_argument("--checkpoint_out", type=str, default="checkpoints/skin_lesion_model.pth")
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
 
    if args.real_data:
        print(f"Loading real HAM10000 data from '{args.data_dir}' ...")
        full_dataset = HAM10000Dataset(data_dir=args.data_dir, transform=get_train_transforms())
    else:
        print("No --real_data flag passed: training on SYNTHETIC demo data. "
              "Predictions from this run are for pipeline-testing only, not clinical use. "
              "See README.md for instructions on training on real HAM10000 data.")
        full_dataset = SyntheticLesionDataset(num_samples=args.num_synthetic_samples,
                                               transform=get_train_transforms())
 
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
 
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
 
    if len(train_loader) == 0:
        raise ValueError(
            f"Training set has only {len(train_ds)} samples, which is smaller than "
            f"--batch_size {args.batch_size} after drop_last=True. Use a smaller "
            f"--batch_size or more samples (--num_synthetic_samples)."
        )
 
    model = build_model(backbone_name=args.backbone, pretrained=True, device=device)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
 
    checkpoint_dir = os.path.dirname(args.checkpoint_out) or "."
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
 
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
 
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for images, labels in batch_bar:
            images, labels = images.to(device), labels.to(device)
 
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
 
            batch_bar.set_postfix(loss=f"{loss.item():.4f}",
                                   running_acc=f"{correct / max(1, total):.4f}")
 
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_acc = evaluate(model, val_loader, device)
 
        print(f"Epoch [{epoch + 1}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
 
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint_out)
            print(f"  -> Saved new best checkpoint to '{args.checkpoint_out}'")
 
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved at: {args.checkpoint_out}")
 
 
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)
 
 
if __name__ == "__main__":
    main()