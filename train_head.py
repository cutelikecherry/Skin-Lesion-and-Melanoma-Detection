
import argparse
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
 
from feature_cache import load_feature_cache
from model import build_model, save_lightweight_checkpoint
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Train the classifier head on cached features.")
    parser.add_argument("--features_cache", type=str, default="features_cache.pkl")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_out", type=str, default="checkpoints/skin_lesion_model.pkl")
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
 
    cache = load_feature_cache(args.features_cache)
    backbone_name = cache["backbone_name"]
    print(f"Loaded feature cache built with backbone '{backbone_name}': "
          f"{cache['train_features'].shape[0]} train vectors, "
          f"{cache['val_features'].shape[0]} val vectors "
          f"(dim={cache['train_features'].shape[1]})")
 
    train_dataset = TensorDataset(cache["train_features"], cache["train_labels"])
    val_dataset = TensorDataset(cache["val_features"], cache["val_labels"])
 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
 
    if len(train_loader) == 0:
        raise ValueError(
            f"Training set has {len(train_dataset)} cached feature vectors, which is smaller "
            f"than --batch_size {args.batch_size} after drop_last=True. Use a smaller "
            f"--batch_size, or re-run extract_features.py with more --train_augment_copies."
        )
 
    
    model = build_model(backbone_name=backbone_name, pretrained=True, device=device)
    for p in model.backbone.parameters():
        p.requires_grad = False
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=args.lr, weight_decay=1e-4)
 
    best_val_acc = 0.0
    log_every = max(1, args.epochs // 20)
 
    epoch_bar = tqdm(range(args.epochs), desc="Training head", unit="epoch")
    for epoch in epoch_bar:
        model.classifier_head.train()
        running_loss, correct, total = 0.0, 0, 0
 
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.classifier_head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item() * features.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
 
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_acc = evaluate_head(model, val_loader, device)
 
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_lightweight_checkpoint(model, args.checkpoint_out)
 
        epoch_bar.set_postfix(loss=f"{train_loss:.4f}", train_acc=f"{train_acc:.4f}",
                               val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
 
        if (epoch + 1) % log_every == 0 or epoch == args.epochs - 1:
            tqdm.write(f"Epoch [{epoch + 1}/{args.epochs}] "
                       f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
 
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Lightweight checkpoint saved at: {args.checkpoint_out}")
    print("This .pkl file only contains the trained classifier head + metadata, "
          "not the full backbone -- point the Streamlit app's sidebar checkpoint "
          "field at this path to use it.")
 
 
@torch.no_grad()
def evaluate_head(model, loader, device):
    model.classifier_head.eval()
    correct, total = 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model.classifier_head(features)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)
 
 
if __name__ == "__main__":
    main()