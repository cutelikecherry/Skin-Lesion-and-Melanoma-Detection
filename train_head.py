import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from feature_cache import load_feature_cache
from model import build_model, save_lightweight_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train the classifier head on cached features with class balancing.")
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
    train_feats, train_labels = cache["train_features"], cache["train_labels"]
    val_feats, val_labels = cache["val_features"], cache["val_labels"]

    print(f"Loaded feature cache built with backbone '{backbone_name}': "
          f"{train_feats.shape[0]} train vectors, {val_feats.shape[0]} val vectors "
          f"(dim={train_feats.shape[1]})")

    # 1. Compute Inverse Class Weights for Loss
    train_labels_np = train_labels.cpu().numpy()
    unique_classes = np.unique(train_labels_np)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_labels_np
    )
    
    # Map class weights array into full 7-class tensor
    num_classes = len(cache["class_names"])
    full_weights = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(unique_classes, class_weights):
        full_weights[cls] = w
    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32).to(device)

    # 2. Build Balanced WeightedRandomSampler for DataLoader
    class_counts = np.bincount(train_labels_np, minlength=num_classes)
    class_sample_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_sample_weights[train_labels_np]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).type(torch.FloatTensor),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = TensorDataset(train_feats, train_labels)
    val_dataset = TensorDataset(val_feats, val_labels)

    # Note: shuffle must be False when sampler is specified
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(backbone_name=backbone_name, pretrained=True, device=device)
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Weighted CrossEntropyLoss directly penalizes majority-class bias
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=args.lr, weight_decay=1e-2)

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
    print(f"Checkpoint saved at: {args.checkpoint_out}")


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