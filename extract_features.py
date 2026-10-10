import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from dataset import HAM10000Dataset, SyntheticLesionDataset, get_eval_transforms, get_train_transforms
from feature_cache import extract_features, save_feature_cache
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache frozen-backbone features for fast CPU classifier-head training."
    )
    parser.add_argument("--backbone", type=str, default="efficientnet_b4",
                        choices=["efficientnet_b4", "resnet50"],
                        help="If extraction is still slow on your CPU, resnet50 is lighter "
                             "than efficientnet_b4 -- try switching.")
    parser.add_argument("--real_data", action="store_true")
    parser.add_argument("--data_dir", type=str, default="./ham10000_data")
    parser.add_argument("--num_synthetic_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Extraction-only batch size. Safe to set much higher than a "
                             "training batch size since no gradients are stored (no backward pass).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker processes for parallel image preprocessing. "
                             "Try 2-4 if you have spare CPU cores.")
    parser.add_argument("--train_augment_copies", type=int, default=3,
                        help="How many independently-augmented feature vectors to cache PER "
                             "training image. Adds cheap data-augmentation diversity into the "
                             "cache at a one-time extraction cost, instead of a per-epoch cost.")
    parser.add_argument("--output", type=str, default="features_cache.pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Backbone: {args.backbone}  (frozen -- used only to extract features, never trained here)")

    if args.real_data:
        print(f"Loading real HAM10000 data from '{args.data_dir}' ...")
        full_eval_ds = HAM10000Dataset(data_dir=args.data_dir, transform=get_eval_transforms())
        full_train_ds = HAM10000Dataset(data_dir=args.data_dir, transform=get_train_transforms())
    else:
        print("No --real_data flag passed: using SYNTHETIC demo data. "
              "See README.md for real HAM10000 download instructions.")
        full_eval_ds = SyntheticLesionDataset(num_samples=args.num_synthetic_samples,
                                              transform=get_eval_transforms())
        full_train_ds = SyntheticLesionDataset(num_samples=args.num_synthetic_samples,
                                               transform=get_train_transforms())

    # Extract target class labels for stratification
    if args.real_data:
        targets = full_eval_ds.metadata["label_name"].values
    else:
        targets = full_eval_ds.labels

    indices = np.arange(len(full_eval_ds))

    # Perform stratified split (80% train, 20% val)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=42
    )

    val_ds = Subset(full_eval_ds, val_indices)
    train_ds = Subset(full_train_ds, train_indices)

    model = build_model(backbone_name=args.backbone, pretrained=True, device=device)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    print(f"\nExtracting VALIDATION features ({len(val_ds)} images, 1 deterministic pass each)...")
    val_features, val_labels = extract_features(model.backbone, val_ds, device=device,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 desc="Validation features")

    print(f"\nExtracting TRAINING features ({len(train_ds)} images, "
          f"{args.train_augment_copies} augmented pass(es) each)...")
    train_feature_chunks, train_label_chunks = [], []
    for copy_idx in range(args.train_augment_copies):
        feats, labels = extract_features(model.backbone, train_ds, device=device,
                                         batch_size=args.batch_size, num_workers=args.num_workers,
                                         desc=f"Train features (pass {copy_idx + 1}/{args.train_augment_copies})")
        train_feature_chunks.append(feats)
        train_label_chunks.append(labels)

    train_features = torch.cat(train_feature_chunks, dim=0)
    train_labels = torch.cat(train_label_chunks, dim=0)

    save_feature_cache(args.output, train_features, train_labels, val_features, val_labels,
                        backbone_name=args.backbone)

    print(f"\nSaved feature cache to '{args.output}'")
    print(f"  train_features: {tuple(train_features.shape)}")
    print(f"  val_features:   {tuple(val_features.shape)}")
    print(f"\nYou can now delete the raw dataset folder ('{args.data_dir if args.real_data else '(synthetic, nothing to delete)'}') "
          f"and run train_head.py using only '{args.output}'.")


if __name__ == "__main__":
    main()