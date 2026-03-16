import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DEFAULT_DATA_ROOT = Path("data/external/food11/food11")
DEFAULT_OUTPUT = Path("artifacts/food11_resnet18.pt")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dirs(data_root):
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "test"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Food-11 train/test dirs not found under: {data_root}. "
            "Expected data/external/food11/food11/train and /test"
        )
    return train_dir, val_dir


def build_loaders(train_dir, val_dir, batch_size, num_workers):
    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_ds, val_ds, train_loader, val_loader


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=0):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    steps = 0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(logits, targets)
        steps += 1

        if max_batches > 0 and batch_idx >= max_batches:
            break

    return running_loss / max(steps, 1), running_acc / max(steps, 1)


def evaluate(model, loader, criterion, device, max_batches=0):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    steps = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item()
            running_acc += accuracy(logits, targets)
            steps += 1

            if max_batches > 0 and batch_idx >= max_batches:
                break

    return running_loss / max(steps, 1), running_acc / max(steps, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 on Food-11")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_dir, val_dir = resolve_dirs(args.data_root)
    train_ds, val_ds, train_loader, val_loader = build_loaders(
        train_dir, val_dir, args.batch_size, args.num_workers
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.max_val_batches,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            class_names = [
                name for name, _ in sorted(train_ds.class_to_idx.items(), key=lambda item: item[1])
            ]
            checkpoint = {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": 224,
                "best_val_acc": best_val_acc,
            }
            torch.save(checkpoint, output_path)
            print(f"Saved checkpoint: {output_path} (val_acc={best_val_acc:.4f})")

    print("Training completed.")


if __name__ == "__main__":
    main()