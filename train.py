from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


ProgressCallback = Callable[[int, int, float, float, float], None]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_data_loaders(
    data_dir: str = "data",
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.92, 1.08),
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def train_model(
    model: nn.Module,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    progress_callback: Optional[ProgressCallback] = None,
    save_path: str = "saved_model.pth",
) -> list[dict[str, float]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)

        if progress_callback is not None:
            progress_callback(
                epoch,
                epochs,
                train_loss,
                train_accuracy,
                test_accuracy,
            )

    torch.save(model.state_dict(), save_path)
    return history


def load_saved_model(
    model: nn.Module,
    save_path: str = "saved_model.pth",
    device: Optional[torch.device] = None,
) -> bool:
    path = Path(save_path)
    if not path.exists():
        return False

    map_location = device if device is not None else "cpu"
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    model.eval()
    return True
