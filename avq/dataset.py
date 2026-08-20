import copy
from PIL import Image
from pathlib import Path
from datasets import load_dataset

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from avq.helpers import seed_worker


def create_index_to_label(words_file, train_loader):
    """
    Create a mapping from class index to human-readable label.

    words_file: Path to Tiny ImageNet words.txt file.
    train_loader: PyTorch DataLoader whose dataset contains the classes attribute.
    """
    # Parse WordNet ID -> Human Label
    wnid_to_label = {}
    with open(words_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                wnid_to_label[parts[0]] = parts[1]

    # Create index -> label mapping
    index_to_label = {
        idx: wnid_to_label[wnid]
        for idx, wnid in enumerate(train_loader.dataset.classes)
    }

    return index_to_label


class TinyImageNetValidationDataset(Dataset):
    """
    Load the original Tiny ImageNet validation layout:

      val/
      ├── images/
      │   ├── val_0.JPEG
      │   ├── val_1.JPEG
      │   └── ...
      └── val_annotations.txt

    Each line of val_annotations.txt begins with:
        image_filename    class_id    ...
    """

    def __init__(self, val_directory, class_to_idx, transform):
        super().__init__()
        self.val_directory = Path(val_directory)
        self.images_directory = self.val_directory / "images"
        self.annotation_path = self.val_directory / "val_annotations.txt"
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples = []
        with self.annotation_path.open("r") as file:
            for line_number, line in enumerate(file, start=1):
                fields = line.strip().split("\t")
                image_name = fields[0]
                class_id = fields[1]
                image_path = self.images_directory / image_name
                class_index = self.class_to_idx[class_id]
                self.samples.append((str(image_path), class_index))
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def build_tiny_imagenet_loaders(
    root_directory,
    batch_size=64,
    num_workers=4,
    image_size=64,
    seed=15,
    drop_last_train=True,
):
    """
    Build Tiny ImageNet train, deterministic train-evaluation,
    and validation loaders.

    Returns:
        train_loader:
            Uses light random augmentation.
        train_eval_loader:
            Uses deterministic preprocessing on the training set.
            Useful for final training-set metrics.
        val_loader:
            Uses deterministic preprocessing.
        class_to_idx:
            Mapping from WordNet class IDs to integer labels.
    """
    root = Path(root_directory)
    train_directory = root / "train"
    val_directory = root / "val"

    # Model outputs use Tanh, so map images from [0, 1] to [-1, 1].
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=str(train_directory),
        transform=train_transform,
    )

    # Same training files (w/ deterministic preprocessing)
    train_eval_dataset = copy.copy(train_dataset)
    train_eval_dataset.transform = evaluation_transform

    class_to_idx = train_dataset.class_to_idx
    
    val_dataset = TinyImageNetValidationDataset(
        val_directory=str(val_directory),
        class_to_idx=class_to_idx,
        transform=evaluation_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    pin_memory = use_cuda
    persistent_workers = num_workers > 0

    common_loader_arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=drop_last_train,
        generator=generator,
        **common_loader_arguments,
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    words_file = root / "words.txt"
    idx_to_class = create_index_to_label(words_file, train_loader)

    print("Tiny ImageNet data loaded")
    print(f"Root: {root}")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Training images: {len(train_dataset):,}")
    print(f"Training-evaluation images: {len(train_eval_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")
    print(f"Training batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")

    return train_loader, train_eval_loader, val_loader, class_to_idx, idx_to_class



