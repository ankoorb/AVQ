from PIL import Image
from pathlib import Path
from datasets import load_dataset

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from avq.helpers import seed_worker


def create_index_to_label(words_file, class_to_idx):
    """
    Create a mapping from class index to human-readable label.

    words_file: Path to Tiny ImageNet words.txt file.
    class_to_idx: Mapping from WordNet class ID to integer class index
    """
    # Parse WordNet ID -> Human Label
    wnid_to_label = {}
    with open(words_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                wnid, label = parts
                wnid_to_label[wnid] = label

    # Create index -> label mapping
    idx_to_label = {
        idx: wnid_to_label[wnid] for wnid, idx in class_to_idx.items()
    }

    return idx_to_label


def create_stratified_split(targets, num_classes, eval_fraction=0.05, seed=15):
    """
    Create deterministic stratified train/eval indices.

    For Tiny ImageNet: 500 images/class => 475 train/class, 25 eval/class
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    eval_indices = []
    train_indices = []
    targets = torch.tensor(targets)
    for class_idx in range(num_classes):
        class_indices = torch.where(targets == class_idx)[0]
        permutation = torch.randperm(len(class_indices), generator=generator)
        class_indices = class_indices[permutation]
        num_eval = int(len(class_indices) * eval_fraction)
        eval_indices.extend(class_indices[:num_eval].tolist())
        train_indices.extend(class_indices[num_eval:].tolist())

    return train_indices, eval_indices


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
    eval_fraction=0.05,
    drop_last_train=True,
):
    """
    Build Tiny ImageNet train, deterministic train-evaluation,
    and validation loaders.

    Returns:
        train_loader:
            Uses light random augmentation.
        train_eval_loader:
            Uses deterministic preprocessing on a held-out 5% split of the 
            original training set. Used for checkpoint selection and early stopping.
        val_loader:
            Uses the official Tiny ImageNet validation set with deterministic 
            preprocessing. Used for final evaluation.
        class_to_idx:
            Mapping from WordNet class IDs to integer labels.
        idx_to_label:
            Mapping from index to label
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

    full_train_dataset = datasets.ImageFolder(
        root=str(train_directory),
        transform=train_transform,
    )

    class_to_idx = full_train_dataset.class_to_idx

    # Same training files (w/ deterministic preprocessing)
    full_train_eval_dataset = datasets.ImageFolder(
        root=str(train_directory),
        transform=evaluation_transform,
    )

    train_indices, train_eval_indices = create_stratified_split(
        targets=full_train_dataset.targets,
        num_classes=len(class_to_idx),
        eval_fraction=eval_fraction,
        seed=seed,
    )

    train_dataset = torch.utils.data.Subset(
        full_train_dataset, train_indices
    )
    train_eval_dataset = torch.utils.data.Subset(
        full_train_eval_dataset, train_eval_indices
    )
    
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
    idx_to_label = create_index_to_label(words_file, class_to_idx)

    print("Tiny ImageNet data loaded")
    print(f"Root: {root}")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Training images: {len(train_dataset):,}")
    print(f"Training-evaluation images: {len(train_eval_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")
    print(f"Training batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")

    return train_loader, train_eval_loader, val_loader, class_to_idx, idx_to_label


class MiniPlacesDataset(Dataset):

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.targets = []

        text_file = root_dir / "devkit" / "data" / f"{split}.txt"
        with open(text_file, 'r') as f:
            for line in f:
                fields = line.strip().split()
                img_name = fields[0]
                class_id = fields[1]
                image_path = self.root_dir / img_name
                class_idx = int(class_id)
                self.samples.append((str(image_path), class_idx))
                self.targets.append(class_idx)
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def build_mini_places_loaders(
    root_directory,
    batch_size=16,
    num_workers=4,
    image_size=128,
    seed=15,
    eval_fraction=0.05,
    drop_last_train=True,
):
    """
    Build MiniPlaces train, deterministic train-evaluation,
        and validation loaders.
    
    Returns:
        train_loader:
            Uses light random augmentation.
        train_eval_loader:
            Uses deterministic preprocessing on a held-out 5% split of the 
            original training set. Used for checkpoint selection and early stopping.
        val_loader:
            Uses the official MiniPlaces validation set with deterministic 
            preprocessing. Used for final evaluation.
        class_to_idx:
            Mapping from class IDs to integer labels.
        idx_to_label:
            Mapping from index to label
    """
    root = Path(root_directory)

    # Get class to idx and idx to label dicts
    categories_file = root / "devkit" / "data" / "categories.txt"
    with open(categories_file, 'r') as f:
        class_to_idx = {}
        idx_to_label = {}
        
        for line in f:
            path, label = line.strip().split()
            scene_name = "_".join(path.strip('/').split('/')[1:])
            class_to_idx[scene_name] = int(label)
            idx_to_label[int(label)] = scene_name

    
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

    full_train_dataset = MiniPlacesDataset(
        root_dir=root, split="train", transform=train_transform,
    )

    full_train_eval_dataset = MiniPlacesDataset(
        root_dir=root, split="train", transform=evaluation_transform,
    )

    train_indices, train_eval_indices = create_stratified_split(
        targets=full_train_dataset.targets,
        num_classes=len(class_to_idx),
        eval_fraction=eval_fraction,
        seed=seed,
    )

    train_dataset = torch.utils.data.Subset(
        full_train_dataset, train_indices
    )
    train_eval_dataset = torch.utils.data.Subset(
        full_train_eval_dataset, train_eval_indices
    )

    val_dataset = MiniPlacesDataset(
        root_dir=root, split="val", transform=evaluation_transform,
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

    print("MiniPlaces data loaded")
    print(f"Root: {root}")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Training images: {len(train_dataset):,}")
    print(f"Training-evaluation images: {len(train_eval_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")
    print(f"Training batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")
    
    return train_loader, train_eval_loader, val_loader, class_to_idx, idx_to_label
    

    

# TODO: COCO 2017 data