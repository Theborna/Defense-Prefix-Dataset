import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

from datasets.configs.sun397_config import templates
from datasets.utils.make_dataset_train import make_image_text


class SUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.

    Args:
        root (string): Root directory of the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        split (str, optional): Specifies the split to use, either 'train' or 'test'.
        num_typographic_images (int, optional): Number of typographic images to generate.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        split: str = 'train',
        num_typographic_images: int = 2
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._data_dir = Path(self.root) / "sun" / "images"
        self._typographic_dir = Path(self.root) / "sun" / "typographic_images"
        self._split = Path(self.root) / "configs" / "split_zhou_SUN397.json"
        self.num_typographic_images = num_typographic_images
        self._typographic_image_classes = []

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        idx_to_class = dict((v, k)
                            for k, v in self.class_to_idx.items())
        self.classes = [idx_to_class[i].replace('_', ' ').replace('/', ' ') for i in range(len(idx_to_class))]
        self.templates = templates

        with open(self._split) as f:
            split_dict = json.load(f)

        if split == 'train':
            split_data = split_dict["train"]
        else:
            split_data = split_dict["test"]

        self._all_image_files = [v[0] for v in split_data]
        self._all_labels = [v[1] for v in split_data]

        self._split_image_files = [v[0] for v in split_data]
        self._split_labels = [v[1] for v in split_data]

        self._make_typographic_attack_dataset()

    def __len__(self) -> int:
        return len(self._split_image_files)

    def __getitem__(self, idx):
        image_file = self._image_files[idx]
        typographic_image_files = self._typographic_image_files[idx]
        label = self._labels[idx]
        typographic_label = self._typographic_image_classes[idx]
        image = PIL.Image.open(image_file).convert("RGB")
        typographic_images = [PIL.Image.open(typographic_image_file).convert("RGB") for typographic_image_file in typographic_image_files]

        if self.transform:
            image = self.transform(image)
            typographic_images = [self.transform(typographic_image) for typographic_image in typographic_images]

        if self.target_transform:
            label = self.target_transform(label)    

        return image, typographic_images, label, typographic_label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()
    
    def _check_exists_synthesized_dataset(self) -> bool:
        return self._typographic_dir.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)

    def _make_typographic_attack_dataset(self) -> None:
        if self._check_exists_synthesized_dataset():
            return
        for i, file in tqdm(enumerate(self._split_image_files), total=len(self._split_image_files)):
            labels = make_image_text(
                file, self.classes, self._data_dir, self._typographic_dir, self._split_labels[i], 
                num_typographic_images=self.num_typographic_images
            )
            self._typographic_image_classes.append(labels)