import json

import torchvision.datasets as Dataset
from PIL import Image

from datasets.configs.food101_config import templates
from datasets.utils.make_dataset_train import make_image_text
from tqdm import tqdm


class Food101(Dataset.Food101):
    def __init__(self, root, split='test', transform=None, download=False, num_typographic_images=2):
        super().__init__(root, split, transform=transform, download=download)
        self._typographic_images_folder = self._base_folder / "typographic_images"
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())
  
        self._typographic_image_files = []
        self._base_image_files = []
        self._typographic_image_classes = []
        self.num_typographic_images = num_typographic_images
        
        for class_label, im_rel_paths in metadata.items():
            self._typographic_image_files += [
                [self._typographic_images_folder.joinpath(*f"{im_rel_path}_{i}.jpg".split("/")) for i in range(self.num_typographic_images)] for im_rel_path in im_rel_paths
            ]
            self._base_image_files += ["/".join(f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths]

        self.classes = [' '.join(class_i.split('_')) for class_i in self.classes]
        self._make_typographic_attack_dataset()

        self.templates = templates

    def __getitem__(self, idx):
        image_file, typographic_image_files, label = self._image_files[idx], self._typographic_image_files[idx], self._labels[idx]
        typographic_label = self._typographic_image_classes[idx]
        image = Image.open(image_file).convert("RGB")
        typographic_images = [Image.open(typographic_image_file).convert("RGB") for typographic_image_file in typographic_image_files]

        if self.transform:
            image = self.transform(image)
            typographic_images = [self.transform(typographic_image) for typographic_image in typographic_images]

        if self.target_transform:
            label = self.target_transform(label)    

        return image, typographic_images, label, typographic_label

    def _check_exists_synthesized_dataset(self) -> bool:
        return self._typographic_images_folder.is_dir()

    def _make_typographic_attack_dataset(self):
        if self._check_exists_synthesized_dataset():
            return
        for i, file in tqdm(enumerate(self._base_image_files), total=len(self._base_image_files)):
            labels = make_image_text(
                file, self.classes, self._images_folder, self._typographic_images_folder,
                self._labels[i], num_typographic=self.num_typographic_images
            )
            self._typographic_image_classes.append(labels)