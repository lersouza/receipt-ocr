import json
import torch

from pathlib import Path
from typing import Union
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from transformers import PreTrainedTokenizer, AutoTokenizer


class WikiTextMaskedDataset(Dataset):
    """
    Dataset for pre-training models on synthetic images based on WikiText.
    """
    DEFAULT_TRANSFORMS = transforms.Compose([
        transforms.Resize((512, 256)),
        transforms.ToTensor(),
    ]) 
    RANDOM_TRANSFORMS = transforms.RandomApply(torch.nn.ModuleList([
        transforms.GaussianBlur(3),
        transforms.ColorJitter(),
        transforms.RandomRotation(degrees=(-5, 5)),
    ]), p=0.5)

    def __init__(self,
                 dataset_dir: str,
                 tokenizer: Union[str, PreTrainedTokenizer],
                 target_max_length: int = 256,
                 enhance_factor: int = 1) -> None:
        """
        Initializes the dataset:
        - `dataset_dir`: Directory containing the dataset files (png, json and mask)
        - `tokenizer`: A `PreTrainedTokenizer` or the name of one.
        - `target_max_length`: The maximum length for a target text to be predicted.
        - `enhance_factor`: A factor to scale the dataset.
        """

        self.images = list(Path(dataset_dir).glob('**/*.png'))
        self.labels = list(Path(dataset_dir).glob('**/*.png.json'))
        self.masks = list(Path(dataset_dir).glob('**/*.png.mask'))

        self.images, self.labels, self.masks = (sorted(self.images),
                                                sorted(self.labels),
                                                sorted(self.masks))

        self.enhance_factor = enhance_factor
        self.original_size = len(self.images)

        assert len(self.images) > 0
        assert len(self.images) == len(self.labels) == len(self.masks)
        assert self.images[0].stem + '.png' == self.labels[0].stem == self.masks[0].stem

        self.label_content = []
        self._load_labels()

        self.tokenizer = (tokenizer if isinstance(tokenizer, PreTrainedTokenizer)
                          else AutoTokenizer.from_pretrained(tokenizer))

        self.target_max_length = target_max_length

    def __len__(self):
        """ Returns the length of the dataset. """
        return len(self.images) * self.enhance_factor

    def __getitem__(self, index):
        """
        Returns a dict representing the sample located at `index`:

        - `['image']`: a PIL image transformed by WikiTextMaskedDataset.DEFAULT_TRANSFORMS + RANDOM_TRANSFORMS.
        - `['mask']`: CharGrid mask for the image.
        - `['target']: A Target Text to be predicted, encoded by `self.tokenizer`.        
        """
        actual_index = index % self.original_size

        image, mask, label = (self.images[actual_index],
                              self.masks[actual_index],
                              self.label_content[actual_index])

        image = Image.open(image)
        image = self.RANDOM_TRANSFORMS(self.DEFAULT_TRANSFORMS(image))

        mask = torch.load(mask).transpose(1, 0)

        target_label = self.tokenizer(label['original'],
                                      max_length=self.target_max_length,
                                      padding='max_length',
                                      truncation=True)

        return {'image': image, 'mask': mask.long(),
                'target': torch.tensor(target_label.input_ids, dtype=torch.long)}

    def _load_labels(self):
        """
        Pre load labels, since they are the smallest files.
        """
        for label in self.labels:
            with open(label, 'r') as label_file:
                label_config = json.load(label_file)
            self.label_content.append(label_config)