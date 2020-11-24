from pathlib import Path

from torch.utils.data import Dataset
from torchvision.transforms import transforms


class WikiTextMaskedDataset(Dataset):
    def __init__(self, dataset_dir: str, enhance_factor: int = 3) -> None:
        self.images = list(Path(dataset_dir).glob('**/*.png'))
        self.labels = list(Path(dataset_dir).glob('**/*.png.json'))
        self.masks = list(Path(dataset_dir).glob('**/*.png.mask'))

        self.images, self.labels, self.masks = (
            sorted(self.images),
            sorted(self.labels),
            sorted(self.masks))            

        self.enhance_factor = enhance_factor
        self.original_size = len(self.images)

        assert len(self.images > 0)
        assert len(self.images) == len(self.labels) == len(self.masks)
        assert self.images[0].stem == self.labels[0].stem == self.masks[0].stem

        self._load_labels()

    def __len__(self):
        return len(self.images) * self.enhance_factor

    def __getitem__(self, index):
        actual_index = index % self.original_size

        image, mask, label = (
            self.images[actual_index], self.masks[actual_index],
            self.labels[actual_index])

        mask = self.masks[index % self.original_size]
        return None

    def _load_labels(self):
        pass