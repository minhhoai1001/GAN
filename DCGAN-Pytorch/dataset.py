import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CelebaDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img = Image.open(os.path.join(self.root, img_name))

        if self.transform is not None:
            img = self.transform(img)

        return img
