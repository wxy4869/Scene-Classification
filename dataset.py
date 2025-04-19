import os

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


LABEL_MAP = {
    'bedroom': 1,
    'Coast': 2,
    'Forest': 3,
    'Highway': 4,
    'industrial': 5,
    'Insidecity': 6,
    'kitchen': 7,
    'livingroom': 8,
    'Mountain': 9,
    'Office': 10,
    'OpenCountry': 11,
    'store': 12,
    'Street': 13,
    'Suburb': 14,
    'TallBuilding': 15,
}


class MyDataset(Dataset):
    def __init__(self, path='./data/train', is_train=True):
        self.path = path
        self.is_train = is_train

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.img_path = []  # e.g. 'data/train/bedroom/image_0001.jpg'
        self.label = []  # 1...15
        for dir in os.listdir(path):
            if not os.path.isdir(os.path.join(path, dir)):
                continue
            for file in os.listdir(os.path.join(path, dir)):
                if file == '.DS_Store':
                    continue
                self.img_path.append(os.path.join(path, dir, file))
                self.label.append(LABEL_MAP[dir])

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        label = self.label[idx]
        img = Image.open(img_path).convert('L')
        transform = self.train_transform if self.is_train else self.test_transform
        img = transform(img)
        return img, label

    def __len__(self):
        return len(self.img_path)
