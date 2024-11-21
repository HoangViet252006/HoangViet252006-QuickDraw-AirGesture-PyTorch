from os import listdir
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision.transforms import Compose, ToTensor


class QuickDraw(Dataset):
    def __init__(self, data_path, is_train, ratio, transform = None):
        self.data = []
        self.labels = []

        subdir = [os.path.join(data_path, sub) for sub in listdir(data_path)]
        self.categories = [os.path.basename(sub) for sub in subdir]
        self.categories = [cate.split("_")[-1].replace(".npy", "") for cate in self.categories]

        for sub in subdir:
            images = np.load(sub)
            length = images.shape[0]
            file_name = os.path.basename(sub)
            label = self.categories.index(file_name.split("_")[-1].replace(".npy", ""))
            if is_train:
                self.data.extend(images[:int(length * ratio)])
                self.labels.extend([label] * int(length * ratio))
            else:
                length -= int(length * ratio)
                self.data.extend(images[-length:])
                self.labels.extend([label] * length)

        self.transform = transform
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.data[index]
        image = np.reshape(image, (28, 28))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
    ])
    data_path = "data"
    dataset = QuickDraw(data_path, False, 0.8, transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        drop_last=False,
        num_workers=0,
        shuffle=False,
    )

    for iter, (i, l) in enumerate(dataloader):
        print(l)





