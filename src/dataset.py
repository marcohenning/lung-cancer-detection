import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class LungCancerDataset(Dataset):

    def __init__(self, transform):
        self.dataframe = pd.read_csv('dataset/labels.csv')
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join('dataset', self.dataframe.iloc[idx, 0])
        image = Image.open(image_path)
        image_transformed = self.transform(image)
        label = self.dataframe.iloc[idx, 1]
        return image_transformed, label
