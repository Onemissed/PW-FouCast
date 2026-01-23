import os
import numpy as np
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    def __init__(self, lists, pangu_list):
        self.lists = lists
        self.pangu_list = pangu_list

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        if 'random' in str(self.lists[index]):
            sequence = np.load(os.path.join('/path/to/weather_data/sevir_lr/data/vil_single/random/', str(self.lists[index]) + '.npy'))
        else:
            sequence = np.load(os.path.join('/path/to/weather_data/sevir_lr/data/vil_single/storm', str(self.lists[index]) + '.npy'))

        pangu_sequence = np.load(os.path.join('/path/to/weather_data/pangu_weather/pangu_interpolate', str(self.pangu_list[index]) + '.npy'))

        return sequence, pangu_sequence