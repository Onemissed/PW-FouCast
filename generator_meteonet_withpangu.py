import os
import numpy as np
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    def __init__(self, lists):
        self.lists = lists

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        sequence = np.load(os.path.join('/path/to/weather_data/meteonet/data/nw/reflectivity_5to20/', str(self.lists[index]) + '.npy'))
        pangu_sequence = np.load(os.path.join('/path/to/weather_data/meteonet_pangu/meteonet_pangu_interpolate/', str(self.lists[index]) + '_upper.npy'))

        return sequence, pangu_sequence