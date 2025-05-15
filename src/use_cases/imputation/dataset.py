import os.path

import pandas as pd
import torchvision
from torch.utils.data import Dataset


class MetraqImputationDataset(Dataset):
    '''
    Metraq imputation dataset.
    '''
    def __init__(self, data_file: str, magnitude_id: int, fixed_size: int = None, transform: callable = None):
        '''
        Creates an instance of the dataset.

        Args:
            data_file (str): CSV file to read the Metraq imputation data from.
            transform (callable, optional): Transform(s) to apply to the dataset.
        '''
        self.data_file = data_file
        self.transform = transform
        self.fixed_size = fixed_size
        self.magnitude_id = magnitude_id

        self.__init_data()

    def __init_data(self):
        self.data = pd.read_csv(self.data_file, parse_dates=["entry_date"])
        self.data = self.data[self.data["magnitude_id"] == self.magnitude_id]
        self.sequence_ids = self.data['sequence'].unique()
        self.stats = self.data['value'].describe()

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        sequence_idx = self.sequence_ids[idx]

        data = self.data[self.data['sequence'] == sequence_idx].copy()
        gt = data[['value', 'is_interpolated']].copy()
        data.loc[data['is_interpolated']==1, 'value'] = None

        result = (data, gt, self.stats)

        if self.transform:
            return self.transform(result)

        return result

if __name__ == "__main__":
    from tqdm import tqdm

    dataset = MetraqImputationDataset("imputation/data/sequences.csv")

    with tqdm(total=len(dataset)) as pbar:
        for sequence,gt in dataset:
            pass
            pbar.update(1)