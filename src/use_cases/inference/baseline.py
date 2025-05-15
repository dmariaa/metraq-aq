import os.path
from typing import List

from torch.utils.data import DataLoader
from dataclasses import dataclass

from src.models.inference.dataset import InferenceDataset, Split

@dataclass
class Config:
    data_file: str
    magnitudes: list
    years: list
    seq_len: int

    output_folder: str

    model: List[str]

    batch_size: int
    workers: int = 4

def data_factory(config: Config, split: Split):
    dataset = InferenceDataset(data_file=config.data_file,
                               magnitudes=config.magnitudes,
                               years=config.years,
                               seq_len=config.seq_len,
                               split=split)

    shuffle = True if split == split.Train else False
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=shuffle,
                            num_workers=config.workers)

    return dataset, dataloader

class BaselineGenerator:
    def __init__(self, config: Config):
        self.config = config

        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)

    def load_data(self, split: Split):
        data_set, data_loader = data_factory(config=self.config, split=split)
        print(f"{split.name} ({len(data_set), len(data_loader)})")
        return data_set, data_loader

    def train(self):
        raise NotImplementedError("Train not implemented for this model")

    def predict(self, x):
        raise NotImplementedError("Predict not implemented for this model")

    def test(self):
        raise NotImplementedError("Test not implemented for this model")