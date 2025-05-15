import time
from enum import Enum

from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.dataset.metraq_air_quality import MetraqAirQualityDataset


class Split(Enum):
    Train = 0
    Val = 1
    Test = 2


class InferenceDataset(MetraqAirQualityDataset):
    coords_columns = ['utm_x', 'utm_y']
    date_columns = ['hour', 'day_of_week', 'day_of_month', 'day_of_year']

    def __init__(self, data_file: str,
                 magnitudes: list,
                 years: list,
                 split: Split,
                 seq_len: int,
                 transform: callable = None):
        super().__init__(data_file, transform=transform, include_interpolations=True,
                         call_init_data=False)

        self._magnitudes = magnitudes
        self._years = years
        self._sensors = None
        self.split = split
        self.seq_len = seq_len

        self._init_data()

    @property
    def sensors(self):
        return self._sensors

    def _filter_chunk(self, chunk):
        chunk['year'] = chunk['entry_date'].dt.year
        condition = pd.Series(True, index=chunk.index)
        condition &= (chunk['magnitude_id'].isin(self._magnitudes))
        condition &= (chunk['year'].isin(self._years))
        if self._exclude_interpolations():
            condition &= (chunk['is_interpolated']==0)
        return chunk[condition]

    def _load_data_filtering(self):
        print("Reading data...", end="", flush=True)
        data = []

        reader = pd.read_csv(self.data_file, parse_dates=['entry_date'], chunksize=2500000)
        for chunk in reader:
            chunk['year'] = chunk['entry_date'].dt.year
            data.append(
                self._filter_chunk(chunk=chunk)
            )
        self._data = pd.concat(data, ignore_index=True)
        self._data = self._data.sort_values(by=['sensor_id', 'magnitude_id', 'entry_date'])
        print(f"done")

    def _init_data(self):
        self._load_data_filtering()

        # replaced with loading and filtering function
        # self._read_data()
        # self._exclude_interpolations()
        # self._filter_data()

        self._sensors = self._data['sensor_id'].unique()
        self._pivot_data()
        self._get_time_features()
        self._generate_dataset()

    def _extract_metadata(self):
        pass

    def _pivot_data(self):
        print("Pivoting data...", end="", flush=True)
        df_values = self._data.pivot_table(
            index=['entry_date', 'sensor_id'],
            columns='magnitude_name',
            values='value'
        ).reset_index()

        self.magnitude_columns = list(df_values.columns.difference(['entry_date', 'sensor_id']))

        cols_to_keep = self._data.columns.difference(['magnitude_id', 'magnitude_name', 'value', 'is_valid',
                                                      'is_interpolated'])
        df_extra = self._data[cols_to_keep].drop_duplicates(subset=['entry_date', 'sensor_id'])
        self._data = df_extra.merge(df_values, on=['entry_date', 'sensor_id'])
        print("done")

    def _filter_data(self):
        print("Filtering data...", end="", flush=True)
        self.__orig_data = self._data   # debugging purposes only
        self._data['year'] = self._data['entry_date'].dt.year

        self._data = self._data[
            (self._data["magnitude_id"].isin(self._magnitudes)) &
            (self._data['year'].isin(self._years))
        ]
        print("done")

    def _generate_dataset(self):
        print(f"Generating dataset data...", end="", flush=True)
        self.columns = self.coords_columns + self.magnitude_columns  + self.date_columns

        # self._df_data = self._data[['entry_date'] + self.columns].groupby('entry_date')

        data = self._data[['entry_date'] + self.columns].set_index('entry_date')
        self.scaler = StandardScaler()
        self.scaler.fit(data[self.magnitude_columns])
        data.loc[:, self.magnitude_columns] = self.scaler.transform(data[self.magnitude_columns])
        self._df_data = data.groupby(data.index)

        self._data = list(self._df_data.groups.keys())

        num_records = len(self._data)
        num_train = int(num_records * 0.7)
        num_test = int(num_records * 0.2)
        num_vali = num_records - num_train - num_test
        border1s = [0, num_train - self.seq_len, num_train + num_vali - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_records]
        split_start = border1s[self.split.value]
        split_end = border2s[self.split.value]
        self._data = self._data[split_start:split_end]
        print("Done")

    def __len__(self):
        return len(self._data) - self.seq_len

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        keys = self._data[s_begin:s_end]
        x = [
            self._df_data.get_group(date)[self.columns].values
            for date in keys
        ]

        if self.transform is not None:
            return self.transform(x)

        return x


if __name__=="__main__":
    import numpy as np
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataset = InferenceDataset("dataset/aq_data_mad.csv", [7], [2019], Split.Train, 6)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Len dataset: {len(dataset)}")
    print(f"Len dataloader: {len(loader)}")
    for i, batch in tqdm(enumerate(loader)):
        pass

