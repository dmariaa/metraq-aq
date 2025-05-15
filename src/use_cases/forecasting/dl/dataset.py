from __future__ import annotations
import pandas as pd
from torch.utils.data import Subset, Dataset
from torchvision.transforms.v2 import Compose
from typing_extensions import override

from src.dataset.metraq_aq_prediction import AQPredictionDataset


class MLDataset(AQPredictionDataset):
    def __init__(self, data_file: str,
                 x_steps: int, y_steps: int,
                 x_magnitudes: list = None, y_magnitudes: list = None,
                 years: list = None,
                 include_interpolations: bool = True, transform: callable = None):
        super().__init__(data_file, x_steps, y_steps, include_interpolations, transform, call_init_data=False)
        self._x_magnitudes = x_magnitudes
        self._y_magnitudes = y_magnitudes
        self._years = years
        super()._init_data()

    @override
    def _preprocess(self):
        self._filter_data()
        self._exclude_interpolations()
        self._pivot_rows()

    @override
    def _get_data_columns(self, magnitudes: list = None):
        if magnitudes is None:
            magnitudes = self._magnitude_columns

        return ['sensor_id', 'entry_date'] + magnitudes

    @override
    def _extract_metadata(self):
        print("Extracting metadata...", end="", flush=True)
        valid_columns = [col for col in self._data.columns if str(col).endswith('_vsq')]

        # list of valid_dates with both sensors
        sensor_count = self._data['sensor_id'].nunique()
        valid_dates = (self._data.groupby('entry_date')[valid_columns].sum().eq(sensor_count).all(axis=1))
        self.entry_dates = list(valid_dates[valid_dates].index)
        self._valid_data = self._data[self._data['entry_date'].isin(self.entry_dates)]

        # self.entry_dates = list(self._valid_data['entry_date'].unique())
        self.sensors = list(self._valid_data['sensor_id'].unique())
        print("done")

    def _filter_data(self):
        '''
        Filters data to include only the records with sensors that have data
        in all requested magnitudes and years
        '''
        if self._x_magnitudes is None and self._y_magnitudes is None:
            return

        self._data['year'] = self._data['entry_date'].dt.year

        magnitude_ids = list(set(self._x_magnitudes + self._y_magnitudes))
        filtered_data = self._data[(self._data['magnitude_id'].isin(magnitude_ids)) &
                                   (self._data['year'].isin(self._years))]

        # Choose only sensors with all years + magnitudes
        presence_table = filtered_data.groupby(['sensor_id', 'magnitude_id', 'year']).size()
        valid_sensors = presence_table.unstack(level=list(range(1, presence_table.index.nlevels)))
        sensors = valid_sensors[~valid_sensors.isnull().any(axis=1)].index

        # Get the data for the sensors that have records for all magnitudes and years
        self._data = self._data[(self._data['sensor_id'].isin(sensors)) &
                                (self._data['magnitude_id'].isin(magnitude_ids)) &
                                (self._data['year'].isin(self._years))]

        magnitudes = self._data.groupby(['magnitude_id', 'magnitude_name']).size().reset_index().iloc[:, :-1].values
        self.x_magnitudes = [(mag_id, mag_name) for mag_id, mag_name in magnitudes if mag_id in self._x_magnitudes]
        self.y_magnitudes = [(mag_id, mag_name) for mag_id, mag_name in magnitudes if mag_id in self._y_magnitudes]

    def _pivot_rows(self):
        pivot_df = self.data.pivot_table(
            index=['sensor_id', 'entry_date'],
            columns='magnitude_id',
            values='value'
        )

        self._magnitude_columns = list(pivot_df.columns)

        pivot_df_valid_sequences = self.data.pivot_table(
             index=['sensor_id', 'entry_date'],
             columns='magnitude_id',
             values='valid_sequence'
        )

        pivot_df_interpolated = self.data.pivot_table(
             index=['sensor_id', 'entry_date'],
             columns='magnitude_id',
             values='is_interpolated'
        )

        pivot_df = pd.concat([
            pivot_df,
            pivot_df_valid_sequences.add_suffix('_vsq'),
            pivot_df_interpolated.add_suffix('_itp')
        ], axis=1).reset_index()

        pivot_df.sort_values(by=['entry_date', 'sensor_id'], inplace=True, ignore_index=True)
        self._data = pivot_df

    def __len__(self):
        return len(self.entry_dates)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise ValueError("Index is not valid.")

        if index < 0 or index >= len(self.entry_dates):
            raise IndexError("Index out of range or not valid.")

        sequence_start = self.entry_dates[index]
        date_range = pd.date_range(start=sequence_start, periods=self.total_steps, freq='h')
        sensors = self._valid_data[self._valid_data['entry_date'].isin(date_range)]['sensor_id'].unique()

        x_dates = date_range[:self.x_steps]
        x = self._data[(self._data['sensor_id'].isin(sensors)) &
                       (self._data['entry_date'].isin(x_dates))][self._get_data_columns(self._x_magnitudes)]
            #.to_dict(orient='split', index=False)

        y_dates = date_range[self.x_steps:]
        y = self._data[(self._data['sensor_id'].isin(sensors)) &
                       (self._data['entry_date'].isin(y_dates))][self._get_data_columns(self._y_magnitudes)]
            #.to_dict(orient='split', index=False)

        row = {
            'x': x,
            'y': y,
            'sensors': list(sensors),
            'x_magnitudes': self.x_magnitudes,
            'y_magnitudes': self.y_magnitudes,
            'x_dates': x_dates.to_pydatetime().tolist(),
            'y_dates': y_dates.to_pydatetime().tolist(),
            'stats': self.stats
        }

        if self.transform is not None:
            row = self.transform(row)

        return row


if __name__ == "__main__":
    from models.ml.transforms import Normalize, AddHour, AddWeekDay, AddWeekNumber, AddMonth, ToKeras
    from tqdm import tqdm
    from rich import print
    from rich.pretty import Pretty

    transforms = Compose([
        Normalize(),
        AddHour(),
        AddWeekDay(),
        AddWeekNumber(),
        AddMonth(),
        ToKeras()
    ])

    dataset_file = "dataset/aq_data_mad.csv"
    dataset: Dataset = MLDataset(data_file=dataset_file, include_interpolations=False,
                        x_steps=12, y_steps=12,
                        years=[2023],
                        x_magnitudes=[7,1000], y_magnitudes=[7])

    with tqdm(total=len(dataset)) as pbar:
        for i, row in enumerate(dataset):
            pbar.update(1)

    # print(f"Iterations over {len(dataset)} rows")
    # for i in range(len(dataset)):
    #     row = dataset[i]
    #     print(f"Iteration {i}", end='\r', flush=True)



    # dataset2 = MLDataset(data_file=dataset_file, include_interpolations=False,
    #                     x_steps=12, y_steps=12,
    #                     years=[2022],
    #                     x_magnitudes=[7,1000], y_magnitudes=[7],
    #                      transform=transforms)
    #
    # print(len(dataset))
    # print(len(dataset2))
