import pandas as pd
import tqdm
from typing_extensions import override

from src.dataset.metraq_air_quality import MetraqAirQualityDataset


class AQPredictionDataset(MetraqAirQualityDataset):
    def __init__(self, data_file: str, x_steps: int, y_steps: int,
                 include_interpolations: bool = False,
                 transform: callable = None, call_init_data: bool = True):
        super().__init__(data_file, include_interpolations=include_interpolations, transform=transform,
                         call_init_data=False)
        self._data_columns = None
        self.x_steps = x_steps
        self.y_steps = y_steps
        self.total_steps = self.x_steps + self.y_steps
        self.include_interpolations = include_interpolations

        # Now initialization can be performed
        if call_init_data:
            self._init_data()

    @override
    def _exclude_interpolations(self, valid_sequence_columns: list = None):
        if valid_sequence_columns is None:
            valid_sequence_columns = ['sensor_id', 'magnitude_id']

        self._data['valid_sequence'] = True

        print("Generating valid sequences...", flush=True)
        # Identify indices of interpolated rows
        interpolated_indices = pd.Index([]) if self.include_interpolations else self._data.index[self._data['is_interpolated'] == 1]

        # Iterate through unique sensor_id and magnitude_id combinations
        idf = self._data.groupby(valid_sequence_columns)
        with tqdm.tqdm(total=len(idf)) as pbar:
            for (sensor_id, magnitude_id), group_df in idf:
                # Find interpolated rows within this group
                group_interpolated_indices = interpolated_indices.intersection(group_df.index)

                # For each interpolated row, mark a range of rows in the boolean mask
                for idx in group_interpolated_indices:
                    current_pos = group_df.index.get_loc(idx)

                    # Determine the valid range for this group
                    first = max(0, current_pos - self.total_steps + 1)
                    last = current_pos
                    indices_to_update = group_df.iloc[first:last + 1].index
                    self._data.loc[indices_to_update, 'valid_sequence'] = False

                # Mark the last `total_steps - 1` rows in each group as invalid
                last_indices_to_update = group_df.iloc[-(self.total_steps - 1):].index
                self._data.loc[last_indices_to_update, 'valid_sequence'] = False

                # Update the number of valid rows for the group
                pbar.update(1)

        print("...done")

    @override
    def _extract_metadata(self):
        super()._extract_metadata()
        self._valid_indices = self._data[self._data['valid_sequence']].index
        self.entry_dates = list(self._data[self._data['valid_sequence']]['entry_date'].unique())

    def _get_data_columns(self):
        if self._data_columns is None:
            self._data_columns = [c for c in self._data if c != 'valid_sequence']
        return self._data_columns

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, index):
        if not isinstance(index, int) or index < 0 or index >= len(self._valid_indices):
            raise ValueError("Index out of range or not valid.")

        # Get the starting index for the sequence
        start_index = self._valid_indices[index]

        start = self._data.index.get_loc(start_index)
        x_end = start + self.x_steps
        y_end = x_end + self.y_steps

        # Extract the full sequence for x and y
        x = self._data.iloc[start:x_end][self._get_data_columns()].to_dict(orient='records')
        y = self._data.iloc[x_end:y_end][self._get_data_columns()].to_dict(orient='records')

        row = {
            'x': x,
            'y': y
        }

        if self.transform is not None:
            row = self.transform(row)

        return row

if __name__ == "__main__":
    from rich import print
    from rich.pretty import Pretty

    dataset_file = "dataset/aq_data_mad.csv"
    dataset = AQPredictionDataset(data_file=dataset_file, x_steps=12, y_steps=12,
                                  include_interpolations=True)
    row = dataset[0]
    print(Pretty(row))
