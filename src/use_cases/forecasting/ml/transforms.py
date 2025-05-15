import numpy as np
from torch import nn
from tqdm import tqdm

from src.models.ml.dataset import MLDataset


class AddWeekDay(nn.Module):
    def forward(self, item):
        item['x'] = item['x'].copy()
        item['x_magnitudes'] = item['x_magnitudes'].copy()

        item['x']['99000'] = item['x']['entry_date'].dt.weekday / 6.0
        item['x_magnitudes'] += [(99000, 'weekday')]
        return item


class AddHour(nn.Module):
    def forward(self, item):
        item['x'] = item['x'].copy()
        item['x_magnitudes'] = item['x_magnitudes'].copy()

        item['x']['99001'] = item['x']['entry_date'].dt.hour / 23.0
        item['x_magnitudes'] += [(99001, 'hour')]
        return item


class AddWeekNumber(nn.Module):
    def forward(self, item):
        item['x'] = item['x'].copy()
        item['x_magnitudes'] = item['x_magnitudes'].copy()

        item['x']['99002'] = item['x']['entry_date'].dt.isocalendar().week.astype(float) / 53.0
        item['x_magnitudes'] += [(99002, 'week_number')]
        return item


class AddMonth(nn.Module):
    def forward(self, item):
        item['x'] = item['x'].copy()
        item['x_magnitudes'] = item['x_magnitudes'].copy()

        item['x']['99003'] = item['x']['entry_date'].dt.month / 12
        item['x_magnitudes'] += [(99003, 'month')]
        return item

class Normalize(nn.Module):
    def forward(self, item):
        stats = item['stats']
        for magnitude_id, magnitude_name in item['x_magnitudes']:
            min = stats[magnitude_id]['min']
            max = stats[magnitude_id]['max']
            item['x'][magnitude_id] = (item['x'][magnitude_id] - min) / (max - min)

        for magnitude_id, magnitude_name in item['y_magnitudes']:
            min = stats[magnitude_id]['min']
            max = stats[magnitude_id]['max']
            item['y'][magnitude_id] = (item['y'][magnitude_id] - min) / (max - min)

        return item


def compact_data(data: MLDataset, common_sensors: list):
    x = []
    y = []
    x_dates = []
    y_dates = []

    with tqdm(total=len(data)) as pbar:
        for i in range(len(data)):
            row = data[i]

            # transform and filter x data
            row_x = row['x']
            row_x = row_x[row_x['sensor_id'].isin(common_sensors)].sort_values(['sensor_id', 'entry_date'])
            x.append(row_x[row_x.columns[2:]].values.flatten())

            # transform and filter y data
            row_y = row['y']
            row_y = row_y[row_y['sensor_id'].isin(common_sensors)].sort_values(['sensor_id', 'entry_date'])
            y.append(row_y[row_y.columns[2:]].values.flatten())

            # collect x and y dates
            x_dates += row['x_dates']
            y_dates += row['y_dates']
            pbar.update(1)

    meta = {
        'x_magnitudes': row['x_magnitudes'],
        'y_magnitudes': row['y_magnitudes'],
        'sensors': common_sensors,
        'x_steps': row['x_steps'],
        'y_steps': row['y_steps'],
        'x_dates': sorted(list(set(x_dates))),
        'y_dates': sorted(list(set(y_dates))),
        'stats': data.stats
    }

    pbar.close()

    return np.array(x), np.array(y), meta


def expand_data(data_x, data_y, metadata):
    n_sensors = len(metadata['sensors'])
    data_x_reshaped = None
    data_y_reshaped = None

    if data_x is not None:
        n_x_magnitudes = len(metadata['x_magnitudes'])
        n_x_steps = metadata['x_steps']
        data_x_reshaped = data_x.reshape(-1, n_sensors, n_x_steps, n_x_magnitudes)

    if data_y is not None:
        n_y_magnitudes = len(metadata['y_magnitudes'])
        n_y_steps = metadata['y_steps']
        data_y_reshaped = data_y.reshape(-1, n_sensors, n_y_steps, n_y_magnitudes)

    return data_x_reshaped, data_y_reshaped