import numpy as np
from torch import nn


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

class ToKeras(nn.Module):
    def forward(self, item: dict):
        x = item['x'].sort_values(['sensor_id','entry_date'])[item['x'].columns[2:]].values.flatten()
        y = item['y'].sort_values(['sensor_id','entry_date'])[item['y'].columns[2:]].values.flatten()

        item['x'] = x
        item['y'] = y
        return item

    def restore_shape(self, item):
        n_sensors = len(item['sensors'])
        n_x_magnitudes = len(item['x_magnitudes'])
        n_y_magnitudes = len(item['y_magnitudes'])
        n_x_steps = len(item['x_dates'])
        n_y_steps = len(item['y_steps'])
        x_reshaped = item['x'].reshape(n_sensors * n_x_steps, n_x_magnitudes)
        y_reshaped = item['y'].reshape(n_sensors * n_y_steps, n_y_magnitudes)
        return x_reshaped, y_reshaped