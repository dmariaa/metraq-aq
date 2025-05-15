import numpy as np
import torchvision
from tqdm import tqdm

from src.use_cases.imputation.dataset import MetraqImputationDataset

import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

class GetValue(object):
    def __call__(self, item):
        simulated_failures, failure_times = item
        y_values = simulated_failures["value"].copy()

        values = simulated_failures.copy()
        values.loc[simulated_failures['is_interpolated'] == 1, "value"] = None
        values['weekday'] = values['entry_date'].dt.weekday
        values['monthday'] = values['entry_date'].dt.day
        x_values = values

        return x_values.reset_index(), y_values.reset_index()

transforms = torchvision.transforms.Compose([ GetValue() ])

dataset = MetraqImputationDataset("data/sequences.csv", transform=None)

sequences = []
error = {}

with tqdm(total=len(dataset)) as pbar:
    for (x, y, _) in dataset:
        x['value'] = x['value'].interpolate(method='spline', order=3)

        interpolated_x_values = x.loc[x['is_interpolated'] == 1, 'value']
        interpolated_y_values = y.loc[x['is_interpolated'] == 1, 'value']
        magnitude_id = int(x.iloc[0]['magnitude_id'].item())

        if magnitude_id not in error:
            error[magnitude_id] = []
        error[magnitude_id].append(np.mean(np.abs(np.array(interpolated_x_values)-np.array(interpolated_y_values))))

        pbar.update(1)

for magnitude_id in error.keys():
    print(f"Mean absolute error ({magnitude_id}): {sum(error[magnitude_id]) / len(error[magnitude_id])}")
