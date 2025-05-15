import os
import random
from functools import partial

import numpy as np

from models.tools.metrics import metric
from src.dataset.tools.interpolator import KrigingInterpolator, Interpolator, IdwInterpolator, RbfInterpolator
from src.models.inference.dataset import Split
from src.models.inference.baseline import BaselineGenerator, Config


class InterpolatorBaseline(BaselineGenerator):
    models = {
        'Kriging': KrigingInterpolator,
        'IDW': IdwInterpolator,
        'RBF_multiquadriq': partial(RbfInterpolator, interpolation_function='multiquadric', epsilon=10),
        'RBF_linear': partial(RbfInterpolator, interpolation_function='linear', epsilon=10)
    }

    def __init__(self, config: Config):
        super().__init__(config)

    def validate(self, model, val_data, val_loader, missing_sensor):
        assert model in self.models, f"Model not in interpolation models"
        model_class: type[Interpolator] = self.models[model]

        gt = []
        pred = []

        for i, batch in enumerate(val_loader):
            for b, item in enumerate(batch):
                item_np = item.numpy()[0]
                x_miss, y_miss, z_miss = item_np[missing_sensor, 0:3]
                x, y, z = np.delete(item_np, missing_sensor, axis=0)[:,0:3].T
                model = model_class(x=x, y=y, z=z)
                z_pred = model.interpolate(x_miss, y_miss)
                gt.append(z_miss)
                pred.append(z_pred.item())

        mae, mse, rmse, mape, mspe = metric(np.array(pred), np.array(gt))
        print(f"Sensor: {val_data.sensors[missing_sensor]}, -> MAE: {mae}, MSE: {mse}")
        return mae, mse, rmse, mape, mspe

    def train(self):
        test_data, test_loader = self.load_data(Split.Test)

        num_sensors = len(test_data.sensors)
        missing_sensor = random.randint(0, num_sensors - 1)

        save_file = os.path.join(self.config.output_folder, "inference_interpolation.txt")
        if not os.path.exists(save_file):
            with open(save_file, 'a') as f:
                f.write(f"missing sensor {missing_sensor}\n")
                f.write("model, sensor, split, mae, mape, mse, mspe, rmse\n")

        with open(save_file, 'a') as f:
            for model in self.config.model:
                mae, mse, rmse, mape, mspe = self.validate(model, test_data, test_loader, missing_sensor)
                f.write(f"{model}, {test_data.sensors[missing_sensor]}, {Split.Test.name}, {mae}, {mape}, {mse}, {mspe}, {rmse}\n")


    def predict(self, x):
        raise NotImplementedError("Predict not implemented for this model")

    def test(self):
        test_data, test_loader = self.load_data(Split.Test)
        self.validate(test_data, test_loader)

if __name__=="__main__":
    config = Config(data_file="dataset/aq_data_mad.csv",
                    output_folder="output/baseline/interpolation",
                    magnitudes=[7],
                    years=[2019],
                    seq_len = 1,
                    batch_size=1,
                    model=["IDW", "Kriging", "RBF_multiquadriq", "RBF_linear"],
                    workers=0)

    interpolator = InterpolatorBaseline(config)
    interpolator.train()
