import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.models.ml.transforms import expand_data


def compute_metrics_keras(y_hat: np.ndarray, y: np.ndarray, metadata: dict, normalize: bool = False):
    _, y_reshaped = expand_data(None, y, metadata)
    _, y_hat_reshaped = expand_data(None, y_hat, metadata)

    mae = []
    mse = []
    rmse = []

    for i, (magnitude_id, magnitude_name) in enumerate(metadata['y_magnitudes']):
        y_eval = y_reshaped[..., i]
        y_hat_eval = y_hat_reshaped[..., i]

        if normalize:
            min = metadata['stats'][magnitude_id]['min']
            max = metadata['stats'][magnitude_id]['max']
            y_eval =  (max - min) * y_eval + min
            y_hat_eval = (max - min) * y_hat_eval + min

        mae.append(np.abs(y_hat_eval - y_eval).mean())
        mse.append(((y_hat_eval - y_eval) ** 2).mean())
        rmse.append(np.sqrt(mse))

    return mae, mse, rmse


def plot_predictions(y_hat: np.ndarray, y: np.ndarray, metadata: dict, output_folder: str, normalize: bool = False):
    _, y_reshaped = expand_data(None, y, metadata)
    _, y_hat_reshaped = expand_data(None, y_hat, metadata)

    dates = list(metadata['y_dates'])
    dates.sort()

    os.makedirs(output_folder, exist_ok=True)

    for i, (magnitude_id, magnitude_name) in enumerate(metadata['y_magnitudes']):
        for j, sensor_id in enumerate(metadata['sensors']):
            y_eval = y_reshaped[..., j, 0, i]
            y_hat_eval = y_hat_reshaped[..., j, 0, i]

            if normalize:
                min = metadata['stats'][magnitude_id]['min']
                max = metadata['stats'][magnitude_id]['max']
                y_eval = (max - min) * y_eval + min
                y_hat_eval = (max - min) * y_hat_eval + min

            df = pd.DataFrame({
                'dates': dates[:len(y)],
                'y': y_eval,
                'y_hat': y_hat_eval
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['dates'], y=df['y'], name="ground truth", mode='lines',
                                     line=dict(color='green', width=1)))
            fig.add_trace(go.Scatter(x=df['dates'], y=df['y_hat'], name="predictions", mode='lines',
                                     line=dict(color='red', width=1)))
            fig.update_layout(title=f"Predictions for sensor {sensor_id}",
                              xaxis_title='Dates',
                              yaxis_title='Values')

            fig.write_html(os.path.join(output_folder, f"sensor-{sensor_id}.html"))




