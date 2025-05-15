import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats


def get_failures_data(df: pd.DataFrame):
    df_failures = df[df["is_interpolated"] == 1].sort_values(by=["sensor_id", "magnitude_id", "entry_date"])

    # Identify the start of a new failure block
    df_failures["new_block"] = df_failures.groupby(["sensor_id", "magnitude_id"])["entry_date"].diff() > pd.Timedelta(hours=1)

    # Create a block ID for each failure block
    df_failures["block_id"] = df_failures.groupby(["sensor_id", "magnitude_id"])["new_block"].cumsum()

    # Compute the total duration of each block (counting consecutive failures)
    block_durations = df_failures.groupby(["sensor_id", "magnitude_id", "block_id"])["entry_date"].count().reset_index()
    block_durations.rename(columns={"entry_date": "duration_hours"}, inplace=True)

    # Merge the computed duration back to the original dataframe
    df_failures = df_failures.merge(block_durations, on=["sensor_id", "magnitude_id", "block_id"], how="left")

    # Compute time differences (MTBF) between failure blocks
    df_failures["time_diff_hours"] = (
        df_failures.groupby(["sensor_id", "magnitude_id"])["entry_date"]
        .transform(lambda x: x.diff().dt.total_seconds() / 3600)
    )

    return df_failures[df_failures["new_block"]]  # Keep only rows marking the start of failure blocks


def plot_distribution_vs_exponential(failures: pd.DataFrame, magnitude_id: int):
    failures_magnitude = failures[failures["magnitude_id"] == magnitude_id]

    if failures_magnitude.empty:
        print(f"⚠️ No data for magnitude {magnitude_id}. Skipping...")
        return None

    mean_diff, mean_duration = (failures_magnitude.groupby(["sensor_id"])[["time_diff_hours", "duration_hours"]].mean()).mean()
    max_diff, max_duration = failures_magnitude[["time_diff_hours", "duration_hours"]].max()

    # plot histogram of time differences
    histogram_trace = go.Histogram(
        x=failures_magnitude['time_diff_hours'],
        nbinsx=100,
        histnorm='probability density',
        marker=dict(color="blue", opacity=0.6),
        name="Datos Reales"
    )

    # plot exponential distribution for time differences
    lambda_hat = 1 / mean_diff
    exp_dist = stats.expon(scale=1/lambda_hat)
    x = np.linspace(0, max_diff, 100)
    pdf_trace = go.Scatter(
        x=x,
        y=exp_dist.pdf(x),
        mode="lines",
        line=dict(color="red", width=2),
        name=f"Distribución Exponencial (λ={lambda_hat:.4f})"
    )

    fig = go.Figure(data=[histogram_trace, pdf_trace])

    fig.update_layout(
        title=f"Ajuste de Distribución Exponencial (Magnitud {magnitude_id})",
        xaxis_title="Tiempo entre fallos (horas)",
        yaxis_title="Densidad de Probabilidad",
        template="plotly_white",
        legend_title="Leyenda"
    )
    return fig


if __name__=="__main__":
    from tqdm import tqdm

    df = pd.read_csv("dataset/aq_data_mad.csv", parse_dates=["entry_date"])
    failures = get_failures_data(df)

    plot_distribution_vs_exponential(failures, magnitude_id=7)
    pass

    # Get the mean time between failures (MTBF) per magnitude
    # mtbf_per_magnitude = failures.groupby(["magnitude_id"])[["time_diff_hours", "duration_hours"]].mean()
    #
    # can also be calculated first by sensor as:
    # mtbf_per_magnitude = failures.groupby([["sensor_id", "magnitude_id"])[["time_diff_hours", "duration_hours"]].mean()
    # mtbf_per_magnitude = failures.groupby(["magnitude_id"])[["time_diff_hours", "duration_hours"]].mean()

    # magnitude_ids = [1, 6, 7, 8, 9, 10, 11, 12, 14, 15, 20, 22, 30, 35, 37, 38, 39, 42, 43, 44, 431]
    #
    # for magnitude_id in tqdm(magnitude_ids):
    #     fig = plot_distribution_vs_exponential(failures, magnitude_id)
    #     if fig:
    #         fig.write_html(f"output/imputation/distribution_exponential_mad_{magnitude_id}.html")