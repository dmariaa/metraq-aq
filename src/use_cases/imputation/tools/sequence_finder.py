import pandas as pd
import scipy.stats as stats
import numpy as np

from models.imputation.tools.exponencial_missing_values import get_failures_data

def generate_random_failures(sequence: pd.DataFrame, mean_mtbf: float, mean_duration: float, min_duration: int = 1,
                             random_seed: int = None):
    """
    Generates random failures based on the estimated failure rate (MTBF) per magnitude.
    """
    df_simulated = sequence.copy()

    # Prevent division by zero
    if pd.isna(mean_mtbf) or mean_mtbf == 0:
        return df_simulated

    start_time = df_simulated["entry_date"].min()
    end_time = df_simulated["entry_date"].max()

    # Generate failure timestamps using an exponential distribution
    failure_times = []
    current_time = start_time
    while current_time < end_time:
        hours_to_next_failure = max(1, int(stats.expon.rvs(scale=mean_mtbf, size=1, random_state=random_seed)[0]))
        failure_duration = max(min_duration, int(np.ceil((stats.expon.rvs(scale=mean_duration, size=1,
                                      random_state=random_seed)[0]))))
        next_failure = current_time + pd.Timedelta(hours=hours_to_next_failure)

        if next_failure > end_time:
            break

        failure_period = pd.date_range(start=next_failure,
                               periods=min(failure_duration, int((end_time - next_failure).total_seconds() // 3600)),
                               freq="h")
        failure_times.extend(failure_period)

        current_time = next_failure + pd.Timedelta(hours=failure_duration)

    # If it doesn't generate any failure, generate at least one random failure
    if len(failure_times) == 0:
        hours_to_next_failure = int(np.random.randint(1, (end_time - start_time).total_seconds() // 3600))
        failure_duration = max(min_duration, int(np.ceil((stats.expon.rvs(scale=mean_duration, size=1)[0]))))
        next_failure = start_time + pd.Timedelta(hours=hours_to_next_failure)

        failure_period = pd.date_range(start=next_failure,
                               periods=min(failure_duration, int((end_time - next_failure).total_seconds() // 3600)),
                               freq="h")

        failure_times.extend(failure_period)

    # Update existing timestamps in the dataset
    df_simulated.loc[df_simulated['entry_date'].isin(failure_times), "is_interpolated"] = 1

    return df_simulated, failure_times

def get_continuous_sequences(df: pd.DataFrame, min_intervals: int = 1, exact_intervals: bool = False):
    df = df.sort_values(by=["sensor_id", "magnitude_id", "entry_date"])

    # Identify breaks in non-interpolated sequences
    df["break"] = (df["is_interpolated"].shift(1) == 1) | (df["is_interpolated"] == 1)

    # Identify large time gaps between consecutive non-interpolated entries
    df["time_diff"] = df.groupby(["sensor_id", "magnitude_id"])["entry_date"].diff()

    # Detect missing years
    df["year_diff"] = df["entry_date"].dt.year.diff()
    df["missing_year"] = df["year_diff"] > 1  # True if more than 1 year is missing

    # Create a new break when missing years are detected
    df["break"] = df["break"] | df["missing_year"]

    # Assign a unique group ID to each contiguous block
    df["group"] = df.groupby(["sensor_id", "magnitude_id"])["break"].cumsum()

    # Filter only non-interpolated data
    df_filtered = df[df["is_interpolated"] == 0]

    # Find the longest sequence per sensor and magnitude
    continuous_sequences = (
        df_filtered.groupby(["sensor_id", "magnitude_id", "group"])
        .agg(start_date=("entry_date", "min"),
             end_date=("entry_date", "max"),
             length=("entry_date", "count"))
        .reset_index()
    )

    if not exact_intervals:
        return continuous_sequences[continuous_sequences['length'] >= min_intervals]

    exact_sequences = continuous_sequences[continuous_sequences['length'] == min_intervals]
    long_sequences = continuous_sequences[continuous_sequences['length'] > min_intervals]

    num_subsequences = long_sequences['length'] // min_intervals
    repeated_rows = long_sequences.loc[long_sequences.index.repeat(num_subsequences)].reset_index(drop=True)

    offsets = np.concatenate([np.arange(n) * min_intervals for n in num_subsequences])

    repeated_rows["start_date"] = repeated_rows["start_date"] + pd.to_timedelta(offsets, unit="h")
    repeated_rows["end_date"] = repeated_rows["start_date"] + pd.Timedelta(hours=min_intervals - 1)
    repeated_rows["length"] = min_intervals

    return pd.concat([exact_sequences, repeated_rows], ignore_index=True)

if __name__ == "__main__":
    import os
    from tqdm import tqdm

    n = 30 * 24
    magnitude_ids = [1, 6, 7, 8, 9, 10, 11, 12, 14, 15, 20, 22, 30, 35, 37, 38, 39, 42, 43, 44, 431]
    df = pd.read_csv("dataset/aq_data_mad.csv", parse_dates=["entry_date"])
    df = df[df["magnitude_id"].isin(magnitude_ids)]

    failures: pd.DataFrame = get_failures_data(df)
    mtbf_per_magnitude = failures.groupby(["sensor_id", "magnitude_id"])[["time_diff_hours", "duration_hours"]].mean()

    continuous_sequences = get_continuous_sequences(df, n, False)
    print(f"Number of continuous sequences with at least {n} intervals: {len(continuous_sequences)}")

    sequences = []

    with tqdm(total=len(continuous_sequences)) as pbar:
        for index, sequence in continuous_sequences.iterrows():
            sensor_id = sequence['sensor_id']
            magnitude_id = sequence['magnitude_id']
            start_date = sequence['start_date']
            end_date = sequence['end_date']

            diff, dur = mtbf_per_magnitude.loc[sensor_id, magnitude_id]

            sequence_data = df[(df["sensor_id"] == sensor_id) &
                               (df["magnitude_id"] == magnitude_id) &
                               (df["entry_date"] >= start_date) &
                               (df["entry_date"] <= end_date)]

            simulated_failures, failure_times = generate_random_failures(sequence_data, diff, dur)
            sequences.append(simulated_failures)

            pbar.update(1)

    for i, sequence in enumerate(sequences):
        sequence['sequence'] = i

    all_sequences = pd.concat(sequences)
    all_sequences.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sequences_720.csv"),
                         index=False)