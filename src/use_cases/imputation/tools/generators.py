import pandas as pd
from sklearn.preprocessing import StandardScaler

def tsl_generator(src: str):
    """
    Time series library dataset generator

    Generates a csv to be used with the Time Series Library (tsl) package for
    data imputation task
    
    :param src: 
    :return: 
    """
    df = pd.read_csv(src, parse_dates=["entry_date"])
    df['year'] = df['entry_date'].dt.year

    magnitude_ids = [6, 7, 8, 12, 14, 20, 30, 35]
    sensor_id = 28079008
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    data = df[(df["magnitude_id"].isin(magnitude_ids)) & (df["sensor_id"] == sensor_id) & df['year'].isin(years)]

    data_pivot = data.pivot(index="entry_date", columns="magnitude_id", values="value")

    scaler = StandardScaler()
    standard_data = scaler.fit_transform(data_pivot)
    standardized_df = pd.DataFrame(standard_data, index=data_pivot.index, columns=data_pivot.columns)

    return standardized_df

if __name__ == "__main__":
    data = tsl_generator("dataset/aq_data_mad.csv")
    data.to_csv("/home/dmariaa/metraq/src/models/imputation/data/sensor-28079008-standard.csv")