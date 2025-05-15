import pandas as pd
from torch.utils.data import Dataset


class MetraqAirQualityDataset(Dataset):
    '''
    Metraq air quality dataset.

    This dataset class provides functionalities to load, preprocess, and interact with air quality data from a CSV file.
    It supports filtering interpolated data and calculates key statistics for the specified air quality measurements.
    The class is compatible with machine learning frameworks that utilize dataset objects, providing convenience
    methods for data access and optional transformations.
    '''

    def __init__(self, data_file: str, include_interpolations: bool, transform: callable = None,
                 call_init_data: bool = True):
        '''
        Creates an instance of the dataset.

        Args:
            data_file (str): CSV file to read the Metraq air quality data from.
            include_interpolations (bool): If True, missing entries are interpolated.
            transform (callable, optional): Transform(s) to apply to the dataset.
        '''
        self.data_file = data_file
        self.transform = transform
        self.include_interpolations = include_interpolations
        self._stats = None
        self._data: pd.DataFrame = None

        if call_init_data:
            self._init_data()

    def _init_data(self):
        '''
        Initializes the dataset by reading, optionally filtering, and preprocessing the data.
        '''
        self._read_data()
        self._preprocess()
        self._extract_metadata()

    def _preprocess(self):
        '''
        Extracts metadata, such as magnitudes, entry dates, and sensors, and calculates summary statistics for the dataset.
        '''
        self._exclude_interpolations()
        self._get_stats()
        self._get_time_features()

    def _read_data(self):
        '''
        Reads the data from the CSV file and sorts it by sensor_id, magnitude_id, and entry_date.
        '''
        print("Extracting data...", end="", flush=True)
        self._data = pd.read_csv(self.data_file, parse_dates=['entry_date'])
        self._data = self._data.sort_values(by=['sensor_id', 'magnitude_id', 'entry_date'])
        print("done")

    def _get_stats(self):
        print("Calculating stats...", end="", flush=True)
        raw_data = self._data[self._data['is_valid'] == 1]
        grouped = raw_data.groupby(['magnitude_id'])
        self._stats = grouped['value'].describe()
        print("done")

    def _get_time_features(self):
        print("Calculating time features...", end="", flush=True)
        self.data['month'] = (self.data['entry_date'].dt.month - 1) / 11.0 - 0.5
        self.data['hour'] = self.data['entry_date'].dt.hour / 23.0 - 0.5
        self.data['day_of_week'] = self.data['entry_date'].dt.dayofweek / 6.0 - 0.5
        self.data['day_of_month'] = (self.data['entry_date'].dt.day - 1) / 30.0 - 0.5
        self.data['day_of_year'] = (self.data['entry_date'].dt.dayofyear - 1) / 365.0 - 0.5
        print("done")

    def _exclude_interpolations(self):
        '''
        Excludes interpolated values from the dataset if include_interpolations is set to False.
        '''
        if not self.include_interpolations:
            print("Removing interpolated data...", end="", flush=True)
            self._data = self.data[self._data['is_interpolated'] == 0]
            print("done")

    def _extract_metadata(self):
        print("Extracting metadata...", end="", flush=True)
        magnitudes = self._data.groupby(['magnitude_id', 'magnitude_name']).size().reset_index().iloc[:, :-1].values
        self.magnitudes = [(mag_id, mag_name) for mag_id, mag_name in magnitudes]
        self.entry_dates = list(self._data['entry_date'].unique())
        self.sensors = list(self._data['sensor_id'].unique())
        print("done")



    @property
    def stats(self):
        '''
        Returns:
            dict: A dictionary containing statistics for each magnitude, indexed by magnitude ID.

            Example usage:
                stats = dataset.stats
                magnitude_stats = stats[7]  # Get statistics for magnitude ID 7
        '''
        return self._stats.to_dict(orient='index')

    @property
    def data(self):
        '''
        Returns:
            DataFrame: The internal DataFrame containing the air quality data.
        '''
        return self._data

    def __len__(self):
        '''
        Returns:
            int: The number of records in the dataset.
        '''
        return len(self._data)

    def __getitem__(self, index):
        '''
        Retrieves a data record by index.

        Args:
            index (int): The index of the data record to retrieve.

        Returns:
            dict: A dictionary representation of the data record, with optional transformations applied.
        '''
        row = self._data.iloc[index].to_dict()

        if self.transform is not None:
            row = self.transform(row)

        return row


if __name__ == "__main__":
    from rich import print
    from rich.pretty import Pretty

    dataset_file = "dataset/aq_data_mad.csv"
    dataset = MetraqAirQualityDataset(data_file=dataset_file, include_interpolations=False)
    row = dataset[0]
    print(Pretty(row))