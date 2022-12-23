import pandas
from datetime import datetime

from sklearn.preprocessing import StandardScaler

from Clustering.constants import WINTER, SPRING, SUMMER, AUTUMN


def load_data(path):
    dataset = pandas.read_csv(path)

    def date_to_season(date_str):
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if date.month == 12 or (1 <= date.month <= 2):
            return WINTER
        if 3 <= date.month <= 5:
            return SPRING
        if 6 <= date.month <= 8:
            return SUMMER
        if 9 <= date.month <= 11:
            return AUTUMN

    dataset['Date'] = dataset['Date'].apply(date_to_season)

    scaler = StandardScaler()
    scaler.fit(dataset.drop('Date', axis=1))
    scaled_features = scaler.transform(dataset.drop('Date', axis=1))
    scaled_data = pandas.DataFrame(scaled_features, columns=dataset.drop('Date', axis=1).columns)

    X = scaled_data[['Open', 'High', 'Low', 'Close']]
    y = dataset['Date']

    return X, y
