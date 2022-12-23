import pandas
from datetime import datetime


WINTER = 1
SPRING = 2
SUMMER = 3
AUTUMN = 4


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

    X = dataset[['Close', 'Open', 'High', 'Low']]
    y = dataset['Date']

    return X, y
