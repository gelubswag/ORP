from glob import glob
import pandas as pd

from config import settings
from dataset_dtypes import dtypes, features

from sklearn.preprocessing import StandardScaler


def parse_csv(
    file_mask: str = settings.DATASET_MASK,
            ) -> list:
    csv_files = glob(file_mask)

    if not csv_files:
        raise FileNotFoundError(f"Не найдены датасеты с маской: {file_mask}")

    if len(csv_files) > 1:
        print("Найдено несколько датасетов: ")

        for file_i in range(len(csv_files)):
            print(f"{file_i}: {csv_files[file_i]}")

        while True:
            choice = int(input("Выберите номер файла: "))

            try:
                file = csv_files[choice]
                break

            except Exception as e:
                print(f"Неверный номер файла: {e}")
    else:
        file = csv_files[0]

    print(f"Выбран датасет: {file}")

    df = pd.read_csv(file, delimiter=settings.CSV_DELIMITER, dtype=dtypes)
    return df, df[[col for col in features]], df.columns.tolist()


def normalize(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler


data, features_unscaled, labels = parse_csv(settings.DATASET_MASK)
features_scaled, scaler = normalize(features_unscaled)