from numpy import array
from glob import glob
import pandas as pd

from config import settings
from dataset_dtypes import dtypes


def parse_csv(
    file_mask: str = settings.DATASET_MASK,
    has_header: bool = settings.HAS_HEADER,
    ignore_attrs: list = settings.IGNORE_ATTRS,
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
    return df.to_numpy(), df.columns.tolist()


def make_avg_marks(data):
    return array([[data[i][0], data[i][1], data[i][2]/data[i][0]]
                 for i in range(len(data))]
                 )


data, labels = parse_csv(settings.DATASET_MASK, settings.HAS_HEADER)
print(data)
# data = make_avg_marks(data)
