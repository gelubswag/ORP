import numpy as np


dtypes = {
    'number_courses': np.int32,
    'time_study': np.float32,
    'Marks': np.float32,
}

features = [
    'Sum of Females  Life Expectancy',
    'Sum of Life Expectancy  (both sexes)',
    'Sum of Males  Life Expectancy',
]

features_indices = {
    1: 0,
    2: 1,
    3: 2,
}
