from Diffusion236610 import DATA_DIR
from typing import Sequence, List

import os


def get_train_val_test_records_from_lines(lines: Sequence[str]) -> List[str]:
    ltafdb_records = [
        os.path.join(DATA_DIR, "ltafdb", f"{rec}.h5")
        for rec in lines[0].strip(os.linesep).split(':')[-1].split(',')
    ]
    afdb_records = [
        os.path.join(DATA_DIR, "afdb", f"{rec}.h5")
        for rec in lines[1].strip(os.linesep).split(':')[-1].split(',')
    ]
    nsrdb_records = [
        os.path.join(DATA_DIR, "nsrdb", f"{rec}.h5")
        for rec in lines[2].strip(os.linesep).split(':')[-1].split(',')
    ]
    mitdb_records = [
        os.path.join(DATA_DIR, "mitdb", f"{rec}.h5")
        for rec in lines[3].strip(os.linesep).split(':')[-1].split(',')
    ]

    records = ltafdb_records + afdb_records + nsrdb_records + mitdb_records

    return records


def get_train_val_test_split():
    split_config = os.path.join(DATA_DIR, "arrhythmia_classification_train_val_test_split.txt")
    with open(split_config, 'r') as f:
        split = f.readlines()

    training_lines = split[1:5]
    validation_lines = split[7:11]
    testing_lines = split[13:17]

    train_records = get_train_val_test_records_from_lines(training_lines)
    val_records = get_train_val_test_records_from_lines(validation_lines)
    test_records = get_train_val_test_records_from_lines(testing_lines)

    return train_records, val_records, test_records
