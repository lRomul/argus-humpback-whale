from os.path import join
import pandas as pd
import numpy as np
import random

from src import config


def make_train_val_df(train_csv_path, train_dir, val_proportion):
    data_df = pd.read_csv(train_csv_path)
    data_df = data_df[data_df.Id != 'new_whale']
    data_df['val'] = np.random.random(size=data_df.shape[0]) < val_proportion
    val_df = data_df[data_df.val].copy()
    train_df = data_df[~data_df.val].copy()
    val_df['Id_in_train'] = val_df.Id.isin(set(train_df.Id))
    train_df['Id_in_train'] = True
    train_val_df = pd.concat([val_df, train_df])
    train_val_df['image_path'] = train_val_df.Image.map(
        lambda x: join(train_dir, x))
    return train_val_df


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    train_val_df = make_train_val_df(config.TRAIN_CSV_PATH,
                                     config.TRAIN_DIR,
                                     config.VAL_PROPORTION)
    train_val_df.to_csv(config.TRAIN_VAL_CSV_PATH)
    print(f"Train val saved to {config.TRAIN_VAL_CSV_PATH}")
