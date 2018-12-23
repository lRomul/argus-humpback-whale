from os.path import join
import pandas as pd
import numpy as np
import random

from src import config


def make_train_val_df(train_csv_path, bbox_csv_path, train_dir, val_proportion):
    data_df = pd.read_csv(train_csv_path)
    data_df = data_df[data_df.Id != 'new_whale']
    id_counts = dict(data_df.Id.value_counts())
    data_df['id_counts'] = data_df.Id.map(lambda x: id_counts[x])

    data_df['val'] = (np.random.random(size=data_df.shape[0]) < val_proportion) \
                     & (data_df['id_counts'] > 2)
    val_df = data_df[data_df.val].copy()
    train_df = data_df[~data_df.val].copy()

    assert set(data_df.Id) == set(train_df.Id)

    id2class = {train_id: cls for cls, train_id in enumerate(set(train_df.Id))}
    train_df['class_index'] = train_df.Id.map(lambda x: id2class[x])
    val_df['class_index'] = val_df.Id.map(lambda x: id2class[x] if x in id2class else -1)

    train_val_df = pd.concat([val_df, train_df])
    train_val_df['image_path'] = train_val_df.Image.map(
        lambda x: join(train_dir, x))

    bboxes_df = pd.read_csv(bbox_csv_path)
    train_val_df = pd.merge(train_val_df, bboxes_df, how='left', on='Image')

    return train_val_df


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    train_val_df = make_train_val_df(config.TRAIN_CSV_PATH,
                                     config.BOUNDING_BOXES_CSV,
                                     config.TRAIN_DIR,
                                     config.VAL_PROPORTION)
    train_val_df.to_csv(config.TRAIN_VAL_CSV_PATH, index=False)
    print(f"Train val saved to {config.TRAIN_VAL_CSV_PATH}")
