from os.path import join
from torch.utils.data import DataLoader

from argus import load_model
from argus.callbacks import MonitorCheckpoint, EarlyStopping,\
    LoggingToFile, ReduceLROnPlateau

from src.transforms import get_transforms
from src.datasets import WhaleDataset, RandomWhaleDataset
from src.argus_models import ArcfaceModel
from src.metrics import CosMAPatK
from src import config


experiment_name = 'arcface_resnet50_016_after_001'
experiment_dir = join(config.EXPERIMENTS_DIR, experiment_name)
train_val_csv_path = config.TRAIN_VAL_CSV_PATH
image_size = (208, 656)
num_workers = 8
batch_size = 32
balance_coef = 0.0
train_epoch_size = 50000
pretrain_model_path = join(config.EXPERIMENTS_DIR,
                           'arcface_resnet50_016',
                           'model-131-0.895746.pth')


if __name__ == "__main__":
    train_transforms = get_transforms(True, image_size)
    train_dataset = RandomWhaleDataset(train_val_csv_path, True,
                                       balance_coef=balance_coef, size=train_epoch_size,
                                       **train_transforms)
    val_transforms = get_transforms(False, image_size)
    val_dataset = WhaleDataset(train_val_csv_path, False, **val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False)

    model = load_model(pretrain_model_path)
    print("Model params:", model.params)

    train_metric_dataset = WhaleDataset(train_val_csv_path, True, **val_transforms)
    monitor_metric = CosMAPatK(train_metric_dataset, k=5,
                               batch_size=batch_size, num_workers=num_workers)
    monitor_metric_name = 'val_' + monitor_metric.name
    callbacks = [
        MonitorCheckpoint(experiment_dir, monitor=monitor_metric_name, max_saves=3),
        EarlyStopping(monitor=monitor_metric_name, patience=50),
        ReduceLROnPlateau(monitor=monitor_metric_name, patience=10, factor=0.64, min_lr=1e-8),
        LoggingToFile(join(experiment_dir, 'log.txt'))
    ]

    with open(join(experiment_dir, 'source.py'), 'w') as outfile:
        outfile.write(open(__file__).read())

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=1000,
              callbacks=callbacks,
              metrics=['accuracy', monitor_metric])
