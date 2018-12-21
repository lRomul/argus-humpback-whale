from os.path import join
from torch.utils.data import DataLoader

from argus.callbacks import MonitorCheckpoint, EarlyStopping,\
    LoggingToFile, ReduceLROnPlateau

from src.transforms import get_transforms
from src.datasets import WhaleDataset
from src.argus_models import CnnFinetune
from src.metrics import MAPatK
from src import config


experiment_name = 'resnet50_002'
experiment_dir = join(config.EXPERIMENTS_DIR, experiment_name)
train_val_csv_path = config.TRAIN_VAL_CSV_PATH
image_size = (176, 560)
num_workers = 8
batch_size = 32


if __name__ == "__main__":
    train_transforms = get_transforms(True, image_size)
    train_dataset = WhaleDataset(train_val_csv_path, True, **train_transforms)
    val_transforms = get_transforms(False, image_size)
    val_dataset = WhaleDataset(train_val_csv_path, False, **val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False)

    params = {
        'nn_module': {
            'model_name': 'resnet50',
            'num_classes': len(train_dataset.id2class),
            'pretrained': True,
            'dropout_p': 0.2
        },
        'optimizer': ('Adam', {'lr': 0.00003}),
        'loss': 'CrossEntropyLoss',
        'device': 'cuda'
    }
    model = CnnFinetune(params)

    monitor_metric = MAPatK(k=5)
    monitor_metric_name = 'val_' + monitor_metric.name
    callbacks = [
        MonitorCheckpoint(experiment_dir, monitor=monitor_metric_name),
        EarlyStopping(monitor=monitor_metric_name, patience=30),
        ReduceLROnPlateau(monitor=monitor_metric_name, patience=7, factor=0.64, min_lr=1e-8),
        LoggingToFile(join(experiment_dir, 'log.txt'))
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=1000,
              callbacks=callbacks,
              metrics=['accuracy', monitor_metric])
