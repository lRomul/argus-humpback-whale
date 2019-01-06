from os.path import join
from torch.utils.data import DataLoader

from argus.callbacks import MonitorCheckpoint, EarlyStopping,\
    LoggingToFile, ReduceLROnPlateau

from src.transforms import get_transforms
from src.datasets import WhaleDataset, RandomWhaleDataset
from src.argus_models import CenterLossModel
from src.metrics import CosMAPatK
from src import config


experiment_name = 'center_loss_resnet50_001'
experiment_dir = join(config.EXPERIMENTS_DIR, experiment_name)
train_val_csv_path = config.TRAIN_VAL_CSV_PATH
image_size = (96, 304)
num_workers = 8
batch_size = 128
balance_coef = 0.0
train_epoch_size = 20000


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

    params = {
        'nn_module': {
            'cnn_finetune': {
                'model_name': 'resnet50',
                'num_classes': len(train_dataset.id2class_idx),
                'pretrained': True,
                'dropout_p': 0.5
            },
            'embedding_size': 512
        },
        'center_loss': {
            'lr': 0.01,
            'weight': 0.5
        },
        'optimizer': ('Adam', {'lr': 0.0003}),
        'loss': 'CrossEntropyLoss',
        'device': 'cuda'
    }
    print("Model params:", params)
    model = CenterLossModel(params)

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
              metrics=['accuracy', 'center_loss', monitor_metric])
