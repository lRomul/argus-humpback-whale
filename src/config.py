from os.path import join

DATA_DIR = '/workdir/data/'
TRAIN_CSV_PATH = join(DATA_DIR, 'train.csv')
SAMPLE_SUBMISSION = join(DATA_DIR, 'sample_submission.csv')
TRAIN_DIR = join(DATA_DIR, 'train')
TEST_DIR = join(DATA_DIR, 'test')

VAL_PROPORTION = 0.1
TRAIN_VAL_CSV_PATH = join(DATA_DIR, 'train_val.csv')
