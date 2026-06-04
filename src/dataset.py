import pandas as pd
import os

from collections import defaultdict


SCRIPT_DIR = os.getcwd()

DATA_ROOT = os.path.join(SCRIPT_DIR, 'data')


def get_uci_har_dataset():
    UCI_HAR_DIR = os.path.join(DATA_ROOT, 'uci_har')

    TRAIN_DIR = os.path.join(UCI_HAR_DIR, 'train')
    TEST_DIR  = os.path.join(UCI_HAR_DIR, 'test')
    TRAIN_IS  = os.path.join(TRAIN_DIR, 'Inertial Signals')
    TEST_IS   = os.path.join(TEST_DIR,  'Inertial Signals')
    
    feat_df = pd.read_csv(os.path.join(UCI_HAR_DIR, 'features.txt'),
                      sep=r'\s+', header=None, names=['idx', 'name'])
    raw_names = feat_df['name'].tolist()

    seen = defaultdict(int)
    feature_names = []
    for n in raw_names:
        if raw_names.count(n) > 1:
            feature_names.append(f'{n}_{seen[n]}')
            seen[n] += 1
        else:
            feature_names.append(n)

    n_dupes = len(raw_names) - len(set(raw_names))
    print(f'Total features : {len(feature_names)}')
    print(f'Duplicate names found and resolved: {n_dupes}')

    # LABEL_MAP = {
    #     1: 'WALKING',
    #     2: 'WALKING_UPSTAIRS',
    #     3: 'WALKING_DOWNSTAIRS',
    #     4: 'SITTING',
    #     5: 'STANDING',
    #     6: 'LAYING'
    # }

    X_train = pd.read_csv(os.path.join(TRAIN_DIR, 'X_train.txt'),
                      sep=r'\s+', header=None, names=feature_names)
    X_test = pd.read_csv(os.path.join(TEST_DIR, 'X_test.txt'),
                        sep=r'\s+', header=None, names=feature_names)
    y_train_raw = pd.read_csv(os.path.join(TRAIN_DIR, 'y_train.txt'),
                            sep=r'\s+', header=None, names=['label'])
    y_test_raw = pd.read_csv(os.path.join(TEST_DIR, 'y_test.txt'),
                            sep=r'\s+', header=None, names=['label'])
    # y_train = y_train_raw['label'].map(LABEL_MAP)
    # y_test = y_test_raw['label'].map(LABEL_MAP)
    
    return X_train, y_train_raw["label"].values.ravel(), X_test, y_test_raw["label"].values.ravel()