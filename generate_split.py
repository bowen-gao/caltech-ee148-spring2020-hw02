import numpy as np
import os
import json

np.random.seed(2020)  # to ensure you always get the same train/test split

data_path = 'data/RedLights2011_Medium'
gts_path = 'data/hw02_annotations'
preds_path = 'data/hw02_preds'
split_path = 'data/hw02_splits'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

split_test = True  # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
file_names_train = []
file_names_test = []
'''
Your code below. 
'''
num_train = int(len(file_names) * train_frac)
train_index = set(np.random.choice(len(file_names), num_train, replace=False))
for i in range(len(file_names)):
    if i in train_index:
        file_names_train.append(file_names[i])
    else:
        file_names_test.append(file_names[i])

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train, file_names_test)) == 0

np.save(os.path.join(split_path, 'file_names_train.npy'), file_names_train)
np.save(os.path.join(split_path, 'file_names_test.npy'), file_names_test)

if split_test:
    with open(os.path.join(gts_path, 'annotations.json'), 'r') as f:
        gts = json.load(f)

    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    gts_train = {}
    gts_test = {}
    '''
    Your code below. 
    '''
    train_set = set(file_names_train)
    test_set = set(file_names_test)
    for file in gts:
        if file in train_set:
            gts_train[file] = gts[file]
        else:
            gts_test[file] = gts[file]
    with open(os.path.join(gts_path, 'annotations_train.json'), 'w') as f:
        json.dump(gts_train, f)

    with open(os.path.join(gts_path, 'annotations_test.json'), 'w') as f:
        json.dump(gts_test, f)

print(len(file_names_train), len(file_names_test))
