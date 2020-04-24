import os
import numpy as np
import json
from PIL import Image, ImageDraw
from matplotlib import cm
import seaborn as sb
import matplotlib.pyplot as plt


def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows, n_cols, n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    (t_rows, t_cols, t_channels) = np.shape(T)
    padding_row_top = int(t_rows / 2)
    padding_col_left = int(t_cols / 2)
    heatmap = np.zeros((n_rows, n_cols))
    padding_I = np.zeros((n_rows + t_rows - 1, n_cols + t_cols - 1, t_channels))
    padding_I[padding_row_top:padding_row_top + n_rows, padding_col_left:padding_col_left + n_cols] = I
    v2 = T.reshape(t_rows * t_cols * t_channels) / 127.5 - 1
    # loop over the map
    for i in range(n_rows):
        for j in range(n_cols):
            v1 = padding_I[i:i + t_rows, j:j + t_cols].reshape(t_rows * t_cols * t_channels) / 127.5 - 1
            # arc cosine between two vectors
            heatmap[i][j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    # fixed size bounding box
    box_height = 10
    box_width = 10
    (n_rows, n_cols) = np.shape(heatmap)
    top_start = int(box_height / 2)
    left_start = int(box_width / 2)
    # loop over heatmap
    for i in range(top_start, n_rows - (box_height - top_start) + 1):
        for j in range(left_start, n_cols - (box_width - left_start) + 1):
            # let the location be the center of bounding box
            tl_row = i - top_start
            tl_col = j - left_start
            br_row = tl_row + box_height
            br_col = tl_col + box_width
            score = (heatmap[i, j] + 1) / 2
            # if score larger than a threshold
            if score > 0.85:
                # create a bounding box
                output.append([tl_row, tl_col, br_row, br_col, score])
    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    # You may use multiple stages and combine the results
    # template 1
    with open("data/hw02_annotations/annotations.json") as f:
        data = json.load(f)
    loc = data["RL-001.jpg"][1]
    loc = [int(i) for i in loc]
    tmp = Image.open("data/RedLights2011_Medium/RL-001.jpg")
    tmp = np.asarray(tmp)
    T1 = tmp[loc[0]:loc[2], loc[1]:loc[3]]
    # Image.fromarray(T1).save("aaa.jpg")
    heatmap1 = compute_convolution(I, T1)
    # template 2
    loc = data["RL-036.jpg"][1]
    loc = [int(i) for i in loc]
    tmp = Image.open("data/RedLights2011_Medium/RL-036.jpg")
    tmp = np.asarray(tmp)
    T2 = tmp[loc[0]:loc[2], loc[1]:loc[3]]
    Image.fromarray(T2).save("bbb.jpg")
    heatmap2 = compute_convolution(I, T2)
    # final heatmap
    heatmap = np.maximum(heatmap1, heatmap2)
    '''
    heat_map = sb.heatmap(heatmap)
    plt.savefig("hm.jpg")
    '''
    output = predict_boxes(heatmap)
    print(len(output))
    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# load splits: 
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'data/hw02_preds'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    print(i)
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds_train.json'), 'w') as f:
    json.dump(preds_train, f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):
        # read image using PIL:
        I = Image.open(os.path.join(data_path, file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path, 'preds_test.json'), 'w') as f:
        json.dump(preds_test, f)
