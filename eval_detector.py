import os
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    t = max(box_1[0], box_2[0])
    l = max(box_1[1], box_2[1])
    b = min(box_1[2], box_2[2])
    r = min(box_1[3], box_2[3])
    inter = abs(max((b - t, 0)) * max((r - l), 0))
    box1a = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    box2a = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
    iou = float(inter) / (box1a + box2a - inter)
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        num_tp = 0
        num_preds = 0
        for i in range(len(pred)):
            if pred[i][4] >= conf_thr:
                num_preds += 1
        for i in range(len(gt)):
            flag = 0
            for j in range(len(pred)):
                if pred[j][4] < conf_thr:
                    continue
                iou = compute_iou(pred[j][:4], gt[i])
                if iou > iou_thr:
                    TP += 1
                    num_tp += 1
                    flag = 1
                    break
            if flag == 0:
                FN += 1
        FP += (num_preds - num_tp)
    '''
    END YOUR CODE
    '''

    return TP, FP, FN


# set a path for predictions and annotations:
preds_path = 'data/hw02_preds'
gts_path = 'data/hw02_annotations'

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path, 'preds_train.json'), 'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'), 'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    '''
    Load test data.
    '''

    with open(os.path.join(preds_path, 'preds_test.json'), 'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'), 'r') as f:
        gts_test = json.load(f)

# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 


# confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],
# dtype=float))  # using (ascending) list of confidence scores as thresholds
confidence_thrs = np.linspace(0.85, 1.0, num=10)
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for iou_thr in [0.25, 0.5, 0.75]:
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr,
                                                               conf_thr=conf_thr)
    # Plot training set PR curves
    precisions = []
    recalls = []
    for i in range(len(confidence_thrs)):
        p = tp_train[i] / (tp_train[i] + fp_train[i])
        precisions.append(p)
        r = tp_train[i] / (tp_train[i] + fn_train[i])
        recalls.append(r)
    plt.plot(recalls, precisions)
plt.legend(["iou_thr=0.25", "iou_thr=0.5", "iou_thr=0.75"])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR curve for different iou thresholds on training set")
plt.savefig("pr_train")
plt.clf()
if done_tweaking:
    print('Code for plotting test set PR curves.')
    confidence_thrs = np.linspace(0.85, 1.0, num=10)
    tp_test = np.zeros(len(confidence_thrs))
    fp_test = np.zeros(len(confidence_thrs))
    fn_test = np.zeros(len(confidence_thrs))
    for iou_thr in [0.25, 0.5, 0.75]:
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thr,
                                                                conf_thr=conf_thr)
        # Plot test set PR curves
        precisions = []
        recalls = []
        for i in range(len(confidence_thrs)):
            if tp_test[i] + fp_test[i] != 0:
                p = tp_test[i] / (tp_test[i] + fp_test[i])
            else:
                p = 1.0
            precisions.append(p)
            r = tp_test[i] / (tp_test[i] + fn_test[i])
            recalls.append(r)
        plt.plot(recalls, precisions)
    plt.legend(["iou_thr=0.25", "iou_thr=0.5", "iou_thr=0.75"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve for different iou thresholds on test set")
    plt.savefig("pr_test")
