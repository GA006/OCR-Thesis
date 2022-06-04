import numpy as np
import pandas as pd

def metrics(gt_label, pred_label, classes):


    total_score = np.zeros((4, len(classes)))
    count = 0
    for i in range(len(pred_label)):
        y_true = gt_label[i]
        y_pred = pred_label[i]
        score = np.zeros((4, len(classes)))

        if len(np.unique(y_true)) != len(np.unique(y_pred)):
            count += 1
            continue

        for j in range(len(classes)):
            curr_true = (y_true == j).astype(int)
            curr_pred = (y_pred[0] == j).astype(int)
 
            tp = np.sum(np.logical_and(curr_pred == 1, curr_true == 1))

            tn = np.sum(np.logical_and(curr_pred == 0, curr_true == 0))

            fp = np.sum(np.logical_and(curr_pred == 1, curr_true == 0))

            fn = np.sum(np.logical_and(curr_pred == 0, curr_true == 1))

            if tp + fp == 0 or tp + fp == 0 or tp == 0:
                score[:, j] = 0
            else:
                score[0, j] = tp / (tp + fp)
                score[1, j] = tp / (tp + fn)
                score[2, j] = (2 * score[0, j] * score[1, j]) / (score[0, j] + score[1, j])
                score[3, j] = (tp + tn) / (tp + tn + fp + fn)
        total_score += score

    total_score /= (len(pred_label) - count)

    return total_score

def save_metrics(score, classes, path_to_save_csv, to_save=True):
    df = pd.DataFrame(score)
    df.rename(columns={i: classes[i] for i in range(len(classes))}, inplace=True)
    df.rename(index={0: 'Precision', 1: 'Recall', 2: 'F1', 3: 'Accuracy'}, inplace=True)
    df['Mean'] = list(df.mean())

    if to_save:
        df.to_csv(path_to_save_csv)
    
    return df