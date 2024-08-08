import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score
from model import MLPDiffusion

def changezereo(ff):
    for i in range(ff.shape[0]):
        for j in range(ff.shape[1]):
            if ff[i, j] < 0:
                ff[i, j] = 0
            if ff[i, j] > 0:
                ff[i, j] = 1
    return ff

def dis_hisc(y_pre, y_test):
    y_test_pre = y_pre.detach().numpy()
    finalloss = np.sum((y_test.reshape(-1) - y_test_pre.reshape(-1))**2) / y_test_pre.shape[0]
    return finalloss

def result_auc1(f1_result, f2_result, gold):
    f3 = f2_result[:, :, 0] - f1_result[:, :, 0]
    f3_1 = f2_result[:, :, 1] - f1_result[:, :, 1]
    f3_2 = f2_result[:, :, 2] - f1_result[:, :, 2]

    f_real = np.zeros(f3.shape)
    i = 0

    for i in range(gold.shape[0]):
        if gold[i, 2] == '0':
            break
        else:
            q1 = gold[i, 0]
            q2 = gold[i, 1]
            q1 = int(q1[1:])
            q2 = int(q2[1:])
            f_real[q1 - 1, q2 - 1] = 1

    f5 = f3 + f3_1 + f3_2
    f3_1 = changezereo(f3_1)
    f3_2 = changezereo(f3_2)
    f3 = changezereo(f3)

    f4 = np.zeros(f3.shape)
    for i in range(f4.shape[0]):
        for j in range(f4.shape[1]):
            f4[i, j] = f3[i, j] + f3_1[i, j] + f3_2[i, j]
            if f4[i, j] > 1:
                f4[i, j] = 1
            else:
                f4[i, j] = 0

            if f5[i, j] < 0:
                f5[i, j] = 0
            else:
                f5[i, j] = 1

    y_pre = f4.reshape(-1)
    y_test = f_real.reshape(-1)

    y_pre2 = f5.reshape(-1, 1)

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pre)
    roc_auc = metrics.auc(fpr, tpr)
    precision1, recall1, _ = metrics.precision_recall_curve(y_test, y_pre)
    aupr1 = metrics.auc(recall1, precision1)
    acc1 = accuracy_score(y_test, y_pre)
    print(aupr1, 'aupr')
    print(roc_auc, 'f4')
    print(acc1, 'acc2')

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pre2)
    roc_auc = metrics.auc(fpr, tpr)
    precision2, recall2, _ = metrics.precision_recall_curve(y_test, y_pre2)
    aupr2 = metrics.auc(recall2, precision2)
    acc2 = accuracy_score(y_test, y_pre2)
    print(aupr2, 'aupr2')
    print(roc_auc, 'f5')
    print(acc2, 'acc2')

