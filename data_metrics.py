import sys
import platform
# print("Python version:\n", sys.version)
# print ("Path to the python executable:\n", sys.executable)


# In[2]:


import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
import pickle
import pandas as pd
import os
import sys
# from sklearn.metrics import log_loss, brier_score_loss
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
from sklearn.metrics import *
from scikitplot.metrics import *
from sklearn.calibration import calibration_curve
# from sklearn.metrics import confusion_matrix, balanced_accuracy_score, matthews_corrcoef, roc_curve, auc, precision_recall_curve
# from scikitplot.metrics import plot_cumulative_gain, plot_lift_curve, plot_roc_curve, plot_roc
import seaborn as sns
import random
import importlib.util


def confusion_(y_true, y_pred,plot=True,figName = None):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred,labels=[0,1]).ravel()
    if plot==True:
        clf_flat = np.array([TN,FP,FN,TP])
        ax= plt.subplot()

        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in
                        clf_flat]
        group_percentages = ["{0:.2%}".format(value) for value in
                             clf_flat/np.sum(clf_flat)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                  zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(np.reshape(clf_flat,(2,2)), annot=labels, fmt='', cmap='Blues')

        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.xaxis.set_ticklabels(['non-tumor', 'tumor'])
        ax.yaxis.set_ticklabels(['non-tumor', 'tumor']);
        
        if figName is not None:
            plt.savefig(figName, dpi=300,bbox_inches='tight')
    return TN, FP, FN, TP

def classical_metrics(TN, FP, FN, TP, y_true,y_pred,verbose=True):
    #sensitivity:
    TPR = TP/(TP+FN)
    if verbose: print("Sensitivity:",TPR)

    #sensitivity:
    TNR = TN/(TN+FP)
    if verbose: print("Specificity:",TNR)

    #precision:
    PPV = TP/(TP+FP)
    if verbose: print("Precision:",PPV)

    #negative predictive value (NPV):
    NPV = TN/(TN+FN)
    if verbose: print("Negative predictive value:",NPV)

    #false negative rate/ miss rate:
    FNR = FN/(FN+TP)
    if verbose: print("False negative rate:",FNR)

    #fall-out / false positive rate:
    FPR = FP/(FP+TN)
    if verbose: print("Fall Out:",FPR)

    #false discovery rate:
    FDR = 1 - PPV
    if verbose: print("False discovery rate:",FDR)

    #false omission rate:
    FOR = FN/(FN+TN)
    if verbose: print("False omission rate:",FOR)

    #Threat score / Critical Success Index:
    TS = TP/(TP+FN+FP)
    if verbose: print("Threat score:",TS)

    #Acccuracy:
    accuracy = (TP + TN)/(TP+TN+FP+FN)
    if verbose: print("Accuracy:",accuracy)

    #Balanced Accuracy:
    BA = balanced_accuracy_score(y_true, y_pred)
    if verbose: print("Balanced Accuracy:",BA)

    #F1 Score:
    F1 = (2*TP)/(2*TP+FP+FN)
    if verbose: print("F1 / Dice Score:",F1)

    #Matthews Correlation Coefficient:
    # MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    MCC = matthews_corrcoef(y_true, y_pred)
    if verbose: print("Matthews Correlation Coefficient:",MCC)

    #Informedness / Bookmarker Informedness:
    BM = TPR + TNR - 1
    if verbose: print("Informedness / Bookmarker Informedness",BM)

    #Markedness:
    MK = PPV + NPV - 1
    if verbose: print("Markedness",MK)
    
    ###COMPILE TOGETHER:
    metric = ['Sensitivity',
            "Specificity",
            "Precision",
            "Negative predictive value",
            "False negative rate",
            "Fall Out",
            "False discovery rate",
            "False omission rate",
            "Threat score",
            "Accuracy",
            "Balanced Accuracy",
            "F1 / Dice Score",
            "Matthews Correlation Coefficient",
            "Informedness / Bookmarker Informedness ",
            "Markedness"]
    value = [TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, accuracy, BA, F1, MCC, BM, MK]
    
    return metric, value
  
    
def similarity_metrics(y_true, y_pred, y_prob):
    
    jaccard = jaccard_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    cohen = cohen_kappa_score(y_true, y_pred) 
    RI = rand_score(y_true, y_pred)
#     ARI = adjusted_rand_score(y_true, y_pred)
    MI = mutual_info_score(y_true, y_pred)
    
    metric = [
    "Jaccard Coefficient",
    "Area under ROC Curve",
    "Cohen Kappa",
    "Rand Index",
#     "Adjusted Rand Index",
    "Mutual Information"]
#     value = [jaccard, roc_auc, cohen, RI, ARI, MI]
    value = [jaccard, roc_auc, cohen, RI, MI]
    return metric, value


# In[56]:

def make_plots(y_true, y_prob, y_2class_prob, figName = None):
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label = "AUC")
    ax1.legend(loc = "lower right")
    ax1.plot([0,1],[0,1], 'k--')
    ax1.set_xlabel("False Positive Rate" , fontsize=12)
    ax1.set_ylabel("True Positive Rate" , fontsize=12)
    ax1.set_title('ROC AUC %.3f' % roc_auc)


    plot_lift_curve(y_true, y_2class_prob, ax=ax2)

    plot_cumulative_gain(y_true, y_2class_prob, ax=ax3)

    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_prob)
    lr_auc = auc(lr_recall, lr_precision)
    # plot the precision-recall curves
    no_skill = len(y_true[y_true==1]) / len(y_true)
    ax4.plot([0, 1], [no_skill, no_skill], linestyle='--')
    ax4.plot(lr_recall, lr_precision, marker='.')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall AUC:%.3f' % lr_auc)

    # fig.text(0.5, -0.05, 'Number of Epochs', ha='center') #Y-axis label
    # fig.text(-0.01, 0.5, 'Dice Coefficient Loss',va='center', rotation='vertical') #X-axis label

    # plt.title('Batch Size {} : Binary Cross Entropy '.format(LAYER_NAME))
    # fig.suptitle('Metric Plots for GBM Model')
    fig.suptitle('Metric Plots for %s' % LAYER_NAME,y=1.08, fontsize=22)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    if figName is not None:
        fig.savefig(figName, dpi=300,bbox_inches='tight')
    plt.show()
