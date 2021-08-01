"""
Implementation of the Deep Temporal Clustering model
Performance metric functions
@author Florent Forest (FlorentF9)
@author Thang Hoang Ta
    (1) https://github.com/FlorentF9/DeepTemporalClustering/blob/master/metrics.py
    (2) https://www.tutorialspoint.com/scikit_learn/scikit_learn_clustering_performance_evaluation.htm

"""

import numpy as np
from scipy.optimize import linear_sum_assignment

try:  # SciPy >= 0.19
    from scipy.special import comb, logsumexp
except ImportError:
    from scipy.misc import comb, logsumexp  # noqa 

from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.cluster import KMeans

def cluster_silhouette(X, labels, metric='euclidean', n_clusters = 2, calculate='no'):
    """
    Calculate Silhoeutte Coefficient 
    # Arguments
        X: input items (list)
        labels: predicted labels (list)
        metric: a metric used for caculating the relationship between items
        n_clusters: the number of clusters used by kmeans (int)
        calculate: if "yes", the kmeans will be calculated to get labels.
    # Return
        score, in [0,1]
    """
    if (calculate == 'yes'):
        kmeans_model = KMeans(k=n_clusters, random_state=1).fit(X)
        labels = kmeans_model.labels_
    return metrics.silhouette_score(X, labels, metric=metric)

def cluster_ri(y_pred, y_true):
    """
    Calculate Rand Index (RI). May be slow.
    # Arguments
        y_true: true labels (list)
        y_pred: predicted labels (list)
    # Return
        score, in [0,1]
    """
    
    tp_plus_fp = comb(np.bincount(y_pred), 2).sum()
    tp_plus_fn = comb(np.bincount(y_true), 2).sum()
    A = np.c_[(y_pred, y_true)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(y_pred))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def cluster_ari(y_true, y_pred):
    """
    Calculate Adjusted Rand Index (ARI)
    # Arguments
        y_true: true labels (list)
        y_pred: predicted labels (list)
    # Return
        score, in [0,1]
    """
    return metrics.adjusted_rand_score(y_true, y_pred)

def cluster_ami(labels_true, labels_pred):
    """
    Calculate Adjusted Mutual Information (AMI)
    # Arguments
        y_true: true labels (list)
        y_pred: predicted labels (list)
    # Return
        score, in [0,1]
    """
    return metrics.adjusted_mutual_info_score (labels_true, labels_pred)

def cluster_nmi(y_true, y_pred):
    """
    Calculate Normalized Mutual Information (NMI)
    # Arguments
        y_true: true labels (list)
        y_pred: predicted labels (list)
    # Return
        score, in [0,1]
    """
    return metrics.normalized_mutual_info_score(y_true, y_pred)

def cluster_fm(y_true, y_pred):
    """
    Calculate Fowlkes-Mallows Score
    # Arguments
        y_true: true labels (list)
        y_pred: predicted labels (list)
    # Return
        score, in [0,1]
    """
    return metrics.fowlkes_mallows_score (y_true, y_pred)

def cluster_acc(y_true, y_pred):
    """
    Calculate unsupervised clustering accuracy. Requires scikit-learn installed
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size

def cluster_purity(y_true, y_pred):
    """
    Calculate clustering purity (should be for reference purposes only)
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        purity, in [0,1]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return metrics.accuracy_score(y_pred_voted, y_true)

def roc_auc(y_true, q_pred, n_classes):
    """
    Calculate area under ROC curve (ROC AUC)
    WARNING: DO NOT USE, MAY CONTAIN ERRORS
    TODO: CHECK IT!
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        q_pred: predicted probabilities, numpy.array with shape `(n_samples,)`
    # Return
        ROC AUC score, in [0,1]
    """
    if n_classes == 2:  # binary ROC AUC
        auc = max(metrics.roc_auc_score(y_true, q_pred[:, 1]), metrics.roc_auc_score(y_true, q_pred[:, 0]))
    else:  # micro-averaged ROC AUC (multiclass)
        fpr, tpr, _ = metrics.roc_curve(label_binarize(y_true, classes=np.unique(y_true)).ravel(), q_pred.ravel())
        auc = metrics.auc(fpr, tpr)
    return auc

def cluster_homogeneity(y_true, y_pred):
    return metrics.homogeneity_score(y_true, y_pred)

def cluster_completeness(y_true, y_pred):
    return metrics.completeness_score(y_true, y_pred)

def cluster_v_measure(y_true, y_pred):
    return metrics.v_measure_score(y_true, y_pred)






    




