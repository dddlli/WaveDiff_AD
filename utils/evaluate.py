import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from utils.affiliation.generics import convert_vector_to_events
from utils.affiliation.metrics import pr_from_events
from utils.spot import SPOT

def getThreshold(init_score, test_score, q=1e-2):
    """
    Get anomaly threshold using SPOT algorithm
    
    Args:
        init_score: Initialization scores from normal data
        test_score: Test scores to detect anomalies
        q: Initial anomaly probability (default: 0.01)
        
    Returns:
        Detection threshold
    """
    s = SPOT(q=q)
    s.fit(init_score, test_score)
    s.initialize(verbose=False)
    ret = s.run()
    threshold = np.mean(ret['thresholds'])

    return threshold

def getAffiliationMetrics(label, pred):
    """
    Get affiliation-based metrics for anomaly detection
    
    Args:
        label: Ground truth anomaly labels
        pred: Predicted anomaly indicators
        
    Returns:
        Precision, recall, and F1 score
    """
    events_pred = convert_vector_to_events(pred)
    events_label = convert_vector_to_events(label)
    Trange = (0, len(pred))

    # If no predicted events, return zeros
    if len(events_pred) == 0:
        return 0, 0, 0
    
    # If no actual events, precision is 0, recall is 1
    if len(events_label) == 0:
        return 0, 1, 0

    try:
        result = pr_from_events(events_pred, events_label, Trange)
        P = result['precision']
        R = result['recall']
        F = 2 * P * R / (P + R) if (P + R) > 0 else 0
    except Exception as e:
        print(f"Error in affiliation metrics: {e}")
        # Fallback to point-wise metrics
        P, R, F, _ = precision_recall_fscore_support(label, pred, average='binary')
    
    return P, R, F

def getPointMetrics(label, pred):
    """
    Get point-wise metrics for anomaly detection
    
    Args:
        label: Ground truth anomaly labels
        pred: Predicted anomaly indicators
        
    Returns:
        Precision, recall, F1 score, and AUC
    """
    P, R, F, _ = precision_recall_fscore_support(label, pred, average='binary')
    
    # Calculate AUC if classes are both present
    if len(np.unique(label)) > 1:
        auc = roc_auc_score(label, pred)
    else:
        auc = 0
        
    return P, R, F, auc

def evaluate(init_score, test_score, test_label=None, q=1e-2, evaluation_type='affiliation'):
    """
    Evaluate anomaly detection performance
    
    Args:
        init_score: Initialization scores from normal data
        test_score: Test scores to detect anomalies
        test_label: Ground truth anomaly labels (optional)
        q: Initial anomaly probability (default: 0.01)
        evaluation_type: Metric type ('affiliation' or 'point')
        
    Returns:
        Dictionary with evaluation results
    """
    res = {
        'init_score': init_score,
        'test_score': test_score,
        'test_label': test_label,
        'q': q,
    }

    # Get anomaly threshold
    threshold = getThreshold(init_score, test_score, q=q)
    test_pred = (test_score > threshold).astype(int)
    res['threshold'] = threshold
    res['test_pred'] = test_pred

    # Calculate metrics if labels are provided
    if test_label is not None:
        if evaluation_type == 'affiliation':
            precision, recall, f1_score = getAffiliationMetrics(test_label.copy(), test_pred.copy())
            res['precision'] = precision
            res['recall'] = recall
            res['f1_score'] = f1_score
        else:  # point-wise metrics
            precision, recall, f1_score, auc = getPointMetrics(test_label.copy(), test_pred.copy())
            res['precision'] = precision
            res['recall'] = recall
            res['f1_score'] = f1_score
            res['auc'] = auc
            
        # Additional statistics
        n_anomalies = np.sum(test_label)
        n_detected = np.sum(test_pred)
        n_correct = np.sum(test_pred * test_label)
        
        res['n_anomalies'] = n_anomalies
        res['n_detected'] = n_detected
        res['n_correct'] = n_correct
        res['detection_rate'] = n_correct / n_anomalies if n_anomalies > 0 else 0
        res['false_alarm_rate'] = (n_detected - n_correct) / (len(test_label) - n_anomalies) if len(test_label) - n_anomalies > 0 else 0

    return res