import numpy as np

def calculate_coverage(y_true, lower_bounds, upper_bounds):
    y_true = np.array(y_true)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    return np.mean(covered)

def calculate_average_width(lower_bounds, upper_bounds):
    return np.mean(np.array(upper_bounds) - np.array(lower_bounds))

def winkler_score(y_true, lower_bounds, upper_bounds, alpha=0.05):
    """
    Computes the Winkler score for prediction intervals.
    Lower score is better. Balances calibration and sharpness.
    """
    y = np.array(y_true)
    l = np.array(lower_bounds)
    u = np.array(upper_bounds)
    delta = u - l
    
    scores = delta + \
             (2 / alpha) * (l - y) * (y < l) + \
             (2 / alpha) * (y - u) * (y > u)
    
    return np.mean(scores)
