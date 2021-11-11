"""
Handle the calculation of entropy, information gain ratio, and
finding the best value to split on
"""
import math

def log(x):
    if x < 1e-4:
        return 0
    else:
        return math.log2(x)

def entropy(sample:list):
    '''
    Calculate the entropy of a categorical r.v. Y where
    elements in data represents different levels

    y in {0, 1}

    :param sample: target labels/outcomes of Y
    '''
    p_y_one = sum(sample)/len(sample) if len(sample) > 0 else 0
    log_one = 0.0 if p_y_one == 0 else math.log2(p_y_one)
    log_two = 0.0 if p_y_one == 1 else math.log2(1-p_y_one)
    est_ent = (p_y_one*log_one + (1-p_y_one)*log_two)
    return est_ent if est_ent >= 0 else -est_ent


def info_gain(data, feature, split_value, categorical):
    '''
    Given two datasets (based on a split of the space) calculate
    the conditional entropy of the target variable

    :param data: dataset containing the training samples
    :param feature: feature that is being used to split dataset
    :param split_value: value of the split
    :param categorical: if this is a categorical variable or not
    :return: information gain ratio for feature and split value
    '''
    left_split = None
    right_split = None
    if categorical:
        left_split = data[data[:,feature] == split_value,-1]
        right_split = data[data[:,feature] != split_value,-1]
    else:
        left_split = data[data[:,feature] >= split_value,-1]
        right_split = data[data[:,feature] < split_value,-1]

    num_data_points = data.shape[0]
    p_split_left = len(left_split)/num_data_points
    p_split_right = len(right_split)/num_data_points
    intrinsic_value = -(p_split_left*log(p_split_left)+p_split_right*log(p_split_right))

    condition_entropy = p_split_left*entropy(left_split)+p_split_right*entropy(right_split)
    information_gain = entropy(data[:,-1]) - (condition_entropy)

    if intrinsic_value == 0:
        return 0.0
    else:
        return information_gain/intrinsic_value
        

def find_candidate_split(data, feature, categorical):
    '''
    Find all candidate splits for the dataset. And calculate
    their information gain

    :param data: numpy.ndarray dataset
    :param feature: feature that we are splitting on
    :param categorical: indicating numerical (0) or categorical variable (1)
    :return: 
    '''

    split_values = list(set(data[:,feature]))
    igr_for_splits = [(split_val, info_gain(data, feature, split_val, categorical)) for split_val in split_values]
    return igr_for_splits