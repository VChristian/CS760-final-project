"""
Handle the calculation of entropy, information gain ratio, and
finding the best value to split on
"""
import math

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
    '''
    left_split = []
    right_split = []
    for data_point in data:
        
        # case when xj. >= c
        if data_point[split[0]] >= split[1]:
            left_split.append(data_point[-1])
        else:
            right_split.append(data_point[-1])
    
    p_x_left = len(left_split)/len(data)
    p_x_right = len(right_split)/len(data) # this should be 1

    # tests
    assert (1-p_x_left)-p_x_right <= 1e-4

    # if the split contains no information then return zero
    if p_x_left == 0 or p_x_left == 1:
        return 0.0, 0.0 # ig ratio, split_entropy

    split_entropy = -(p_x_left*math.log2(p_x_left) + p_x_right*math.log2(p_x_right))
    conditional_entropy = p_x_left*entropy(left_split) + p_x_right*entropy(right_split)
    ig = entropy(left_split+right_split) - conditional_entropy

    # calculate entropy of split
    return ig/split_entropy, split_entropy
        

def find_candidate_split(data, feature, categorical):
    '''
    Find all candidate splits for the dataset. And calculate
    their information gain

    :param data: numpy.ndarray dataset
    :param feature: feature that we are splitting on
    :param categorical: indicating numerical or categorical variable
    :return: 
    '''

    # only unique splits
    # dim1_splits = list(set([(0, x_1)for x_1,_,_ in data]))
    # dim2_splits = list(set([(1, x_2) for _,x_2,_ in data]))
    # splits = dim1_splits + dim2_splits
    # ig_split_ent = [info_gain(data, split) for split in splits]
    # return splits, ig_split_ent

    split_values = list(set(data[:,feature]))
    igr_for_splits = [(split_val, info_gain(data, feature, split_val, categorical)) for split_val in split_values]
    return igr_for_splits