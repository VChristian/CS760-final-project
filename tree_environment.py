import numpy as np
from sklearn.metrics import auc, roc_curve
import random 

from entropy import find_candidate_split

"""
What this will have to do:
    
    Paper Model:
    -----------
    1) keep track of currents splits
    2) construct the tree based on splits 
        - turn features into nodes
        - turn features into leaves
    3) evaluate model on test set
        - this will be the reward signal for the agent

    Enhancements:
    -------------
    1) construct a new reward signal at each step
        - then the objective would be to maximize total information gained from the tree
"""

class node():
    def __init__(self, node_type, feature = None, split_value = None, positive_label_prob = None):
        """
        Node class for the decision tree algorithm

        :param node_type: leaf or feature node (0 for leaf 1 for feature node)
        :param index: index of this node
        :param feature: index of feature we are splitting on
        :param split_value: value we are splitting on -- if it is an float then this is numerical -- if it is list then categorical
        :param positive_label_prob: for leafs - probability we label it 1
        """
        self.type = node_type
        self.feature = feature
        self.split_value = split_value
        self.positive_label_prob = positive_label_prob
        self.left_child = None
        self.right_child = None

    def get_type(self):
        """
        Get the type of node we are working (leaf or feature)
        """
        return self.type
    
    def set_left_child(self, child):
        """
        Set the left child
        """
        self.left_child = child

    def set_right_child(self, child):
        """
        Set the right child
        """
        self.right_child = child

    def get_left_child(self):
        """
        Get left child (return a leaf or feature)
        """
        return self.left_child
    
    def get_right_child(self):
        """
        Get right child (return a leaf or feature)
        """
        return self.right_child

class feature_node(node):
    def __init__(self, feature, split_value, categorical, node_type = 1):
        super().__init__(node_type, feature = feature, split_value = split_value)
        self.feature = feature
        self.split_value = split_value
        self.categorical = categorical

    def __call__(self, point):
        if self.categorical:
            if point[self.feature] == self.split_value:
                return super().get_left_child()
            else:
                return super().get_right_child()
        else:
            if point[self.feature] >= self.split_value:
                return super().get_left_child()
            else:
                return super().get_right_child()

class leaf_node(node):
    def __init__(self, index, positive_label_prob, node_type = 0):
        super().__init__(node_type, index, positive_label_prob = positive_label_prob)
    
    def __call__(self):
        return self.positive_label_prob

class tree():
    def __init__(self, head):
        self.head = head
    
    def classify(self, x, prob = True, prob_cut_off = .5):
        start_node = self.head
        while isinstance(start_node, feature_node) and start_node is not None:
            start_node = start_node(x)
        if prob:
            return start_node()
        else:
            prob0 = start_node()
            if prob0 >= prob_cut_off:
                return 1
            else:
                return 0

class decision_tree_env():
    """
    Tree environment
    """
    def __init__(self, tree_depth, dataset, feature_types, train_split = .6666):
        """
        :param tree_depth: depth of tree
        :param dataset: this is assumed to contain our training and valuation set 
        :param feature_types: list indicated if the variable is numeric or categorical
        """
        
        # features selected by our controller
        self.feature_to_split = [] # note that this is also our trajectory
        self.constructed_tree = None

        # dataset splits
        self.dataset = dataset
        self.dtrain = dataset[:int(len(dataset)*train_split)]
        self.dvalidation = dataset[int(len(dataset)*train_split):]
        self.feature_types = feature_types

        # stopping condition
        self.max_nodes = (2**tree_depth)-1
        self.k = tree_depth

        # rewards
        self.rewards = []

    def reward(self):
        """
        Calculate information gain ratio based off of split
        """
        pass

    def step(self, action):
        """
        NOTE: this might need to be changed if depending on our model

        Once step of our tree environment - add action to our feature_to_split list
        
        :param action: integer indexing the feature to split on
        :return: 
            reward - reward for the current step
            done - if the algorithm is done
        """
        self.feature_to_split.append(int(action))
        reward = 0
        done = False
        if len(self.feature_to_split) >= self.max_nodes:
            done = True
        return reward, done

    def reset(self):
        """
        Reset environment - clear the features to split, shuffle train, valuation, test
        """
        self.feature_to_split = []
        self.constructed_tree = None
        #random.shuffle(self.dataset)
        #self.dtrain = self.dataset[:int(3*len(self.dataset)/4)]
        #self.dvalidation = self.dataset[int(3*len(self.dataset)/4):]
        self.rewards = []

    def make_tree(self, index, dataset=None):
        """
        Given a list of features created by the policy create the tree 
        """
        def check_stop_conditions_igr(splits):
            """
            Check stop conditions - currently we are using a simplified version
            """
            index = 0
            for _, igr in splits:
                if igr == 0.0:
                    index += 1
            if index == len(splits):
                return True
            else:
                return False

        def check_stop_condition_leaf(index):
            if self.max_nodes == 1:
                return True
            elif index < self.max_nodes and index >= self.max_nodes-(2**(self.k-1)):
                return True
            else:
                return False

        def get_positive_prob(labels):
            """
            Return the probability of a positive class label
            """
            return sum(labels)/len(labels)
        
        if dataset is None:
            dataset = self.dataset

        if check_stop_condition_leaf(index):
            return leaf_node(index, get_positive_prob(dataset[:,-1]))

        # get candidate split
        feature = self.feature_to_split[index]
        candidate_splits = find_candidate_split(dataset, feature, self.feature_types[feature])

        # case when there is nothing left to split on
        if len(candidate_splits) == 0:
            return None

        # check if leaf conditions
        if check_stop_conditions_igr(candidate_splits): 
            return leaf_node(index, get_positive_prob(dataset[:,-1]))

        split_val, _ = sorted(candidate_splits, key=lambda x:x[1])[-1]
        head = feature_node(feature, split_val, self.feature_types[feature])

        left_split = None
        right_split = None
        if self.feature_types[feature]:
            left_split = dataset[dataset[:,feature] == split_val,:]
            right_split = dataset[dataset[:,feature] != split_val,:]
        else:
            left_split = dataset[dataset[:,feature] >= split_val,:]
            right_split = dataset[dataset[:,feature] < split_val,:]

        left_sub_tree = self.make_tree(2*index+1,left_split)
        head.set_left_child(left_sub_tree)
        right_sub_tree = self.make_tree(2*index+2, right_split)
        head.set_right_child(right_sub_tree)

        return head

    def set_tree(self, head):
        self.constructed_tree = tree(head)

    def get_tree(self):
        return self.constructed_tree

    def evaluate_tree(self, tree_object:tree, dataset = None):
        """
        Evaluate tree_object on the validation set -- report auc
        """
        if dataset is None:
            dataset = self.dvalidation

        labels = dataset[:,-1]
        data_for_auc = []
        for i in range(dataset.shape[0]):
            prob_pos_label = tree_object.classify(dataset[i,:-1])
            data_for_auc.append((labels[i], prob_pos_label))
        data = sorted(data_for_auc, key=lambda x:x[1])
        y = [label for label,_ in data]
        pred = [prob1 for _,prob1 in data]
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        score = auc(fpr, tpr)
        return score

def get_data(filename):
    '''
    Given a filename, get data

    Data format is is

    x11 x21 y1
    x12 x22 y2
    .
    .
    .
    x1n x2n yn

    :param filename: data file name
    :return: list of lists [x1i, x2i, yi]
    '''
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datum = line.strip('\n').split(' ')
            datum = [float(d) for d in datum]
            data.append(datum)
    return data

if __name__ == "__main__":
    dataset = np.asarray(get_data("data/D1.txt"))
    tree_env = decision_tree_env(2,dataset,[0,0], train_split=1)
    tree_env.feature_to_split = [1,0,1]
    head = tree_env.make_tree(0,dataset)
    tree_env.set_tree(head)
    print(tree_env)