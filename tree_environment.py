import numpy as np
import random 

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
    def __init__(self, node_type, index, feature = None, split_value = None, positive_label_prob = None):
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
        self.index = index
        self.left_child = None
        self.right_child = None

    def get_type(self):
        return self.type
    
    def set_left_child(self, child_index):
        self.left_child = child_index

    def set_right_child(self, child_index):
        self.right_child = child_index

class feature_node(node):
    def __init__(self, node_type, index, feature, split_value):
        super().__init__(node_type, index, feature = feature, split_value = split_value)
    
    def __call__(self, point):
        pass

class leaf_node(node):
    def __init__(self, node_type, index, positive_label_prob):
        super().__init__(node_type, index, positive_label_prob = positive_label_prob)
    
    def __call__(self):
        return self.positive_label_prob

class tree_environmnet():
    """
    Tree environment
    """
    def __init__(self, tree_depth, dataset):
        """
        :param tree_depth: depth of tree
        :param dataset: this is assumed to contain our training and valuation set 
        """
        
        # features selected by our controller
        self.feature_to_split = [] # note that this is also our trajectory
        
        # dataset splits
        self.dataset = dataset
        self.train = dataset[:int(3*len(dataset)/4)]
        self.valuation = dataset[int(3*len(dataset)/4):]

        # stopping condition
        self.max_nodes = (2**tree_depth)-1

    def reward(self):
        """
        Calculate information gain ratio based off of split
        """
        pass

    def step(self, action):
        """
        Once step of our tree environment - add action to our feature_to_split list
        
        :param action: integer indexing the feature to split on
        :return: 
            reward - reward for the current step
            done - if the algorithm is done
        """
        self.feature_to_split.append(action)
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
        random.shuffle(self.dataset)
        self.train = self.dataset[:int(3*len(self.dataset)/4)]
        self.valuation = self.dataset[int(3*len(self.dataset)/4):]

    def get_tree(self):
        """
        Given a list of features created by the policy create the tree 
        """
        def check_stop_conditions(self):
            pass

    def evaluate_tree(self):
        pass
