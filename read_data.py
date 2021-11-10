"""
Read data and data manipulation

Output should be a matrix of elements
"""

import pandas as pd
import random

def get_data(file_name):
    """
    Return train, test, validation set
    """

    # 0 - quantitative, 1 - categorical
    heart_variable = [0,1,0,1,0,1,0,0,0,1,1,0]
    breast_cancer_variables = [0]*9

    header = 0
    flag = False
    if file_name == "breast-cancer-wisconsin.data":
        header = None
        flag = True
    data = pd.read_csv(file_name, header = header)

    if file_name == "heart.csv":
        df_x = data[["age", "creatinine_phosphokinase", "ejection_fraction", \
            "platelets", "serum_creatinine", "serum_sodium", "time"]]
        data[["age", "creatinine_phosphokinase", "ejection_fraction", \
            "platelets", "serum_creatinine", "serum_sodium", "time"]] = (df_x-df_x.mean())/df_x.std()

    dmatrix = data.values
    data_indices = [i for i in range(dmatrix.shape[0])]
    random.shuffle(data_indices)
    train_indices = data_indices[:int(3*len(data_indices)/4)]
    test_indices = data_indices[int(3*len(data_indices)/4):]
    if flag:
        for i in range(len(data_indices)):
            label = dmatrix[i,-1]
            if int(label) == 2:
                dmatrix[i,-1] = 0
            else:
                dmatrix[i,-1] = 1
        return dmatrix[:,1:], train_indices, test_indices, breast_cancer_variables 
    else:
        return dmatrix, train_indices, test_indices, heart_variable


