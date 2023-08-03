import pandas as pd
import numpy as np


#create deviations column
def deviations_col(df, score_col, prediction_col):
    deviations=pd.DataFrame(abs(df[score_col]-df[prediction_col]))
    df = df.assign(deviations=deviations)
    return df

#tag as 1 when larger then number. else 0.
def binary_tag(df, threshold, new_col_name):
    df.loc[:, new_col_name] = pd.DataFrame(np.where(df['deviations'] > threshold, 1, 0))
    return df

#return only tagged most deviated sorted reviews where prediction>overallscore
def sorted_tagged_and_larger_pred(df, prediction_col, score_col, deviations_col):
    sorted = df[df[prediction_col]>df[score_col]].sort_values(by=deviations_col, ascending=False)
    return sorted

#function for trees based regression models. printing train & test mse.
def score_summary(y_train, y_train_pred, y_test, y_test_pred, scorer):
    score_train = scorer(y_train, y_train_pred)
    score_test = scorer(y_test, y_test_pred)
    print(f"{scorer.__name__} (Train):", score_train)
    print(f"{scorer.__name__} (Test):", score_test)
    return score_train, score_test


def summary_df(data, prediction_col, score_col, tag_threshold, tagged_col):
    new_df= data[['Title','Review', 'OverallScore']]
    new_df['prediction'] = prediction_col
    new_df= deviations_col(new_df, score_col, 'prediction')
    new_df= binary_tag(new_df, tag_threshold, tagged_col)
    new_df= sorted_tagged_and_larger_pred(new_df, 'prediction', score_col, 'deviations')
    return new_df

def tagged_df(data, tag_col):
    new_data=data[data[tag_col]==1]
    return new_data




