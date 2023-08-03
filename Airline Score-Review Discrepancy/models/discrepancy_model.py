from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Discrepancy_Model:
    def __init__(self, params):
        # hyper params
        self.params = params
        self.model_class = params.get('model_class', None)
        self.model = self.model_class(**self.get_model_params(params))
        self.min_split_gain = params.get('min_split_gain', None)
        self.num_leaves = params.get('num_leaves', None)
        self.features = params.get('features', None)
        self.learning_rate = params.get('learning_rate', None)
        self.tag_threshold = params.get('tag_threshold')

    def get_model_params(self, params):
        # Extract only the relevant hyperparameters for the model
        model_params = {
            'objective': self.params.get('objective', None),
            'min_split_gain': self.params.get('min_split_gain', None),
            'num_leaves': self.params.get('num_leaves', None),
            'random_state': self.params.get('random_state', None),
            'verbose': self.params.get('verbose', None)
        }
        return model_params

    def train_test_split(self, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, y, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        self.y_test_pred = self.model.predict(self.X_test)
        self.y_test_pred = np.clip(self.y_test_pred, 1, 10)

        self.y_train_pred = self.model.predict(self.X_train)
        self.y_train_pred = np.clip(self.y_train_pred, 1, 10)
        
        self.prediction_col = self.model.predict(X)
        self.prediction_col = np.clip(self.prediction_col, 1, 10)
        return self
    
    #create deviations column
    def deviations_col(self, df, score_col, prediction_col):
        deviations=pd.DataFrame(abs(df[score_col]-df[prediction_col]))
        df = df.assign(deviations=deviations)
        return df

    #tag as 1 when larger then number. else 0.
    def binary_tag(self, df, new_col_name):
        df.loc[:, new_col_name] = pd.DataFrame(np.where(df['deviations'] > self.tag_threshold, 1, 0))
        return df

    #return only tagged most deviated sorted reviews where prediction>overallscore
    def sorted_tagged_and_larger_pred(self, df, prediction_col, score_col, deviations_col):
        sorted = df[df[prediction_col]>df[score_col]].sort_values(by=deviations_col, ascending=False)
        return sorted
    
    def full_summary_df(self, data, score_col, tagged_col):
        new_df= data[['Title','Review', 'OverallScore']]
        new_df['prediction'] = self.prediction_col
        new_df= self.deviations_col(new_df, score_col, 'prediction')
        new_df= self.binary_tag(new_df, tagged_col)
        self.full_summary_df= new_df
        return self
    

    def summary_df(self, data, score_col, tagged_col):
        new_df= data[['Title','Review', 'OverallScore']]
        new_df['prediction'] = self.prediction_col
        new_df= self.deviations_col(new_df, score_col, 'prediction')
        new_df= self.binary_tag(new_df, tagged_col)
        new_df= self.sorted_tagged_and_larger_pred(new_df, 'prediction', score_col, 'deviations')
        self.summary_df= new_df
        return self
    
    #function for trees based regression models. printing train & test mse.
    def score_summary(self, scorer):
        self.score_train = scorer(self.y_train, self.y_train_pred)
        self.score_test = scorer(self.y_test, self.y_test_pred)
        print(f"{scorer.__name__} (Train):", self.score_train)
        print(f"{scorer.__name__} (Test):", self.score_test)
        print(f"tagged samples number: ", len(self.tagged_df))
        return self
    
    def tagged_df(self, tag_col):
        new_data=self.summary_df[self.summary_df[tag_col]==1]
        self.tagged_df=new_data
        return self