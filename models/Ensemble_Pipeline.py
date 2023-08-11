import numpy as np
import pandas as pd

class Ensemble_Pipeline:
    def __init__(self, params3):
        self.model1_predict = params3.get('model1_predict', None)
        self.model2_predict = params3.get('model2_predict', None)
        self.model1_y_train = params3.get('model1_y_train', None)
        self.model1_y_test = params3.get('model1_y_test', None)
        self.tag_threshold = params3.get('tag_threshold', None)
        self.threshold_low_score = params3.get('threshold_low_score', None)
        self.model2_base_weight = params3.get('model2_base_weight', None)
        self.model2_adjusted_weight = params3.get('model2_adjusted_weight', None)
        self.ensemble_scoring()

    #only if the title embedding is low enough- it gets 50% of score.
    def calc_weights(self):
        self.model2_weights = np.where(self.model2_predict < self.threshold_low_score, self.model2_adjusted_weight, self.model2_base_weight)
        self.model1_weights = 1 - self.model2_weights
        return self
    
    def calc_new_pred(self):
        self.new_prediction= (self.model1_predict*self.model1_weights)+(self.model2_predict*self.model2_weights)
        return self

    def ensemble_scoring(self): 
        # Calculate the adjusted weight for models
        self.calc_weights()
        #calculate new ensembled prediction column
        self.calc_new_pred()
        return self

    def deviations_col(self, df, score_col, prediction_col):
        deviations=pd.DataFrame(abs(df[score_col]-df[prediction_col]))
        df = df.assign(deviations=deviations)
        return df

    #tag as 1 when larger then number. else 0.
    def binary_tag_large(self, df, new_col_name):
        df.loc[:, new_col_name] = pd.DataFrame(np.where(df['new_prediction']-df['OverallScore'] > self.tag_threshold, 1, 0))
        return df

    #create df with model1 pred, model2 pre, new pred, deviation from new pred, tag, etc.
    def ensemble_df(self, data, score_col, tagged_col):
        new_df= data[['Title','Review', 'OverallScore']]
        new_df['prediction1'] = self.model1_predict
        new_df['prediction2'] = self.model2_predict
        new_df['new_prediction'] = self.new_prediction
        new_df= self.deviations_col(new_df, 'OverallScore', 'new_prediction')
        new_df= self.binary_tag_large(new_df, tagged_col)
        self.ensemble_summary_df= new_df
        self.tagged_col=tagged_col
        return self

    #screen for tagged samples and sort by top deviations
    def sorted_tags(self): #data, score_col, tagged_col
        #self.ensemble_df(data, score_col, tagged_col)
        self.tagged_df=self.ensemble_summary_df[self.ensemble_summary_df[self.tagged_col]==1].sort_values(by='deviations', ascending=False)
        return self

    #function for trees based regression models. printing train & test mse.
    def score_summary(self, scorer):
        self.score_train = scorer(self.model1_y_train, self.new_prediction[self.model1_y_train.index])
        self.score_test = scorer(self.model1_y_test, self.new_prediction[self.model1_y_test.index])
        print(f"{scorer.__name__} (Train):", self.score_train)
        print(f"{scorer.__name__} (Test):", self.score_test)
        print(f"tagged samples number: ", len(self.tagged_df))
        return self