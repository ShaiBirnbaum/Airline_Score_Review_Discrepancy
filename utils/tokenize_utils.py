
import pandas as pd
import numpy as np

#stack and save embeddings to a feather and csv file, and return rmbeddings stacked df.
def save_embeddings_list(embeddings_list, file_name):
    stacked_embeddings=np.stack(embeddings_list) #stack
    df_embeddings = pd.DataFrame(stacked_embeddings) 
    df_embeddings.columns = df_embeddings.columns.astype(str) #needed for feather file
    df_embeddings.to_feather(file_name + ".feather")
    df_embeddings.to_csv(file_name + ".csv") #backup
    return df_embeddings