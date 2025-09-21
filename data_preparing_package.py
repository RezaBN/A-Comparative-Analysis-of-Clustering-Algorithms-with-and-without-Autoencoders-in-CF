import numpy as np
import pandas as pd
import sklearn

def read_data(selected_data):
    if selected_data == 'MovieLens_100K':
        df = pd.read_csv('Datasets\Data.csv')
        return df
    else:
        raise ValueError("Selected dataset is not supported.")



# Create the user-item matrix
def create_useritem_matrix(df):
    
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating')
    
    # Fill missing values with zeros (NaN values become zeros)
    user_item_matrix = user_item_matrix.fillna(0)

    return user_item_matrix



# Split the user-item matrix into training and testing sets
def test_train(user_item_matrix):

    from sklearn.model_selection import train_test_split
    
    train_matrix, test_matrix = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

    return train_matrix, test_matrix


