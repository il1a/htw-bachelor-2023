"""
utils.py
Additional python script, containing a custom function and a custom class that are being used across the notebooks.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn


def split_data(df, features):
    """
    Split data based on the given features.

    Arguments:
    - df: DataFrame to split
    - features: list of feature names

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test

    WARNING! This function implements specific logic that only implies to certain columns of the DataFrame being used
    for model training. Not suitable for generalized usage.
    """

    # Stack the specified features to form the input matrix
    X = np.hstack([df[feature].values.reshape(-1, 1) if feature in ['norp_count', 'vader_compound', 'gpe_count']
                   else np.stack(df[feature].values) for feature in features])

    y = df['label_encoded'].values

    # Split the data into train and a temporary set (80%-20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=1)

    # Split the temporary set into test and validation sets (50%-50%)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

    print("Training set:", len(X_train))
    print("Validation set:", len(X_val))
    print("Test set:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


class NewsClassifier(nn.Module):
    """
    A simple LSTM-based classifier for news categorization.
    By default, Bi-LSTM (bidirectional=True)

    Attributes:
    - lstm: LSTM layer
    - fc: Fully connected layer for classification
    - sigmoid: Sigmoid activation function
    """
    def __init__(self, input_dim=300, hidden_dim=50, output_dim=1, bidirectional=True):
        super(NewsClassifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_out = lstm_out[:, -1, :]
        out = self.fc(final_out)
        return self.sigmoid(out)
