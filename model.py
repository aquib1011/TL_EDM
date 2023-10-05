from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


def build_model_using_sequential(hidden_units1, hidden_units2, hidden_units3, hidden_units4):
    """
    Build a sequential model with 4 hidden layers and 1 output layer
    
        Parameters: 
             hidden_units1 (int): number of neurons in first hidden layer
             hidden_units2 (int): number of neurons in second hidden layer
             hidden_units3 (int): number of neurons in third hidden layer
             hidden_units4 (int): number of neurons in fourth hidden layer
        Returns:
            model (Sequential): sequential model
    """
    model = Sequential([
        Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
        Dropout(0.1),
        Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
        Dropout(0.1),
        Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
        Dropout(0.1),
        Dense(hidden_units4, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='linear')
    ])
    return model
 