import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_datasets(x_train, x_test):

  """
  Standard Scale test and train datasets

    Parameters: 
        x_train (DataFrame): training dataset
        x_test (DataFrame): test dataset

    Returns:
        x_train_scaled (DataFrame): scaled training dataset
        x_test_scaled (DataFrame): scaled test dataset
  """
  standard_scaler = StandardScaler()
  x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns=x_train.columns
  )
  x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns = x_test.columns
  )
  return x_train_scaled, x_test_scaled
