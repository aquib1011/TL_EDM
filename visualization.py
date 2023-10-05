import matplotlib.pyplot as plt

def plot_history(history, key, category):
  '''
    Plot the history of the model
    Parameters:
        history (History): history of the model
        key (str): key of the history
        category (str): category of the history
    Returns:
        None
        
  '''
  plt.rcParams['font.family'] = 'Times New Roman'
  plt.rcParams['font.size'] = 15
  plt.rcParams['axes.linewidth'] = 1.5
  plt.rcParams["figure.dpi"] = 500
  plt.rcParams["figure.figsize"] = (6,4)
    
  plt.plot(history.history[key])
  plt.plot(history.history['val_'+key])
  plt.legend(["MSLE", 'val_MSLE'])

  plt.xlabel("Epochs")
  plt.ylabel("MSLE" +" "+ + category)
  plt.show()
