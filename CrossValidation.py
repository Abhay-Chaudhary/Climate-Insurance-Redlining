import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, root_mean_squared_error as rmse

pd.options.display.float_format = '{:.6f}'.format

def calculate_diff(y_true, preds):
    return round(abs((sum(y_true)-sum(preds)) / (1000 * 1000)), 2)

def calculate_metrics(y_true, preds, fix_pct=True):
# hack to set to zero percentages below 0 for unbounded regressors
    if fix_pct:
        preds = np.where(np.array(preds) < 0, 0, preds)

    return [mse(y_true, preds), 
            rmse(y_true, preds), 
            mae(y_true, preds), 
            calculate_diff(y_true, preds),
            sum(y_true) / (1000 * 1000)]

def create_eval_table(vals):
    cols = ['Feature Set', 'Target Variable' ,'Model', 'MSE', 'RMSE', 'MAE', 'Global Diff', 'Total Val']
    res = pd.DataFrame(vals, columns=cols)
    return res.groupby(['Feature Set', 'Target Variable' ,'Model']).mean()    

def plot_metrics(eval_table, feature_set, metrics=['RMSE', 'MAE']):
    eval_table.loc[feature_set].droplevel(0).plot.bar(y=metrics, rot=0, width=.8, figsize=(8,3))

    plt.xlabel(None)
    plt.title(f'Model Comparison\n (Feature set: {feature_set})')
    
    plt.show()