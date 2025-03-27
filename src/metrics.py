import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

def best_validation_threshold(y_val, val_anomaly_scores):
    fpr, tpr, thresholds = roc_curve(y_val, val_anomaly_scores)
    df_val_roc = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds})
    df_val_roc['youden-index'] = df_val_roc['tpr'] - df_val_roc['fpr']
    df_val_roc = df_val_roc.sort_values('youden-index', ascending=False, ignore_index=True).drop_duplicates('fpr')
    return df_val_roc.loc[0]

def accuracy(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
  return (tp+tn)/(tp+tn+fp+fn)

def get_overall_metrics(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  acc = (tp+tn)/(tp+tn+fp+fn)
  tpr = tp/(tp+fn)
  fpr = fp/(fp+tn)
  precision = tp/(tp+fp)
  f1 = (2*tpr*precision)/(tpr+precision)
  return {'acc':acc,'tpr':tpr,'fpr':fpr,'precision':precision,'f1-score':f1}

def plot_confusion_matrix(y_true, y_pred, name=""):
  plt.cla()
  plt.clf()
  cm = confusion_matrix(y_true, y_pred)
  group_counts = [f'{value:.0f}' for value in confusion_matrix(y_true, y_pred).ravel()]
  group_percentages = [f'{value*100:.2f}%' for value in confusion_matrix(y_true, y_pred).ravel()/np.sum(cm)]
  labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
  labels = np.array(labels).reshape(2,2)
  sns.heatmap(cm, annot=labels, cmap='Oranges', xticklabels=['Predicted Benign', 'Predicted Malicious'], yticklabels=['Actual Benign', 'Actual Malicious'], fmt='')
  plt.savefig(f"out_matrix_{name}.png")
  return

def plot_roc_curve(y_true, y_score, max_fpr=1.0, name=""):
  plt.cla()
  plt.clf()
  fpr, tpr, thresholds = roc_curve(y_true, y_score)
  aucroc = roc_auc_score(y_true, y_score)
  plt.plot(100*fpr[fpr < max_fpr], 100*tpr[fpr < max_fpr], label=f'ROC Curve (AUC = {aucroc:.4f})')
  plt.xlim(-2,102)
  plt.xlabel('FPR (%)')
  plt.ylabel('TPR (%)')
  plt.legend()
  plt.title('ROC Curve and AUCROC')
  plt.savefig(f"out_curve_{name}.png")