import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import matplotlib.pyplot as plt


def do_boxplot(ax, acc, positions , color, label):
  acc_bplot = ax.boxplot(acc, whis= 3e5, patch_artist=True ,
                        whiskerprops = {'linestyle':'-','linewidth':2.0, 'color': color},
                       medianprops= {'linestyle':'-','linewidth':5.0, 'color': color},
                       positions = positions, widths = 2
                       )
  for box in acc_bplot['boxes']:
    box.set_facecolor(color)
    box.set_alpha(0.45)
  
  for cap in acc_bplot['caps']:
    cap.set_color(color)
    cap.set_linewidth(2.)
  return acc_bplot


def put_tpr_and_tnr_values(ax, tpr_values, tnr_values, positions, medians):
  for i,xpos in enumerate(positions):
    ax.annotate(np.round(np.median(tpr_values, axis = 0)[i],2), (xpos-10,medians[i]+0.01), fontsize = 15, color = 'red',weight = 'semibold')
    ax.annotate(np.round(np.median(tnr_values, axis = 0)[i],2), (xpos-10,medians[i]-0.02), fontsize = 15, color = 'green',weight = 'semibold')


def plot_conf_matrix(cm,title):
  classes = ['Mistake', 'Correct']
  fig, ax = plt.subplots(figsize = (5,5))
  im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  
  
  
  
  
  cbar = ax.figure.colorbar(im, ax=ax)
  cbar.ax.tick_params(labelsize=15)
  # We want to show all ticks...
  ax.set_xticks((0,1))
  ax.set_yticks((0,1))
  # ax.set_yticks(np.arange(cm.shape[0]))
  # ... and label them with the respective list entries
  ax.set_xticklabels(classes, fontsize = 15)
  ax.set_yticklabels(classes,rotation = 'vertical', fontsize = 15, ha = 'center', va = 'center')
  ax.set_ylabel('True labels', fontsize = 16)
  ax.set_xlabel('Predicted labels', fontsize = 16)


  fmt = 'f'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        
        ax.text(j, i, "{:0.2f}".format(cm[i, j]),
                ha="center", va="center", fontsize = 25,
                color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  # plt.suptitle('C')
  plt.ylim([1.5, -.5])
  plt.savefig('cm_mua.png')
  plt.show()
  



