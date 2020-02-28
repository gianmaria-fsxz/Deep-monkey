def manual_weighted_roc(preds,trues, plot_ind = True):
  """Computes tpr and fpr of NN preds prediction using prevalence i.e. weighting the occurence of one class in respect of the other. 
  The result will be a ROC curve weighted on classes examples numbers. POSITIVE means class 1, NEGATIVE is class 0 for a binary classifier
  Parameters:
  --------
  preds: outputs of the classifier of shape (batch_size,)\n
  trues: labels for ground truth, same shape of preds\n
  plot_ind: if you want to plot the roc_curve\n
  Returns:
  -------
  fpr,tpr (invert them to get tnr and fnr): they are computed for 100 threshold values, np.arange(0.0,1.00,0.01). 
  You can calculate your threshold with get_th and then extract the tpr and fpr for that given threshold. For example, if 
  the threshold is 0.61 , you'll check the tpr with index =  np.where(np.arange(0.,1.,0.01) == 0.61)[0].item()
  This way tpr[index] will be your tpr at Youden Index
  
  """
  thresh = np.arange(0.0,1.00,0.01)
  tpr,fpr,auc = np.zeros((thresh.shape)), np.zeros((thresh.shape)),np.zeros((thresh.shape))
  pt = np.count_nonzero(trues)/trues.shape[0]
  pf = 1-pt
  for j,t in enumerate(thresh):
    tp,fp,tn,fn = 0,0,0,0
    for w in range(trues.shape[0]):
      if trues[w] == 1.:
        if preds[w] > t:
          tp+=1
        else:
          fn += 1
          
      else:
        if preds[w] < t:
          tn+=1
        else:
          fp += 1
        
    tpr[j] = (tp*pt)/(tp*pt+fn*pt)
    fpr[j] = (fp*pf)/(tn*pf+fp*pf)
  if plot_ind == True:
    plt.plot(fpr,tpr)
    
    plt.plot(np.arange(0.,1.1,.1),np.arange(0.,1.1,.1), linewidth = .1,linestyle = '--')
    plt.title('ROC CURVE',fontsize = 15)
    plt.xlabel('False Positive Rate (Specificity)',fontsize = 13)
    plt.ylabel('True Positive Rate (Sensitivity)',fontsize = 13)
    plt.savefig('roc.png')
    plt.show()
  return fpr,tpr

def get_th(preds, lab):
  """Computes the threshold value that best separate the two classes by Youden criterion 
  i.e. the value that maximes true positive and true negative. This function should be used on the train set 
  and succesively used as threshold for test set
  Parameters:
  --------
  preds: numpy array that represents the outputs of the NN\n
  lab: numpy binary array of 0 or 1 representing the labels
  Returns:
  --------
  th: float value of the threshold
  """
  tp_tn = []

  vals =  np.arange(0.0,1.00, 0.01)
  
  fpr,tpr = manual_weighted_roc(preds,lab,plot_ind = False)
  fnr = 1. - tpr
  tnr = 1. - fpr
  ind = np.argmax(tpr+tnr)
  return vals[ind]

def calc_acc(preds,trues,thresh):
  """Computes the accuracy using the threshold obtained from get_th function
  Parameters:
  -------
  preds: numpy array that represents the outputs of the NN\n
  trues: numpy binary array of 0 or 1 representing the labels\n
  thresh: float value, can be computed by the function get_th\n
  Returns:
  -------
  accuracy: fraction of corrected predicted trials """
  

  score = 0
  for pred, gtr in zip(preds, trues):
    if pred > thresh and gtr == 1.:
      score +=1
    if pred < thresh and gtr == 0.:
      score += 1
  return score/preds.shape[0]  


def plot_preds_hist(preds,trues,mist_ind, title = None):
  """Plots the histogram of the outputs of the NN
  Parameters:
  -------
  preds: numpy array that represents the outputs of the NN\n
  trues: numpy binary array of 0 or 1 representing the labels\n
  mist_ind: ground truth value for mistake trials (1 or 0)\n
  title: optional\n
  """
  plt.hist(preds[trues==mist_ind], bins = np.arange(0.,1.01,.02), color = 'red', trues = 'Mistake', alpha = 0.55 )
  plt.hist(preds[trues==np.abs(1-mist_ind)], bins = np.arange(0.,1.01,.02), color = 'green', trues = 'Correct', alpha = 0.55)
  plt.legend()
  plt.title(title)
  plt.show()

def plot_rnn_outputs(preds,trues,mis_ind):
  """Plot the outputs of the RNN over 'time'
  Parameters:
  -------
  preds: outputs of the RNN. Must be a numpy array of shape (num_trials, time_steps)\n
  trues: ground truths for preds, same shape
  mis_ind: value for 'mistake' class (0 or 1)"""              



  if len(preds.shape) == 1:
    print('Are you sure you are passing the right values?')
    return


  for j in range(preds.shape[0]):
    ind =  trues[j,0,mis_ind]
    if mis_ind == 0:
      color = 'red' if ind == 1. else 'green'
    else: 
      color = 'red' if ind == 0. else 'green'
    

    plt.plot(range(0,TIME_LENGHT*TIME_BIN,TIME_BIN),preds[j], color = color)
    red_patch = mpatches.Patch(color='red', label='mistake')
  green_patch = mpatches.Patch(color='green', label='correct')

  plt.legend(handles=[red_patch,green_patch], bbox_to_anchor=(1.,1.))    
  plt.title('Readout of the NN over time for single double-neuron classifier'.format(i)) 
  plt.xlabel('time [ms]')
  plt.show()  

