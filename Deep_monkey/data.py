class Trial():
  def __init__(self,mua,events_code,times,st,re,events,ID):
    self.mua = mua
    self.events_code = events_code
    self.times = times
    self.st = st #1 if it's stop trial, 0 if it's nostop trial
    self.re = re #1 if it's correct and 0 if it's mistake
    self.events = events
    self.ID = ID


SESS_LIST = ['cornelio_10',
             'cornelio_12',
             'cornelio_13',
             'cornelio_14',
             'cornelio_17',
             'piero_0506', #nan values in electrode 89, be careful
             'piero_0606',
             'piero_0906']

def mua_smoother(mua_tensor,windowSize):
  '''Applies smoothing to the whole tensor mua of a single trial
  Parameters:
  -------
  mua_tensor: numpy tensor of shape (num_electrodes, time_bins)
  windowSize: length of window to applay causal smoothing
  Return:
  -------
  mua_tensor: the tensor mua smoothed
  '''
  

  for i in range(mua_tensor.shape[0]): #for loop over the trial electrodes
    mua_tensor[i,:] = scipy.signal.lfilter(np.ones(windowSize) / windowSize, 1, mua_tensor[i,:], )
  return mua_tensor

def extract_trials(filepath,session,windowSize = 16):
  """Extract the trials data from the matlab script that you can find on the repository.
  The functions uses mua_smoother function in order to perform causal smoothing
  Arguments
  --------
  filepath: the filepath of the single matlab file of the session to extract
  session: a string that will be used to track the trial ID after mixing them. Use one of the string
  in the variable SESS_LIST\n
  windowSize: algorithm will perform a causal smoothing over time in order to reduce noisy fluctuations.
  Provide the time steps range on which the smoothing will be performed. Default is 16 (80 ms), probably not
  important to the Deep Learning algorithm\n
  Returns
  -------
  trials: list of Trial objects associated to the selected countermanding session


  """
  all_matlab = scipy.io.loadmat(filepath)
  
  id_session = 'p{}_'.format(session[-4:-2]) if session[:3] == 'pie' else 'c{}_'.format(session[-2:])
  print('For session {} ID of trials will be: '.format(session),id_session)
  stop_corr_cond = [164,4004,4019,1444,179,169,4009,174,4014]


  Trials_data = all_matlab['Trials']
  MUA_cells = Trials_data['mua']



  Events_time = all_matlab['Events']['time']
  num_trials = len(MUA_cells[0][0][0])
  print('Found {} trials'.format(num_trials))
  trials = []
  p = 0 #index that slide trough the times variable that is unique
  for i in range(num_trials): #this is actually the number of trials

    #smoothing of the MUA
    smoothed_MUA = mua_smoother(MUA_cells[0][0][0][i], windowSize)


    #let's start by creating a simple array of int codes for events (reshape is to pass from (1,13) to (13,1) array size)
    trial_ev_codes = np.reshape(Trials_data['Events'][0][0][0][i], Trials_data['Events'][0][0][0][i].shape[1])
    # print(trial_ev_codes)
    #how much event indexes there are in this trial? Usually 12 or 13 so we need to check each time
    len_of_events = trial_ev_codes.shape[0] 
    
    #because of the structure of times (12-13 codes for each trial all in one line), 
    #we have to take slices of size len_of_events
    times = Events_time[0][0][0][p:p+len_of_events]

    #we need to convert the absolute times of events (calculated in seconds) to trials time bin. 
    #Every time rescale for the start time in order to have each trial with its own time. We will use the int times to but labels for the events
    # times = np.array(np.floor((times-times[0])*200.1148),dtype = 'int')


    #check for trial type: stop no stop, mistake-correct ecc.
    if 31 in trial_ev_codes:
      isstop = 1
      if 5000 in trial_ev_codes:
        iscorrect = 1
      elif 5006 in trial_ev_codes:
        iscorrect = 0
    else:
      isstop = 0
      if 3 in trial_ev_codes:
        iscorrect = 1
      else:
        iscorrect = 0


    #get the go time and stop time, and eventually the mistake/ reward time
    try:
       go_index = np.where(trial_ev_codes == 33)[0].item()
       go_time = int(np.floor((times[go_index]-times[0])*200.1148))
     
    except(ValueError):
      p+= len_of_events
      continue #if there's no go it is for sure a trial that we don't want :D

    try:
      stop_index = np.where(trial_ev_codes == 31)[0].item()
      stop_time = int(np.floor((times[stop_index]-times[0])*200.1148))
    except(ValueError):
      stop_time = np.nan

    try:
      mistake_index = np.where(trial_ev_codes == 5006)[0].item()
      mistake_time = int(np.floor((times[mistake_index]-times[0])*200.1148))
    except(ValueError):
      mistake_time = np.nan
    
    try:
      reward_index = np.where(trial_ev_codes == 5000)[0].item()
      reward_time = int(np.floor((times[reward_index]-times[0])*200.1148))
    except(ValueError):
      #3 is the index for reward in no-stop trials
      if 3 in trial_ev_codes:
        reward_index = np.where(trial_ev_codes == 3)[0].item()
        reward_time = int(np.floor((times[reward_index]-times[0])*200.1148))
      else:
        reward_time = np.nan

    event_dic = {'Go Signal': go_time, 'Stop Signal': stop_time, 'Mistake': mistake_time, 'Reward': reward_time}
    

    #let's store into the Trial object the events code of the single trial, the MUA of the single trial and the times of the events of the single trial
    t = Trial(smoothed_MUA,
              trial_ev_codes,
              times, isstop, iscorrect,
              event_dic, id_session+str(i))
    #and then put it into a list that will contain all the trials
    trials.append(t)

    #don't forget to update p! At next iteration the times of the trial will start from p, 
    #then the need of update it. Otherwise all trials will have the same times :D
    p+= len_of_events
  return trials
    
    

def take_stop_trials(trials):
  """Takes only the stop trials from a list of trials"""
  stop_t = []
  for t in trials:
    if t.st == 1:
      stop_t.append(t)

  print('{} stop trials was found'.format(len(stop_t)))
  return stop_t

def count_corr_mis(trials):
	"""Counts the number of correct trials and the number of mistake trials"""
	print('In {} trials were found:'.format(len(trials)))
	cor = 0
	mis = 0
	for t in trials:
		if t.re == 1: cor+=1
		else: mis +=1
	print('Correct trials: ',cor, '\tMistake Trials: ',mis)


def extract_tensor(trials,before_length,after_length, align_control,channels = 96):
  """Extract the mua tensor from a list of trial
  Parameters
  -------
  trials: list of Trial object from which the mua will be extracted\n
  before_length: time steps to take before the align event (each step is 5 ms)\n
  after_length: time steps to take after the align event (each step is 5 ms)\n
  align_control: event of the trial on which you want to align. Must be one key inside Trial.events dictionary\n
  channels: provide the number of channels that the tensor returned will have. Default is 96\n
  Returns
  --------
  x: the tensor associated to the trials mua. It is a numpy array of shape (num_trials, time_steps, channels)"""
  
  x = np.zeros((len(trials),before_length + after_length,channels))
  for i,t in enumerate(trials):
    start = t.events[align_control] - before_length
    end = t.events[align_control] + after_length
    useful = t.mua[:channels,start:end]
    x[i,:,:useful.shape[0]] = np.swapaxes(useful,0,1)
    x[i] = np.log(x[i])
    # x[i] = x[i] - np.mean(x[i], axis = 1, keepdims = True)
  return x


def ResampleLinear1D(original, targetLen):
    #initial array
    original = np.array(original, dtype=np.float)
    # create array indexes in the range of original but with targetLEN values
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)

    index_floor = np.array(index_arr, dtype=np.int) #Round down
    
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp


def extract_strecth_tensor(trials,stretch_length, align_controls,channels):
  """Extract the stretched mua tensor from a list of trial. Interpolations is performed by ResampleLinear1D function
  Parameters
  -------
  trials: list of Trial object from which the mua will be extracted\n
  stretch_length: time steps in which the trial will be interpolated between the two events given\n
  align_controls: tuple of Trial.events dictionary keys. Example: ('Go Signal', 'Stop Signal')
  channels: provide the number of channels that the tensor returned will have. Default is 96\n
  Returns
  --------
  x: the tensor associated to the trials mua. It is a numpy array of shape (num_trials, stretch_length, channels)"""
  
  x = np.zeros((len(trials),stretch_length,channels))
  for i,t in enumerate(trials):
    start = t.events[align_controls[0]] 
    end = t.events[align_controls[1]]
    useful = t.mua[:channels,start:end]
    useful = np.swapaxes(useful,0,1)
    for c in range(useful.shape[1]): #loop over electrodes
      x[i,:,c] = ResampleLinear1D(useful[:,c],stretch_length)
    x[i] = np.log(x[i])
    # x[i] = x[i] - np.mean(x[i], axis = 1, keepdims = True)
  return x

def calc_ssd(trial):
  """Compute the ssd of a single Trial object. 
  The ssd is the 'Stop Signal delay' meaning the time distance (in time bins, each one of 5 ms)
  between the Go signal and the Stop signal"""
  return trial.events['Stop Signal']-trial.events['Go Signal']


def remove_bad_electrodes(tensor):
  """Clean a single trial numpy tensor from bad electrodes. Use only if you are analyzing Cornelio alone
  Arguments:
  --------
  tensor: numpy tensor of shape (time_steps, channels) representing the mua of the SINGLE trial\n
  Returns
  -------
  tensor: new tensor of shape (time_steps, channels - bad_channels)"""
  
  bad_ele = [22,23, 25,55,57, 59, 61, 63,65, 67, 69, 71,90]


    
  new_ten = np.zeros((tensor.shape[0],tensor.shape[-1]-len(bad_ele)))
  j= 0
  for i in range (tensor.shape[-1]):
    if i not in bad_ele:
      new_ten[:,j] = tensor[:,i]
      j+=1
  return new_ten

def create_random_folds(trials,foldnum):
  """Create a set of indipendent folds in order to perform K-cross validation
  Parameters
  --------
  trials: list of Trial objects\n
  foldnum: number of folds to generate
  Returns
  -------
  kfolds: a python dictionary with keys 'traini' 'testi' where i is the fold index. 
  The union of all kfolds['testi'] will be the all dataset. kfolds['traini'] and kfolds['testi'] are independent, 
  so can be used as x_train and x_test by using the functions extract_tensor or extract_stretch_tensor"""
  kfolds = {}
  testsize = len(trials)//foldnum
  for i in range(foldnum):
    kfolds['test{}'.format(i)] = trials[testsize*i:testsize*(i+1)]
    kfolds['train{}'.format(i)] = trials[:testsize*i]+trials[testsize*(i+1):]
    print('fold {}: train: {} test: {}'.format(i, len(kfolds['train{}'.format(i)]), len(kfolds['test{}'.format(i)])))
  return kfolds


def get_ids(trials):
  """Returns a list of the trial IDs"""
  
  ids = []
  for t in trials:
    ids.append(t.idnum)
  return ids

def take_class_trials(trials,selected_class):
  '''Returns the trials of "one" class i.e. just stop and NOT no-stop. '''
  class_atts = {'stop':'st','nostop':'st', 'left':'lr', 'right':'lr', 'reward':'re', 'mistake':'re'}
  class_return = {'stop':1,'nostop':0, 'left':1, 'right':0, 'reward':1, 'mistake':0}
  zero_trials, one_trials = [],[]
  for t in trials:
    if t.__dict__[class_atts[selected_class]] == 0:
      
      zero_trials.append(t)
    else:
      one_trials.append(t)
  if class_return[selected_class] == 1:
    return one_trials
  else:
    return zero_trials  
  

def plot_trial_from_tensor(tensor,time_bin = 5, xlim = None,events= None):
  """Plots the MUA of a single trial
  Parameters:
  -------
  tensor: numpy tensor of shape (time_steps, channels)\n
  time_bin: value of each time_bin. Default is 5 ms\n
  xlim = time range (provide it in time_bin units, not real time)\n
  events = Trial.events keys (such as 'Go Signal'). Vertical lines will be plott in correspondence of selected events.\n
  Returns:
  -------
  None, just plot """
  ascisse = range(0,tensor.shape[0]*time_bin,time_bin)
  fig = plt.figure(figsize = (10,6))

  for i in range (tensor.shape[-1]):
    plt.plot(ascisse,tensor[:,i],linewidth = .4)
  plt.plot(ascisse, np.mean(tensor, axis = 1), color = 'black')
  if events:
    for e in events:
      plt.axvline(e*time_bin, linestyle = 'dashed')
  if xlim:
    plt.xlim([x*time_bin for x in xlim])
  plt.xlabel('Time [ms]')
  
  plt.show()



def plot_trial(trial, xlims = None,ylims = None):
  """Plots the MUA for a Trial object
  Parameters:
  -------
  trial: Trial object\n
  xlims: optional time range, express it in time bins and not in real time
  ylims: optional, range for MUA values

  """
  time_bin = 5
  colors = {'Go Signal': 'blue', 'Stop Signal': 'orange', 'Detouches Central': 'hotpink', 
            'Touches Target': 'darksalmon','Reward': 'green', 'Mistake': 'red'}
  x_range = range(0,trial.mua.shape[-1]*time_bin,time_bin)
  plt.figure(figsize = (10,6))
  for i in range(trial.mua.shape[0]):
    plt.plot(x_range,trial.mua[i,:], linewidth = .4)
  mean_mua = np.mean(trial.mua,axis = 0)
  
  plt.plot(x_range, mean_mua, linestyle = 'solid', color = 'black', label = 'mean')
  if xlims:
    #xlims = np.array(lims)*time_bin
    plt.xlim([x*time_bin for x in xlims])
  
  else:
    plt.xlim((trial.events['Trial Start']-10)*time_bin, (trial.events['Trial End']+10)*time_bin)
  if ylims:
      plt.ylim(ylims)  
  dont_plot = ['Trial Start', 'Target Appears', 'Trial End', 'Central Spot Appears', 'Touches Central']
  for j,i in enumerate(trial.events):
    if i not in dont_plot and trial.events[i] != 'nan':
      plt.axvline(trial.events[i]*time_bin, linestyle = 'dashed', color = colors[i], label = i)
      #plt.annotate(i, (trial.events[i]-30,6-0.35*j), color = 'darkgreen')
  plt.legend()
  plt.title('Multi Unit Activity')
  plt.xlabel('Steps of 5 ms each')
  plt.show()    



def demean_tensor(tensor):
  """Subtract the spatial mean to the data tensor
  Arguments
  -------
  tensor: provide the 3d tensor data of shape (num_trials, time_steps, channels)
  Returns
  -------
  the same tensor with mean subtracted
  """
  return tensor - np.mean(tensor, axis = 2, keepdims= True)


def get_class_weights(labs,time_lenght):
  """Computes the weights to use for unbalanced datasets:
  Parameters
  --------
  labs: pass them in the shape (trials_num,)\n
  Returns
  --------
  weights: matrix of shape (trials_num,time_lenght)
  """
  
  num_one = np.count_nonzero(labs)
  num_zero = labs.shape[0]-num_one
  tot_num = labs.shape[0]
  weights = np.zeros((labs.shape[0],time_lenght))
  for w in range(weights.shape[0]):
    weights[w] = tot_num/num_one if labs[w] == 1. else tot_num/num_zero
   
  return weights  




