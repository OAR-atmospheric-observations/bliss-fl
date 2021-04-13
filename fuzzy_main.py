
"""
Created on Jan 5 2021

@author: elizabeth.smith/jacob.carlin
"""
"""
This is a starter code from ES's read-in/plotter script. We are collaborating now to 
convert code(s) shared by Tim Bonin from matlab to python for our needs. Tim's code(s)
help us build the fuzzy logic algorithm we want to apply to our PBLTops datasets!
"""
#########################################################
#All the imports
#########################################################
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
import scipy.optimize as sp
import scipy.signal as ss
import pandas as pd
from siphon.catalog import TDSCatalog
from datetime import datetime, timezone, timedelta
from suntime import Sun
import cmocean as cm

#this is making the plots pretty!
plt.style.use('bmh')
#this is making the font big enough to read
plt.rcParams['font.size'] = 16
x_tick_labels = ['00','03','06','09','12','15','18','21']

#################################################################################################################

#########################################################
#FUNCTIONS
#########################################################


################ ES temporal variance profiles ##############
def calcSigma(stare_t, stare_z, stare_field, window=.25):
    """
    This function computes the variance of a field within a specified time window.
    Adapted from existing ES code for variance of CLAMPS data.
    
    Args:
        stare_t (array): 1-D array of times [hr]
        stare_z (array): 1-D array of heights [km]
        stare_field (array): 2-D array of field to compute variance of
        window (float): Time window for computing variance [h] (Default = 0.25)
        
    Returns:
        var_field (array): 2-D array of variances over specified window
        time_axis (array): 1-D array of new times
    """
    
    time_axis = np.arange(0, stare_t[-1], window) # New time axis
    var_field = np.full((len(time_axis), len(stare_z)), np.nan) # Variance over window
    prev_index = 0 # previous index for looping below
    for j in range(0, len(stare_z)):
        work = stare_field[:, j]
        for i in range(0, len(time_axis) - 1):
            X = np.where(stare_t < time_axis[i+1])[0]
            if X.size < 1:
                continue
            end = X[-1]
            var_field[i, j] = np.nanvar(work[prev_index:end])
            prev_index = end
            
    return var_field, time_axis

  
##########################################################


################### ES Moving Avg ##########################
def moving_average(a, n=3):
    """
    This function calculates a moving average over some window.
    
    Args:
        a (array): 1-D array of field
        n (int): Window size, default 3 point
    Returns:
        ret (array): 1-D array of averaged field
    """
    
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    return ret[n - 1:] / n

  
##########################################################
 

################ ES uv frm spd/dir########################
def uv_from_spd_dir(speed, wdir):
    """
    This function computes the u- and v-components of the wind
    given speed and direction.
    
    Args:
        speed (array): Wind speeds
        wdir (array): Wind direction in degrees
    Returns:
        u (array): u-component of the wind
        v (array): v-components of the wind
    """
    
    wdir = np.deg2rad(wdir)
    u = -speed * np.sin(wdir)
    v = -speed * np.cos(wdir)
    
    return u, v
  
  
##########################################################


##########################################################
def find_closest(A, target):
    """
    This function finds the index of the closest value to a 
    target value with a specified array.
    Source: Jeremy Gibbs (NOAA-NSSL)
    
    Args:
        A (array): Data to search for closest value
        target (float): Value being searched for
    Returns:
        idx (int): Index of A with value closest to target.
    """

    idx = A.searchsorted(target) # A must be sorted
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    
    return idx
  
  
##########################################################



##########################################################
#our fuzzy functions start here -- we wrote them all
##########################################################

##########################################################
def find_peaks(vals, num_peaks=5):
    """
    This function finds x number of highest peak values from a 1D arrary.
    
    Args: 
        vals (array): 1-D array of values 
        num_peaks (int): default 5, number of peaks desired
        
    Returns:
        ret (array): 1-D array of peak indicies
    """
    
    temp = np.argpartition(-vals, num_peaks)
    result_args = temp[:num_peaks]
    
    return result_args
  
  
##########################################################


##########################################################
def mean_u(ws, t, z, new_t, lo=.1, hi=1.):
  """
  This function computes a mean wind speed over a layer.
  
  Args:
      ws (array): 2-D windspeed (time, height)
      t (array): 1-D time
      z (array): 1-D height
      new_t (array): 1-D array of desired output time
      lo (float): default = 0.1, lower limit for averaging layer, [km]
      hi (float): default = 1.0, upper limit for averaging layer, [km]
  
  Returns:
        mean_u (array): 1-D array of layer averaged windspeed
  """
  mean_U = np.full(ws.shape[0], np.nan)
  lo_ind = np.where(z > lo)[0][0] - 1
  hi_ind = np.where(z > hi)[0][0]
  for i in range(ws.shape[0]):
    mean_U[i] = np.nanmean(ws[i, lo_ind:hi_ind]) 
  f = interp1d(t, mean_U, fill_value="extrapolate")
  mean_u = f(new_t)
  
  return mean_u


##########################################################


##########################################################
def HF_varcalc(field, time, height):
  """
  This function computes the high frequency variance fields. It first
  does a high-pass filter, then uses the calcSigma function to get
  varaince.
  
  Args:
      field (array): 2-D field for which high frequency variance is needed
                       [time, height]
      time (array): 1-D array of time
      height (array): 1-D array of height
  
  Returns:
      filt_field (array): 2-D field of same shape as field after high-pass filter
      HFsigw (array): 2-D array of the high frequency variance computed from filt_field
      HFsigw_t (array): 1-D array of new time axis for HFsigw
  """
  
  filt_field = np.full(field.shape,np.nan)
  HFThresh = 0.01667 #this frequency is corresponding to a 1 min period
                     # any freq higher than this is considered high frequency
  order = 6
  b, a = ss.butter(order, HFThresh, btype='lowpass', output='ba')
  # note this is lowpass, but data-lowpass == highpass and vice-versa 
  for i in range(len(height)):
      field_nan = field[:,i]
      field_nan[np.isnan(field[:, i])] = 0.0
      filt = ss.filtfilt(b, a, field_nan)      
      filt_field[:, i] = field[:, i] - filt
  HFsigw, HFsigw_t = calcSigma(time, height, filt_field, window=0.166) #computing the vertical velocity variance from raw w in 5 min=.083 (10=0.166)windows
  HFsigw[np.where(HFsigw > 5)] = np.nan
  return filt_field, HFsigw, HFsigw_t
  
  
##########################################################


##########################################################
def fuzzify(var_input, bound_low, bound_high):
  """
  This function produces membership values given appropriate cutoffs for 
  the fuzzy logic application (half-trapezoidal).
  
  Args:
      var_input (array): values for which the membership function is needed
      bound_low (float): x1 value, lower bound for membership, below which = 1
      bound_high (float): x2 value, upper bound for membership, above which = 0
  
  Returns:
      var_output (array): array same shape as var_input holding membership values
  """
  # Create output array 
  var_output = np.zeros_like(var_input) 
  # Keeps nans where the are from input values
  var_output[np.isnan(var_input)] = np.nan
  # set values based on x1, x2 bounds provided
  var_output[var_input > bound_high] = 1.0
  var_output[var_input < bound_low] = 0.0
  # set values between 0,1 given slope between bound points
  slope = 1.0 / (bound_high - bound_low)
  g = np.logical_and((var_input >= bound_low), (var_input <= bound_high))
  var_output[g] = slope * (var_input[g] - bound_low)
  # Clean up output and force to be between (0, 1)
  var_output = np.clip(var_output, 0.0, 1.0)
  
  return var_output


##########################################################


##########################################################
def fuzzGrid(input_var, input_x, input_y, start_time=0., end_time=1440., low_z = 0.01):
  """
  This function regrids input to 10-min/10-m grid
  
  Args:
      input_var (array): 2-D array field to be regridded
      input_x (array): 1-D x-axis of input_var
      input_y (array): 1-D y-axis of input_var
      start_time (int): the beginning of the desired output time array (default 0 min)
      end_time (int): the end of the desired output time array (default 1440 min)
      low_z (int): the lowest index of the desired output height array (lowest lidar height, default 10 m)
      
  Returns:
      output_var (array): 2-D array field now on 10-min/10-m grid
  """
  
  # Define function characterizing original data grid
  f = RegularGridInterpolator(points=[input_x, input_y], values=input_var)#, bounds_error=False, fill_value=np.nan)
  
  x_new = np.arange(start_time, end_time+10., 10.) # Interpolate to 10-min intervals
  y_new = np.arange(low_z,4.001,0.01) # Interpolate to 10-m intervals
  
  #check the end of the file to see if the end time is too early (15 min allowance)
  diff=x_new[-1]-input_x[-1]
  if diff> 0:
      if diff<0.26: # if the input time ends 15 min or less earlier than the desired time axis
          input_x[-1]=x_new[-1] #set it to the end of the desired time axis
      else: #if it ends more than 15 min earlier than the time axis
          print('THIS FILE IS TOO SHORT YO... Figure it out.') #we do nothing and print a message
          #the below code will fail in this case. Should we add an exit? 
          
  g,h = np.meshgrid(y_new,x_new)
  G=g.reshape((np.prod(g.shape),))
  H=h.reshape((np.prod(h.shape),))
  coords=zip(H,G)
  pts = list(coords)
  pts = np.asarray(pts)
  output = f(pts) # Interpolate to new grid points
  output_var = output.reshape(np.shape(g)) # Reshape to new array size
  
  return output_var


##########################################################


def run_haar_wavelet(profile, z, haarDilationLength=100):
  """
  Perform a Haar wavelet analysis on a given profile at heights z.
  Default haarDilationLength 
  
  Args:
    profile: Data profile to perform wavelet analysis on.
    z: Attendant height vector [km]
    haarDilationLength = Size of Haar wavelet window (default = 100 m)
    
  Returns:
  	Wavelet transform
  """
  
  # Compute the number of dilation gates
  z = z * 1e3 # Convert to m
  rGateSize = np.ediff1d(z)
  numDilationGates = int(np.ceil(haarDilationLength / (2 * rGateSize))[0])
  
  # Define Haar wavelet function
  haar_func = np.full((2 * int(numDilationGates) + 1,), np.nan)
  haar_func[0:numDilationGates] = 1;
  haar_func[numDilationGates] = 0
  haar_func[(numDilationGates+1):2*numDilationGates+1] = -1

  # Convolve the wavelet over all range gates
  nHeights = len(z)
  w_trans = np.full((nHeights), 0, dtype='float')
  for j in range(numDilationGates, nHeights - numDilationGates):
    for k in range(0, len(haar_func)):
      tmp = ((1./haarDilationLength) * rGateSize[j] * haar_func[k] * profile[j - numDilationGates + k])
      if ~np.isnan(tmp):
        w_trans[j] += tmp

  return w_trans


##########################################################
def calc_haar_membership(profile, z, BLhgt):
  """
  This function dynamically computes membership functions based on haar wavelet transfrom as computed
  by the run_haar_wavelet function.
  Membership = 1 at all levels below the lowest considered peak. Membership = 0 at all levels above 
  the highest considered peak. For each peak in between, the membership function decreases in steps.
  D(mem)/dz)=peak value/(sum of all retained peak values).  See Bonin et al. (2018) for details. 
  
  Args: 
      profile (array): 1-D profile of transformed wavelet profile
      z (array): 1-D height data
      BLhgt (array): 1-D timeseries of first guess BL heights
      
  Returns:
      prof_fuzz (array): same shape as profile holding membership values based on haar wavelet method
      top5_hgt (list): list of heights of the retained peaks used in membership function
  """
  heights = z

  #find top five peaks in cD output
  top5_idx = np.sort(find_peaks(profile))
  #print('Top5', top5_idx)

  top5_hgt = heights[top5_idx]
  top5_val = profile[top5_idx]
  
  prof_fuzz = np.full(heights.shape, np.nan) 
  
  
  # Eliminate peaks outside +/- 25% of round one Zi
  upperlim = 1.25*BLhgt[i] # km
  lowerlim = .75*BLhgt[i] # km
    
  if np.any((top5_hgt <= upperlim) & (top5_hgt >= lowerlim)):
  # Retain only points within +/- 25% of Zi
      top5_val = top5_val[(top5_hgt <= upperlim) & (top5_hgt >= lowerlim)]
      top5_hgt = top5_hgt[(top5_hgt <= upperlim) & (top5_hgt >= lowerlim)]
  else:
      return prof_fuzz, []
  
  if len(top5_hgt) >= 1:
    top5_idx = np.full(top5_hgt.shape, np.nan)
    for j in range(len(top5_hgt)):
      top5_idx[j] = find_closest(heights, top5_hgt[j])
      
  top5_idx = top5_idx[(top5_hgt <= upperlim) & (top5_hgt >= lowerlim)]

  # Get highest andlowest maxima heights
  if len(top5_idx) == 0:
      return prof_fuzz, top5_hgt
  if len(top5_idx) == 1:
    top_hgt = top5_hgt
    bot_hgt = top5_hgt
  else:
    top_hgt = max(top5_hgt)
    bot_hgt = min(top5_hgt)

  # Membership = 1 at all levels below the lowest considered peak. 
  # Membership = 0 at all levels above the highest considered peak. 
  # For each peak in between, the membership function decreases in 
  # steps = peak value/(sum of all retained peak values).  
    
  for j in range(len(heights)):
    if (heights[j] > top_hgt):
      prof_fuzz[j] = 0.0
    elif (heights[j] < bot_hgt):
      prof_fuzz[j] = 1.0
    else: # interpolate membership function
      if j in top5_idx:
        tmp_idx = np.where(top5_idx == j)[0][0] # Grab index of current height within top5_idx
        scaling = top5_val[tmp_idx] / np.sum(top5_val)
        prof_fuzz[j] = prof_fuzz[j-1] - scaling
      else:
        prof_fuzz[j] = prof_fuzz[j-1]
  
  return prof_fuzz, top5_hgt 
##########################################################


##########################################################
def BLhgtFiltSmooth(BLhgt_vals, window=7):
    """
    This function takes a timeseries of PBL heights from the fuzzy logic algorithm
    and smooths them via a triangle weighted function. It provides the smoothed 
    timeseries and a timeseries of the range of values that went into each mean. It
    also removes outlier values based on day-long standard deviation filter.
  
    Args:
      BLhgt_vals (1-D array): values of BLhgt from fuzzy logic algorithm 
      window (int): the number of values included in the rolling mean, default 7
  
    Returns:
      BLhgt_sm (1-D array): smoothed BL hgts
      BLhgt_sm_range (2-D array): [time, level (bottom=0, top=1)] range of values that go into mean
    """
    BLhgt=BLhgt_vals
    #standard deviation based filter
    BL_std = np.nanstd(BLhgt)
    for j in range(len(BLhgt)):
        if BLhgt[j]>BLhgt[j]+BL_std:
            BLhgt[j]=np.nan
        if BLhgt[j]<BLhgt[j]-BL_std:
            BLhgt[j]=np.nan
    
    #triangle mean over +- 30 min (7 points or defined by window)
    BLdf = pd.DataFrame(BLhgt)
    BLhgts = BLdf.rolling(7, center=True, win_type='triang',min_periods=1).mean()
    #BLhgt_sm_upper = BLdf.rolling(7, center=True,min_periods=1).max()
    #BLhgt_sm_lower = BLdf.rolling(7, center=True,min_periods=1).min()
    BLhgts_sm_std = BLdf.rolling(7, center=True,min_periods=1).std()
    
    BLhgt_sm = BLhgts.values
    BLhgt_sm = BLhgt_sm.ravel()
    #BLhgt_sm_upper = BLhgt_sm_upper.values
    #BLhgt_sm_upper = BLhgt_sm_upper.ravel()
    #BLhgt_sm_lower = BLhgt_sm_lower.values
    #BLhgt_sm_lower = BLhgt_sm_lower.ravel()
    BLhgt_sm_std = BLhgts_sm_std.values
    BLhgt_sm_std = BLhgt_sm_std.ravel()
        
    #BLhgt_sm_range = np.vstack((BLhgt_sm_lower, BLhgt_sm_upper)).T
    #BLhgt_sm_range = np.vstack((BLhgt_sm-BLhgt_sm_std, BLhgt_sm+BLhgt_sm_std)).T
    
    return BLhgt_sm, BLhgt_sm_std

def findBLhgt(BL_logical,gate_min):
    """
    This function takes a 2 D logical array relecting BL membership in/out (1/0) status and
    a min usuable lidar range gate and finds the BL hgt based on that membership. 
  
    Args:
      BLlogical (2-D array): values of BL logical membership, 1=in 0=out 
      gate_min (int): index (in y, interp, space) of lowest useable lidar range gate
  
    Returns:
      BLhgt (1-D array): top of layer where BL membership is at least .5 (top of layer of 1s) attached
      to the surface.
     """
    BLhgt = np.full(np.shape(BL_logical)[0], np.nan)
    for j in range(len(BLhgt)):  # Loop through times
        starting_search = np.where(BL_logical[j, :] == 1)[0]
        if len(starting_search) == np.count_nonzero(np.isnan(starting_search)):
            #there are no ones, so nothing is "in" the BL
            continue
        elif starting_search[0]>gate_min+2: #beyond 2 gates of the useable range gates
            #bottom of ones layer (mixed layer) is elevated
            print("ELEVATED LAYER DETECTED... skipping")
            continue
        spots = np.where(BL_logical[j, starting_search[0]:] != 1)[0]
        if len(spots) == np.count_nonzero(np.isnan(spots)):
            continue
        else:
            BLhgt[j] = y[spots[0]]
    return BLhgt

##########################################################
def xcorr(x, y, scale='none'):
    """
    This function comes from Tyler Bell and computes a correlation between x and y.
    
    Args:
      x (array): values for correlation calculation
      y (array): values for correlation calculation
                 if y==x, auto-correlation!
    
    Returns:
      corr (): correlation values
      lags (): lags
    """
    # Pad shorter array if signals are different lengths
    if x.size > y.size:
        pad_amount = x.size - y.size
        y = np.append(y, np.repeat(0, pad_amount))
    elif y.size > x.size:
        pad_amount = y.size - x.size
        x = np.append(x, np.repeat(0, pad_amount))
        
    corr = np.correlate(x, y, mode='full')  # scale = 'none'
    lags = np.arange(-(x.size - 1), x.size)
    if scale == 'biased':
        corr = corr / x.size
    elif scale == 'unbiased':
        corr /= (x.size - abs(lags))
    elif scale == 'coeff':
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    return corr, lags


##########################################################


##########################################################        
def lenshow(x, freq=1, tau_min=3, tau_max=12, plot=False):
    """
    This function comes from Tyler Bell     
    Reads in a timeseries. Freq is in Hz. Default taus are from avg values from Bonin Dissertation (2015)
    Returns avg w'**2 and avg error'**2
    """
    # Find the perturbation of x
    mean = np.mean(x)
    prime = x - mean
    # Get the autocovariance 
    acorr, lags = xcorr(prime, prime, scale='unbiased')
    acov = acorr# * var
    # Extract lags > 0
    lags = lags[int(np.ceil(len(lags)/2)):] * freq
    acov = acov[int(np.ceil(len(acov)/2)):]
    # Define the start and end lags
    lag_start = int(tau_min / freq)
    lag_end = int(tau_max / freq)
    # Fit the structure function
    fit_funct = lambda p, t: p[0] - p[1]*t**(2./3.) 
    err_funct = lambda p, t, y: fit_funct(p, t) - y
    p1, success = sp.leastsq(err_funct, [1, .001], args=(lags[lag_start:lag_end], acov[lag_start:lag_end]))
    if plot:
        new_lags = np.arange(tau_min, tau_max)
        plt.plot(lags, acov)
        plt.plot(new_lags, fit_funct(p1, new_lags), 'gX')
        plt.plot(0, fit_funct(p1, 0), 'gX')
        plt.xlim(0, tau_max+20)
        plt.xlabel("Lag [s]")
        plt.ylabel("$M_{11} [m^2s^{-2}$]")
    return p1[0], acov[0] - p1[0]

  
##########################################################


##########################################################
def lenshowVar(time, height, vertvel, intensity, window=.25):
    """
    This function applied the Lenshow method to vertical velocity data. Calls
    Tyler Bell's lenshow function (which calls his correlate fucntion). Note 
    vertical velocity data should not be filtered/have nans yet
    Add reference information
    
    Args:
        time (array): 1-D array of times
        height (array): 1-D array of heights
        vertvel (array): 2-D array of vertical velocity observations
        intensity (array): 2-D array of intensity (SNR+1)
        window (float): Time window for computing variance [h] (Default = 0.25)
    Returns:
        sigw (array): 2-D array of vertical velocity variance (new time res.)
        sigw_t (array): 1-D array new time resolution
    """
    #rename variables because I am a bad coder
    HR=time
    Z=height
    w=vertvel
    snr = intensity
    # provide an initial index for slicing
    prev_idx = 0
    # New time axis
    lenshow_t = np.arange(0, HR[-1], window)
    # interpolate snr to the new time axis for filtering later
    snr_l=np.full((lenshow_t.shape[0],Z.shape[0]),np.nan)
    for ii in range(len(Z)):
        snr_l[:,ii] = np.interp(lenshow_t,HR,snr[:,ii])
    # array for sigw 
    lenshow_sigw = np.full((lenshow_t.shape[0],Z.shape[0]),np.nan)
    # loop over all heights
    for j in range(0,len(Z)):
        # loop over all NEW times (stepping via window)
        for i in range(len(lenshow_t)-1):
            # find the index of the last time for the relevant averaging window
            last = np.where(HR < lenshow_t[i+1])[0]
            # if this search turns up nothing, go to the next loop step without doing anythong
            if last.size < 1:
                continue
            # the final index for slicing is the end of the averaging window
            last_idx = last[-1]
            # slice the w field using the indicies we've found to define the averaging window
            work_data = w[prev_idx:last_idx,j]
            # if all w data in this slice is nan, set the first index for the next step to the current last index
            # amd go on to the next loop step
            if np.isnan(work_data).all():
                prev_idx = last[-1]
                continue
            # Use the lenshow method which is inside Tyler Bell's function. We only want the first return (2nd is avg w)
            std = lenshow(work_data)[0]
            lenshow_sigw[i,j]=std
            # set the first index for the next step to the current last index
            prev_idx = last_idx
    
    # now we want to filter the output of the lenshow-ed vert velocity by SNR
    # this cutoff value is hardcoded and less restrictive than usual since lenshow already does some 'filtering' of its own in effect
    lenshow_sigw[np.where(snr_l<1.0075)] = np.nan
    
    return lenshow_sigw, lenshow_t
  
  
##########################################################

##################################################################################################################
#%%
#Actual code
##################################################################################################################


##################################################################################################################    
#siphon THREDDS for CHEESEHEAD
##################################################################################################################

# # Catalog for the CLAMPS2 stares
# STRcatURL = "https://data.nssl.noaa.gov/thredds/catalog/FRDD/CLAMPS/campaigns/CHEESEHEAD/CLAMPS2/dlfp/catalog.xml"
# # Open the catalog
# STRcat = TDSCatalog(STRcatURL)
# # Print some of the datasets
# #print(STRcat.datasets[:])
# # Get the dates for all the netCDF datasets
# #STRnc_dates = np.array([datetime.strptime(ds, "clampsdlfpC1.b1.%Y%m%d.%H%M%S.cdf") for ds in STRcat.datasets if '.cdf' in ds])
# # Find the index of the date we want
# #ind = np.argmin(np.abs(dt - STRnc_dates))
# # Get the dataset
# #STRds = STRcat.datasets[ind]


# # Catalog for the CLAMPS2 VADs
# VADcatURL = "https://data.nssl.noaa.gov/thredds/catalog/FRDD/CLAMPS/campaigns/CHEESEHEAD/CLAMPS2/dlvad/catalog.xml"
# # Open the catalog
# VADcat = TDSCatalog(VADcatURL)
# # Print some of the datasets
# #print(VADcat.datasets[:])
# # Get the dates for all the netCDF datasets
# #VADnc_dates = np.array([datetime.strptime(".".join(ds.split['.'][-3:-1]), "%Y%m%d.%H%M%S") for ds in VADcat.datasets if '.cdf' in ds])
# # Find the index of the date we want
# #ind = np.argmin(np.abs(dt - VADnc_dates))
# # Get the dataset
# #VADds = VADcat.datasets[ind]


# # Catalog for the CLAMPS2 AERIoe
# AOEcatURL = "https://data.nssl.noaa.gov/thredds/catalog/FRDD/CLAMPS/campaigns/CHEESEHEAD/CLAMPS2/mwronly/catalog.xml"
# # Open the catalog
# AOEcat = TDSCatalog(AOEcatURL)
# # Print some of the datasets
# #print(AOEcat.datasets[:])
# # Get the dates for all the netCDF datasets
# #AOEnc_dates = np.array([datetime.strptime(ds, "clampsaerieoe1turnC1.c1.%Y%m%d.%H%M%S.cdf") for ds in AOEcat.datasets if '.cdf' in ds])
# # Find the index of the date we want
# #ind = np.argmin(np.abs(dt - AOEnc_dates))
# # Get the dataset
# #AOEds = AOEcat.datasets[ind]


# stares=[]
# vads=[]
# rets=[]
# for i in range(len(STRcat.datasets)):
#     stares.append(str(STRcat.datasets[i]))
# for i in range(len(VADcat.datasets)):
#     vads.append(str(VADcat.datasets[i]))
# for i in range(len(AOEcat.datasets)):
#     rets.append(str(AOEcat.datasets[i]))
   
## Build the range of dates you need to loop over    
# dates=np.concatenate((np.arange(20190920,20190931,1),np.arange(20191000,20191002,1)))
## Option: replace the full range with a subset for debugging or for a case specific test
# #dates=dates[8:9] 

## look for dates where Stare, VAD, and Retrieval file are all present
# dates_flags = np.full(dates.shape, np.nan)
# for i in range(len(dates)):
#     if np.all([any(str(dates[i]) in string for string in stares), 
#                any(str(dates[i]) in string for string in vads), 
#                any(str(dates[i]) in string for string in rets)]):
#         dates_flags[i] = True
#     else: 
#         dates_flags[i] = False
        
# #repeat to toss bad files (non .cdf)
# stares=[]
# vads=[]
# rets=[]
# indexing_original_cats = []
# count = 0
# for i in range(len(STRcat.datasets)):
#     if str(STRcat.datasets[i])[-3:]=='cdf': #ignore png/pdf files in directory
#         stares.append(str(STRcat.datasets[i]))
#         count=count+1
#         if count<2:
#             indexing_original_cats.append(i)
# count = 0
# for i in range(len(VADcat.datasets)):
#     if str(VADcat.datasets[i])[-3:]=='cdf':
#         vads.append(str(VADcat.datasets[i]))
#         count=count+1
#         if count<2:
#             indexing_original_cats.append(i)
# count = 0
# for i in range(len(AOEcat.datasets)):
#     if str(AOEcat.datasets[i])[-3:]=='cdf':
#         rets.append(str(AOEcat.datasets[i]))
#         count=count+1
#         if count<2:
#             indexing_original_cats.append(i)

## only operate on cases where Stare, VAD, and Retrieval file are all present
# for case in range(0,len(dates)):#if the loop stops for whatever reason and you want to start 
#                                 #where you left off, change 0 to whatever number
#     Case=dates[case]
#     if dates_flags[case]==0:
#         print('Missing obs on '+str(Case))
#         continue
#     print('PROCESSING'+str(Case))
#     #search for correct index for each set of files
#     idx_s = [i for i, s in enumerate(stares) if str(Case) in s][0]
#     idx_v = [i for i, s in enumerate(vads) if str(Case) in s][0]
#     idx_a = [i for i, s in enumerate(rets) if str(Case) in s][0]
    
#     # grab proper info for day where Stare, VAD, and Retrieval file are all present
#     STRds = STRcat.datasets[idx_s + indexing_original_cats[0]]
#     VADds = VADcat.datasets[idx_v + indexing_original_cats[1]]
#     AOEds = AOEcat.datasets[idx_a + indexing_original_cats[2]]
#     # Modify catURL to get the dataset URLs
#     STRdsURL = STRcatURL.replace('catalog.xml', STRds.name).replace('catalog', 'dodsC')
#     VADdsURL = VADcatURL.replace('catalog.xml', VADds.name).replace('catalog', 'dodsC')
#     AOEdsURL = AOEcatURL.replace('catalog.xml', AOEds.name).replace('catalog', 'dodsC')

    #end thredds CHEESHEAD
    ##################################################################################################################
    
    
##################################################################################################################    
#siphon THREDDS for PBLTOPS
##################################################################################################################

#Catalog for the CLAMPS2 stares
STRcatURL = "https://data.nssl.noaa.gov/thredds/catalog/FRDD/CLAMPS/campaigns/PBLtops/clamps1/kaefs/stare/catalog.xml"
#Open the catalog
STRcat = TDSCatalog(STRcatURL)
# Print some of the datasets
#print(STRcat.datasets[:])
# Get the dates for all the netCDF datasets
#STRnc_dates = np.array([datetime.strptime(ds, "clampsdlfpC1.b1.%Y%m%d.%H%M%S.cdf") for ds in STRcat.datasets if '.cdf' in ds])
# Find the index of the date we want
#ind = np.argmin(np.abs(dt - STRnc_dates))
# Get the dataset
#STRds = STRcat.datasets[ind]


# Catalog for the CLAMPS2 VADs
VADcatURL = "https://data.nssl.noaa.gov/thredds/catalog/FRDD/CLAMPS/campaigns/PBLtops/clamps1/kaefs/vad/catalog.xml"
# Open the catalog
VADcat = TDSCatalog(VADcatURL)
# Print some of the datasets
#print(VADcat.datasets[:])
# Get the dates for all the netCDF datasets
#VADnc_dates = np.array([datetime.strptime(".".join(ds.split['.'][-3:-1]), "%Y%m%d.%H%M%S") for ds in VADcat.datasets if '.cdf' in ds])
# Find the index of the date we want
#ind = np.argmin(np.abs(dt - VADnc_dates))
# Get the dataset
#VADds = VADcat.datasets[ind]


# Catalog for the CLAMPS2 AERIoe
#AOEcatURL = "https://data.nssl.noaa.gov/thredds/catalog/FRDD/CLAMPS/campaigns/PBLtops/clamps2/oun/tropoe_v1/catalog.xml"
AOEcatURL = "https://data.nssl.noaa.gov/thredds/catalog/FRDD/CLAMPS/campaigns/PBLtops/clamps1/kaefs/tropoe_v1/catalog.xml"
# Open the catalog
AOEcat = TDSCatalog(AOEcatURL)
# Print some of the datasets
#print(AOEcat.datasets[:])
# Get the dates for all the netCDF datasets
#AOEnc_dates = np.array([datetime.strptime(ds, "clampsaerieoe1turnC1.c1.%Y%m%d.%H%M%S.cdf") for ds in AOEcat.datasets if '.cdf' in ds])
# Find the index of the date we want
#ind = np.argmin(np.abs(dt - AOEnc_dates))
# Get the dataset
#AOEds = AOEcat.datasets[ind]

stares=[]
vads=[]
rets=[]
for i in range(len(STRcat.datasets)):
    stares.append(str(STRcat.datasets[i]))
for i in range(len(VADcat.datasets)):
    vads.append(str(VADcat.datasets[i]))
for i in range(len(AOEcat.datasets)):
    rets.append(str(AOEcat.datasets[i]))

# Build the range of dates you need to loop over    
dates=np.concatenate((np.arange(20200821,20200832,1),np.arange(20200900,20200931,1)))
# Option: replace the full range with a subset for debugging or for a case specific test
#dates=dates[27:28] 

# look for dates where Stare, VAD, and Retrieval file are all present
dates_flags = np.full(dates.shape, np.nan)
for i in range(len(dates)):
    if np.all([any(str(dates[i]) in string for string in stares), 
              any(str(dates[i]) in string for string in vads), 
              any(str(dates[i]) in string for string in rets)]):
        dates_flags[i] = True
    else: 
        dates_flags[i] = False

# only operate on cases where Stare, VAD, and Retrieval file are all present
for case in range(0,len(dates)):#if the loop stops for whatever reason and you want to start 
                                #where you left off, change 0 to whatever number
    Case=dates[case]
    if dates_flags[case]==0:
        print('Missing obs on '+ str(Case))
        continue
    print('PROCESSING'+str(Case))
    #search for correct index for each set of files
    idx_s = [i for i, s in enumerate(stares) if str(Case) in s][0]
    idx_v = [i for i, s in enumerate(vads) if str(Case) in s][0]
    idx_a = [i for i, s in enumerate(rets) if str(Case) in s][0]

    # grab proper info for day where Stare, VAD, and Retrieval file are all present
    STRds = STRcat.datasets[idx_s]
    VADds = VADcat.datasets[idx_v]
    AOEds = AOEcat.datasets[idx_a]
    # Modify catURL to get the dataset URLs
    STRdsURL = STRcatURL.replace('catalog.xml', STRds.name).replace('catalog', 'dodsC')
    VADdsURL = VADcatURL.replace('catalog.xml', VADds.name).replace('catalog', 'dodsC')
    AOEdsURL = AOEcatURL.replace('catalog.xml', AOEds.name).replace('catalog', 'dodsC')
    #end thredds PBLtops
    #################################################################################################################
    
        
##################################################################################################################
#Local files
##################################################################################################################

# use this block for local file read in option

# import glob
# # local filepath  information
# path_to_files = '/Users/elizabeth.smith/Documents/PBLTops/data/'
# dlfp = 'clamps1/shv/stare/'
# dlvad = 'clamps1/shv/vad/'
# retrieval = 'clamps1/shv/aerioe_v0/'

# # path_to_files = '/Users/elizabeth.smith/Documents/CHEESEHEAD/data/'
# # dlfp = 'CLAMPS2/dlfp/'
# # dlvad = 'CLAMPS2/dlvad/'
# # retrieval = 'CLAMPS2/mwronly/'

# stares = glob.glob(path_to_files+dlfp+'*.cdf')
# vads = glob.glob(path_to_files+dlvad+'*.cdf')
# rets = glob.glob(path_to_files+retrieval+'*.cdf')

# # Build the range of dates you need to loop over
# dates=np.concatenate((np.arange(20200824,20200832,1),np.arange(20200900,20200931,1))) #PBL TOPS
# #dates = np.concatenate((np.arange(20190919,20190931,1),np.arange(20191001,20191012,1))) #CHEESEHEAD

# # Option: replace the full range with a subset for debugging or for a case specific test
# #dates = dates[1:6] #aug25-29

# # look for dates where Stare, VAD, and Retrieval file are all present
# dates_flags = np.full(dates.shape, np.nan)
# for i in range(len(dates)):
#     if np.all([any(str(dates[i]) in string for string in stares), 
#               any(str(dates[i]) in string for string in vads), 
#               any(str(dates[i]) in string for string in rets)]):
#         dates_flags[i] = True
#     else: 
#          dates_flags[i] = False
# # only operate on cases where Stare, VAD, and Retrieval file are all present
# for case in range(0,len(dates)): #if the loop stops for whatever reason and you want to start 
#                                  #where you left off, change 0 to whatever number
#     Case=dates[case]
#     if dates_flags[case]==0:
#         print('Missing obs on '+ str(Case))
#         continue
#     print('PROCESSING'+str(Case))
#     #search for correct index for each set of files
#     idx_s = [i for i, s in enumerate(stares) if str(Case) in s][0]
#     idx_v = [i for i, s in enumerate(vads) if str(Case) in s][0]
#     idx_a = [i for i, s in enumerate(rets) if str(Case) in s][0]

#     STRds = stares[idx_s]
#     VADds = vads[idx_v]
#     AOEds = rets[idx_a]
#     # grab proper info for day where Stare, VAD, and Retrieval file are all present
#     STRdsURL = STRds
#     VADdsURL = VADds
#     AOEdsURL = AOEds
    
    
    
    ##################################################################################################################
    #Setting parameters for the present dataset
    ##################################################################################################################
    
    # Here we set file paths, location parameters, writeout attributes, etc.
    # These are generalized to work regardless of the data access method used above (local vs THREDDS)
    
    plot_me = True # set to true to get plots
    show_me = False # set to true to show plots. Set to false to save plots
    write_me = True # set to true to write out final PBL height info to netcdf
    
    C1_cheese_lat = 45.92
    C1_cheese_lon = -89.73
    C2_cheese_lat = 45.54
    C2_cheese_lon = -90.28
    C1_pblkaefs_lat = 34.9824
    C1_pblkaefs_lon = -97.5206
    C1_pblkshv_lat = 32.4511
    C1_pblkshv_lon = -93.8414
    C2_pblkoun_lat = 35.2368
    C2_pblkoun_lon = -97.4637
    
    author_list = 'Elizabeth Smith (NOAA-NSSL); Jacob Carlin (OU-CIMMS/NSSL)'
    contact_list = 'elizabeth.smith@noaa.gov; jacob.carlin@noaa.gov'
    campaign = 'PBLTops2020'
    
    sun_lat = C2_pblkoun_lat
    sun_lon = C2_pblkoun_lon
    
    
    #files
    path_to_write = '/Users/jacob.carlin/Documents/Data/PBL Tops/test/'
    path_to_plots = '/Users/jacob.carlin/Documents/Data/PBL Tops/test/'
    # path_to_write = '/Users/elizabeth.smith/Documents/CHEESEHEAD/PBLTopFuzzy/output_files/'
    # path_to_plots = '/Users/elizabeth.smith/Documents/CHEESEHEAD/PBLTopFuzzy/'
    
    #stares
    date_files = STRdsURL
    #date_files = path_to_files+'clampsdlfpC1.b1.20200920.000017.cdf'
    #date_files = path_to_files+'clampsdlfpC2.b1.20200915.000000.cdf'
    
    #vads
    date_filev = VADdsURL
    #date_filev = path_to_files+'clampsdlvad1turnC1.c1.20200920.000859.cdf'
    #date_filev = path_to_files+'clampsdlvadC2.c1.20200915.000000.cdf'
    
    #aeri retrievals 
    date_filea = AOEdsURL
    #data_filea = path_to_files+'clampsaerioe1turnC1.c1.20200920.001005.cdf'
    #data_filea = path_to_files+'clampsaerioe1turnC2.c1.20200915.152005.cdf'
    
    # label text for plots/save names
    date_label = date_files[-19:-11]
    platform = date_files[-25:-23]
    Case=date_label
    
    #basetime variable 
    base_time = datetime(int(str(Case)[0:4]), 
                         int(str(Case)[4:6]), 
                         int(str(Case)[6:8])).replace(tzinfo=timezone.utc).timestamp()
    timestamp = datetime.utcfromtimestamp(base_time)
    
    ##################################################################################################################
    #read-in
    ##################################################################################################################
    
    #read in stare file
    dl = nc.Dataset(date_files)
    Z = dl.variables['height'][:278].data #278 gets us up to ~5km
    Z[np.where(Z>100)] = np.nan
    lower_limit = np.ceil(Z[0]*100.)/100. # rounding the lowest Z value to the upper 100 m value
    lowest_gate = Z[2] #min usable range gate 
    HR = (dl.variables['time_offset'][:].data)/3600. #avg dt is 1.33s thus 5 minutes=225.5 records; 10 minutes=451.1
    HR[np.where(HR>100.) ]= np.nan 
    t_s = HR * 60.0 # minutes
    snr = dl.variables['intensity'][:,:278].data
    bscat = np.log10(dl.variables['backscatter'][:,:278].data)
    bscat[np.where(bscat>100)] = np.nan
    w = dl.variables['velocity'][:,:278].data
    w[np.where(w>100)] = np.nan
    dl.close()
    
    # find vert velocity variance via lenshow methods 
    # this must be done prior to filtering out any data as it doesnt handle nans
    sigw, sigw_t = lenshowVar(HR, Z, w, snr, window=.166)
    sigw_t = sigw_t * 60.0 # minutes
    
    ## this chunk helps us understand what signal-to-noise ratio limit we should use below
    ## can ignore if the cutoff value is already set
    #plt.plot(wr[0::10,0::5],snrr[0::10,0::5],'bo',alpha=.3)
    #plt.xlim(-20,20)
    #plt.xlabel('w')
    #plt.ylim(1,1.02)
    #plt.ylabel('SNR')
    #plt.axhline(1.01,color='lightblue') #typical rule of thumb value
    #plt.title(str(platform)+' '+str(date_label))
    #plt.show()
    ## the 1.01 value looks pretty good so lets use it
    snr_cutoff=1.01
    
    # filtering any data that doesn't have sufficient signal-to-noise ratio
    bscat[np.where(snr<snr_cutoff)] = np.nan
    w[np.where(snr<snr_cutoff)] = np.nan
    # remove noisy data in the lowest two range gates
    w[:,:2] = np.nan 
    bscat[:,:2] = np.nan
    
    # computing the SNR variance from raw snr in 10 min windows
    sigsnr, sigsnr_t = calcSigma(HR,Z,snr,window=0.166) 
    sigsnr[np.where(sigsnr>10)] = np.nan
    print('Stare read in complete')
    #%%
    
    #dl vad
    dlv = nc.Dataset(date_filev)
    Zv = dlv.variables['height'][:278].data #278 gets us up to ~5km
    Zv[np.where(Z>100)] = np.nan
    HRv = (dlv.variables['time_offset'][:].data)/3600. 
    t_v = HRv * 60.0 # minutes
    snrv = dlv.variables['intensity'][:,:278].data
    ws = dlv.variables['wspd'][:,:278].data
    wd = dlv.variables['wdir'][:,:278].data
    u, v = uv_from_spd_dir(ws, wd)
    dlv.close()
    
    #filtering any data that doesn't have sufficient signal-to-noise ratio
    ws[np.where(snrv<snr_cutoff)] = np.nan
    wd[np.where(snrv<snr_cutoff)] = np.nan
    u[np.where(snrv<snr_cutoff)] = np.nan
    v[np.where(snrv<snr_cutoff)] = np.nan
    
    
    # Cleaning up files with weird timestamps and bad retrievals (hard target scans)
    stupid = any(np.ediff1d(t_v) < 0)
    while stupid:
        #while there are any stupid negative time differences
        g = np.where(np.ediff1d(t_v) < 0)[0] + 1 # Find times that differ from regular 10-min intervals
        snrv = np.delete(snrv, g, 0)
        ws = np.delete(ws, g, 0)
        wd = np.delete(wd, g, 0)
        u = np.delete(u, g, 0)
        v = np.delete(v, g, 0)
        HRv = np.delete(HRv, g)
        t_v = np.delete(t_v, g)
        stupid = any(np.ediff1d(t_v) < 0)
    
    
    print('VAD read in complete')
    #%%
    #aeri data
    aeri = nc.Dataset(date_filea)
    HRa = aeri.variables['time_offset'][:].data/3600.
    t_a = HRa * 60.0 # minutes
    Za = aeri.variables['height'][:]
    T_orig = aeri.variables['temperature'][:,:]
    pt_orig = aeri.variables['theta'][:,:]
    wv_orig = aeri.variables['waterVapor'][:,:]
    aeri.close()
    
    #doing some moving average temporal smoothing of AERI retreival before we feed it into anything
    T=np.full(T_orig.shape,np.nan)
    pt=np.full(pt_orig.shape,np.nan)
    wv=np.full(wv_orig.shape,np.nan)
    T[0,:]=T_orig[0,:]
    T[-1,:]=T_orig[-1,:]
    pt[0,:]=pt_orig[0,:]
    pt[-1,:]=pt_orig[-1,:]
    wv[0,:]=wv_orig[0,:]
    wv[-1,:]=wv_orig[-1,:]
    for i in range(1,len(Za)-1):
        T[1:-1,i]=moving_average(T_orig[:,i])
        pt[1:-1,i]=moving_average(pt_orig[:,i])
        wv[1:-1,i]=moving_average(wv_orig[:,i])
    print('AERI read in complete')
    ##################################################################################################################
    
    # find start and end time relative to all observation files we have
    start_time = np.ceil(np.max([t_a[0],t_v[0],t_s[0],sigw_t[0]])/10.)*10. # the latest start time rounded to the inner 10 min
    end_time = np.floor(np.min([t_a[-1],t_v[-1],t_s[-1],sigw_t[-1]])/10.)*10. # the earliest end time rounded to the inner 10 min
    
    # set up new common x (time) and y (height) dimensions for all data to be regridded to later using start and end times
    x = np.arange(start_time, end_time+10., 10.)
    y = np.arange(lower_limit, 4.001, 0.01)
    gate_min = np.where(y>lowest_gate)[0][0] #gives us the index needed for "spots" in BL_logical
    
    #%%
    ##################################################################################################################
    #fuzzy logic
    ##################################################################################################################
    
    #dict for x1,x2, weight vals for various inputs -- for membership functions
    # x1,x2 vals defined by bonin et al 2018
    membership_vals = {'wVar': {'lo': 0.02,
                                'hi': 0.08,
                                'wt': 1.0},
                       'HFwVar': {'lo': 0.0025,
                                  'hi': 0.01,
                                  'wt': 1.0},
                       'snr': {'wt': 2.0},
                       'snrVar': {'wt': 2.0},
                       'uv': {'wt': 2.0},
                       'wv': {'wt': 2.0},
                       'pt': {'wt': 2.0}}
    
    
        
    # Compute high frequency variance 
    HFw, HFwVar, HFt = HF_varcalc(w, HR, Z)
    HFwVar[np.where(HFwVar<.0001)]=np.nan #this was pulling first guess down with all zeros otherwise
    U = mean_u(ws, HRv, Zv, HFt, lo=.1, hi=1.)
    # tip: if you run into an error where U (or ws rather) has a bad shape, check to see if you have a wonky extra 
    # lidar file on the day you are trying to process. That can result in funny things. Remove the wonky file if so.
    for i in range(len(Z)):
        HFwVar[:,i] = HFwVar[:,i] / U #normalize by mean wind speed to prevent removal of large eddies
                       
    ##########################################################
    # 1st generation steps: 
    #########################################################
      
    # Perform fuzzification to membership values (0, 1)
    wVar_fuzz = fuzzify(sigw, membership_vals['wVar']['lo'], membership_vals['wVar']['hi'])
    wVarHF_fuzz = fuzzify(HFwVar, membership_vals['HFwVar']['lo'], membership_vals['HFwVar']['hi'])
    
    # Interpolate to common grid for AERI and DL data
    wVar_fuzz_intp = fuzzGrid(wVar_fuzz, sigw_t, Z, start_time = start_time, end_time = end_time, low_z=lower_limit)
    wVarHF_fuzz_intp = fuzzGrid(wVarHF_fuzz, sigw_t, Z, start_time = start_time, end_time = end_time, low_z=lower_limit)
    
    # Add weights and calculate aggregate value
    wVar_fuzz_intp_wt = np.full(wVar_fuzz_intp.shape, np.nan)
    wVar_fuzz_intp_wt[~np.isnan(wVar_fuzz_intp)] = membership_vals['wVar']['wt']
    
    wVarHF_fuzz_intp_wt = np.full(wVarHF_fuzz_intp.shape, np.nan)
    wVarHF_fuzz_intp_wt[~np.isnan(wVarHF_fuzz_intp)] = membership_vals['HFwVar']['wt']
    
    # Compute time-weighted membership function for incorporating AERI data
    sun = Sun(sun_lat, sun_lon) #sun for given lat lon, "actual code" block
    sunup = sun.get_sunrise_time(timestamp).replace(tzinfo=timezone.utc).timestamp()
    sundown = sun.get_sunset_time(timestamp).replace(tzinfo=timezone.utc).timestamp()
    prev_sundown = sun.get_sunset_time(timestamp - timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp()
    next_sundown = sun.get_sunset_time(timestamp + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp()
    
    real_sunup = sunup
    real_sundown = sundown
    real_prev_sundown = prev_sundown
    real_next_sundown = next_sundown

    sunup_offset = -20 * 60.
    sundown_offset = -20 * 60.
    
    sunup += sunup_offset
    sundown += sundown_offset
    prev_sundown += sundown_offset
    next_sundown += sundown_offset
    
    membership_times = np.arange(base_time, base_time+87000, 600)
    membership_value_vec = np.full(len(membership_times), np.nan)
    transition_half_window = 3600 # 30-min [in s] for a 1-h total transition window centered on sunup/sundown
    
    # Nighttime
    if sundown < sunup:
        g = np.where(np.logical_and(membership_times > (sundown + transition_half_window),
                                    membership_times < (sunup - transition_half_window)))
    else: # sunrise for date comes first
        g = np.where(np.logical_and(membership_times > (prev_sundown + transition_half_window),
                                   membership_times < (sunup - transition_half_window)))
    membership_value_vec[g] = 2.0 
    
    
    # Daytime
    if sundown < sunup:
        g = np.where(np.logical_and(membership_times < (next_sundown - transition_half_window),
                                   membership_times > (sunup + transition_half_window)))
    else:
        g = np.where(np.logical_and(membership_times < (sundown - transition_half_window),
                                   membership_times > (sunup + transition_half_window)))    
    membership_value_vec[g] = 0.0 
    
    # Transition zones
    # Dawn
    g = np.where(np.abs(membership_times - sunup) <= transition_half_window)[0]
    membership_value_vec[g] = -erf((membership_times[g] - sunup) / 1200.) + 1.0 
    # 1200.0 (20-min periods) gets to vals = (0, 2) at (-1 hr, +1 hr)
    
    # Dusk
    g = np.where(np.abs(membership_times - sundown) <= transition_half_window)[0]
    membership_value_vec[g] = erf((membership_times[g] - sundown) / 1200.) + 1.0
    
    g = np.where(np.abs(membership_times - next_sundown) <= transition_half_window)[0]
    membership_value_vec[g] = erf((membership_times[g] - next_sundown) / 1200.) + 1.0
    
    g = np.where(np.abs(membership_times - prev_sundown) <= transition_half_window)[0]
    membership_value_vec[g] = erf((membership_times[g] - prev_sundown) / 1200.) + 1.0
    
    
    # interpolating
    f = interp1d((membership_times - base_time)/60., membership_value_vec) # Convert time to minutes vec
    # compute the T gradient in an array
    T_grad=np.full(T.shape,np.nan)
    T_grad[:,:-1]=np.diff(T,axis=1) #vertical temperature gradients
    # fuzzify the T gradient
    inv_bin = fuzzGrid(T_grad, t_a, Za, start_time = start_time, end_time = end_time, low_z=lower_limit)
    # prepare and array for the T gradient weighting
    inv_fuzz_wt = np.full(inv_bin.shape, np.nan)
    inv_fuzz = np.full(inv_bin.shape, np.nan)
    # this list is a holder for debug plotting purposes
    inversion=[]
    # looping over times
    for i in range(len(x)):
        asign = np.sign(inv_bin[i,:]) #find sign of gradient vals
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int) #return 1 for sign change, 0 for not
        signchange[0]=0 #setting since numpy.roll does a circular shift, so if the last element has different 
        # sign than the first, the first element in the signchange array will be 1
        #note -1, +1, and 0 are all considered unique signs here
        inversion_idx = np.where(signchange==1)[0]
        if inversion_idx.size < 1:
            # if there are no sign changes, we skip this loop and move on 
            inversion.append(np.nan)
            continue
        else:
            inversion_idx=inversion_idx[0]
        if y[inversion_idx] > 1.25: #above 1.25 km
          	# we don't want want to retain any info about elevated inversions, so if the inversion
            # identified is above 1.25, we break out and move foward with the time loop
            inversion.append(np.nan)
            continue 
        else:
            # use inversion index to define membership in the boundary layer
            inv_fuzz[i,:inversion_idx+1] = 1.0 # in the boundary layer
            inv_fuzz[i,inversion_idx+1:] = 0 # outside the boundary layer
        inv_fuzz_wt[i, :] = f(x[i]) # apply weighting as defined above
        inversion.append(y[inversion_idx])
      
    # Compute overall aggregate values
    sum_fuzz = np.zeros_like(wVar_fuzz_intp_wt)
    for i in range(np.shape(wVar_fuzz_intp_wt)[0]):
      for j in range(np.shape(wVar_fuzz_intp_wt)[1]):
          sum_fuzz[i, j] = ((np.nansum((wVar_fuzz_intp[i, j] * wVar_fuzz_intp_wt[i, j], 
                                        wVarHF_fuzz_intp[i, j] * wVarHF_fuzz_intp_wt[i, j],
                                        inv_fuzz[i, j] * inv_fuzz_wt[i, j]))) / 
                            (np.nansum((wVar_fuzz_intp_wt[i, j],
                                        wVarHF_fuzz_intp_wt[i, j],
                                        inv_fuzz_wt[i, j]))))       
    
    # Compute overall aggregate values for overnight mechanical only terms (zero out fuzzwt)
    sum_fuzz_ovn_mech = np.zeros_like(wVar_fuzz_intp_wt)
    for i in range(np.shape(wVar_fuzz_intp_wt)[0]):
      for j in range(np.shape(wVar_fuzz_intp_wt)[1]):
          sum_fuzz_ovn_mech[i, j] = ((np.nansum((wVar_fuzz_intp[i, j] * wVar_fuzz_intp_wt[i, j], 
                                        wVarHF_fuzz_intp[i, j] * wVarHF_fuzz_intp_wt[i, j],
                                        inv_fuzz[i, j] * 0))) / 
                            (np.nansum((wVar_fuzz_intp_wt[i, j],
                                        wVarHF_fuzz_intp_wt[i, j],
                                        0))))    
    
    # Compute overall aggregate values for overnight buoyant only terms (zero out var wts)
    sum_fuzz_ovn_buoy = np.zeros_like(wVar_fuzz_intp_wt)
    for i in range(np.shape(wVar_fuzz_intp_wt)[0]):
      for j in range(np.shape(wVar_fuzz_intp_wt)[1]):
          sum_fuzz_ovn_buoy[i, j] = ((np.nansum((wVar_fuzz_intp[i, j] * 0, 
                                        wVarHF_fuzz_intp[i, j] * 0,
                                        inv_fuzz[i, j] * inv_fuzz_wt[i, j]))) / 
                            (np.nansum((0,
                                        0,
                                        inv_fuzz_wt[i, j]))))                   
    
    # Codifying BL status using 0.5 threshold of mem aggregrate (1=in, 0=out)
    BL_thresh = 0.5
    BL_logical = np.full(wVar_fuzz_intp.shape,np.nan)
    BL_logical_ob = np.full(wVar_fuzz_intp.shape,np.nan)
    BL_logical_om = np.full(wVar_fuzz_intp.shape,np.nan)
    BL_logical[sum_fuzz >= BL_thresh] = 1.0
    BL_logical[sum_fuzz < BL_thresh] = 0.0
    BL_logical_om[sum_fuzz_ovn_mech >= BL_thresh] = 1.0
    BL_logical_om[sum_fuzz_ovn_mech < BL_thresh] = 0.0
    BL_logical_ob[sum_fuzz_ovn_buoy >= BL_thresh] = 1.0
    BL_logical_ob[sum_fuzz_ovn_buoy < BL_thresh] = 0.0
    BLhgt = findBLhgt(BL_logical,gate_min)
    BLhgt_ob = findBLhgt(BL_logical_ob,gate_min)
    BLhgt_om = findBLhgt(BL_logical_om,gate_min)
    
    #smoothing BL height and defining variability based on standard deviation 
    BLhgt_sm, BLhgt_sm_range = BLhgtFiltSmooth(BLhgt, window=7)
    BLhgt_om_sm, BLhgt_om_sm_range = BLhgtFiltSmooth(BLhgt_om, window=7)
    BLhgt_ob_sm, BLhgt_ob_sm_range = BLhgtFiltSmooth(BLhgt_ob, window=7)
    
    
    
    #%%
    if plot_me == True:
        #########################################################
        # Plot first generation aggregrate
        #########################################################
        plt.figure(figsize=(18,6))
        plt.pcolormesh(x,y,sum_fuzz.transpose(),vmin=0,vmax=1,cmap='inferno')
        plt.plot(x,BLhgt,'bo',zorder=1000,alpha=.5)
        plt.plot(x,BLhgt_sm,'b')
        plt.colorbar(label='Aggregrate First Gen')
        plt.ylim(0.06,2.75)
        #plt.ylim(0,2)
        plt.xticks(np.arange(0,1410.1,180),x_tick_labels)
        plt.xlim(0.0,1410)
        plt.title(str(platform)+' '+str(date_label))
        plt.xlabel('Hour [UTC]')
        plt.ylabel(' Height AGL [km]')
        if show_me == True:
            plt.show()
        else:
            plt.savefig(path_to_plots+date_label+'_FirstGenAg_'+platform+'.png')
            plt.close()
    #%%
    
    print('First Gen. Fuzzy Complete')
    ##########################################################
    # 2nd generation steps: 
    #########################################################
    
    # Regridding
    snr_bin = fuzzGrid(snr, HR*60., Z, start_time = start_time, end_time = end_time, low_z=lower_limit)
    sigsnr_bin = fuzzGrid(sigsnr, sigsnr_t*60., Z, start_time = start_time, end_time = end_time, low_z=lower_limit)
    u_bin = fuzzGrid(u, HRv*60., Zv, start_time = start_time, end_time = end_time, low_z=lower_limit)
    v_bin = fuzzGrid(v, HRv*60., Zv, start_time = start_time, end_time = end_time, low_z=lower_limit)
    pt_bin = fuzzGrid(pt, HRa*60., Za, start_time = start_time, end_time = end_time, low_z=lower_limit)
    wv_bin = fuzzGrid(wv, HRa*60., Za, start_time = start_time, end_time = end_time, low_z=lower_limit)
    
    # Membership arrays
    snr_fuzz = np.full(snr_bin.shape, np.nan)
    sigsnr_fuzz = np.full(sigsnr_bin.shape,np.nan)
    uv_fuzz = np.full(u_bin.shape, np.nan)
    pt_fuzz = np.full(pt_bin.shape, np.nan)
    wv_fuzz = np.full(wv_bin.shape, np.nan)
    
    # Haar wavelet arrays for visualization
    x_wave_shape = snr_bin.shape[0]
    y_wave = np.linspace(min(y),max(y),len(y))
    y_wave_shape = len(y_wave)
    snr_fuzz_wavelet = np.full((x_wave_shape, y_wave_shape), np.nan)
    sigsnr_fuzz_wavelet = np.full((x_wave_shape, y_wave_shape), np.nan)
    u_fuzz_wavelet = np.full((x_wave_shape, y_wave_shape), np.nan)
    v_fuzz_wavelet = np.full((x_wave_shape, y_wave_shape), np.nan)
    uv_fuzz_wavelet = np.full((x_wave_shape, y_wave_shape), np.nan)
    pt_fuzz_wavelet = np.full((x_wave_shape, y_wave_shape), np.nan)
    wv_fuzz_wavelet = np.full((x_wave_shape, y_wave_shape), np.nan)    
    
    # Loop through times
    snr_wavelet_peak_hgts = []
    sigsnr_wavelet_peak_hgts = []
    uv_wavelet_peak_hgts = []
    pt_wavelet_peak_hgts = []
    wv_wavelet_peak_hgts = []
    for i in range(len(x)):
      snr_fuzz_wavelet[i,:] = run_haar_wavelet(snr_bin[i, :], y)
      snr_fuzz[i, :], hgts = calc_haar_membership(snr_fuzz_wavelet[i, :], y[:], BLhgt_sm)
      snr_wavelet_peak_hgts.append(hgts)
      
      sigsnr_fuzz_wavelet[i, :] = run_haar_wavelet(sigsnr_bin[i, :], y)
      sigsnr_fuzz[i, :], hgts = calc_haar_membership(sigsnr_fuzz_wavelet[i, :], y[:], BLhgt_sm)
      sigsnr_wavelet_peak_hgts.append(hgts)
      
      # u & v will be passed through wavelet then the vector magnitude of the wavelet transform
      # is used in the membership function computation process
      u_fuzz_wavelet[i, :] = run_haar_wavelet(u_bin[i, :], y)
      v_fuzz_wavelet[i, :] = run_haar_wavelet(v_bin[i, :], y)
      uv_fuzz_wavelet[i, :] = np.sqrt(u_fuzz_wavelet[i, :]**2 + v_fuzz_wavelet[i, :]**2)
      uv_fuzz[i, :], hgts = calc_haar_membership(uv_fuzz_wavelet[i, :], y[:], BLhgt_sm)
      uv_wavelet_peak_hgts.append(hgts)     
      
      pt_fuzz_wavelet[i, :] = run_haar_wavelet(pt_bin[i, :], y)
      pt_fuzz[i, :], hgts = calc_haar_membership(pt_fuzz_wavelet[i, :], y[:], BLhgt_sm)
      pt_wavelet_peak_hgts.append(hgts)
      
      wv_fuzz_wavelet[i, :] = run_haar_wavelet(wv_bin[i, :], y)
      wv_fuzz[i, :], hgts = calc_haar_membership(wv_fuzz_wavelet[i, :], y[:], BLhgt_sm)
      wv_wavelet_peak_hgts.append(hgts)
          
    # Create 2D arrays of weights for each variable
    snr_fuzz_wt = np.full(snr_fuzz.shape, np.nan)
    snr_fuzz_wt[~np.isnan(snr_fuzz)] = membership_vals['snr']['wt']
    
    sigsnr_fuzz_wt = np.full(sigsnr_fuzz.shape, np.nan)
    sigsnr_fuzz_wt[~np.isnan(sigsnr_fuzz)] = membership_vals['snrVar']['wt']
    
    uv_fuzz_wt = np.full(uv_fuzz.shape, np.nan)
    uv_fuzz_wt[~np.isnan(uv_fuzz)] = membership_vals['uv']['wt'] 
    
    pt_fuzz_wt = np.full(pt_fuzz.shape, np.nan)
    pt_fuzz_wt[~np.isnan(pt_fuzz)] = membership_vals['pt']['wt']
    
    wv_fuzz_wt = np.full(wv_fuzz.shape, np.nan)
    wv_fuzz_wt[~np.isnan(wv_fuzz)] = membership_vals['wv']['wt']
    
    # creating 2nd generation aggregate! 
    sum_fuzz_2 = np.zeros_like(snr_fuzz_wt)
    for i in range(np.shape(snr_fuzz_wt)[0]):
      for j in range(np.shape(snr_fuzz_wt)[1]):
          sum_fuzz_2[i, j] = ((np.nansum((wVar_fuzz_intp[i, j] * wVar_fuzz_intp_wt[i, j], 
                                         wVarHF_fuzz_intp[i, j] * wVarHF_fuzz_intp_wt[i, j],
                                         inv_fuzz[i, j] * inv_fuzz_wt[i, j],
                                         snr_fuzz[i, j] * snr_fuzz_wt[i, j],
                                         sigsnr_fuzz[i, j] * sigsnr_fuzz_wt[i, j],
                                         uv_fuzz[i, j] * uv_fuzz_wt[i, j],
                                         pt_fuzz[i, j] * pt_fuzz_wt[i, j],
                                         wv_fuzz[i, j] * wv_fuzz_wt[i, j]))) / 
                              (np.nansum((wVar_fuzz_intp_wt[i, j],
                                                 wVarHF_fuzz_intp_wt[i, j],
                                                 inv_fuzz_wt[i, j],
                                                 snr_fuzz_wt[i, j],
                                                 sigsnr_fuzz_wt[i, j],
                                                 uv_fuzz_wt[i, j],
                                                 pt_fuzz_wt[i, j],
                                                 wv_fuzz_wt[i, j]))))
          
    # creating 2nd generation aggregate! 
    sum_fuzz_ovn_mech2 = np.zeros_like(snr_fuzz_wt)
    for i in range(np.shape(snr_fuzz_wt)[0]):
      for j in range(np.shape(snr_fuzz_wt)[1]):
          sum_fuzz_ovn_mech2[i, j] = ((np.nansum((wVar_fuzz_intp[i, j] * wVar_fuzz_intp_wt[i, j], 
                                         wVarHF_fuzz_intp[i, j] * wVarHF_fuzz_intp_wt[i, j],
                                         inv_fuzz[i, j] * 0,
                                         snr_fuzz[i, j] * snr_fuzz_wt[i, j],
                                         sigsnr_fuzz[i, j] * sigsnr_fuzz_wt[i, j],
                                         uv_fuzz[i, j] * uv_fuzz_wt[i, j],
                                         pt_fuzz[i, j] * 0,
                                         wv_fuzz[i, j] * 0))) / 
                              (np.nansum((wVar_fuzz_intp_wt[i, j],
                                                 wVarHF_fuzz_intp_wt[i, j],
                                                 0,
                                                 snr_fuzz_wt[i, j],
                                                 sigsnr_fuzz_wt[i, j],
                                               uv_fuzz_wt[i, j],
                                                 0,
                                                 0))))
          
    # creating 2nd generation aggregate! 
    sum_fuzz_ovn_buoy2 = np.zeros_like(snr_fuzz_wt)
    for i in range(np.shape(snr_fuzz_wt)[0]):
      for j in range(np.shape(snr_fuzz_wt)[1]):
          sum_fuzz_ovn_buoy2[i, j] = ((np.nansum((inv_fuzz[i, j] * inv_fuzz_wt[i, j],
                                         pt_fuzz[i, j] * pt_fuzz_wt[i, j],
                                         wv_fuzz[i, j] * wv_fuzz_wt[i, j]))) / 
                              (np.nansum((inv_fuzz_wt[i, j],
                                                 pt_fuzz_wt[i, j],
                                                 wv_fuzz_wt[i, j]))))
    
    
    # Codifying BL status using 0.5 threshold of mem aggregrate (1=in, 0=out)
    BL_thresh = 0.5
    BL_logical_2 = np.full(snr_fuzz_wt.shape,np.nan)
    BL_logical_ob2 = np.full(wVar_fuzz_intp.shape,np.nan)
    BL_logical_om2 = np.full(wVar_fuzz_intp.shape,np.nan)
    BL_logical_2[sum_fuzz_2 >= BL_thresh] = 1.0
    BL_logical_2[sum_fuzz_2 < BL_thresh] = 0.0
    BL_logical_om2[sum_fuzz_ovn_mech2 >= BL_thresh] = 1.0
    BL_logical_om2[sum_fuzz_ovn_mech2 < BL_thresh] = 0.0
    BL_logical_ob2[sum_fuzz_ovn_buoy2 >= BL_thresh] = 1.0
    BL_logical_ob2[sum_fuzz_ovn_buoy2 < BL_thresh] = 0.0
    BLhgt_2 = findBLhgt(BL_logical_2,gate_min)
    BLhgt_om2 = findBLhgt(BL_logical_om2,gate_min)
    BLhgt_ob2 = findBLhgt(BL_logical_ob2,gate_min)
    
    BLhgt_2_sm, BLhgt_2_sm_range = BLhgtFiltSmooth(BLhgt_2, window=7)
    BLhgt_om2_sm, BLhgt_om2_sm_range = BLhgtFiltSmooth(BLhgt_om2, window=7)
    BLhgt_ob2_sm, BLhgt_ob2_sm_range = BLhgtFiltSmooth(BLhgt_ob2, window=7)
    
    sunup_min = (sunup - base_time)/60.
    sundown_min = (sundown - base_time)/60.
    
    if x[-1] > sunup_min: # if there are enough data to care about cutting the overnight terms off
        cut_start = np.where(x > sunup_min)[0][0] 
        if sundown_min < 1410.: # 23:30 Z
            cut_end = np.where(x > sundown_min)[0][0]+1
        else: #if sundown is inbetween 23:30 and 00Z
            cut_end = len(x)-1
        if cut_end < cut_start: #sunset is after 0
            BLhgt_om2_sm[cut_start:]=np.nan 
            BLhgt_om2_sm_range[cut_start:]=np.nan
            BLhgt_ob2_sm[cut_start:]=np.nan
            BLhgt_ob2_sm_range[cut_start:]=np.nan
            BLhgt_om2_sm[:cut_end]=np.nan 
            BLhgt_om2_sm_range[:cut_end]=np.nan
            BLhgt_ob2_sm[:cut_end]=np.nan
            BLhgt_ob2_sm_range[:cut_end]=np.nan
        else:
            BLhgt_om2_sm[cut_start:cut_end]=np.nan 
            BLhgt_om2_sm_range[cut_start:cut_end]=np.nan
            BLhgt_ob2_sm[cut_start:cut_end]=np.nan
            BLhgt_ob2_sm_range[cut_start:cut_end]=np.nan
    
    # Output BL_hgt time in epoch seconds
    base_time = datetime(int(str(Case)[0:4]), 
                         int(str(Case)[4:6]), 
                         int(str(Case)[6:8])).replace(tzinfo=timezone.utc).timestamp()
    base_time_vec = base_time + (x * 60.0)
    
    
      
    
    #%%
    if plot_me == True:
        #########################################################
        # Plot second generation aggregrate
        #########################################################
        plt.figure(figsize=(18,6))
        plt.pcolormesh(x,y,sum_fuzz_2.transpose(),vmin=0,vmax=1,cmap='inferno')
        plt.plot(x,BLhgt_2,'go',zorder=1000,alpha=.5)
        plt.plot(x,BLhgt_2_sm,'g',label='2nd gen')
        plt.plot(x,BLhgt,'bo',zorder=1000,alpha=.5)
        plt.plot(x,BLhgt_sm,'b',label='1st gen')
        plt.colorbar(label='Aggregrate Sec. Gen')
        plt.ylim(0.06,2.75)
        plt.xticks(np.arange(0,1410.1,180),x_tick_labels)
        plt.xlim(0.0,1410)
        plt.xlabel('Hour [UTC]')
        plt.ylabel(' Height AGL [km]')
        plt.title(str(platform)+' '+str(date_label))
        plt.legend()
        if show_me == True:
            plt.show()
        else:
            plt.savefig(path_to_plots+date_label+'_SecondGenAg_'+platform+'.png')
            plt.close()
    
    print('Second Gen. Fuzzy Complete')    
    
    
    
    ##################################################################################################################
    # Basic variable visualizations 
    ##################################################################################################################
    
    #can add BL height plotting to them by adding:
    #plt.plot(x,BLhgt,'ko',zorder=1000, label='2nd gen')
    #plt.plot(x,BLhgt_2,'go',zorder=1000, label='1st gen')
    #plt.plot(x,BLhgt_sm,'b',label='1st gen')
    #plt.plot(x,BLhgt_2_sm,'g',label='2nd gen')
    
    if plot_me == True:
        #snr
        plt.figure(figsize=(18,6))
        plt.pcolormesh(HR,Z,snr.transpose(),vmin=.99,vmax=1.1,cmap='inferno')
        plt.colorbar(label='Intensity [SNR+1]')
        plt.fill_between(x/60.,BLhgt_2_sm-BLhgt_2_sm_range,BLhgt_2_sm+BLhgt_2_sm_range,color='grey',alpha=.3)
        plt.plot(x/60.,BLhgt_2_sm-BLhgt_2_sm_range,color='grey',lw=2)
        plt.plot(x/60.,BLhgt_2_sm+BLhgt_2_sm_range,color='grey',lw=2)
        plt.plot(x/60.,BLhgt_2_sm,color='w',lw=3)
        plt.plot(x/60.,BLhgt_2_sm,color='k',lw=2,label='PBLH')
        plt.ylim(0.06,2.75)
        plt.xticks(np.arange(0,24.1,3))
        plt.xlim(0.0,24)
        plt.xlabel('Hour [UTC]')
        plt.ylabel(' Height AGL [km]')
        plt.title(str(platform)+' '+str(date_label))
        if show_me == True:
            plt.show()
        else:
            plt.savefig(path_to_plots+date_label+'_SNR_'+platform+'.png')
            plt.close()
        
        
        #sigw
        plt.figure(figsize=(18,6))
        plt.pcolormesh(sigw_t,Z,sigw.transpose(),vmin=0,vmax=.6,cmap='summer')
        plt.colorbar(label='Vertical Velocity Variance [m/s]')
        plt.ylim(0.06,2.75)
        plt.xticks(np.arange(0,1410.1,180),x_tick_labels)
        plt.xlim(0.0,1410)
        plt.xlabel('Hour [UTC]')
        plt.ylabel(' Height AGL [km]')
        plt.title(str(platform)+' '+str(date_label))
        plt.fill_between(x,BLhgt_2_sm-BLhgt_2_sm_range,BLhgt_2_sm+BLhgt_2_sm_range,color='grey',alpha=.3)
        plt.plot(x,BLhgt_2_sm-BLhgt_2_sm_range,color='grey',lw=2)
        plt.plot(x,BLhgt_2_sm+BLhgt_2_sm_range,color='grey',lw=2)
        plt.plot(x,BLhgt_2_sm,color='w',lw=3)
        plt.plot(x,BLhgt_2_sm,color='k',lw=2,label='PBLH')
        plt.legend()
        if show_me == True:
            plt.show()
        else:
            plt.savefig(path_to_plots+date_label+'_sigw_'+platform+'.png')
            plt.close()
        
        
        
        # #T grad
        plt.figure(figsize=(18,6))
        plt.pcolormesh(HRa,Za,T_grad.transpose(),vmin=-1,vmax=1,cmap=cm.cm.delta)
        plt.colorbar(label='Temperature gradient [C]')
        plt.ylim(0.06,2.75)
        plt.xticks(np.arange(0,24.1,3))
        plt.xlim(0.0,24)
        plt.xlabel('Hour [UTC]')
        plt.ylabel(' Height AGL [km]')
        # plt.plot(x/60.,BLhgt_sm,'w',lw=3)
        # plt.plot(x/60.,BLhgt_sm,'k',lw=2,label='1st gen')
        # plt.plot(x/60.,BLhgt_2_sm,'w',lw=3)
        # plt.plot(x/60.,BLhgt_2_sm,'b',lw=2,label='2nd gen')
        plt.fill_between(x/60.,BLhgt_2_sm-BLhgt_2_sm_range,BLhgt_2_sm+BLhgt_2_sm_range,color='grey',alpha=.3)
        plt.plot(x/60.,BLhgt_2_sm-BLhgt_2_sm_range,color='grey',lw=2)
        plt.plot(x/60.,BLhgt_2_sm+BLhgt_2_sm_range,color='grey',lw=2)
        plt.plot(x/60.,BLhgt_2_sm,color='w',lw=3)
        plt.plot(x/60.,BLhgt_2_sm,color='k',lw=2,label='PBLH')
        plt.legend()
        plt.title(str(platform)+' '+str(date_label))
        if show_me == True:
            plt.show()
        else:
            plt.savefig(path_to_plots+date_label+'_Tgrad_'+platform+'.png')
            plt.close()
        
        # #WV
        plt.figure(figsize=(18,6))
        plt.pcolormesh(HRa,Za,wv.transpose(),vmin=0,vmax=18,cmap='viridis')
        plt.colorbar(label='WV Mixing Ratio')
        plt.ylim(0.06,2.75)
        plt.xticks(np.arange(0,24.1,3))
        plt.xlim(0.0,24)
        plt.xlabel('Hour [UTC]')
        plt.ylabel(' Height AGL [km]')
        plt.title(str(platform)+' '+str(date_label))
        # plt.plot(x/60.,BLhgt_sm,'w',lw=3)
        # plt.plot(x/60.,BLhgt_sm,'k',lw=2,label='1st gen')
        # plt.plot(x/60.,BLhgt_2_sm,'w',lw=3)
        # plt.plot(x/60.,BLhgt_2_sm,'b',lw=2,label='2nd gen')
        plt.fill_between(x/60.,BLhgt_2_sm-BLhgt_2_sm_range,BLhgt_2_sm+BLhgt_2_sm_range,color='grey',alpha=.3)
        plt.plot(x/60.,BLhgt_2_sm-BLhgt_2_sm_range,color='grey',lw=2)
        plt.plot(x/60.,BLhgt_2_sm+BLhgt_2_sm_range,color='grey',lw=2)
        plt.plot(x/60.,BLhgt_2_sm,color='w',lw=3)
        plt.plot(x/60.,BLhgt_2_sm,color='k',lw=2,label='PBLH')
        plt.legend()
        if show_me == True:
            plt.show()
        else:
            plt.savefig(path_to_plots+date_label+'_WV_'+platform+'.png')
            plt.close()
        
        #summary figure
        fig,ax=plt.subplots(1,1,sharey=True,sharex=True,figsize=(18,6))
        ax.plot(x/60.,BLhgt_sm, lw=6, color='k',label='1st Gen')
        ax.plot(x/60.,BLhgt_2_sm, lw=3, color='b',label='2nd Gen')
        ax.plot(x/60.,BLhgt_om2_sm, 'co',lw=None, label='2nd Gen Mech')
        ax.plot(x/60.,BLhgt_ob2_sm, 'mo', lw=None, label='2nd Gen Buoy')
        plt.legend()
        plt.xlim(0,24)
        plt.ylim(0.06,2.75)
        if show_me == True:
            plt.show()
        else:
            plt.savefig(path_to_plots+date_label+'_summary_'+platform+'.png')
            plt.close()
    
    ##################################################################################################################
    # Netcdf writeout
    ##################################################################################################################
    if write_me==True:  
        hourly_t = x[np.where(x%60==0)]*60. #units of seconds, hourly on the hour
        hourly_t = np.insert(hourly_t, 0, 0., axis=0) #0 dosent work because we started at 10 so... 
        hourly_epoch = base_time_vec[np.where(x%60==0)]
        hourly_epoch = np.insert(hourly_epoch,0,base_time_vec[0])
        hourly_pblh = BLhgt_2_sm[np.where(x%60==0)]
        hourly_pblh = np.insert(hourly_pblh, 0, BLhgt_2_sm[0], axis=0)
        hourly_pblh_std = BLhgt_2_sm_range[np.where(x%60==0)]
        hourly_pblh_std = np.insert(hourly_pblh_std, 0, BLhgt_2_sm_range[0], axis=0)
        hourly_pblh_om = BLhgt_om2_sm[np.where(x%60==0)]
        hourly_pblh_om = np.insert(hourly_pblh_om, 0, BLhgt_om2_sm[0], axis=0)
        hourly_pblh_ob = BLhgt_ob2_sm[np.where(x%60==0)]
        hourly_pblh_ob = np.insert(hourly_pblh_ob, 0, BLhgt_ob2_sm[0], axis=0)
        
        ten_t = x*60. #units of seconds, 10 min data
        ten_epoch = base_time_vec
        ten_pblh = BLhgt_2_sm
        ten_pblh_std = BLhgt_2_sm_range
        ten_pblh_om = BLhgt_om2_sm
        ten_pblh_ob = BLhgt_ob2_sm
        
        # create output file nc4.Dataset(name, write mode, clear if it exists, file format)
        output_file = nc.Dataset(path_to_write+date_label+'_'+platform+'fuzzyPBLh.nc', 'w', clobber=True, format='NETCDF3_64BIT')
        
        
        # global attributes
        output_file.title = 'CLAMPS Multi-Instrument Fuzzy Logic Estimated PBL Heights'
        output_file.author = author_list
        output_file.contact = contact_list
        output_file.reference = 'coming soon... contact for more info'
        output_file.campaign = campaign
        output_file.basetime = str(base_time)
        output_file.platform = platform
        output_file.latitude = sun_lat
        output_file.longitude = sun_lon
        
        
        
        # define dimensions       (name,value)
        output_file.createDimension('t', len(x)) #time dimension, 10min
        output_file.createDimension('t_1hr', len(hourly_t)) #time dimension, 10min
        
        #ten min output
        # create a variable file.createVariable(name, precision, dimensions) = values (usually some array)    
        output_file.createVariable('t','f8',('t'))[:] = ten_t
        # set attributes of variable file.variables[name], name, value)
        setattr(output_file.variables['t'],'units','seconds since 00Z UTC/basetime')
        setattr(output_file.variables['t'],'description','time axis for 10-minute data')
        
        output_file.createVariable('t_epoch','f8',('t'))[:] = ten_epoch
        setattr(output_file.variables['t_epoch'],'units','second epoch time (since 00UTC on 1/1/1970)')
        setattr(output_file.variables['t_epoch'],'description','time axis for 10-minute data in epoch seconds')
                
        output_file.createVariable('pblh','f4',('t'))[:] = ten_pblh
        setattr(output_file.variables['pblh'],'units','km a.g.l.')
        setattr(output_file.variables['pblh'],'description','PBL height as estimated every 10min by fuzzy logic from CLAMPS')
        
        output_file.createVariable('stdev','f4',('t'))[:] = ten_pblh_std
        setattr(output_file.variables['stdev'],'units','km')
        setattr(output_file.variables['stdev'],'description','standard deviation of all PBL estimates in hour-wide triangle rolling window')
        
        output_file.createVariable('pblh_mech','f4',('t'))[:] = ten_pblh_om
        setattr(output_file.variables['pblh_mech'],'units','km a.g.l.')
        setattr(output_file.variables['pblh_mech'],'description','PBL height including only "mechanical" terms" from lidar')
        
        output_file.createVariable('pblh_therm','f4',('t'))[:] = ten_pblh_om
        setattr(output_file.variables['pblh_therm'],'units','km a.g.l.')
        setattr(output_file.variables['pblh_therm'],'description','PBL height including only "thermal" terms" from thermo retrievals')
        
        
        
        # hourly output
        output_file.createVariable('t_1hr','f8',('t_1hr'))[:] = hourly_t
        setattr(output_file.variables['t_1hr'],'units','seconds since 00Z UTC/basetime')
        setattr(output_file.variables['t_1hr'],'description','time axis for 1-hour data')
        
        output_file.createVariable('t_1hr_epoch','f8',('t_1hr'))[:] = hourly_epoch
        setattr(output_file.variables['t_1hr_epoch'],'units','second epoch time (since 00UTC on 1/1/1970)')
        setattr(output_file.variables['t_1hr_epoch'],'description','time axis for 1-hour data in epoch seconds')
        
        output_file.createVariable('pblh_1hr','f4',('t_1hr'))[:] = hourly_pblh
        setattr(output_file.variables['pblh_1hr'],'units','km a.g.l.')
        setattr(output_file.variables['pblh_1hr'],'description','PBL height as estimated hourly by fuzzy logic from CLAMPS')
        
        output_file.createVariable('stdev_1hr','f4',('t_1hr'))[:] = hourly_pblh_std
        setattr(output_file.variables['stdev_1hr'],'units','km')
        setattr(output_file.variables['stdev_1hr'],'description','standard deviation of all PBL estimates in hourly averaging window')
        
        output_file.createVariable('pblh_1hr_mech','f4',('t_1hr'))[:] = hourly_pblh_om
        setattr(output_file.variables['pblh_1hr_mech'],'units','km a.g.l.')
        setattr(output_file.variables['pblh_1hr_mech'],'description','PBL height including only "mechanical" terms" from lidar')
        
        output_file.createVariable('pblh_1hr_therm','f4',('t_1hr'))[:] = hourly_pblh_om
        setattr(output_file.variables['pblh_1hr_therm'],'units','km a.g.l.')
        setattr(output_file.variables['pblh_1hr_therm'],'description','PBL height including only "thermal" terms" from thermo retrievals')
        
        # close it up
        output_file.close()
        print("File written: "+str(path_to_write+date_label+'_'+campaign+'_'+platform+'_fuzzyPBLh.nc'))
    
    

