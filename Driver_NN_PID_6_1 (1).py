#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Driver_NN_PID   -  Version 0.6
'''Latest version:  Combines NN_for_PID and PID_test module codes.
In this version, NeuralNet serves as a "judge" using only its
accuracy to discard batches if there is no improvement.  That is,
we are not using the 3 output node results to directly input to the PID.
   
We use batch'es' which have 20 rows with 11 entries as a simulation,
the first 8 of each row are MAP (mean arterial pressure) values;
the next 3 of which are the PID parameters Ki, Kd, and Kp; and
the last is the PID's performance score in approaching a target MAP.
This driver will ouptput a new baseBatch.csv for each i-th loop,
which is fed to the NN changes the parms (K) by the gradient value.

An inner j1-loop determines a gradient, starting with the i-th batch
and varying each parm in K, one at at time, by a slight amount as deltaBatches
to obtain "coordinates" of the gradient.  EXPLANATION NEEDS WORK

We start by training the NN - PID combination with two
simple scenarios, 50 lines of each in this dataset:
(1) measured MAP remains about targetMAP, for which PID should succeed.
(2) measured MAP starts 16 mmHg below target, and PID should "fail" to reach targetMAP.
Each line has 8 MAP measurements, and these are used (each) to train the NN-PID.
Since time step is 10 seconds, each profile only constitutes 10 x 8 seconds of each dataset.
We might easily extend this to 5 to 10 minutes of profile.
See Driver_NN_PID_setup, which creates a dataset as well as a fluid losses file.

Outer_loop:
   runs the dataset through the PID,
   computing a score for each row depending on degree of success
   of PID in reaching a target MAP.  In the combined modules, 
   Driver_PID_NN, the neural net 
   runs batches of PID results seeking to improve the PID's
   internal parameters.

Conventions:
i - indexes the outer loop which runs through a batch of profiles
j1 - runs over the PID parm numbers, 1,...N, varying each as a batch is run
k - 
m - the element of a batch row picked out for the PID
n - cycles the PID - NN combination through batches

We start with two simple scenarios: 
(1) measured MAP starts 16 mmHg below target, and PID should "fail" to reach targetMAP,
(2) measured MAP remains about targetMAP, for which PID should succeed.

There are other files ready to be used as batches, such as:
  keras_3parms_glitches
  keras_3parms_sinusoids

Notes:
the PID outputs are all appended to controls.csv for a record of PID progress.
There is a lag, L, built in to the proportional component of the PID.
The pump model depends on a delay and a resistance_factor usually near 1.
The delay is assumed to be 1/10 second, i.e. one time step.

u1 is an infusion rate in micro-gm/(kg x min).
v1 is fluid loss (mL / sec) for one time interval (hemorrhage, urine, compartment, etc.)
e1 is the error between targetMAP and observed MAP (mmHg).
'''
# Debugging and flow control
dbg = True  # top level results
dbg0 = False  # body response model
dbg1 = False  # more detail
dbg2 = False  # set_traces on PID
dbg3 = False  # set traces on NN
dbg4 = False  # Keras errors
dbg5 = False # end of n-loop
dbg6 = False # trap index error near PID
dbg7 = False # create batch methods
dbg8 = False # stop at NN stages

# flow control
open_loop = True # training with NN updating PID parameters
use_exponential_damping_in_pump = False
train = True    # training versus closed loop runs
test= False
preview = True
summary = True

import sys
import csv
import math
import random
import inspect

import numpy as np
from numpy import array

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from IPython.core.debugger import set_trace
from IPython.core.debugger import Tracer; debug_here = Tracer()

# Pump simulator parameters
delay = 1   # assumed to be about one time step for now (10 sec.).
resistance_factor = 0.9  # degree to which the pump can actually infuse.
pump_noise_level = 3.0
use_exponential_damping_in_pump = True
eMax = 20
# this factor also includes the linear transfer function constant.

# global parameters
u1 = []
v1 = []
e1 = []
VB3 = 0  
VB2 = 0
VB1 = 0
VB0 = 0
Um1 = 0    # no reaction of model for first two steps
Vm1 = 0
Um2 = 0
Vm2 = 0

M = 20    # PID's batch size
scores = [0 for i in range(M)]  # given by PID for how close to target MAP it reaches
delta_K = []
deriv_limit = 0.5  # limits the derivative estimate in derivative
ave_score = 0
num_scores = 0

noise = 0.1    # noise * random  added to each coordinate in create_batch_i_j
epsilon = 0.05 # wiggle epsilon * random(0,1) if on a flat splot
fraction = 0.1 # maximum step limited by fraction of K.  May need to be larger
#                for faster convergence of the PID parameters, K.

out_f = ""     # output file names
accuracy = 0   # current accuracy of current batch
K = [0,0,0]         # parameters for PID
fileRoot = "batch_"  # prefix for batch file names
i = 0          # loop batch numbering starts with 0

# Hyperparameters
numCycles = 5     # n-loop range.  file of 100 lines can be split into 5 batchs of 20
numParms = 3      # number of PID parameters to be optimized
batchSize = 20    # NN's batches (may be different from baseBatches for PID
noise = 5         # noise level with uniform distribution, i.e. white noise
quartile = 0.2    # null hypothesis for random white noise.  Lower is better.
percentile = 0.05 # for null hypothesis for Gaussian noise in NN accuracies
P = percentile    # short name for percentile
# N = 3           # dimension of PID's parameters, currently hard-coded 3
targetMAP = 65    # desireable mean arterial blood pressure in pigs
initialMAP = 60   # This is for a typical hypovolemic (hemorrhaged) MAP.
initialError = 5
initialLoss = 0   # this gets the fluid loss model recursion started.
initialInfusion = 11
initialBleed = 7

L = 3     # pure time delay, or lag in the impulse response, i.e. 10 sec.
Kp = 1.2  # proportional gain (g kg-1 min-1) mmHg-1,  Might be as low as 0.3.
Ki = 0.8  # integral gain
Kd = 0.1  # differential gain
Ci = 2.1  # 1 mmHg rise in MAP proportional to 1 mL infusion
Cm = 1    # Set Cm = 0 to eliminate bleeding component entirely from PID control.
#    Set Cm = 0 and Cl= 1 both to eliminate the fluid loss file input and
#    there could be a built-in updating of the loss "v" for steady bleed.
M = 1     # multiplier of basic time steps, 10 steps over 1 second
TI = M * 10   # total simulation time
T = 5    # time constant in the step response of "proportional"

# Set Cm = 0 to eliminate bleeding, i.e. fluid response VB0 from error:
#      e1 = (targetMAP - Ci * infusion  +  Cm * VB0)

# Following parametrize the body response function due to Hahn et al. slide 6
VB = 2500     # estimate of blood volume for a 43 kg swine
K = 1         # feedback gain
alpha = 2.5   # fluid transfer rate in body compartments for 900 mL bleed (Hahn)

del_t = 1     # discrete time difference or sampling interval (10  sec)
targetMAP = 80 # mmHg typical for swine
initialMAP = 60 # This is for a typical hypovolemic (hemorrhaged) MAP.

# Here is the final combination in PID.  Set Cm = 0 to eliminate bleeding.
#      self.e.append(targetMAP - Ci * infusion + Cm * self.VB0)
        
# Following parametrize the body response function due to Hahn et al. slide 6
VB = 2500     # estimate of blood volume for a 43 kg swine
K = 1         # feedback gain
alpha = 2.5   # fluid transfer rate in body compartments for 900 mL bleed (Hahn)
step = 1

#------- Helper functions -------------

def print_frame():
    callerframerecord = inspect.stack()[1]    # 0 represents this line
                                              # 1 represents line at caller
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    if(dbg1): 
        print(info.filename)                      # __FILE__
        print(info.function)                      # __FUNCTION__
        print(info.lineno)                        # __LINE__
        print(" ")
    return info.lineno

#-----------------------------------------
#-- Search and gradient descent process --
# acc and acc_prev are using profile scores in this version
def step_j_th_parm(acc, acc_prev, j1, k, P, K, epsilon):
    # Limits change to P % of acc_m1, e.g. 1 %.
    change = 0
    if(dbg7):
        set_trace()
    diff = float(acc[k]) - float(acc_prev[k])
    
    # Remember, Newton's method is
    #      X(n+1)  =  X(n)  -  f(X(n) / f'(X(n))
    # The crude gradient descent method uses this method in all coordinates. 
    if (diff > epsilon and acc > epsilon):
         change  =  -  acc / diff
    elif (diff > epsilon and acc < epsilon):
        # Note that a flat spot occurs at f(Xn) = 0, f'(Xn) != 0.
        # As long as this is not the last base batch, we randomize the 
        # batch and keep moving!
        change = 0
    else:
        #  An indeterminant spot occurs when both f and f' = 0 closely.
        #  We should kick things off a flat spot randomly some % of f
        change = epsilon * random.uniform(0, 1)
        
    if (change > P * K[j1]):
        change = P * K[j1]

    delta_K[j1][k] = change + K[j1]

    # Note, scores and accuracy may continue to improve even if we keep the
    # same batch data randomized somewhat.
    # After 3 inner loops, all coordinaes of K are updated.
    return delta_K
#--------------------------------------

def create_new_batch (i, bBatch, nBatch, delta_K, noise):
    # This creates a new batch base on the the best gradient steps
    # as the PID  parameters are extracted and recorded in the
    # next base_batch for the NN.
    success = 1
    rows = []
    b_size = len(bBatch) # baseBatch from which deltaBatches are derived
    # Read baseBatch here as rows and row
    try:
        with open(bBatch, 'r', newline='', encoding="utf8") as inp:
            count = 0
            csv_in = csv.reader(inp, delimiter=',')
            for row in csv_in:
                rows.append(row)
                count += 1     
            fileSize = count
        inp.close()
        if(fileSize != M):
            print("*** create_new_batch: fileSize = ",fileSize," at i = ",i)
            print("    bBatch = ",bBatch)
            print(print_frame())
            return 0
    except OSError as err:
        print("OS error: {0}".format(err))
        print(print_frame())
        success = 0
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 0
    except IOError:
        print ("File error with :", bBatch)
        print(print_frame())
        success = 0
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(print_frame())
        success = 0
        raise
    finally:
        if (success == 0):
            return 0       
    try:
        with open(nBatch, 'w+', newline ='', encoding = "utf8") as outp:
            #-write intermediate batch i_j with variation in the j-coordinate
            csv_out = csv.writer(outp, delimiter=',')
            k = 0
            for k in range(0, M):
                row = rows[k]
                meas = []
                # measurments at 8 preceeding time steps
                for m in range(0,8):
                    val = row[m]
                    meas.append(val)
                
                # these steps are decreased as good MAP value approached
                Kp = K[0] + delta_K[k][0]
                meas.append(Kp)
                Ki = K[1] + delta_K[k][1]
                meas.append(Ki)
                Kd = K[2] + delta_K[k][0]
                meas.append(Kd)
                meas.append(score[k])
                # Note:  we now append scores in these newBatches
                # for the NN 
                csv_out.writerow(meas)
        outp.close()
    except OSError as err:
        print("OS error: {0}".format(err))
        print(print_frame())
        success = 0
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 0
    except IOError:
        print ("File error with :", deltaBatch)
        print(print_frame())
        success = 0
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(print_frame())
        success = 0
        raise
    finally:
        outp.close()
    return success
# end create_new_batch

##-- Prepare for (revised) base batch in i-th direction (a deltaBatch)
# Note, we keep previous batches for later optimization of hyperparameters
#------------------------------------
def create_batch_i_j (i, j1, bBatch, dBatch, K, noise):
    # These are the Inner-loop's batches where PID parameters are
    # varied one at a time.  The best gradient steps in each of
    # the N PID  parameters are extracted and recorded in the
    # next base_batch for the NN.
    success = 1
    rows = []
    
    file = open(bBatch)
    fileSize = len(file.readlines()) # baseBatch from which deltaBatches are derived
    if(fileSize != M):
        print("*** create_batch_i_j: fileSize = ",fileSize," at i = ",i,", j = ",j1)
        print(print_frame())

    # Read baseBatch here as rows and row
    try:
        with open(bBatch, 'r', newline='', encoding="utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            for row in csv_in:
                rows.append(row)
        inp.close()
        if(dbg7):
            set_trace() ##################################
        if(fileSize != M):
            print("*** create_batch_i_j: fileSize = ",fileSize," at i = ",i)
            success = 0
            print(print_frame())
    except OSError as err:
        print("OS error: {0}".format(err))
        print(print_frame())
        success = 0
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 0
    except IOError:
        print ("File error with :", baseBatch)
        print(print_frame())
        success = 0
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(print_frame())
        success = 0
        raise
    finally:
        if (success == 0):
            return 0
    try:
        with open(dBatch, 'w+', newline ='', encoding = "utf8") as outp:
            #-write intermediate batch i_j with variation in the j-coordinate
            csv_out = csv.writer(outp, delimiter=',')
            k = 0
            for k in range(0, M):
                row = rows[k]
                meas = []
                # measurments at 8 preceeding time steps
                for m in range(0,8):
                    val = row[m]
                    meas.append(val)
                # vary the j1-th PID parm
                if (j1 == 0):
                    # Note: this does not alter the K params
                    Kp = K[0] + step   # this step is decreased as good value approached
                    meas.append(Kp)
                    meas.append(K[1])
                    meas.append(K[2])
                elif (j1 == 1):
                    meas.append(K[0])
                    Ki = K[1] + step
                    meas.append(Ki)
                    meas.append(K[2])
                elif (j1 == 2):
                    meas.append(K[1])
                    meas.append(K[2])
                    Kd = K[2] + step
                    meas.append(Kd)
                    # Note:  we don't append scores in these deltaBatches
                    # scores are only appended in new batches created for the NN
                else:
                    if(dbg):
                        print("*** Wrong j1 = ",j1,", at i = ",i)
                    success = 0
                csv_out.writerow(meas)
        outp.close()
    except OSError as err:
        print("OS error: {0}".format(err))
        print(print_frame())
        success = 0
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 0
    except IOError:
        print ("File error with :", deltaBatch)
        print(print_frame())
        success = 0
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(print_frame())
        success = 0
        raise
    finally:
        outp.close()
    return success
#- end create_batch_i_j
#--------------------------------------

def create_batch_i (i, bBatch, nBatch, K, scores, noise):
    # These are the i-loop's batches where PID parameters have
    # been optimized.  The best gradient steps in each of
    # the N PID  parameters have been extracted and recorded in K
    # for the next baseBatch for the NN.
    # Note: We do include scores in newBatch for the NN to use.
    success = 1
    rows = []
    if(dbg7):
        set_trace() ###################
    file = open(bBatch)
    fileSize = len(file.readlines())
    if(fileSize != M):
        print("*** create_batch_i: fileSize = ",fileSize," at i = ",i)
        print(print_frame())
        ##return 0
    try:
        count = 0
        with open(bBatch, 'r', newline='', encoding="utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            for row in csv_in:
                rows.append(row)
                count += 1
        inp.close()
        fileSize = count
        if(dbg7):
            set_trace() ##########################

        if(fileSize != M):
            print("*** create_batch_i: fileSize = ",fileSize," at i = ",i)
            success = 0
            return success            
    except OSError as err:
        print("OS error at 278: {0}".format(err))
        print(print_frame())
        success = 0
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 0
    except IOError:
        print ("File error with :", filename)
        print(print_frame())
        success = 0
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        print(print_frame())
        success = 0
        raise
    finally:
        inp.close()
        if (success == 0):
            return success
    try:
        with open(nBatch, 'w+', newline ='', encoding = "utf8") as outp:
            csv_out = csv.writer(outp, delimiter=',')
            if(dbg7):
                set_trace() #####################
            n_size = len(nBatch)
            if(dbg1): print("-------n_size = ",n_size)
            for k in range(0, M):
                meas = []
                # measurments at preceeding time steps
                for m in range(0,8):
                    val = float(rows[k][m]) + noise * random.uniform(0, 1)
                    meas.append(val)
                Kp = K[0] + noise * random.uniform(0, 1)
                meas.append(Kp)
                Ki = K[1] + noise * random.uniform(0, 1)
                meas.append(Ki)
                Kd = K[2] + noise * random.uniform(0, 1)
                meas.append(Kd)

                if (scores[k] > 0.99999):  # higher and the NN goes crazy
                    print("*** score = ",scores[k])
                    scores[k] = 0.99999

                datum = float(scores[k])
                meas.append(datum)
                csv_out.writerow(meas)
        outp.close()
    except OSError as err:
        print("OS error at 317: {0}".format(err))
        print(print_frame())
        success = 0
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 0
    except IOError:
        print ("File error with :", filename)
        print(print_frame())
        success = 0
    except:
        print("Unexpected error at 326:", sys.exc_info()[0])
        print(print_frame())
        success = 0
        raise
    finally:
        outp.close()
    return success
#- end reate_batch_i
#--------------------------------------------

# Here the neural net as a helper function instead of a class, unlike PID.
# May want to class-ify this in future for modularity.

def NeuralNet (filename):
    """
    NN_for_PID vers 1
    to classify time series used in Driver_NN_PID, based on
    https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
    See SimpleNN for correct original with
    10 inputs, 3 outputs
    """
    import numpy as np
    from numpy import array
    import csv
    import math

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam

    data = []
    try:
        with open(filename, 'r', newline = '', encoding = "utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            for row in csv_in:
                data.append(row)
        inp.close
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
        print("***Input file couldn't not be openned")

    finally:
        inp.close
        print("test completed normally.")

    dataset = array(data)
    xy = dataset.shape
    print("data.shape = ",xy)

    # separate data
    split = 10
    train, test = dataset[:split,:], dataset[split:,:]
    print("MAIN CODE: train----------------")
    print(train)
    print("MAIN CODE: test----------------")
    print(test)
    print("==========================================")
    print(" ")
    print(" As it is split:")
    train_x = train[:,0:8]
    ##train_x = train[:,0:6]
    train_y= train[:,-4:-1]

    print(" ")
    print("train_x ----------------")
    print(train_x)
    print("train_y ----------------")
    print(train_y)

    print(" ")
    print(" test_x,----------------")
    test_x = test[:,0:8]
    #####test_x = test[:,0:6]
    print(test_x)
    print(" test_y,----------------")
    test_y = test[:,-4:-1]
    print(test_y)
    print(" ")
    # THIS IS TRICKY - CREATED ERROR IN SIMILAR CODE ??
    #train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

    if(dbg8):
        set_trace()
    # Build the model
    model = Sequential()

    if(dbg8):
        set_trace()
    #####model.add(Dense(10, input_shape=(6,), activation='relu', name='fc1'))
    model.add(Dense(10, input_shape=(8,), activation='relu', name='fc1'))
    model.add(Dense(10, activation='relu', name='fc2'))
    model.add(Dense(3, activation='softmax', name='output'))

    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=0.001)
    if(dbg8):
        set_trace()
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print('Neural Network Model Summary: ')
    print(model.summary())
    '''Layer (type)                 Output Shape              Param #   
    =================================================================
    fc1 (Dense)                  (None, 10)                50        
    _________________________________________________________________
    fc2 (Dense)                  (None, 10)                110       
    _________________________________________________________________
    output (Dense)               (None, 3)                 33        
    =================================================================
    Total params: 193
    Trainable params: 193
    Non-trainable params: 0'''

    # Train the model
    if(dbg8):
        set_trace()
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)

    # Test on unseen data
    if(dbg8):
        set_trace()
    results = model.evaluate(test_x, test_y)

    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))
    return results
#===================================
# This class is from from PID_test -- tests PID_incremental_controller
# if further module testing is needed for the PID.
# These results were obtained for rabbits with norepinephrine infusion.
# May have to tinker with the constants a bit for other animals and infusants.
'''
u is an infusion rate in micro-gm/(kg x min).
v is fluid loss (mL / sec) for one time interval (hemorrhage, urine, compartment, etc.)
e is the error between targetMAP and observed MAP (mmHg).'''

# Model_increment is a time-stepped linear body response model to infusion and fluid loss.
# It follows: Bighamian et al. A Lumped-Parameter Blood Volume Model.

# T is sampling interval
# VB is the response to u and v as change in blood volume output normalized by VB0
# To convert this to control "delta V", we have delta V = VB * (VB0 - VB1),
# where VB is blood volume estimate,
# K is feedback gain (specifying the speed of ﬂuid shift),
# alpha is the ratio between the volume changes in the intravascular and interstitial ﬂuids.
#
# The controller retains the 1/(1+α) fraction of the input ﬂuid volume in the intravascular compartment 
# while shifting the remaining α/(1 + α) fraction to the interstitial compartment in the steady state. 
# The ﬂuid shift from the intravascular to interstitial compartment (q) acts as feedback control to steer VB 
# to the target change in blood volume (rB) [Hahn].

# u is total infusant  OR incremental?
# i is time step counter, each step being T

#--- Helper functions
# Infusion pump transfer function
# We can give pump some history in the error array
def pump_model(error, infusion, delay, pump_noise_level):
    if(use_exponential_damping_in_pump):
        if (len(error) >= delay):
            exponent = -(error[-delay] / eMax)
            if (exponent < 100 and exponent > -100):
                slowdown = 1 - math.exp(exponent)  # prevents going too fast as error nears 0
                if(dbg): print("slowdown = ",slowdown)
            else:
                slowdown = 1
                print("*** pump exponent too large or too small")
            result = infusion * slowdown * resistance_factor
            return result + pump_noise_level * random.uniform(0, 1)
        else:
            return infusion  + pump_noise_level * random.uniform(0, 1)
    else:
        result = infusion * resistance_factor  + pump_noise_level * random.uniform(0, 1)
        return result
    
# Impulse response, included in the "proportional" component
def impulse(i):
    val = 0
    if (i < T and i >= L):
        val = (Ki / T) * math.exp(-((i - L) / T))       # Kashihara equ (1)
    if(dbg1):
        print("impulse: val = ",val)
    return val

# e1 is the "error" between target MAP and measured MAP.
# calling sequence:  self.i - T, self.i - 1, self.i, e1
def integral(t0, tN, i, e):
    sum = 0
    len_e = len(e)
    if(dbg2):print("integral: len_e = ",len_e)
    for j in range(t0, tN - 1):
        if ( ((i - j) > -1) and (j <= len_e)):
            sum = sum + impulse(i - j) * e[j] * del_t
        return sum

def derivative(h1, i):
    len_h = len(h1)
    if(dbg2):print("derivative: len_h = ",len_h,", i = ",i," h1 = ",h1)
    if ((i < T - 1) and (i > 0) and (i <= len_h) and (len_h > 2)):
        deriv = (h1[i] - h1[i-2]) / 2 * del_t
    elif ((i < T - 1) and (i > 0) and (i <= len_h) and (len_h > 1)):
        deriv = (h1[i] - h1[i-1]) / del_t
    else:
        deriv = 0
        
    return deriv

# Fill the losses list, v1, with the second column.
# First column is just a line number, skipped.
def get_losses(losses_f):
    if(dbg): print("-getLosses entry")
    count = 0
    try:
        with open(losses_f, 'r', newline = '', encoding = "utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            count += 1
            for row in csv_in:
                v1.append(row[1])
        inp.close()
        if(dbg):
            print("-getLosses returned ",count," lines from ",losses_f)
    except csv.Error as e1:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e1))
        print("***getLosses Fluid losses or bleeding file couldn't be openned")
        print(print_frame())
    except IOError:
        print ("***getLosses file error with :",losses_f)
        print(print_frame())
    except:
        print("***getLosses: Unexpected error at 572:", sys.exc_info()[0])
        print(print_frame())
        raise
    finally:
        print("-getLosses: Fluids test opened normally.  398")
        
#==============================
class PID:
    def __init__(self,i,j,k,m,losses_f,MAP,control_f,VB=2500,K0=2.1,alpha=0.5):
        self.VB = VB
        self.K0 = K0
        self.alpha = alpha
        self.losses_f = losses_f
        self.MAP  = MAP            # this is usually a row of data from a batch
        self.control_f = control_f
        self.finalMAP = 0
        self.control = ""
        self.total_outputs = 0
        self.count = 0  #
        self.i = i  # time step number (now row number in this test)
        self.j = j  # controller param number = 0, 1, 2 (not used)
        self.k = k  # controller now reading row k
        self.m = m  # at row element m

    def run_model(self):
        A = 0 # It's easier to debug large formulas in pieces
        B = 0
        C = 0
        D = 0
        E = 0
        global VB3  
        global VB2
        global VB1
        global VB0

        global Um1
        global Vm1

        global Um2
        global Vm2
        
        if self.i < 2:  # recursion begins at second time for volume response 
            VB3 = 0  
            VB2 = 0
            VB1 = 0
            VB0 = 0

            Um1 = initialInfusion
            Vm1 = initialBleed
            
            Um2 = 0 
            Vm2 = 0
        else:
            if(dbg1):
                print(" ")
                print("--- Model: preparing model at i = ",str(self.i))," ----  448"
                print("Um2 = ",Um2,", Um1 = ",Um1)
                print("Vm2 = ",Vm2,", Vm1 = ",Vm1)
            if(dbg2):
                set_trace() ################## 
            A = 2 * VB1 - VB2
            B = - self.K0 * (VB1 - VB2)
            C = (self.K0 / self.VB) * ((float(Um1) - float(Um2)) - (float(Vm1) - float(Vm2)))
            D = (self.K0 * self.K0 / (self.VB * (1 + self.alpha))) * (float(Um2) - float(Vm2))
            VB0 = (A + B + C + D)
            
            # keep short memory for recursion
            Um2 = Um1
            if (len(u1) > 1):
                Um1 = u1[-1]
            else:
                Um1 =0
            Vm2 = Vm1
            if (len(v1) > 1):
                Vm1 = v1[-1]
            else:
                Vm1 = 0
            VB3 = VB2
            VB2 = VB1
            # Careful, this model tends to oscillate with period 2
            VB1 = (VB0 + VB1) / 2

            if(dbg0):
                print(" ")
                print("----run_model: at i = ",str(self.i)," m = ",str(self.m))
                print("-run_model: A = ",A,", B = ",B,", C = ",C,", D = ",D)
                print("VB3 = ",VB3,", VB2 = ",VB2,", VB1 = ",VB1,", VB0 = ",VB0)
                print("Um2 = ",Um2,", Um1 = ",Um1)
                print("Vm2 = ",Vm2,", Vm1 = ",Vm1)
    
    def advance_PID(self):
        A1 = 0 # It's easier to debug large formulas in pieces
        B1 = 0
        C1 = 0
        D1 = 0
        E1 = 0
        if(dbg2):
            set_trace() #########################################
        if (dbg1): print("-advance_PID: at entry, i = ",self.i,", m = ",self.m)
        try:
            out_f = open(self.control_f,"a",newline='',encoding="utf8")
            csv_out = csv.writer(out_f, delimiter=',')
            controller = " "

            # start the filter as best can for first time step
            if (self.m != None and self.m < 2):
                e1.append(initialError)
                v1.append(initialLoss)
                u1.append(initialInfusion)
                
                # Here is the simulated pump, supplying u1
                result = pump_model(e1, initialInfusion, delay, pump_noise_level)
                
                u1.append(result)   # initial infusion. (This accumulates total)
                controller = str(self.i) + "," + str(self.m) + "," + str(u1[0]) + "," + str(v1[0]) + "," + str(e1[0])
                out_f.write(controller)
                self.finalMAP = 0
            
            # subsequent time steps
            elif(self.m != None):
                # -------- PID control function
                if (dbg1): 
                    print("-advance_PID: at i = ",str(self.i),", m = ",str(self.m))
                # (Easier debugging to break up formula into these 4 parts)
                if(dbg2):
                    set_trace() ##########################
                
                # proportional term
                B1 = Kp * impulse (e1[self.m - 1])
                
                # integral term
                C1 = 0
                if (self.m >= T):  # integral goes back T - 1 steps
                    C1 = Ki * integral (self.m - T, self.m - 1, self.m, e1)
                D1 = 0

                # differential term
                if (self.m > 1):
                    deriv = derivative (e1, self.m - 1)
                    if (abs(deriv) <= deriv_limit):
                        D1 = Kd + deriv
                    else:
                        if(dbg):
                            print("-- advance_PID: Had to truncate deriv ",deriv)
                        if (deriv > deriv_limit):
                            D1 = Kd * 0.5
                        elif (deriv < 0.5):
                            D1 = - Kd * 0.5
                            
                infusion = B1 + C1 + D1   # increment of infusion recommended
                
                pump_out = pump_model(e1, infusion, delay, pump_noise_level)  # pump transfer function
                # pump may decrease the infusion rate if it is too fast, depending on e1
                
                tot_infusion = pump_out +  u1[self.m - 1]
                
                u1.append(tot_infusion) # total infusions up to and including m-th time step

                #--- Here is the combination: total infusion - fluid loss
                self.finalMAP = Ci * tot_infusion  -  Cm * VB0
                er = float(self.MAP[-1]) - self.finalMAP
                e1.append (er)

                ln_u = len(u1)
                ln_v = len(v1)
                if (self.m <= ln_u and self.m < ln_v):
                    aa = str(self.m)
                    ba = str(u1[self.m])
                    ca = str(v1[self.m])
                    da = str(e1[self.m])
                else:
                    aa = "0"
                    ba = "0"
                    ca = "0"
                    da = "0"
                    if(dbg1):
                        print("-advance_PID: ln_u = ",ln_u,", ln_v = ",ln_v,". Can't compute self.control")

                if(dbg2):
                    print(" ")
                    print("-advance_PID  A1 = ",A1,", B1 = ",B1,", C1 = ",C1,", D1 = ",D1)
                    print(" - e is now : ",e1[-m:-1])
                    print(" - infusion : ",infusion)
                    print(" - u is now : ",u1[-m:-1])
                    print(" - v is now : ",v1[-m:-1])
                    print(" - m = ",aa,", u1 = ",ba,", v1 = ",ca,", e1 = ",da)   
                    
                self.control = aa + "," + ba + "," + ca + "," + da   # keeps a record of control

                if (dbg): print("-advance_PID:  control = " + self.control)

                # Note: this is the file needed in a trained, closed-loop control.
                # Be sure to delete this file before starting a closed-loop run.
                out_f.write(self.control)
                out_f.close()
                
                self.total_outputs += 1

                #print("-advance_PID: updating i to ",self.i)
                if(dbg1): print("-advance_PID returning error ",e1[-m:-1])
            else:
                print("*** self.i  is a None!")
                out_f.close()
        except OSError as err:
            print("OS error at 775: {0}".format(err))
        except ValueError:
            print("Could not convert data to an integer.")
        except IOError:
            print ("File error with :", self.control_f)
        except:
            print("Unexpected error at 781: ", sys.exc_info()[0])
            raise
        finally:
            out_f.close()
            if(dbg1): print('PID_incremental_controller iteration completed normally.')
        # End advance_PID
        
        def get_average_error(self):
            leng = len(e1)
            for i in range(0,leng):
                e_sum = e1[i]
            if (leng > 0):
                return e_sum / leng
            else:
                print("*** error list has 0 length")
        
        def get_control(self):
            return self.control
#-- End PID controller class

#------------------------------------
# Gradient descent process
# acc and acc_prev are using NN's accuracies in this version
def step_j_th_parm(acc, acc_prev, j1, K, epsilon):
    # Limits change to P % of acc_m1, e.g. 1 %.
    change = 0

    delta_acc = acc - acc_prev
    
    # Remember, Newton's method is
    #      X(n+1)  =  X(n)  -  f(X(n) / f'(X(n))
    # The crude gradient descent method uses this method in all coordinates. 
    if (delta_acc > epsilon and acc > epsilon):
         change  =  -  acc / delta_acc
    elif (delta_acc > epsilon and acc < epsilon):
        # Note that a flat spot occurs at f(Xn) = 0, f'(Xn) != 0.
        # As long as this is not the last base batch, we randomize the 
        # batch and keep moving!
        change = 0
    else:
        #  An indeterminant spot occurs when both f and f' = 0 closely.
        #  We should kick things off a flat spot randomly some % of f
        change = epsilon * random.uniform(0, 1)
    if (change > K[j1] * fraction):
        change = K[j1] * fraction

    K[j1] = change + K[j1]

    # Note, scores and accuracy may continue to improve even if we keep the
    #s ame batch data randomized somewhat.
    # Note also that K does accumulate over j, so after 3 inner loops, all
    # coordinaes of K are updated.
    return K
#------------------------------------

# Here is where the neural network gets to "vote" on keeping settings of K.
# acc is NN's curent accuracy; acc_last is last accuracy achieved; acc_prev next to last.
def improvement (acc, acc_last, acc_prev, Gaussian_noise):
    # After running the last base batch, 
    # THIS NEEDS TO ACCUMULATE A GAUSSIAN NOISE OUTSIDE
    difference1 = abs(acc - acc_last)
    difference2 = abs(acc_last - acc_prev)
    if (difference1 < percentile * Gaussian_noise and difference1 < percentile * Gaussian_noise):
        return 0   # no improvement, discard last batch
    else:
        return 1   # improvement -- move last batch to new baseBatch


#========================= MAIN =======================
if (train):
    '''It may help to have an outline. In the training loops:
    j1-loop makes baseBatche's which are then copied to 3 incremental batch's.
    After the loop there are predicted steps in best directions of each of
    the 3 PID parameters individually as the k-loop runs 0,..,2.
    This  is a crude way of estimating a gradient.
    In the i-loop, the three steps are added to the parameters, in K,
    for a new baseBatch for the next itereation.
    The NN is then run on this batch, which is split into training and test
    files equally.
    This may show an "improvement", in which case the baseBatch is kept, else discarded.
    outer n-loop continues.  We consider breaking if no improvement, but here the
    n-loop just selects a next batch of 20 rows out of the batch_0 file, which has 100.

    Kp = 1.2  # proportional gain (g kg-1 min-1) mmHg-1,  Might be as low of 0.3.
    Ki = 0.8  # integral gain
    Kd = 0.1  # differential gain

    Here we  read the training file, of 100 rows, in "batches" of 20,
    giving 5 training cycles.
    '''
    K = [Kp, Ki, Kd]  # initial parameters for PID and NN
    Gaussian = 0.0    # Kalman estimate of the distribution of NN accuracies
    training_cycles = 5   # keep it short for a while
    M = 20  # batch size short name
    batchSize = M
    training_rows = []
    
    # Read entire training file, selecting batches from it
    try:
        filename = "training.csv"
        filename = "kerasNN_3parms_0.csv"
        with open(filename, 'r', newline='', encoding="utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            count = 0
            for row in csv_in:
                training_rows.append(row)
                count += 1
        fileSize = count
        if(dbg): print("fileSize = ",fileSize)
        inp.close()
    except OSError as err:
        print("OS error at 884: {0}".format(err))
        success = 0
    except ValueError:
        print("Could not convert data to an integer.")
        success = 0
    except IOError:
        print ("File error with :", filename)
        success = 0
    except:
        print("Unexpected error at 893: ", sys.exc_info()[0])
        success = 0
        raise
    finally:
        inp.close()
        
    # Note: instead of this n1 loop, we could run through a list of training batchs
    # from other sources such as NoisySinusoids or Glitches.
    for n1 in range(0, training_cycles - 1):   #  Run several training sets through PID-NN
        if(dbg2):
            set_trace() ###################
        batch_rows = []
        # Peel out the section of training file from M*n to M*(n+1) as a baseBatch.
        # In n-th loop peel off the batch from M * n to M * (n+1)
        start = n1 * M
        endit = (n1+1) * M
        count = 0
        print("--- start = ",start,", endit = ",endit)
        
        for peel in range(start, endit):
            batch_rows.append(training_rows[peel])
            count += 1
        if(dbg1):
            if (count != M):
                print("*** Peeled ",count," rows for next baseBatch. breaking n loop")
                print(print_frame())
                break
                
        # write baseBatch for the record, to be read later and parms varied in "deltaBatch"es
        baseBatch = fileRoot+str(n1)+".csv"   # base batch for n-th cycle
        try:
            with open(baseBatch, 'w+', newline ='', encoding = "utf8") as outp:
                csv_out = csv.writer(outp, delimiter=',')
                for k in range(0, M):
                    csv_out.writerow(batch_rows[k])
            outp.close()
        except OSError as err:
            print("OS error : {0}".format(err))
            print(print_frame())
            success = 0
        except ValueError:
            print("Could not convert data to an integer.")
            print(print_frame())
            success = 0
        except IOError:
            print ("File error with :", baseBatch)
            print(print_frame())
            success = 0
        except:
            print("Unexpected error : ", sys.exc_info()[0])
            success = 0
            print(print_frame())
            raise
        finally:
            outp.close()
            if(dbg): print("Successfuly wrote batch ",baseBatch)
               
        for i in range(0, batchSize):   #  Run varied batches through PID-NN
            if(dbg7):
                set_trace() ########################
            if (i == 0):
                baseBatch = fileRoot+str(n1)+".csv"
                newBatch = baseBatch
            else:
                baseBatch = newBaseBatch   # returned from end of this loop
                newBatch = baseBatch
                
            file = open(newBatch)
            fileSize = len(file.readlines())
            if (fileSize != M):
                    print("newBatch size = ",fileSize," at i = ",i,", n = ",n1)
                    print(print_frame())
                    break
            if(dbg): 
                print("--- At beginning looping over baseBatch = ",baseBatch)
                print("    fileSize = ",fileSize)
            
            #--Inner loop creates rows of batch_i_j as small steps in each component of K.
            #  The PID parameter array, K, will be collected after the end of the loop.
            #  Run Driver_NN_PID_setup to get initial "_0" batch.

            average_row_error = 0
            ave_score = 0
            scores = [0 for i in range(0,batchSize)]
            acc_prev = 0
            acc_last = 0

            # This loop varies the j1-th coordinate of K to create deltaBatch, and
            # runs this file with PID, collecting a score base on how close to MAP target
            # the PID gets at the end of the profile (batch row).

            for j1 in range(0,2):  # runs PID with variations in each of 3 PID parameters

                # Change the j1-th coordinate in the best gradient direction
                delta_K = step_j_th_parm (accuracy, acc_last, j1, K, epsilon)
                
                # Fill in the j-th coordinate of params, K, to create a "delta"
                deltaBatch = fileRoot+str(i)+"_"+str(j1)+".csv"

                if(dbg7):
                    set_trace() ########################
                success = create_batch_i_j (i, j1, baseBatch, deltaBatch, delta_K, noise)
                if(dbg1): 
                    print("-Main: create_batch_i_j baseBatch = ",baseBatch,", deltaBatch = ",deltaBatch)
                if (success == 0):
                    break

                if(dbg): 
                    print("- j1 loop new batch: ",deltaBatch," at i = ",i,", j = ",j1)
                    print("  batchSize = ",batchSize)
                # Read the new "batch" and run PID on each measure in each row,
                # computing a score from the final MAP reached by the PID.
                if(dbg7):
                    set_trace()
                try:
                    rows = []
                    with open(deltaBatch, 'r', newline='', encoding="utf8") as inp:
                        csv_in = csv.reader(inp, delimiter=',')
                        count = 0
                        for row in csv_in:
                            rows.append(row)
                            count += 1
                    fileSize = count
                    if (fileSize != M):
                        print("*** Wrong deltaBatch file size")
                        print(print_frame())
                    inp.close()

                    for k in range(0,fileSize):
                        # For each row of batch_i_j, advance PID across simulated measurements
                        row = rows[k]
                        row_size = len(row)
                        mp = 0

                        # Run PID et al. across the k-th row
                        for m in range(0,row_size):
                            if(dbg1): print("Inner loop m ",m," entering PID")
                            pid = PID(i,j1,k,m,"bleeding.csv",row,"controls.csv",VB=2500,K0=2.1,alpha=0.5)
                            pid.run_model()
                            pid.advance_PID()   # value reached at end of row test
                            mp = pid.finalMAP
                            if(dbg1): print(" finalMAP = ",mp)
                        if(dbg2):
                            set_trace() ###################################

                        scores[k] = min( 0.99, 1 - abs((mp - targetMAP) / targetMAP))
                        
                        ave_score += scores[k]

                except OSError as err:
                    print("OS error : {0}".format(err))
                    print(print_frame())
                    break
                except ValueError:
                    print("Could not convert data to an integer.")
                    print(print_frame())
                    break
                except IOError:
                    print ("File error with : ", baseBatch)
                    print(print_frame())
                    break
                except:
                    print("Unexpected error :", sys.exc_info()[0])
                    print(print_frame())
                    break
                    raise
                finally:
                    inp.close()
                    if(dbg1):
                        print("deltaBatch ",deltaBatch," i = ",i,", j = ",j1,", scores = ",scores)
                        print("  K[",j1,"] = ",K[j1])
                #--end inner loop

                ave_score = ave_score / 3
                
                # sync up named parameters with K array
                Kp = K[0]
                Ki = K[1]
                Kd = K[2]

                if(dbg):
                    print("Finished inner loops.  Now K = ",K)
                    print("  ave_score = ",ave_score)

                # create new batch for NN based on best K's incremented in gradient directions
                newBatch = fileRoot+str(i+1)+".csv"
                create_batch_i(i+1, baseBatch, newBatch, K, scores, noise)
                # Next iteration of i-loop should pick the new baseBatch up.
                if(dbg7):
                    set_trace() #####################
                acc_prev = acc_last
                acc_last = accuracy

                if(dbg1): 
                    print("--- End of Loops i = ",i,", n = ",n1)
                    print("   accuracy = ",accuracy)
                    print("   acc_last = ",acc_last)
                    print("   acc_prev = ",acc_prev)
                    print("   ave_score = ",ave_score)
                    print(" ")

                # Re-train NN on the revised i-th batch
                if(dbg7):
                    set_trace() #####################
                if(dbg): print("At NN, newBatch = ",newBatch)

                file = open(newBatch)
                fileSize = len(file.readlines())
                if (fileSize < M):
                    print("*** newBatch is short : ",fileSize," Not running NN")
                else:
                    RESULTS = NeuralNet(newBatch)

                    loss = RESULTS[0]
                    accuracy = RESULTS[1]
                    '''Loss function measures the difference between the predicted label and the ground truth label. 
                    E.g., square loss is  L(y^,y)=(y^−y)^2 , hinge loss is  L(y^,y)=max{0,1 − y^ x y} ...'''
                    if (dbg):
                        print("--NN RESULTS:",RESULTS)
                        print("    accuracy  at 816 = ",accuracy)
                        print(print_frame())
                        print("  After NN call Kp = ",K[0],", Ki = ",K[1],", Kd = ",K[2])
                # Estimate presumed Gaussian noise of the NN's accuracy
                if (i == 0):
                    W1 = 1
                    W2 = 0
                else:
                    W1 = 1 / float(i)
                    W2 = 1 - W1
                Gaussian = W1 * accuracy + W2 * Gaussian
                if(dbg1): print("Gaussian noise level is ",Gaussian)

            #-- end j1-Loop over batch and delta batches

            improve = improvement(accuracy, acc_last, acc_prev, Gaussian)
            
            if(dbg5):
                set_trace()  #############################
            if (improve == 0): 
                print("-- No significant impovement in accuracy. discarding newBatch")
                file = open(baseBatch)
                fileSize = len(file.readlines())
                if (fileSize != M):
                    print("*** baseBatch is wrong length : ",fileSize, " Breaking i-loop")
                    print(print_frame())
                    break
                else:
                    newBaseBatch = baseBatch
            else: # replace old baseBatch with last batch from PID
                file = open(newBatch)
                fileSize = len(file.readlines())
                if (fileSize != M):
                    print("*** newBatch is wrong length : ",fileSize, " Breaking i-loop")
                    print(print_frame())
                    break
                else:
                    newBaseBatch = newBatch
            print("--- At end of batch loops, newBaseBatch = ",newBaseBatch)
        #--end batch i-loop
    #--end training n-loop
#=========================================================
if (test):
    if(dbg): print("========== STARTING VALIDATION =========")
    '''Validation test
    Read line from control.csv control file of past control;
    run PID (which may input a fluid losses file as an option);
    run pump simulator (basically a linear transfer function with lag)
    write line to control file;
    run NN to make a correction to PID output (trained separately for that as above);
    run "model" of body responses;
    compute error between body response and desired targetMAP.
    
    As prerequisite, this depends on the parameters, K, being adequately trained
    by the training loop.  Keeping its trained node weights, the NN now serves
    to make a small correction to the PID output.'''
    score = 0
    total_score = 0
    num_scores = 0
    finalMAP = 0
    num_test_batches = 100
    fileRoot_test = "NoisySinusoids.csv"  # could be random impulses for example
    fileRoot_test = "Glitches.csv"  # could be random impulses for example
    fileRoot_test = Batch_3.csv
    
    if(dbg): 
        print("Before outer loop, entering get_losses")
    if(Cm == 1): get_losses("bleeding.csv")   # opens losses file for the fluid loss model
    # Otherwise the PID appends new "v" values loop-by-loop
    
    i = 0
    j = 0
    k = 0
    baseBatch = fileRoot_test
    if(dbg):
        print("--- Beginning test loop")
        print("    Batch: ",baseBatch," at i = ",i)
            
    baseBatch = fileRoot_test
    
    # ADDED VERS 6.1
    num_scores = 0
    score = 0
    tot_score = 0
    # Read and run PID on each measure in each row
    try:
        rows = []
        with open(baseBatch, 'r', newline='', encoding="utf8") as inp:
            count = 0
            csv_in = csv.reader(inp, delimiter=',')
            for row in csv_in:
                rows.append(row)
                count += 1
        length = count
        if(dbg): print("Outer loop: Batch length is ",length)

        for i in range(0,length):
            # For each row of test_batch_i, advance PID across the simulated measurements
            # keeping track of score and total score.
            row = rows[i]
            row_size = len(row)

            # Run PID et al. across the i-th row
            e1 = []   # empty this for a run over a row
            u1 = []
            v1 = []
            for m in range(0,row_size):
                if(dbg1): print("Inner loop m ",m," entering PID")

                pid = PID(i,j,k,m,"bleeding.csv",row,"controls.csv",VB=2500,K0=2.1,alpha=0.5)
                pid.run_model()
                pid.advance_PID()   # value reached at end of row test            
                mp = pid.finalMAP
            if(dbg1):
                print(" finalMAP = ",mp)  
            ###score = 0
            ###tot_score = 0
            
            # After the loop, mp is the last and hopefully best value achieved.

            score = min(0.99, 1 - abs((mp - targetMAP) / targetMAP))
            tot_score += score
            num_scores += 1
        inp.close()
        ###if (num_scores > 0):
        ###    ave_score = tot_score / num_scores
        ###    print("")
        ###    print("--- ave_score = ",ave_score)
        ###else:
        ###    print("*** num_scores = 0 at 1168")
    except OSError as err:
        print("OS error: {0}".format(err))
        print(print_frame())
        stop = True
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        stop = True
    except:
        print("Unexpected error at 1159: ", sys.exc_info()[0])
        print(print_frame())
        raise
    finally:
        print("Input file",baseBatch," closed")

    if (num_scores > 0):
        ave_score = tot_score / num_scores
    else:
        print("** Can't report ave_score since num_scores = 0")

        
    if(dbg):
        print(" ")
        print("--- End of trained run at i = ",i)
        print("    tot_score = ",tot_score)

