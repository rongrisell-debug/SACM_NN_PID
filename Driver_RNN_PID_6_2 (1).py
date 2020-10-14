#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# Driver_RNN_PID_6_2   -  Version 6.2  - WORKING VERSION
get_ipython().run_line_magic('pylab', 'inline')
import sys
import random
import math
import csv
import ast
from IPython.core.debugger import set_trace
from IPython.core.debugger import Tracer; debug_here = Tracer()
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Debugging
dbg = True  # top level results
dbg0 = True  # body response model
dbg1 = False  # print print_frame
dbg2 = False # set_traces in PID and MAIN
dbg3 = False  # batches
dbg4 = False  # step_j1
dbg5 = False # end of n-loop - improvement ?
dbg6 = False  # set trace at i = 2 in run_model
dbg7 = False # create batch methods
dbg8 = False # stop at NN stages
dbg9 = False # Error at dot product
dbg11 = False # check pump model
dbg12 = False # deep in PID for error
dbg13 = False # summary PID and set trace
dbg14 = False # 
dbg15 = False # Shapes in NN
dbg16 = False # advance PID
dbg18 = False # improvement and accuracy parameters

# flow control
fluidLoss = False
open_loop = True # train with NN updating PID parameters
useExponentialDampingInPump = False
train = True    # train versus closed loop runs
test= False     # not yet implemented
preview = True
summary = True
cutNeurons = True  # Just run the controller (PID, fuzzy, etc)

sine = False # for testing Neural-Net
rand = False  # for testing Neural-Net
# If both sine and rand- are False, it uses actual train data

'''Latest version:  Combines NN_for_PID and PID_test module codes.
In this version, Neural Net serves as a "judge" using only its
"accuracy" to discard batches if there is no improvement.  That is,
we are not using the 3 output node results to directly input to the PID.

Here we create batches in a different way, compatile with the Neural Net.
Instead of inptting a file with many short sequences,
we need to create one relatively long sequence and create rows for the PID
in a sliding-window fashion.  See the new "slicer" below.

Batches have 20 rows with 50 entries each to test the controller,
the first 47 of each row are measured MAP (mean arterial pressure) values
and the next 3 of which are the PID parameters Ki, Kd, and Kp; and
the last is the PID's performance score in approaching a target MAP,
which is the train-ing "y" fed to the NN, or "score".

The controller ouptputs a new base batch.csv with each i-th loop,
which after stepping slightly in each of the dimensions of the parameers,
is then fed to the NN with judges the stepped parmeters (Kparms) by its own 
minibatch stochastic gradient descent method.  The accuracy and loss of a CNN or 
rmse and loss of a RNN;
then modulate a gradient step for the controller parameters by slowing
down or speeding up the steps.  We show that these feedback connections
will not run the system into instability if parameters are held witn a
certain Lipshitz-type constraint.   This is believed to prevent blow-up or 
zeroing of the NN, and walk-away by the essentially Newtonian step of the PID.
This combinatoin is a semi-autonomous learner in the sense that both 
have access to an external "true" but that itself is noisy and
there is in addition an unknown fluid loss.  The two, controller and NN
together, are each adapting to noise and unknown fluid loss (if included).
But the NN advises or "judges" from its point of view and allows the PID to
make the decision about infusion with its (hopefully) improving parameters.
The reason for choice of an RNN is that there is an unnkown internal state, the
fluid loss, and the RNN will tend to perform like a Markov process in its search.
Along these lines, another choice is a NN with short-term long-term memroy.

Details
An inner j1-loop determines a gradient, starting with the i-th batch
and varying each parm in K, one at at time, by a slight amount to create deltaBatches.
Thus we obtain "coordinates" of the gradient. This is crude but it works.

We start by train-ing the NN - PID combination with two
simple scenarios:
(1) measured MAP remains about target MAP, for which PID should succeed.
(2) measured MAP starts 16 mmHg below target, and PID should "fail" to reach target MAP.
Each line has 47 MAP measurements, and these are used (each) to train the NN-PID.
Since time step is 10 seconds, each profile constitutes 10 x 47 seconds of each dataset.

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
(1) measured MAP starts 16 mmHg below target, and PID should "fail" to reach target MAP,
(2) measured MAP remains about target MAP, for which PID should succeed.

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
e1 is the error between target MAP and observed MAP (mmHg).
'''
### SORRY ABOUT SO MANY GLOBALS - IT'S MY CLUMSY STYLE!!!

# Pump simulator parameters
delay = 1   # assumed to be about one time step for now (10 sec.).
resistance_factor = 0.95  # degree to which the pump can actually infuse.
pumpNoiseLevel = 0.1
useExponentialDampingInPump = True
eMax = 20
# this factor also includes the linear transfer function constant.

# Response model global parameters -- NEED THESE FOR GLOBAL RECURSION.
# THERE'S PROBABLY A WAY TO REMEMBER ACROSS CLASSES BUT I HAVEN'T FOUND.
u1 = []
v1 = []
e1 = []
VB3 = 0  # These are recursion parameters
VB2 = 0
VB1 = 0
VB0 = 0
Um1 = 0    # no reaction of model for first two steps
Vm1 = 0
Um2 = 0
Vm2 = 0

delta_K = []
deriv_limit = 0.5  # limits the deriv-ative estimate in differential term of PID if used
ave_score = 0
num_scores = 0
noise = 0.1       # noise * random  added to each coordinate in create batch_i_j
epsilon = 0.05    # wiggle epsilon * random(0,1) if on a flat spot
fraction = 0.01   # maximum step limited by fraction of K.  May need to be larger
#                    for faster convergence of the PID parameters, K.
out_f = ""        # output file names
accuracy = 0      # current accuracy of current batch (if using CNN)
K = [0,0,0]         # parameters for PID
fileRoot = "batch_"  # prefix for batch file names
i = 0             # loop batch numbering starts with 0

dataSize = 400    # length of data, which is now only one row, Changes with data.
batchSize = 20    # NN's batches same as baseBatches for PID ???
trainingCycles = int(dataSize / batchSize)    # n1-loop range.
trainingCycles = 5     ############ Debugging purposes
M = batchSize     # PID's batch size - short name for batchSize
Nparms = 3        # dimension of controller's parameters
rowSize = 50      # earlier the "T" in NN_for_PID_4
timeSteps = rowSize - Nparms    # we add Nparms parameters to the test sequence
scores = [0 for i in range(M)]  # given by PID for how close to target MAP it reaches

noise = 5         # noise level with uniform distribution, i.e. white noise
quartile = 0.2    # null hypothesis for random white noise.  Lower is better.
percentile = 0.0001 # for null hypothesis for Gaussian noise in NN accuracies

targetMAP = 65    # desireable mean arterial blood pressure in pigs
initialMAP = 60   # This is for a typical hypovolemic (hemorrhaged) MAP,
                  # in mmHg typical for hypovolumic swine
initialError = 5
initialLoss = 0   # this gets the fluid loss model recursion started.
initialInfusion = 60
initialBleed = 7

# Simulation parameters
L = 3     # pure time delay, or lag in the im-pulse response, i.e. 10 sec.
Kp = 1.2  # proportional gain (g kg-1 min-1) mmHg-1,  Might be as low as 0.3.
Ki = 0.8  # inte-gral gain
Kd = 0.1  # differential gain
Ci = 2.1  # 1 mmHg rise in MAP proportional to 1 mL infusion
Ci = 5    # ???
Cm = 0    # Set Cm = 0 to eliminate fluid loss component entirely from PID control.
Cp = 1    # Set Cp = 0 to eliminate the pump model.
#    Set Cm = 0 and Cl= 1 both to eliminate the fluid loss file input and
#    there could be a built-in updating of the fluid loss "v" for steady bleed.
Mult = 1     # multiplier of basic time steps, 10 steps over 1 second
TI = Mult * 10   # total simulation time
Tc1 = 5   # time constant in the step response of "proportional".
del_t = 1     # discrete time difference or sampling interval (10  sec)
step = 1 # initializes the constrained gradient method in create_batch_i
         # currently we are not using this but instead, create a new batch which
         # computes a controller parameter gradient modulated by
         # the accuracy achieved for the current batch.

# Note: Set Cm = 0 to eliminate bleeding, i.e. fluid response VB0 from error:
#      e1 = (targetMAP - Ci * infusion  +  Cm * VB0)

# The following further parametrize the body response function due to Hahn et al. (slide 6)
VB = 2500     # estimated blood volume for a 43 kg swine
K = 1         # feedback gain
alpha = 2.5   # fluid transfer rate between vascular and extravascular body 
              # compartments given a 900 mL bleed (Hahn)

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

#--------------------------------------

##-- Prepare for (revised) base batch in i-th direction (a deltaBatch)
# Note, we keep previous batches for later optimization of hyperparameters
#------------------------------------
def create_batch_i_j (n, i, j1, bBatch, dBatch, K, noise):
    # These are the Inner-loop's batches where PID parameters are
    # varied one at a time.  The best gradient steps in each of
    # the Nparm PID  parameters are extracted and recorded in the
    # next base_batch for the NN.
    success = 1
    rows = []
    
    # REDUNDANT:  THIS IS ONE WAY TO DETERMINE A FILE SIZE
    file = open(bBatch)
    size = len(file.readlines()) # baseBatch from which deltaBatches are derived
    if(size != M):
        print("*** create_batch_i_j: on entry, bad batchSize = ",size," at i = ",i,", j = ",j1,", n = ",n)
        print(print_frame())

    # Read baseBatch here as rows and row
    try:
        with open(bBatch, 'r', newline='', encoding="utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            count = 0
            for row in csv_in:
                rows.append(row)
                count += 1    # IT WOULD BE JUST AS WELL TO USE len(rows)
        if(count != M):
            print("*** create_batch_i_j: reading bBatch, wrong batchSize = ",count," at i = ",i,", n = ",n)
            success = 0
            print(print_frame())
            if(dbg7): set_trace() #######################  count, rows
    except OSError as err:
        print("OS error: {0}".format(err))
        print(print_frame())
        success = 2
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 3
    except IOError:
        print ("File error with :", baseBatch)
        print(print_frame())
        success = 4
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(print_frame())
        success = 5
        raise
    finally:
        inp.close()
        if(dbg7): set_trace() ################ end of read loop create_batch_i
        if (success != 1):
            return success
    if(dbg7): set_trace() ############### Now writing dBatch  
    meas = []
    try:
        with open(dBatch, 'w', newline ='', encoding = "utf8") as outp:
            #-write intermediate batch i_j with variation in the j-coordinate
            csv_out = csv.writer(outp, delimiter=',')
            k = 0
            for k in range(0, M):
                row = rows[k]
                rowLength = len(row)
                if (dbg7): print("create_batch_i_j: rowLength = ",rowLength,", timeSteps = ",timeSteps)
                #if(dbg7):
                #    print("create_batch_i_j: rowLength = ",rowLength,", timeSteps = ",timeSteps)
                #    set_trace() ##################

                # measurements at preceeding time steps
                for m in range(0,timeSteps):
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
                    print("*** Wrong j1 = ",j1,", at i = ",i)
                    success = 0
                csv_out.writerow(meas)
        if(dbg7): set_trace()  ##################### meas
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
        if(dbg7): set_trace() ################ end of write loop create_batch_i
        if (dbg):
            print("create_batch_i_j successful")
    return success
#- end create_batch_i_j
#--------------------------------------

def create_batch_i (n, i, bBatch, nBatch, K, scores, noise):
    # These are the i-loop's batches where PID parameters have
    # been optimized.  The best gradient steps in each of
    # the N PID  parameters have been extracted and recorded in K
    # for the next baseBatch for the NN.
    # Note: We do include scores in newBatch for the NN to use.
    success = 1
    rows = []
    if(dbg7): set_trace() ################### entry
    file = open(bBatch)
    lngF = len(file.readlines())
    if(batchSize != lngF):
        print("*** create_batch_i: wrong batchSize = ",lngF," at i = ",i,", n = ",n)
        print(print_frame())
    try:
        count = 0
        with open(bBatch, 'r', newline='', encoding="utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            for row in csv_in:
                rows.append(row)
                count += 1
        if(dbg7): set_trace() ##########################

        if(count != M):
            print("*** create_batch_i: wrong batchSize = ",count," at i = ",i)
            success = 0        
    except OSError as err:
        print("OS error at 278: {0}".format(err))
        print(print_frame())
        success = 2
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 3
    except IOError:
        print ("File error with :", bBatch)
        print(print_frame())
        success = 4
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        print(print_frame())
        success = 5
        raise
    finally:
        inp.close()
        if(dbg7): set_trace() ###################
        if (success != 1): 
            return success
    if(dbg2): set_trace() ################### Got passed reading in create_batch_i
    meas = []
    try:
        with open(nBatch, 'w', newline ='', encoding = "utf8") as outp:
            csv_out = csv.writer(outp, delimiter=',')
            for k in range(0, M):
                # Noise-up measurements at preceeding time steps -- WILL THIS ACCUMULATE ???
                for m in range(0,timeSteps):
                    val = float(rows[k][m]) + noise * random.uniform(0, 1)
                    meas.append(val)
                Kp = K[0] + noise * random.uniform(0, 1)
                meas.append(Kp)
                Ki = K[1] + noise * random.uniform(0, 1)
                meas.append(Ki)
                Kd = K[2] + noise * random.uniform(0, 1)
                meas.append(Kd)
                csv_out.writerow(meas)
        if(dbg7): set_trace() ##################### meas
    except OSError as err:
        print("OS error at 317: {0}".format(err))
        print(print_frame())
        success = 1
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 2
    except IOError:
        print ("File error with :", bBatch)
        print(print_frame())
        success = 3
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(print_frame())
        success = 4
        raise
    finally:
        outp.close()
        if(dbg7): set_trace() ################ at end of create_batch_i.  Check meas
        return success
#- end reate_batch_i
#--------------------------------------------
# https://www.analyticsvidhya.com/blog/2019/01/fundamentals-deep-learning-recurrent-neural-networks-scratch-python/
# Here the neural net as a helper function instead of a class, unlike PID.
# May want to class-ify this in future for modularity.

def NeuralNet (training):
    if(dbg8): 
        print("-----Entering the NN")
        #set_trace() ####################### step on!
    
    # NN_for_PID_6
    # See module for Resusc/Driver_NN_PID_6_xxx
    #https://www.analyticsvidhya.com/blog/2019/01/fundamentals-deep-learning-recurrent-neural-networks-scratch-python/

    '''This inputs single sequences of length, seq_num,
    each of which is a "minibatch". 
    So the shape of the input data is:

        (number_of_records x length_of_sequence x types_of_sequences)

    Here, types_of_sequences is 1, because there is only one type of sequence.
    On the other hand, the output will have only one value for each record,
    the seq_num'th value in the input sequence. So it’s shape is:

        (number_of_records x types_of_sequences)

    where types_of_sequences is 1
    '''
    rows = []
    X = []
    Y = []
    X_val = []
    Y_val = []

    '''Assuming an input file of dataSize = dataSize / 2 we can test with a sinewave
    sin_wave = np.array([math.sin(x) for x in np.arange(dataSize)]).

    If sin = False we will read a .csv file as output from "RNNs"
    '''
    seq_len = 50
    if (sine == True):
        sin_wave = np.array([math.sin(x/0.2) for x in np.arange(200)]) 
        + np.array([math.sin(x/0.5) for x in np.arange(200)])
        dataSize = len(sin_wave) - seq_len  # set aside last 50 as validation data
        for i in range(dataSize - seq_len):
            X.append(sin_wave[i:i+seq_len])
            Y.append(sin_wave[i+seq_len])
        '''((dataSize, seq_len, 1), (numrecords, 1))
        Create the validation data:
        '''
        for i in range(dataSize - seq_len, dataSize):
            X_val.append(sin_wave[i:i+seq_len])
            Y_val.append(sin_wave[i+seq_len])
        plt.plot(sin_wave[:seq_len])
    elif (rand == True):
        wave = np.array([random.random() for x in np.arange(200)])
        dataSize = len(wave) - seq_len  # set aside last 50 as validation data
        for i in range(dataSize - seq_len):
            X.append(wave[i:i+seq_len])
            Y.append(wave[i+seq_len])
        for i in range(dataSize - seq_len, dataSize):
            X_val.append(wave[i:i+seq_len])
            Y_val.append(wave[i+seq_len])
        plt.plot(wave[:seq_len])
    else:
        waveStrings = np.array(training)
        wave = waveStrings.astype(np.float)
        dataSize = len(wave) - seq_len  # set aside last 50 as validation data
        for i in range(dataSize - seq_len):
            X.append(wave[i:i+seq_len])
            Y.append(wave[i+seq_len])
        for i in range(dataSize - seq_len, dataSize):
            X_val.append(wave[i:i+seq_len])
            Y_val.append(wave[i+seq_len])
    if(dbg8): set_trace() ################# what is wave and waveStrings ???
    if(dbg0): print("dataSize = ",dataSize)
    # plt.plot(wave[:seq_len])
    #if(dbg8): set_trace() ##################### at end of X, Y, X_val and Y_val packing
    #==========================================
    X = np.array(X)
    X = np.expand_dims(X, axis=2)

    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=1)

    # Print the shape of the data:
    if(dbg15): print("X.shape ",X.shape,", Y.shape ",Y.shape)

    X_val = np.array(X_val)
    X_val = np.expand_dims(X_val, axis=2)

    Y_val = np.array(Y_val)
    Y_val = np.expand_dims(Y_val, axis=1)
    
    if(dbg15): print("X_val.shape ",X_val.shape,", Y_val.shape ",Y_val.shape)

    '''Step 1: Create the Architecture for an RNN model
    The model will take in the input sequence, 
    process it through a hidden layer of 100 units, and produce a single valued 
    output.'''

    learning_rate = 0.0001    
    nepoch = 6               
    T = 50                   # length of sequence
    hidden_dim = int(dataSize / 2)        
    output_dim = 1

    bptt_truncate = 5
    min_clip_value = -10
    max_clip_value = 10

    #Define the weights of the network:
    U = np.random.uniform(0, 1, (hidden_dim, T))
    W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
    V = np.random.uniform(0, 1, (output_dim, hidden_dim))

    if(dbg15): print("Step 1 U.shape ",U.shape,", W shape ",W.shape,", V shape ",V.shape)
    '''Here,
    U the weight matrix for weights between input and hidden layers
    V is the weight matrix for weights between hidden and output layers
    W is the weight matrix for shared weights in the RNN layer (hidden layer)
    Finally, we will define the activation function, sigmoid, to be used in the hidden layer:
    '''
    '''
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    '''
    def sigmoid(x):
        return np.where(x >= 0, 
            1 / (1 + np.exp(-x)), 
            np.exp(x) / (1 + np.exp(x)))
    
    '''Step 2: Train the Model
    Now that we have defined our model, we can finally move on with training 
    it on sequence data. 
    We can subdivide the training process into smaller steps, namely:

    Step 2.1 : Check the loss on training data
    Step 2.1.1 : Forward Pass
    Step 2.1.2 : Calculate Error
    Step 2.2 : Check the loss on validation data
    Step 2.2.1 : Forward Pass
    Step 2.2.2 : Calculate Error
    Step 2.3 : Start actual training
    Step 2.3.1 : Forward Pass
    Step 2.3.2 : Backpropagate Error
    Step 2.3.3 : Update weights

    We need to repeat these steps until convergence. If the model starts to overfit, 
    stop! Or simply pre-define the number of epochs.

    Step 2.1: Check the loss on training data
    We will do a forward pass through our RNN model and calculate the 
    squared error for the predictions for all records in order to get the loss value.'''

    for epoch in range(nepoch):
        # check loss on train
        loss = 0.0

        # do a forward pass to get prediction
        for i in range(Y.shape[0]):
            x, y = X[i], Y[i]                    # get input, output values of each record
            prev_s = np.zeros((hidden_dim, 1))   # here, prev-s is the value of the previous 
            # activation of hidden layer; which is initialized as all zeroes
            for t in range(T):
                new_input = np.zeros(x.shape)    # we then do a forward pass for every timestep 
                # in the sequence
                new_input[t] = x[t]              # for this, we define a single input for that timestep
                mulu = np.dot(U, new_input)
                mulw = np.dot(W, prev_s)
                add = mulw + mulu
                s = sigmoid(add)
                mulv = np.dot(V, s)
                prev_s = s
                if(i == 0 and t == 0):
                    if(dbg15): print("Step 2.1 mulu.shape ",mulu.shape,", mulw.shape ",mulw.shape,", mulv.shape ",mulv.shape)
            # calculate error
            #if(dbg8): set_trace() ##################### at error
            loss_per_record = (float(y) - float(mulv))**2 / 2.0
            #loss_per_record = (y - mulv)**2 / 2
            loss += loss_per_record
        loss = loss / float(y.shape[0])

        '''Step 2.2: Check the loss on validation data
        We will do the same thing for calculating the loss on validation data (in the same loop):
        '''
        # check loss on val
        ##denom = y.shape[1] * 2
        val_loss = 0.0
        for i in range(Y_val.shape[0]):
            x, y = X_val[i], Y_val[i]
            prev_s = np.zeros((hidden_dim, 1))
            for t in range(T):
                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                mulu = np.dot(U, new_input)
                mulw = np.dot(W, prev_s)
                add = mulw + mulu
                s = sigmoid(add)
                mulv = np.dot(V, s)
                prev_s = s
                if(i == 0 and t == 0):
                    if(dbg15): print("Step 2.2 mulu.shape ",mulu.shape,", mulw.shape ",mulw.shape,", mulv.shape ",mulv.shape)
            # calculate error
            loss_per_record = (float(y) - float(mulv))**2 / 2
            val_loss += loss_per_record
        val_loss = val_loss / float(y.shape[0])

        print('Epoch: ', epoch + 1, ', Loss: ', loss, ', Val Loss: ', val_loss)

        '''Should get the output:

        Epoch:  1 , Loss:  [[101185.61756671]] , Val Loss:  [[50591.0340148]]
        ...
        (gets down to 16, 17 or something similar.  We stay at 30, 40)

        Step 2.3: Start actual training
        We will now start with the actual training of the network. In this, we will first do a forward pass to calculate the errors and a backward pass to calculate the gradients and update them. Let me show you these step-by-step so you can visualize how it works in your mind.

        Step 2.3.1: Forward Pass
        In the forward pass:

        . We first multiply the input with the weights between input and hidden layers.
        . Add this with the multiplication of weights in the RNN layer. This is because we 
        want to capture the knowledge of the previous timestep.
        . Pass it through a sigmoid activation function.
        . Multiply this with the weights between hidden and output layers.
        . At the output layer, we have a linear activation of the values so we do not 
        explicitly pass the value through an activation layer
        . Save the state at the current layer and also the state at the previous 
        timestep in a dictionary
        Here is the code for doing a forward pass 
        (note that it is in continuation of the above loop):
        '''
        # train model
        for i in range(Y.shape[0]):
            x, y = X[i], Y[i]

            layers = []
            prev_s = np.zeros((hidden_dim, 1))
            dU = np.zeros(U.shape)
            dV = np.zeros(V.shape)
            dW = np.zeros(W.shape)

            dU_t = np.zeros(U.shape)
            dV_t = np.zeros(V.shape)
            dW_t = np.zeros(W.shape)

            dU_i = np.zeros(U.shape)
            dW_i = np.zeros(W.shape)

            # forward pass
            for t in range(T):
                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                mulu = np.dot(U, new_input)
                mulw = np.dot(W, prev_s)
                add = mulw + mulu
                s = sigmoid(add)
                mulv = np.dot(V, s)
                layers.append({'s':s, 'prev_s':prev_s})
                prev_s = s
                if(i == 0 and t == 0):
                    if(dbg15): print("2.3.1 mulu.shape ",mulu.shape,", mulw.shape ",mulw.shape,", mulv.shape ",mulv.shape)
            # calculate error
            '''Step 2.3.2 : Backpropagate Error
            Calculate the gradients at each layer, and 
            backpropagate the errors. Use truncated back propagation through time (TBPTT), 
            instead of vanilla backprop.
            '''
            # derivative of pred
            dmulv = (float(mulv) - float(y))   # dmulv IS A VECTOR.

            # backward pass
            for t in range(T):
                dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
                dsv = np.dot(np.transpose(V), dmulv)

                ds = dsv
                dadd = add * (1 - add) * ds

                dmulw = dadd * np.ones_like(mulw)

                dprev_s = np.dot(np.transpose(W), dmulw)


                for i in range(t-1, max(-1, t-bptt_truncate-1), -1):
                    ds = dsv + dprev_s
                    dadd = add * (1 - add) * ds

                    dmulw = dadd * np.ones_like(mulw)
                    dmulu = dadd * np.ones_like(mulu)

                    dW_i = np.dot(W, layers[t]['prev_s'])
                    dprev_s = np.dot(np.transpose(W), dmulw)

                    new_input = np.zeros(x.shape)
                    new_input[t] = x[t]
                    dU_i = np.dot(U, new_input)
                    dx = np.dot(np.transpose(U), dmulu)

                    dU_t += dU_i
                    dW_t += dW_i

                dV += dV_t
                dU += dU_t
                dW += dW_t

                '''Step 2.3.3 : Update weights
                Lastly, we update the weights with the gradients of weights calculated. 
                One thing we have to keep in mind that the gradients tend to explode if you 
                don’t keep them in check.This is a fundamental issue in training neural 
                networks, called the exploding gradient problem. So we have to clamp them 
                in a range so that they dont explode. We can do it like this'''

                if dU.max() > max_clip_value:
                    dU[dU > max_clip_value] = max_clip_value
                if dV.max() > max_clip_value:
                    dV[dV > max_clip_value] = max_clip_value
                if dW.max() > max_clip_value:
                    dW[dW > max_clip_value] = max_clip_value

                if dU.min() < min_clip_value:
                    dU[dU < min_clip_value] = min_clip_value
                if dV.min() < min_clip_value:
                    dV[dV < min_clip_value] = min_clip_value
                if dW.min() < min_clip_value:
                    dW[dW < min_clip_value] = min_clip_value

                # update
                U -= learning_rate * dU
                V -= learning_rate * dV
                W -= learning_rate * dW
                if(i == 0 and t == 0):
                    if(dbg15): print("Step 2.3.3 U.shape ",U.shape,", W shape ",W.shape,", V shape ",V.shape)

    '''Step 3: Get predictions
    We will do a forward pass through the trained weights to get our predictions:'''

    preds = []
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]
        prev_s = np.zeros((hidden_dim, 1))
        # Forward pass
        for t in range(T):
            mulu = np.dot(U, x)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s
            if(i == 0 and t == 0):
                if(dbg15): print("Step 3  mulu.shape ",mulu.shape,", mulw.shape ",mulw.shape,", mulv.shape ",mulv.shape)
        preds.append(mulv) 
    preds = np.array(preds)
    
    #if(dbg8): set_trace() ############################
    #Plotting these predictions alongside the actual values:
    plt.plot(preds[:, 0, 0], 'g')
    plt.plot(Y[:, 0], 'r')
    plt.show()

    # Step 4. Validation

    preds = []
    mulu = 1.0
    mulw = 2.0
    mulv = 3.0
    for i in range(Y_val.shape[0]):
        x, y = X_val[i], Y_val[i]
        prev_s = np.zeros((hidden_dim, 1))
        # For each time step...
        for t in range(T):
            mulu = np.dot(U, x)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s
            if(i == 0 and t == 0):
                if(dbg15): print("Step 4  mulu.shape ",mulu.shape,", mulw.shape ",mulw.shape,", mulv.shape ",mulv.shape)
        preds.append(mulv)   
    preds = np.array(preds)
    #if(dbg8): set_trace() ############################
        
    plt.plot(preds[:, 0, 0], 'g')
    plt.plot(Y_val[:, 0], 'r')
    plt.show()

    from sklearn.metrics import mean_squared_error
    max_val = 1
    rmse = math.sqrt(mean_squared_error(Y_val[:, 0] * max_val, preds[:, 0, 0] * max_val))
    if(dbg0): print("rmse = ",rmse,", loss = ",loss)

    return loss, rmse
    # end NN
#===========================================================================

#===================================
# This class is from from PID_test -- tests PID_incremental_controller
# if further module testing is needed for the PID.
# These results were obtained for rabbits with norepinephrine infusion.
# May have to tinker with the constants a bit for other animals and infusants.
'''
u is an infusion rate in micro-gm/(kg x min).
v is fluid loss (mL / sec) for one time interval (hemorrhage, urine, compartment, etc.)
e is the error between target MAP and observed MAP (mmHg).'''

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
def pump_model(error, infusion, delay, pumpNoiseLevel):
    if(dbg11): set_trace()
    # Step thru pump model
    if(useExponentialDampingInPump):
        if (len(error) >= delay):
            exponent = -(error[-delay] / eMax)
            if (exponent < 30 and exponent > -30):
                slowdown = 1 - math.exp(exponent)  # prevents going too fast as error nears 0
                if(dbg): print("slowdown = ",slowdown)
            else:
                slowdown = 1
                print("*** pump exponent too large or too small")
            result = infusion * slowdown * resistance_factor
            return result + pumpNoiseLevel * random.uniform(0, 1)
        else:
            return infusion  + pumpNoiseLevel * random.uniform(0, 1)
    else:
        result = infusion * resistance_factor  + pumpNoiseLevel * random.uniform(0, 1)
        return result
    
# Impulse response, included in the "proportional" component
def impulse (i):
    val = 0
    val1 = 0
    if (i < Tc and i >= L):
        val1 = math.exp(-((i - L) / Tc))
        val = (Ki / Tc) * val1                                 # Kashihara equ (1)
    if(dbg16):
        print(" - impulse: val = ",val,", exp = ",val1,", i-L = ",i - L,", i = ",i,)
    return val

# e1 is the "error" between target MAP and measured MAP.
# Calling sequence:  self.i - Tc, self.i - 1, self.i, e1
def integral (t0, tN, i, e):
    sum = 0
    len_e = len(e)
    if(dbg16): print(" - inte-gral: len_e = ",len_e)
    for j in range(t0, tN - 1):
        if(dbg16): 
            print("Start integral: t0 = ",t0,", tN - 1 = ",tN - 1,", j = ",j,", i-j = ",i - j,", len_e = ",len_e)
            #set_trace() ######### inte-gral loop
        if ( ((i - j) > -1) and (j <= len_e)):
            if(dbg16): print("integral before impulse: e[j] = ",e[j]," del_t = ",del_t)
            sum = sum + impulse(i - j) * e[j] * del_t
            if(dbg16): print("integral sum after impulse: sum = ",sum)
        return sum

def first_deriv(h1, i):   # WE'VE GOT TO FILTER THIS A LOT
    #hd = [0] * 200
    global hd
    derType = 0
    diff = 0
    deriv = 0
    lenh1 = len(h1)
    if(dbg23):
        print("i = ",i,", h1 : ",h1)

    if(i >= 4 and lenh1 > 3):
        hd.append((h1[-4] + h1[-3] + h1[-2] + h1[-1]) / 4)
    elif(i >= 3 and lenh1 > 2):
        hd.append((h1[-3] + h1[-2] + h1[-1]) / 3)
    elif(i >= 2 and lenh1 > 1):
        hd.append((h1[-2] + h1[-1]) / 2)
    elif(i >= 1 and lenh1 > 0):
        hd.append(h1[-1])  # THIS MAY BE JERKY
    else:
        hd.append(0)
    if(dbg3): set_trace() #############       
    len_h = len(hd)
    if ((i < Tc1 - 1) and (i > 2) and (i <= len_h) and (len_h > 2)):
        derType = 2
        diff = hd[i-1] - hd[i-3]
        deriv = diff / 2 * del_t
    elif ((i < Tc1 - 1) and (i > 1) and (i <= len_h) and (len_h > 1)):
        derType = 1
        diff = hd[i-1] - hd[i-2]
        deriv = diff / del_t
    else:
        dertype = 0
        deriv = 0
    if(dbg23):
        print("---first deriv: len h = ",len_h,", i = ",i,", derivative Type = ",derType)
        print(" - diff = ",diff,", hd = ",hd)
        print(" - Tc - 1 = ",Tc1-1,", returning deriv = ",deriv)
    
    if(dbg23): set_trace() #############
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
        print("***getLosses: Unexpected error:", sys.exc_info()[0])
        print(print_frame())
        raise
    finally:
        inp.close()
        print("-getLosses: Fluids test opened normally.  398")
        
#==============================
class PID:
    #PID(i,j1,k,m,"bleeding.csv",row,"controls.csv",finalMAP,VB=2500,K0=2.1,alpha=0.5)
    def __init__(self,i,j,k,m,losses_f,MAP,control_f,finalMAP,VB=2500,K0=2.1,alpha=0.5):
        self.VB = VB
        self.K0 = K0
        self.alpha = alpha
        self.losses_f = losses_f
        self.MAP  = MAP            # this is usually a row of data from a batch
        self.control_f = control_f
        self.finalMAP = finalMAP
        self.control = ""
        self.total_outputs = 0
        self.count = 0  #
        self.i = i  # time step number (now row number in this test)
        self.j = j  # controller param number = 0, 1, 2 (not used)
        self.k = k  # controller now reading row k
        self.m = m  # at row element m

    def run_model(self):
        global VB0
        global VB1
        global VB2
        global VB3
        global Um1
        global Um2
        global Vm1
        global Vm2
        
        #if(dbg0): set_trace() ############## run model entry
        if(self.i > 2):
            sys.exit() ###############
        try:
            A = 0 # It's easier to debug large formulas in pieces
            B = 0
            C = 0
            D = 0
            E = 0
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
                if(dbg0):
                    print(" ")
                    print("--- Model: preparing model at i = ",self.i)
                    #print("Um2 = ",Um2,", Um1 = ",Um1)
                    #print("Vm2 = ",Vm2,", Vm1 = ",Vm1)
                    
                if(dbg6): set_trace() ################## 
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
        except OSError as err:
            print("OS error : {0}".format(err))
            print(print_frame())
        except ValueError:
            print("Could not convert data to an integer.")
            print(print_frame())
        except IOError:
            print ("File error with : ", baseBatch)
            print(print_frame())
        except:
            print("Unexpected error  run model :", sys.exc_info()[0])
            print(print_frame())
        finally:
            if(dbg0): print("Finished run model")
            return
                
    def advance_PID(self):
        A1 = 0 # It's easier to debug large formulas in pieces
        B1 = 0
        C1 = 0
        D1 = 0
        E1 = 0
        
        ma = "0"
        ua = "0"
        va = "0"
        ea = "0"

        if (dbg16 and self.m == 1): 
            print("---------------advance_PID: at entry, i = ",self.i,", m = ",self.m)
            #set_trace() ############## advance PID entry CLICK ONE MORE TIME
        try:
            out_f = open(self.control_f,"a",newline='',encoding="utf8")
            csv_out = csv.writer(out_f, delimiter=',')
            controller = " "
            header = "  loop,   infusion,   loss,    error"
            out_f.write(header)

            # start the filter as best can for first time step
            if (self.m != None and self.m < 1):
                e1.append(initialError)
                v1.append(initialLoss)
                
                # Here is the pump simulation
                if (Cp == 1):
                    result = pump_model(e1, initialInfusion, delay, pumpNoiseLevel)
                else:
                    result = initialInfusion
                
                u1.append(result)   # initial infusion. (This accumulates total)
                controller = str(self.i) + "," + str(self.m) + "," + str(u1[0]) + "," + str(v1[0]) + "," + str(e1[0])
                out_f.write(controller)
                self.finalMAP = initialMAP
            
            # subsequent time steps
            elif(self.m != None):
                # -------- PID control function
                if (dbg16): print("--advance_PID: at i = ",self.i,", m = ",self.m)
                # (Easier debugging to break up formula into these 4 parts)
                if(dbg16): set_trace() ####################### start details of advance PID

                # proportional term
                B1 = 0
                val3 = 0
                lengE = len(e1)
                if (self.m > 0 & self.m - 1 < lengE):
                    val3 = impulse (e1[self.m - 1])
                    B1 = Kp * val3
                else:
                    B1 = 0
                    if(dbg): 
                        print("--- self.m - 1 getting ahead of e1 at B1")
                        print(print_frame())
                if(dbg16): 
                    print("proportional, B1 = ",B1,", Kp = ",Kp,", impulse = ",val3,", m = ",self.m,", lengE = ",lengE)
                
                # integration term
                C1 = 0
                if (self.m >= Tc):  # inte-gral goes back T - 1 steps
                    ## def integral (t0, tN, i, e):
                    C1 = Ki * integral(self.m - Tc, self.m - 1, self.m, e1)
                if(dbg16 and self.m >= Tc):
                    print("integration: C1 = ",C1,", Ki = ",Ki,", e 1 = ",e1,", Tc = ",Tc,", m = ",self.m)
                    #set_trace() ######### after integral C1, Ki, e1, Tc
                    
                D1 = 0
                # differential term
                if (self.m > 1):
                    deriv = first_deriv (e1, self.m - 1)
                    if (abs(deriv) <= deriv_limit):
                        D1 = Kd + deriv
                    else:
                        if(dbg): print("-- advance_PID: Had to truncate deriv ",deriv)
                        if (deriv > deriv_limit):
                            D1 = Kd * 0.5
                        elif (deriv < 0.5):
                            D1 = - Kd * 0.5
                if(dbg16 and self.m > 1):
                    print("differential: D1 = ",D1,", Kd = ",Kd,", e 1 = ",e1,", deriv = ",deriv,", limit = ",deriv_limit,", m = ",self.m)
                    
                infusion = B1 + C1 + D1   # increment of infusion recommended
                
                if (Cp == 1):
                    pump_out = pump_model(e1, infusion, delay, pumpNoiseLevel)  # pump transfer function
                    # pump may decrease the infusion rate if it is too fast, depending on e1
                else:
                    pump_out = infusion
                if(dbg16): print("pump out = ",pump_out,", m = ",self.m)
                    
                lng_u1 = len(u1)
                if (self.m - 1 < lng_u1):
                    tot_infusion = pump_out +  u1[self.m - 1]
                else:
                    if(dbg): 
                        print("--- self.m - 1 getting ahead of u1 at tot_infusion")
                        print(print_frame())
                    tot_infusion = pump_out
                
                u1.append(tot_infusion) # total infusions up to and including m-th time step
                if(dbg16): print("tot infusion = ",tot_infusion,", m = ",self.m)
    
                lnMAP = len(self.MAP)
                if (self.m < lnMAP):
                    er = float(self.MAP[self.m]) - self.finalMAP
                    e1.append (er)
                else:
                    er = initialError
                    e1.append (er)
                    if(dbg): 
                        print("--- self.m - 1 getting ahead of u1")
                        print(print_frame())
                    
                # Here is the combination: total infusion - fluid loss (if Cm > 0)
                self.finalMAP = Ci * tot_infusion  -  Cm * VB0
                if(dbg16): print("finalMAP = ",self.finalMAP,", Ci = ",Ci,", Cm = ",Cm,", m = ",self.m)
    
                ln_u = len(u1)
                ln_v = len(v1) # be careful, this array may not be used if Cm == 0
                ln_e = len(e1)
                
                # Controller output parameters
                if (self.m < ln_u and self.m < ln_e):
                    ma = str(self.m)
                    ua = str(u1[self.m])
                    if (Cm > 0):
                        va = str(v1[self.m])
                    else:
                        va = "None"
                    ea = str(e1[self.m])
                else:
                    ma = "0"
                    ua = "0"
                    if (Cm > 0):
                        va = "0"
                    else:
                        va = "Wrong"
                    ea = "0"
                    if(dbg2):
                        print("-advance_PID: ln_u = ",ln_u,", ln_e = ",ln_e,". Can't compute self.control")
                if(dbg16):
                        print("advance_PID: ln_u = ",ln_u,", ln_v = ",ln_v,", ln_e = ",ln_e,", m = ",self.m)
                        #set_trace() ############ 

                if(dbg13):
                    print(" ")
                    print("---advance_PID  A1 = ",A1,", B1 = ",B1,", C1 = ",C1,", D1 = ",D1)
                    print(" - e 1 is : ",e1[-self.m:-1])
                    print(" - infusion : ",infusion)
                    print(" - pump_out  : ",pump_out)
                    print(" - u 1 is now : ",u1[-self.m:-1])
                    if(fluidLoss == True & ln_v > self.m):
                        print(" - v 1 is now : ",v1[-self.m:-1])
                    print(" - m = ",ma,", u1 = ",ua,", v1 = ",va,", e1 = ",ea)
                if(dbg13): set_trace() ######################## end of advance PID
                    
                self.control = ma + ", " + ua + ", " + va + ", " + ea   # keeps record of control

                # Note: this is the file needed in a trained, closed-loop control.
                # Be sure to delete this file before starting a closed-loop run.
                out_f.write(self.control)
                
                self.total_outputs += 1

                #print("-advance_PID: updating i to ",self.i)
                if(dbg16): print("-advance_PID returning error ",e1[-self.m:-1])
            else:
                print("*** self.i  is a None!")
        except OSError as err:
            print("OS error: {0}".format(err))
            print(print_frame())
            raise
        except ValueError:
            print("Could not convert data to an integer.")
            print(print_frame())
            raise
        except IOError:
            print ("File error with :", self.control_f)
            print(print_frame())
            raise
        except:
            print("Error in advance PID: ", sys.exc_info()[0])
            print(print_frame())
            raise
        finally:
            out_f.close()
            if(dbg1): print('PID_incremental_controller iteration completed normally.')
            return
        # End advance_PID
        
        def get_average_error(self):
            leng = len(e1)
            for i in range(0,leng):
                e_sum = e1[i]
            if (leng > 0):
                return e_sum / leng
            else:
                print("*** error list has 0 length")
                return 0
        
        def get_control(self):
            return self.control
#-- End PID controller class

#------------------------------------
# Gradient descent process
# acc and acc_prev are using NN's accuracies in this version
def step_j_th_parm (acc, acc_prev, j1, K, epsilon):
    if(dbg12): set_trace() ############## entry step_j_th_parm
    try:
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
        # change, then, is the Neutonian step.  Very crude.  Instead we step
        # with an arithmetic progression which begins more slowly.
        
        if (change > K[j1] * fraction):
            change = K[j1] * fraction

        Klast = K[j1]
        # check the derivation in code document for this formula
        if(acc_prev != 0):
            K[j1] = Klast + K[j1] * change / acc_prev
        
        if(dbg4):
            print("--- step_j_th_parm:")
            print(" - K[",j1,"] = ",K[j1],", Klast = ",Klast)
            print(" - change = ",change,", acc = ",acc,", accPrev = ",accPrev)
            set_trace() ##################

        # Note, scores and accuracy may continue to improve even if we keep the
        #s ame batch data randomized somewhat.
        # Note also that K does accumulate over j, so after 3 inner loops, all
        # coordinaes of K are updated.
    except OSError as err:
        print("OS error : {0}".format(err))
        print(print_frame())
        raise
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        raise
    except IOError:
        print ("File error with : ", baseBatch)
        print(print_frame())
        raise
    except:
        print("Unexpected error step_j_th_parm:", sys.exc_info()[0])
        print(print_frame())
        raise
    finally:
        if(dbg2): print("ending step-j-th_parm")
    return K
# end step_j_th_parm
#------------------------------------

# Here is where the neural network gets to "vote" on keeping settings of K.
# acc is NN's curent accuracy; acc_last is last accuracy achieved; acc_prev next to last.
def improvement (acc, acc_last, acc_prev, Gaussian_noise):
    # After running the last base batch, 
    # THIS NEEDS TO ACCUMULATE A GAUSSIAN NOISE OUTSIDE
    difference1 = abs(acc - acc_last)
    difference2 = abs(acc_last - acc_prev)
    if (difference1 < percentile * Gaussian_noise and difference2 < percentile * Gaussian_noise):
        if(dbg2 or dbg18): 
            print("no improvement, discard last batch")
            print("differnece1 = ",difference1,", difference2 = ",difference2)
            if(dbg18): set_trace() #################
        return 0   # no improvement, discard last batch
    else:
        if(dbg2 or dbg18): 
            print("improvement -- move last batch to new baseBatch")
            print("differnece1 = ",difference1,", difference2 = ",difference2)
        return 1   # improvement -- move last batch to new baseBatch


#========================= MAIN =======================
if (train):
    '''It may help to have an outline. In the training loops,
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
    '''
    K = [Kp, Ki, Kd]  # initial parameters for PID and NN
    Gaussian = 0.0    # Kalman estimate of the distribution of NN accuracies
    trainingRows = []
    finalMAP = initialMAP
    
    #Here we  read the training file, of 100 rows, in "batches" of 20,
    #giving dataSize / 20  training cycles.
    training = []
    # Read entire training file, selecting batches from it.
    # Actually there is only one row, of length, dataSize.
    success = 1
    try:
        dataName = "RNN_training.csv"
        with open(dataName, 'r', newline='', encoding="utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            count = 0
            for row in csv_in:
                trainingRows.append(row)
                rowS = len(row)
                count += 1
                for i2 in range(0,rowS):
                    training.append(row[i2])
        if(dbg7): print("MAIN: input fileSize = ",count)
        dataSize = count
    except OSError as err:
        print("OS error: {0}".format(err))
        print(print_frame())
        success = 2
        raise 
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 3
        raise 
    except IOError:
        print ("File error with :", dataName)
        print(print_frame())
        success = 4
        raise 
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        print(print_frame())
        success = 5
        raise 
    finally:
        inp.close()
        if(dbg7):
            print("MAIN: data file entry, success = ",success)
        if (success != 1):
            set_trace() ###########  BAD DATA ENTRY
        
    # Note: instead of this n1 loop, we could run through a list of training batchs
    # from other sources such as NoisySinusoids or Glitches.
    for n1 in range(0, trainingCycles):   #  Run training sets through PID-NN
        batchRows = []
        # Peel out the section of training file from M*n to M*(n+1) as a baseBatch.
        # In n-th loop peel off the batch from M * n to M * (n+1)
        start = n1 * timeSteps
        endit = (n1+1) * timeSteps
        stopit = 0
        begin = 0
        count = 0
        print("--- start = ",start,", endit = ",endit)

        for peel in range(0, M):
            begin = start + peel
            stopit = endit + peel
            batchRows.append(training[begin:stopit])
        if(dbg3): set_trace() ###################  
        baseBatch = fileRoot+str(n1)+".csv"   # base batch for n1-th training cycle
        # write baseBatch to be read later and parms varied in "deltaBatch"es
        try:
            with open(baseBatch, 'w', newline ='', encoding = "utf8") as outp:
                csv_out = csv.writer(outp, delimiter=',')
                for k in range(0, M):
                    csv_out.writerow(batchRows[k])
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
        finally:
            outp.close()
            if(dbg7): print("MAIN: Successfuly wrote base batch ",baseBatch)
               
        for i in range(0, batchSize):   #  Run varied batches through PID-NN
            if(dbg3): set_trace() ########################
            if (i == 0):
                baseBatch = fileRoot+str(n1)+".csv"
                newBatch = baseBatch
            else:
                baseBatch = newBaseBatch   # returned from end of this loop
                newBatch = baseBatch
                
            file = open(newBatch)
            lengF = len(file.readlines())
            if (batchSize != lengF):
                    print("*** newBatch size wrong = ",lengF," at i = ",i,", n = ",n1)
                    print(print_frame())
                    break
            if(dbg7): 
                print("--- At beginning looping over baseBatch = ",baseBatch)
                print("    batchSize = ",batchSize)
            
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

                success = create_batch_i_j (n1, i, j1, baseBatch, deltaBatch, delta_K, noise)
                if(dbg7): 
                    print("-Main: create_batch_i_j baseBatch = ",baseBatch,", deltaBatch = ",deltaBatch,", success = ",success)
                if (success != 1):
                    break
                if(dbg7): set_trace() ################## Passed the early break in j1 - loop
                if(dbg): 
                    print("- j1 loop new : ",deltaBatch," at i = ",i,", j = ",j1,", n = ",n1)
                    print("  batchSize = ",batchSize)
                # Read the new "batch" and run PID on each measure in each row,
                # computing a score from the final MAP reached by the PID.
                if(dbg7): set_trace() #######################
                try:
                    rows = []
                    count = 0
                    with open(deltaBatch, 'r', newline='', encoding="utf8") as inp:
                        csv_in = csv.reader(inp, delimiter=',')
                        for row in csv_in:
                            rows.append(row)
                            count += 1
                    if (batchSize != count):
                        print("*** MAIN: Wrong deltaBatch file size ",count)
                        print(print_frame())
                        if(dbg7): set_trace() #######################

                    for k in range(0,batchSize):
                        # For each row of batch_i_j, advance PID across simulated measurements
                        row = rows[k]
                        row_size = len(row)
                        if (row_size > rowSize):
                            if(dbg): print("---Some long rows in k-loop: ",row_size)
                            row_size = rowSize
                            if(dbg): print("---Resetting row_size = : ",row_size)
                        rowLength = len(row)
                        if(dbg1): print("MAIN: before PID, rowLength = ",rowLength,", timeSteps = ",timeSteps)

                        mp = 0

                        if(dbg12): set_trace() ###################### step through
                        # Run PID et al. across the k-th row
                        for m in range(0,rowSize):
                            owr = [ast.literal_eval(ep) for ep in row]
                            if(dbg): print("MAIN: Inner loop m ",m," entering PID")
                            pid = PID (i,j1,k,m,"bleeding.csv",owr,"controls.csv",finalMAP,VB=2500,K0=2.1,alpha=0.5)
                            pid.run_model()
                            pid.advance_PID()   # value reached at end of row test
                            mp = pid.finalMAP
                            if(dbg): print("PID's finalMAP = ",mp)
                        if(dbg12): set_trace() ################################ end m-loop

                        scores[k] = min( 0.99, 1 - abs((mp - targetMAP) / targetMAP))
                        ave_score += scores[k]
                    if(dbg2):
                        print("rowSize = ",rowSize,", row_size = ",row_size," at j1 = ",j1,", k = ",k)
                        set_trace() #################### row_size, rowSize
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
                    print("Unexpected error in maain:", sys.exc_info()[0])
                    print(print_frame())
                    break
                    raise
                finally:
                    inp.close()
                    if(dbg2):
                        print("deltaBatch ",deltaBatch," i = ",i,", j = ",j1,", scores = ",scores)
                        print("  K[",j1,"] = ",K[j1])
                #--end j1 loop
                

                ave_score = ave_score / 3
                
                # sync up named parameters with K array
                Kp = K[0]
                Ki = K[1]
                Kd = K[2]

                if(dbg2 or dbg18):
                    print("Finished j1 and inner loops.  Now K = ",K,", ave_score = ",ave_score)

                # create new batch for NN based on best K's incremented in gradient directions
                newBatch = fileRoot+str(i+1)+".csv"
                create_batch_i(n1, i+1, baseBatch, newBatch, K, scores, noise)
                # Next iteration of i-loop should pick the new baseBatch up.
                
                if(dbg3 or dbg18): set_trace() #####################
                # step forward for accuracy parameters
                acc_prev = acc_last
                acc_last = accuracy
                accuracy = ave_score
                
                # THIS accuracy SETTING NEEDS TO BE CHANGED FOR EACH CONFIGURATION:
                # Accuracy can be set be either controller or neural net.  If
                # neural net is not configured, the default will be the aveScore as below:

                if(dbg or dbg18): 
                    print("--- End of Loops i = ",i,", n = ",n1)
                    print("   accuracy = ",accuracy)
                    print("   acc_last = ",acc_last)
                    print("   acc_prev = ",acc_prev)
                    print("   ave_score = ",ave_score)
                    print(" ")

                # Re-train NN on the revised i-th batch
                if(dbg3): set_trace() ############ check numbers above
                if(dbg): print("Near NN, newBatch = ",newBatch)

                file = open(newBatch)
                lengF = len(file.readlines())
                #if(dbg8): set_trace() ###################### lengF   starts NN 
                if (batchSize < lengF):
                    print("*** newBatch is short : ",lengF," Not running NN")
                else:
                    if(cutNeurons):
                        loss = 0
                        rmse = 0
                    else:
                        print("--- Starting NeuralNet --- ")

                        loss, rmse = NeuralNet(training)
                        accuracy = rmse   #in this case

                        if(dbg8): set_trace() ###################### ends NN 
                        '''Loss function measures the difference between the predicted label and the ground truth label. 
                        E.g., square loss is  L(y^,y)=(y^−y)^2 , hinge loss is  L(y^,y)=max{0,1 − y^ x y} ...'''
                        if (dbg):
                            print("--NN RESULTS:")
                            print("    loss = ",loss,", rmse = ",rmse)
                            print("    After NN call Kp = ",K[0],", Ki = ",K[1],", Kd = ",K[2])
                            print(print_frame())
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
            
            if(dbg5): set_trace()  #############################
            if (improve == 0): 
                print("-- No significant impovement in accuracy. discarding newBatch")
                file = open(baseBatch)
                lengF = len(file.readlines())
                if (batchSize != lengF):
                    print("*** baseBatch is wrong length : ",lengF, " Breaking i-loop")
                    print(print_frame())
                    break
                else:
                    newBaseBatch = baseBatch
            else: # replace old baseBatch with last batch from PID
                file = open(newBatch)
                lengF = len(file.readlines())
                if (batchSize != lengF):
                    print("*** newBatch is wrong length : ",lengF, " Breaking i-loop")
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
    Read line from RNN_training.csv data file;
    run PID  which may input a fluid losses file as an option;
    run pump simulator (basically a linear transfer function with lag);
    write line to control file;
    run "model" of body responses;
    compute error between body response and desired target MAP;
        
    As prerequisite, this depends on the parameters, K, being adequately trained
    by the training loop.  Keeping its trained node weights, the NN now serves
    to make a small correction to the PID output.
    
    Run NN to make a correction to PID output (trained separately for that as above);
    use accuracy to adjust step size (and other search parameter as needed).'''
    ABC = 0

