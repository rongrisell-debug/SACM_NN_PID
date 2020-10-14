#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Driver_Kramer_Tables   -  Version 1.4
# THIS IS NOW INTENDED TO BE A MORE GENERAL DRIVER
#### Bug at the end of the batch loop - trying Driver_RNN_PID_6_3 routines

# Debugging and flow control
dbg =  False  # top level results
dbg0 = False  # body response model
dbg1 = False  # more detail in print_frame
dbg2 = False  # prints and set_traces in controller and MAIN
dbg3 = False  # traces in n, l, k, m loops
dbg4 = False  # batch wrap-up process
dbg5 = False  # create batch_i_j
dbg6 = False  # trap index error near PID
dbg7 = False  # create batch methods
dbg8 = False  # inner-most m1-loop
dbg9 = False  # set traces on pump model
dbg10 = False # trace pump model
dbg11 = False # track newBatch after create batch_i
dbg15 = False # strange bugs in loops
dbg18 = False # improvement behavior
dbg19 = False # trap error after pump went negative
dbg20 = False # Table parms
createCount = 0    # limits error messages

# flow control
fluidLoss = False  # True to include fluid loss model
train = True    # train versus closed loop runs
test = False
preview = True
summary = True
sine = True
rand = False
'''
Here we use batches differently than with PID, compatile with the RNN testing.
Instead of inptting a file with many short sequences,
we create one relatively long sequence and peel-off rows rows
in a sliding-window fashion.  See the "peel" below.

Batches have 20 rows with 50 entries each to test the controller,
the first 50 - Nparms of each row are measured MAP (mean arterial pressure) values
and the next 3 of which are the Table parameters Ki, Kd, and Kp; and
the last is the controller's performance score in approaching a target MAP, or "score".

The controller ouptputs a new base batch.csv with each i-th loop,
which after stepping slightly in each of the dimensions of the parameers,
is then fed back into the controller for another run.

We show that these feedback connections
will not run the system into instability if parameters are held witn a
certain Lipshitz constraint.   This will prevent blow-up or zeroing
and walk-away by the essentially Newtonian stepping of the controller parameters.
This will have access to an external "true" but that itself is noisy and
there is in addition an unknown fluid loss. 

Details
Outer_loop:
   runs the dataset through the PID,
   computing a score for each row depending on degree of success
   of Table in reaching a target MAP.  In the combined modules, 
   Driver_PID_NN, the neural net 
   runs batches of Table results seeking to improve the Table's
   internal parameters.
An inner j1-loop determines a gradient, ...

starting with the i1-th batch ??? ???
then j1 varying each parm in K parms, one at at time, by a slight amount as deltaBatches
to obtain "coordinates" of the gradient.  ===== EXPLANATION NEEDS WORK

See Driver_RNN_setup, which creates a dataset.
This verion does not use a fluid losses file.

Conventions:
n1 - cycles the Table through batches
i1 - indexes an intermediate loop which runs through a batch of MAP profiles ???
j1 - runs over the parameter Kparm numbers, 1,...Nparms, varying each as batch scored
m1 - the element of a batch row appended up to timeSteps = batchSize - NParms

We start with a simple scenario: 
(1) measured MAP remains about target MAP, +/- 18 mmHg

Remember, there are other files ready to be used as batches, such as:
  keras_3parms_glitches
  keras_3parms_sinusoids

Notes:
the outputs are all appended to controls.csv for a record of progress.
There is a lag, L, built in to the proportional component of the controller.  ???
The pump model depends on a delay and a resistance_factor usually near 1.
The delay is assumed to be 1/10 second, i.e. one time step.

u1 is accumulated infusion in micro-gm/(kg x min) x number of pump cycles.
v1 is fluid loss (mL / sec) for one time interval (hemorrhage, urine, compartment, etc.)
e1 is the error between target MAP and observed MAP (mmHg).
'''
import sys
import csv
import ast
import math
import random
import inspect
import numpy as np
from numpy import array
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from IPython.core.debugger import set_trace
from IPython.core.debugger import Tracer; debug_here = Tracer()

# Pump simulator parameters
delay = 1   # assumed to be about a time step for now (10 sec.).
pumpNoiseLevel = 0.1
resistanceFactor = 1.05 # degree to which the pump can infuse.
useExponentialDampingInPump = True
eMax = 5
# this factor also includes the linear transfer function constant in pump model.
Umax = 10  # maximum infusion rate (see above)  ???

# model parameters
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

# controller parameters
delta_K = []
deriv_limit = 0.5  # limits the derivative estimate in derivative
ave_score = 0
num_scores = 0
noise = 0.1       # noise * random  added to each coordinate in create batch_i_j
epsilon = 0.05    # wiggle epsilon * random(0,1) if on a flat splot
fraction = 0.01   # maximum step limited by fraction of K parms.  May need to be larger
#                    for faster convergence of the Table parameters, K.
out_f = ""        # output file names
accuracy = 0      # current accuracy of current batch, based upon scores

# Driver parameters
fileRoot = "batch_"  # prefix for batch file names
dataName = "RNN_training.csv"

# Hyperparameters
dataSize = 200    # length of data, which is now only one row
##hidden_dim = int(dataSize / 2)  # For NN Driver
batchSize = 20    # batches same as baseBatches for the controller
modulator = 0.01  # moderates the proportional time step in step j_th_parm
trainingCycles = int(dataSize / batchSize)    # n1-loop range.

# Simulation parameters
seqLength = 50    # includes the Kparms at the end
Nparms = 6        # dimension of controller's parameters
timeSteps = seqLength - Nparms    # we add Nparms parameters to the test sequence
scores = [0 for i in range(batchSize)]  # given by Tablefor how close to target MAP it reaches
noise = 5         # noise level with uniform distribution, i.e. white noise
quartile = 0.2    # null hypothesis for random white noise.  Lower is better.
percentile = 0.0001 # for null hypothesis for Gaussian noise in NN accuracies
#P = percentile    # short name for percentile

targetMAP = 65.0  # desireable mean arterial blood pressure in pigs, 80 for humans
initialMAP = 60.0  # This is for a typical hypovolemic (hemorrhaged) MAP,
                   # in mmHg typical for hypovolumic swine
initialError = 5.0
initialLoss = 0.0  # this gets the fluid loss model recursion started.
initialInfusion = 30.0
initialBleed = 7.0

# Parameters specific to the Table in the controller
Kparms = [40.,50.,60.,70.,80.,90.]  # level parameters for Table
Del0 = 3          # lowwest recommended pump step by Table
Del1 = (targetMAP - Kparms[0]) / Del0
Del2 = (targetMAP - Kparms[1]) / Del0
Del3 = (targetMAP - Kparms[2]) / Del0  # Note, steps go negative above targetMAP
Del4 = (targetMAP - Kparms[3]) / Del0  # Some day we might consider a future pump which
Del5 = (targetMAP - Kparms[4]) / Del0  # actually sucks blood out !!
Del6 = (targetMAP - Kparms[5]) / Del0
Dparms = [Del0, Del1, Del2, Del3, Del4, Del5, Del6] # recommended delta's to the pump

Cm = 0    # Set Cm = 0 to eliminate bleeding component entirely from Tablecontrol.
Cp1 = 1    # Set Cp = 0 to eliminate the pump model.
Ci = 1
#    Set Cm = 0 and Cl= 1 both to eliminate the fluid loss file input and
#    there could be a built-in updating of the fluid loss "v" for steady bleed.
Mult = 1     # multiplier of basic time steps, 10 steps over 1 second
TI = Mult * 10   # total simulation time
##T = 5    # time constant in the step response of "proportional".
del_t = 1     # discrete time difference or sampling interval (10  sec)
step = 1 # initializes the constrained gradient method in create batch_i
         # currently we are not using this but instead, create a new batch which
         # computes a controller parameter gradient modulated by
         # the accuracy achieved for the current batch.

# Note: Set Cm = 0 to eliminate bleeding, i.e. fluid response VB0 from error:
#      e1.append(targetMAP - Ci * infusion  +  Cm * VB0)

# The following parametrize the body response function due to Hahn et al. (slide 6)
VB = 2500     # estimated blood volume for a 43 kg swine
K0 = 2.1
alpha = 0.5   # fluid transfer rate between vascular and extravascular body 
gain = 1      # feedback gain if used
##alpha = 2.5   # original model of fluid transfer rate between vascular and extravascular body 

# Variance estimate at end of n1-loop
W1 = 1
W2 = 0
Gaussian = 0
countImprove = 0

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
##-- Prepare for (revised) base batch in i-th direction (a "deltaBatch").
# Note, we keep previous batches for later optimization of hyperparameters.
#-----------------------------------------
def create_batch_i_j (i1, j1, bBatch, dBatch, Kparms, noise):
    # These are the Inner-loop's batches where Tableparameters are
    # varied one at a time.  The best gradient steps in each of
    # the Nparm Table parameters are extracted and recorded in the
    # next base_batch for the NN.
    success = 1
    rows = []
    
    #if(dbg15): set_trace() ################### 
    # create_batch_i_j Read baseBatch here as rows and row
    try:
        with open(bBatch, 'r', newline='', encoding="utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            count = 0
            for line in csv_in:
                rows.append(line)
                count += 1    # IT WOULD BE JUST AS WELL TO USE len(rows)
        # Checking
        lengRow = len(rows[0])
        if(dbg7): print("row leng = ",lengRow)
        #if(dbg9): set_trace() ################ rows[0], count, leng, seqLength
        if (lengRow != seqLength):
            print("*** create_batch_i_j leng row = ",lengRow,", not seqLength = ",seqLength)
            success = 0
            print(print_frame())
        if(dbg7): set_trace() ########################### look: row, lengRow, seqLength, count
        # Here, leng = 50 but
        # When writing, create_batch_i_j: leng =  30 , timeSteps =  44 ???
        # if(dbg9): set_trace() ######################  count, leng, j1
        if(count != batchSize):
            print("*** create_batch_i_j: wrong batchSize ",count," at i = ",i1)
            success = 0
            print(print_frame())
        else:
            if(dbg2): print("create_batch_i_j: count = ",count," at i = ",i1,", j = ",j1)
    except OSError as err:
        print("OS error: {0} ".format(err))
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
        print("Unexpected error:", sys.exc_info()[0], sys.exc_info()[1])
        print(print_frame())
        success = 0
    finally:
        inp.close()
        if (success == 0):
            return 0  # handle this in caller
        
    #if(dbg15): set_trace() ######################
    success = 1
    try:
        with open(dBatch, 'w', newline ='', encoding = "utf8") as outp:
            # Write intermediate batch i_j with variation in the j-coordinate
            csv_out = csv.writer(outp, delimiter=',')
            createCount = 0
            for k1 in range(0, batchSize):
                row = rows[k1]
                if(dbg7): set_trace() ########################### row to be APPENDED
                rowLength = len(row)
                #if(dbg4): set_trace() ###################### list index out  of range: rowLength
                if (rowLength != seqLength):
                    createCount += 1
                    if (createCount < 2):
                        if(dbg4): print("create_batch_i_j: rowLength = ",rowLength,", not seqLength = ",seqLength)
                meas = []
                parm = 0
                if(dbg5):
                    print("rowLength = ",rowLength)
                    set_trace() ######### create_batch_i_j, check rowLength.
                for m3 in range(0,timeSteps):
                    val = row[m3]
                    lenR = len(row)
                    if (timeSteps > lenR):
                        print("*** timeSteps = ",timeSteps,", lenR = ",lenR,", i = ",i1,", j = ",j1,", n = ",n1)
                    meas.append(val)
                # Vary the j1-th parm
                if(dbg5): set_trace() ########################### row pealed to timeSteps

                if (j1 == 0):
                    # Note: this does not alter the original Kparams until create batch_i advances all
                    parm = Kparms[0] + step   # this step is decreased as good value approached
                    meas.append(parm)
                    meas.append(Kparms[1])
                    meas.append(Kparms[2])
                    meas.append(Kparms[3])
                    meas.append(Kparms[4])
                    meas.append(Kparms[5])
                elif (j1 == 1):
                    meas.append(Kparms[0])
                    parm = Kparms[1] + step
                    meas.append(parm)
                    meas.append(Kparms[2])
                    meas.append(Kparms[3])
                    meas.append(Kparms[4])
                    meas.append(Kparms[5])
                elif (j1 == 2):
                    meas.append(Kparms[0])
                    meas.append(Kparms[1])
                    parm = Kparms[2] + step
                    meas.append(parm)
                    meas.append(Kparms[3])
                    meas.append(Kparms[4])
                    meas.append(Kparms[5]) 
                elif (j1 == 3):
                    meas.append(Kparms[0])
                    meas.append(Kparms[1])
                    meas.append(Kparms[2])
                    parm = Kparms[3] + step
                    meas.append(parm)
                    meas.append(Kparms[4])
                    meas.append(Kparms[5])
                elif (j1 == 4):
                    meas.append(Kparms[1])
                    meas.append(Kparms[2])
                    meas.append(Kparms[3])
                    parm = Kparms[4] + step
                    meas.append(parm)
                    meas.append(Kparms[5])
                elif (j1 == 5 ):
                    meas.append(Kparms[1])
                    meas.append(Kparms[2])
                    meas.append(Kparms[3])
                    meas.append(Kparms[4])
                    parm = Kparms[5] + step
                    meas.append(parm)
                else:
                    if(dbg):
                        print("*** Wrong j = ",j1,", at i = ",i1)
                    success = 0
                csv_out.writerow(meas)
        if(dbg5): print("create_batch_i_j finished: i = ",i1,", j = ",j1," success = ",success)
        if(dbg5):
            print("rows to be written i create_batch_i_j, i = ",i1,", j = ",j1," success = ",success)
            print(meas)
            set_trace() ################# rows to be written, j1, i1
    except OSError as err:
        print("OS error: {0}".format(err))
        print(print_frame())
        success = 0
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 2
    except IOError:
        print ("File error with :", deltaBatch)
        print(print_frame())
        success = 3
    except:
        print("Unexpected error:", sys.exc_info()[0], sys.exc_info()[1])
        print(print_frame())
        success = 4
    finally:
        outp.close()
        if (dbg):
            print("create_batch_i_j completed success = ",success)
    return success    # handle exception in caller
#- end create_batch_i_j
#--------------------------------------

def create_batch_i (n1, i1, bBatch, nBatch, Kparms, scores, noise):
    # These are the i or i1-loop's batches where Tableparameters have
    # been optimized.  The best gradient steps in each of
    # the N Table parameters have been extracted and recorded in Kparms
    # for the next baseBatch for the controller.
    # Note: We include scores in the new batch for the controller to use.
    # These scores accelerate ir decellerate the gradent step.
    success = 1
    rows = []

    #if(dbg15): set_trace() ################## ABOUT TO READ IN create_batch_i
    success = 1
    try:
        count = 0
        with open(bBatch, 'r', newline='', encoding="utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            for row in csv_in:
                rows.append(row)
                count += 1
        if(count != batchSize):
            print("*** create_batch_i: reading, wrong batch size = ",count," at i = ",i1)
            print(print_frame())
            success = 0         
    except OSError as err:
        print("OS error: {0}".format(err))
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
        print("Unexpected error: ", sys.exc_info()[0], sys.exc_info()[1])
        print(print_frame())
        success = 5
    finally:
        inp.close()
        if (success != 1):
            print("*** Trouble in create_batch_i reading -- handling in caller")
            return success
        
    #if(dbg15): set_trace() ###################### 
    success = 1
    try:
        with open(nBatch, 'w', newline ='', encoding = "utf8") as outp:
            csv_out = csv.writer(outp, delimiter=',')
            for k2 in range(0, batchSize):
                meas = []
                # Noise-up measurements at preceeding time steps -- WILL THIS ACCUMULATE TOO MUCH???
                for m2 in range(0,timeSteps):
                    val = float(rows[k2][m2]) + noise * random.uniform(0, 1)
                    meas.append(val)
                for m2 in range(0,Nparms):
                    val = Kparms[m2] + noise * random.uniform(0, 1)
                    meas.append(val)
                csv_out.writerow(meas)
        lengM = len(meas)
        if(dbg11): 
            print("create batch_i, writing, lengM = ",lengM,", i = ",i1,", j = ",j1)
            print("   meas: ")
            print(meas)
            set_trace() ################# create_batch_i writing nBatch ############### check lengM
    except OSError as err:
        print("OS error : {0}".format(err))
        print(print_frame())
        success = 6
    except ValueError:
        print("Could not convert data to an integer.")
        print(print_frame())
        success = 7
    except IOError:
        print ("File error with :", nBatch)
        print(print_frame())
        success = 8
    except:
        print("Unexpected error:", sys.exc_info()[0], sys.exc_info()[1])
        print(print_frame())
        success = 9
    finally:
        outp.close()
        if (success != 1):
            print("*** Trouble in create_batch_i writing -- handling in caller")
        return success
#- end create_batch_i

#===================================
# This class is adapted from PID_test -- an incremental PID controller
# These results were obtained for sheep.
# May have to tinker with the constants a bit for other animals and infusants.
'''
u is an infusion rate in micro-gm/(kg x min).
v is fluid loss (mL / sec) for one time interval (hemorrhage, urine, compartment, etc.)
e is the error between target MAP and observed MAP (mmHg).'''

# run-model is a time-stepped linear body response model to infusion and fluid loss.
# It follows: Bighamian et al. A Lumped-Parameter Blood Volume Model.

Tsample = 5  # is sampling interval   (NOT USED)
# VB is the response to u and v as change in blood volume output normalized by VB0
# To convert this to control "delta V", we have delta V = VB * (VB0 - VB1),
# where VB is blood volume estimate,
# gain is feedback gain (specifying the speed of ﬂuid shift),
# alpha is the ratio between the volume changes in the intravascular and interstitial ﬂuids.
#
# The controller(OR MODEL???) retains the 1/(1+α) portion of the input ﬂuid volume in the intravascular compartment 
# while shifting the remaining α/(1 + α) portion to the interstitial compartment in the steady state. 
# The ﬂuid shift from the intravascular to interstitial compartment (q) acts as feedback control to steer VB 
# to the target change in blood volume (rB) [Hahn].

# u is total infusant  OR incremental?
# i is time step counter, each step being T

#--- Helper functions
# Infusion pump transfer function
# We can give pump some history in the error array

def pump_model (error, infusion, delay, pumpNoiseLevel):
    result = 0
    exponent = 0
    if(dbg19): set_trace() ####### entering pump
    if(useExponentialDampingInPump):
        if (len(error) >= delay):
            exponent = -(error[-delay] / eMax)
            if (exponent < 30 and exponent > -30):
                slowdown = 1 - math.exp(exponent)  # prevents going too fast as error nears 0
                if(dbg2): print("slowdown = ",slowdown)
            else:
                slowdown = 1
                if(dbg19): 
                    print(" too large exponent = ",exponent,",slowdown = ",slowdown,", infusion = ",infusion)
                    set_trace() #################### exponent, slowdown, infusion
                print("*** pump exponent too large or too small")
            result = infusion * slowdown * resistanceFactor +  pumpNoiseLevel * random.uniform(0,1)
        else:
            result = infusion  + pumpNoiseLevel * random.uniform(0,1)
    else:
        result = infusion * resistanceFactor  + pumpNoiseLevel * random.uniform(0,1)
    if(dbg10):
        print("Pump: last error ",error[-1],", infusion ",infusion,", delay ",delay,", eMax ",eMax)
        print("      exponent ",exponent,", slowdown ",slowdown,", resistance ",resistanceFactor)
        print("      noise level ",pumpNoiseLevel,", result ",result)
        #set_trace() ############################
    if (result < 0):
        result = 0
        if(dbg19): 
            print("went negative = ",exponent,",slowdown = ",slowdown,", infusion = ",infusion)
            set_trace() #################### exponent, slowdown, infusion
    return result

#=====================
class TableController:  # REMOVE THE SETTINGS OF VB ETC
    def __init__(self,i1,j1,k1,m1,map1,losses_f,row,control_f,finalMAP):
        self.i1 = i1  # time step number (now row number in this test)
        self.j1 = j1  # controller param number = 0, 1, ..., 5
        self.k1 = k1  # controller now reading row k
        self.m1 = m1  # at row element m
        self.map1  = map1   # this is the final MAP, or mp, as called 
        self.losses_f = losses_f
        self.row = row   # this is usually a row of data from a batch
        self.control_f = control_f
        self.finalMAP = finalMAP  # total MAP at this step
        
        self.VB = VB
        self.K0 = K0
        self.alpha = alpha
        #self.Kparms = Kparms  ?? global
        
        self.control = ""  # This will be a row of output from the controller
        self.total_outputs = 0
        self.count = 0  #

        #if(dbg15): set_trace() ################### TC Entry: i1, j1, k1, m1, map1, control_f,finalMAP,Kparms

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
        
        ##if(dbg8): set_trace() ################## step thru
        if self.i1 < 2:  # recursion begins at second time for volume response 
            VB3 = 0  
            VB2 = 0
            VB1 = 0
            VB0 = 0

            Um1 = initialInfusion
            Vm1 = initialBleed
            
            Um2 = 0 
            Vm2 = 0
        else:
            if(dbg2):
                print(" ")
                print("--- Model: preparing model at i = ",str(self.i))," ----  448"
                print("Um2 = ",Um2,", Um1 = ",Um1)
                if(fluidLoss == True):
                    print("Vm2 = ",Vm2,", Vm1 = ",Vm1)
            if(dbg2):
                set_trace() ################## step and check A, B, C, D
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
            if (fluidLoss == True and len(v1) > 1):
                Vm1 = v1[-1]
            else:
                Vm1 = 0
            VB3 = VB2
            VB2 = VB1
            # Careful, this model tends to oscillate with period 2 so smooth
            VB1 = (VB0 + VB1) / 2

            if(dbg0):
                print(" ")
                print("-- output run_model: at i = ",self.i1," m = ",self.m1)
                print("-run_model: A = ",A,", B = ",B,", C = ",C,", D = ",D)
                print("VB3 = ",VB3,", VB2 = ",VB2,", VB1 = ",VB1,", VB0 = ",VB0)
                print("Um2 = ",Um2,", Um1 = ",Um1)
                if(fluidLoss == True):
                    print("Vm2 = ",Vm2,", Vm1 = ",Vm1)
    
    def advance_Table (self, mapRow, m1):
        A1 = 0 # It's easier to debug large formulas in pieces
        B1 = 0
        C1 = 0
        D1 = 0
        E1 = 0
        nt = 0

        map1 = mapRow[m1]
        if(dbg20):
            print("On entry to advance Table, map1 = ",map1,", at m = ",m1)
            set_trace() #################
            
        def Table1 (self, map1):  # Kramer's infusion rate table
            if(dbg20): 
                print("On entry to Table map1 = ",map1)
                set_trace() #################
            del_U = 0.0
            Kparms1half = Kparms[1] / 2  # this is for Level 0

            if map1 >= Kparms[0] and map1 < Kparms[1]:
                if(dbg20): print("Level 1 MAP: ",map1,", del_U = ",del_U)
                del_U = Dparms[1]
            elif map1 >= Kparms[1] and map1 < Kparms[2]:
                if(dbg20): print("Level 2 MAP: ",map1,", del_U = ",del_U)
                del_U = Dparms[2]
            elif map1 >= Kparms[2] and map1 < Kparms[3]:
                if(dbg20): print("Level 3 MAP: ",map1,", del_U = ",del_U)
                del_U = Dparms[3]
            elif map1 >= Kparms[3] and map1 < Kparms[4]:
                if(dbg20): print("Level 4 MAP: ",map1,", del_U = ",del_U)
                del_U = Dparms[4]
            elif map1 >= Kparms[4] and map1 < Kparms[5]:
                if(dbg20): print("Level 5 MAP: ",map1,", del_U = ",del_U)
                del_U = Dparms[5]
            elif map1 >= Kparms[5]:
                if(dbg20): print("Level 6 MAP: ",map1,", del_U = ",del_U)
                del_U = Dparms[6]
            elif map1 >= Kparms1half and map1 < Kparms[0]:
                if(dbg20): print("Level 0 MAP: ",map1,", del_U = ",del_U)
                del_U = Dparms[0]   # keep feeding it with lowest value
            else:
                print("*** Table1 map1 ",map1," dropped below lowest Level 1 ",Kparms[0])
            if(dbg20):
                print("del_U = ",del_U)
                set_trace() ##############
            return del_U
        
        #if(dbg15): set_trace() #####################
        if (dbg20): print("--advance able: after Table, i = ",self.i1,", m = ",self.m1)
        delU = 0
        tot_infusion = 0
        ln_u = 0
        ln_v = 0
        controller = " "
        result = 0
        pump_out = 0
        success = 1
        try:
            out_f = open(self.control_f,"a",newline='',encoding="utf8")
            csv_out = csv.writer(out_f, delimiter=',')

            # start the filter as best can for first time step  ???
            if (self.m1 != None and self.m1 < 1):
                e1.append(initialError)
                v1.append(initialLoss)
                
                if(dbg20): set_trace() ###############
                # Here is the simulated pump starting up
                if (Cp1 == 1):
                    if(dbg15): set_trace() #####################
                    pump_out = pump_model(e1, initialInfusion, delay, pumpNoiseLevel)
                    if(pump_out == 0):
                        set_trace() ########################
                else:
                    pump_out = 0
                    
                if(dbg20): set_trace() ########################   
                firstInfusion = pump_out + initialInfusion
                
                u1.append(firstInfusion)   # infusion. (This accumulates total)
                controller = str(self.i1) + "," + str(self.m1) + "," + str(u1[0]) + "," + str(v1[0]) + "," + str(e1[0])
                out_f.write(controller)
                self.finalMAP = firstInfusion
            
            # subsequent time steps
            elif(self.m1 != None):  # m1 > 0
                if(dbg20): set_trace() ###############
                # -------- control function
                if (dbg8): print("--advance_controller: at i = ",self.i1,", m = ",self.m1," MAP = ",self.map1)

                delU = Table1 (self, map1)
                
                pump_out = 0
                infusion = delU    # emphasizes that this is a "delta" change, not total infusion change
                if(dbg8): print("infusion = ",infusion)
                
                if (Cp1 == 1):
                    pump_out = pump_model(e1, infusion, delay, pumpNoiseLevel)  # pump transfer function
                    # pump may decrease the infusion rate if it is too fast, depending on e1
                    #if(pump_out == 0):
                        #set_trace() ########################
                else:
                    pump_out = initialInfusion
                # Note: pump output is a "delta"
                tot_infusion = pump_out +  u1[self.m1 - 1]
                
                if(dbg20): set_trace() ###############
                u1.append(tot_infusion) # total infusions up to and including m-th time step

                #--- Here is the combination: total infusion - fluid loss
                self.finalMAP = Ci * tot_infusion  -  Cm * VB0
                
                er = float(self.map1 - self.finalMAP)
                e1.append(er)
                
                if(dbg8): print("tot_infusion = ",tot_infusion,", Ci = ",Ci,", er = ",er)

                ln_u = len(u1)
                ln_v = len(v1)
                ln_e = len(e1)
                va = " "
                if (self.m1 <= ln_u and self.m1 < ln_e):
                    ma = str(self.m1)
                    ua = str(u1[self.m1])
                    if(fluidLoss == True and self.m1 < ln_v):
                        va = str(v1[self.m1])
                    ea = str(e1[self.m1])
                else:
                    ma = "0"
                    ua = "0"
                    va = "0"
                    ea = "0"
                    if(dbg):
                        print("---advance: ln_u = ",ln_u,", ln_v = ",ln_v,", ln_e = ",ln_e,". Can't compute self.control")

                if(dbg15 and self.m1 > 30):
                    print(" ")
                    print("---advance Table output parameters")
                    print(" - Kparms= ",Kparms)
                    print(" - e is now : ",e1[-self.m1:-1])
                    print(" - infusion : ",infusion)
                    print(" - pump_out  : ",pump_out)
                    print(" - u is now : ",u1[-self.m1:-1])
                    if (fluidLoss == True):
                        print(" - v is now : ",v1[-self.m1:-1])
                    print(" - time step ma = ",ma,", u1 = ",ua,", v1 = ",va,", e1 = ",ea)
                    print(" - ln_u = ",ln_u,", ln_v = ",ln_v,", ln_e = ",ln_e)
                    
                self.control = ma + ", " + ua + ", " + va + ", " + ea   # keeps record of control

                # Note: this is the file needed in a trained, closed-loop control.
                # Be sure to delete this file before starting a final closed-loop run.
                out_f.write(self.control)
                
                self.total_outputs += 1

                #print("-advance controller : updating i to ",self.i1)
                if(dbg2): print("-advance controller, returning error ",e1[-self.m1:-1])
            else:
                print("*** self.i1  is a None!")
        except OSError as err:
            print("OS error {0}".format(err))
            print(print_frame())
            success = 0
        except ValueError:
            print("Could not convert data to an integer.")
            print(print_frame())
            success = 2
        except IOError:
            print ("File error with :", self.control_f)
            print(print_frame())
            success = 3
        except UnboundLocalError:
            print("Unbound local: ", sys.exc_info()[0])
            print(print_frame())
            print("At m = ",self.m1)
            success = 4
        except:
            print("Unexpected error: ", sys.exc_info()[0], sys.exc_info()[1])
            print(print_frame())
            success = 5
        finally:
            out_f.close() 
            if(dbg): print('Controller iteration completed.')
            if(success != 1):
                print("*** End advance Table: success = ",success)
        #if(dbg15): set_trace() #######################
        return success
        # End advance_PID
        
    def get_average_error(self):
        leng = len(e1)
        e_sum = 0
        for i in range(0,leng):
            e_sum += e1[i]
        if (leng > 0):
            return e_sum / leng
        else:
            print("*** error list has 0 length")
            return 0

    def get_control(self):
        return self.control
#-- End Table Controller class

#------------------------------------
# Gradient descent process
# acc and accLast are using accuracies in this version to
# modulate gradient step size.

def step_j_th_parm (acc, accLast, j1, Kparms, epsilon):
    # Limits change to modulator % of acc_m1, e.g. 1 %.
    step_acc = 0  # accuracy should be stepped according to Newton
    delta_acc = acc - accLast
    
    # Remember, Newton's method is
    #      X(n+1)  =  X(n)  -  f(X(n) / f'(X(n))
    # The crude gradient descent method uses this method in all coordinates. 
                          
    if (delta_acc > epsilon and acc > epsilon and j1 > 0):
         step_acc  =  -  acc / delta_acc
    elif (delta_acc > epsilon and acc < epsilon and j1 > 0):
        # Note that a flat spot occurs at f(Xn) = 0, f'(Xn) != 0.
        # As long as this is not the last base batch, we randomize the 
        # batch and keep moving!
        step_acc = 0
    elif (j1 == 0):
        step_acc = accLast
    else:
        #  An indeterminant spot occurs when both f and f' = 0 closely.
        #  We should kick things off a flat spot randomly some % of f
        step_acc = epsilon * random.uniform(0, 1)

    if (accLast == 0):
        multiplier = 1
    else:
        multiplier = modulator * step_acc / accLast
        if(dbg): print("--- accLast = 0 at j1 = ",j1)
    
    Kparms[j1] = Kparms[j1] * multiplier
    if(dbg18):
        print("--- End of step_j_th_parm")
        print(" - multiplier = ",multiplier,", accLast = ",accLast)
        print(" - step_acc = ",step_acc,", acc = ",acc,", delta_acc = ",delta_acc)
        print(" - step_j_th_parm: changing Kparms[",j1,"] to ",Kparms[j1])

    # Note, scores and accuracy may continue to improve even if we keep the
    #s ame batch data randomized somewhat.
    # Note also that K does accumulate over j, so after 3 inner loops, all
    # coordinaes of K are updated.
    return Kparms
#------------------------------------

# Here is where the system gets to "vote" on keeping settings of Kparms.
# acc is NN's curent accuracy; accLast is last accuracy achieved; accPrev next to last (the oldest)
def improvement (acc, accLast, accPrev, Gaussian_noise):
    difference1 = abs(acc - accLast)
    difference2 = abs(accLast - accPrev)
    if (difference1 < percentile * Gaussian_noise and difference2 < percentile * Gaussian_noise):
        if(dbg18):
            print("improvement: acc = ",acc,", accLast = ",accLast," percentile = ",percentile) 
            print("             accPrev = ",accPrev,", Gaussian_noise = ",Gaussian_noise)
            print("             difference1 = ",difference1,", difference2 = ",difference2)
            print("Improvement -- No improvement, discard last batch")
            set_trace() ########### NO IMPROVEMENT
        return 0   # no improvement, discard last batch
    else:
        if(dbg18):
            print("improvement: acc = ",acc,", accLast = ",accLast," percentile = ",percentile) 
            print("             accPrev = ",accPrev,", Gaussian_noise = ",Gaussian_noise)
            print("             difference1 = ",difference1,", difference2 = ",difference2)
            print("Improvement -- move last batch to new baseBatch")
        return 1

#========================= MAIN =======================
if (train):
    '''It may help to have an outline. In the train loops,
    j1-loop makes baseBatche's which are then copied to 3 incremental batch's.
    After the loop there are predicted steps in best directions of each of
    the 3 Tableparameters individually as the k-loop runs 0,..,2.
    This  is a crude way of estimating a gradient.
    In the i-loop, the three steps are added to the parameters, in Kparms,
    for a new baseBatch for the next itereation.
    The NN is then run on this batch, which is split into train and test
    files equally (???).
    This may show an "improvement", in which case the baseBatch is kept, else discarded.
    outer n-loop continues.  We consider breaking if no improvement.
    '''
    Gaussian = 0.0    # Kalman estimate of the distribution of accuracies
    trainingRows = []  # this will become a list of lists  ???
    finalMAP = initialMAP
    control_f = "control.csv"  # Clear this for a final run.  It accumulates
    i1 = 0
    j1 = 0
    k1 = 0
    m1 = 0
    if(dbg8): set_trace() ################### Start of MAIN
    countImprove = 0

    training = []
    # Make first batch. Read entire training file, selecting batches from it.
    # In this type of Driver there is only one row, of length, dataSize, used also for NN's.
    success = 1
    try:
        with open(dataName, 'r', newline='', encoding="utf8") as inp:
            csv_in = csv.reader(inp, delimiter=',')
            count = 0
            for row in csv_in:
                trainingRows.append(row)  # A row is a list so this is a list of lists
                seq = len(row)
                count += 1
                for i1 in range(0,seq):  # On the other hand, this is a simple list.  We use.
                    training.append(row[i1])
        #if(dbg9): set_trace() ###################### check trainingRows and training
        if(dbg): print("MAIN: input fileSize = ",count)
    except OSError as err:
        print("OS error: {0}".format(err))
        print(print_frame())
        success = 0
    except ValueError as ex:
        print("Could not convert data to an integer.")
        print(ex)
        print(print_frame())
        success = 2
    except IOError:
        print ("File error with :", dataName)
        print(print_frame())
        success = 3
    except:
        print("Unexpected error: ", sys.exc_info()[0], sys.exc_info()[1])
        print(print_frame())
        success = 4
    finally:
        inp.close()
        if(success != 1):
            print("MAIN: - exiting. success = ",success)
            sys.exit()
        
    # Note: instead of this n1 loop, we could run through a list of training batchs
    # from other sources such as NoisySinusoids or Glitches.
                        
    for n1 in range(0, trainingCycles):   #  Run training batches through controller
        # Create the first batch and drive later batch creations thru create_Batch_i_j
        batchRows = []
        # Peel out the section of training file from M*n to M*(n+1) as a baseBatch.
        # In n-th loop peel off the batch from M * n to M * (n+1)
        start = n1 * timeSteps
        endit = (n1+1) * timeSteps
        count = 0
        print("--- start = ",start,", endit = ",endit)

        #if(dbg15): set_trace() ################ 
        for peel in range(0, batchSize):
            begin = start + peel
            stopit = endit + peel
            #if(dbg3):
                #set_trace() ################### 
            batchRows.append(training[begin:stopit] + Kparms)

        baseBatch = fileRoot+str(n1)+".csv"   # base batch for n1-th training cycle
        # write baseBatch to be read later and parms varied in "deltaBatch"es
        
        #if(dbg15): set_trace() ################ STEP IT
        success = 1
        try:
            with open(baseBatch, 'w', newline ='', encoding = "utf8") as outp:
                csv_out = csv.writer(outp, delimiter=',')
                for k in range(0, batchSize):
                    csv_out.writerow(batchRows[k])
        except OSError as err:
            print("OS error : {0}".format(err))
            print(print_frame())
            success = 0
        except ValueError:
            print("Could not convert data to an integer.")
            print(print_frame())
            success = 2
        except IOError:
            print ("File error with :", baseBatch)
            print(print_frame())
            success = 3
        except:
            print("Unexpected error : ", sys.exc_info()[0], sys.exc_info()[1])
            success = 4
            print(print_frame())
        finally:
            outp.close()
            if(success != 1):
                print("**** MAIN: Could not process base batch ",baseBatch," at loop n = ",n1)
                print("success = ",success)
                sys.exit()
        if(dbg15): set_trace() ###################
            
        #---- create gradient-varied batches
        for i1 in range(0, batchSize):
            if (i1 == 0):
                baseBatch = fileRoot+str(n1)+".csv"
                newBatch = baseBatch
                #if(dbg4): set_trace() ################# check newBatch size 
            else:
                baseBatch = newBaseBatch   # returned from end of this loop ???
                newBatch = baseBatch
                #if(dbg4): set_trace() ################# check new Batch and new Base Batch size
            file = open(newBatch)
            lenBatch = len(file.readlines())
            file.close()
            if (batchSize != lenBatch):
                print("*** newBatch size = ",lenBatch," at i1 = ",i1,", j = ",j1,", n = ",n1)
                print(print_frame())
                print("Breaking i1 loop at ",i1)
                break
            if(dbg): 
                print("--- At beginning looping over baseBatch = ",baseBatch)
                print("    batch Size = ",lenBatch)
            
            #--Inner loop creates rows of batch_i_j as small steps in each component of Kparms.
            #  The gradient adjusted Kparms will be collected after the end of the loop.
            #  Run Driver_NN_PID_setup to get initial "_0" base batch.

            average_row_error = 0
            aveScore = 0
            scores = [0] * batchSize
            accPrev = 0
            accLast = 0

            if(dbg18):
                print("Setting accPrev and accLast  = 0")
                set_trace() ########## initializing accPrev and accLast

            # This loop varies the j1-th coordinate of K to create deltaBatch, and
            # runs this file with PID, collecting a score base on how close to MAP target
            # the Tablegets at the end of the profile (batch row).

            for j1 in range(0,Nparms):  # runs Table with variations in each of 6 Table parameters
                
                if(dbg18):
                    print("Enterning j1: accPrev = ",accPrev,", accLast = ",accLast,", accuracy = ",accuracy)
                    set_trace() ######## updating acc's
                    
                # Change the j-th coordinate of Kparms in the best gradient direction
                delta_K = step_j_th_parm (accuracy, accLast, j1, Kparms, epsilon)
                
                # Fill in the j-th coordinate of Kparams to create a "delta"
                deltaBatch = fileRoot+str(n1)+"_"+str(j1)+".csv"

                if(dbg3):
                    set_trace() ############# accuracy, accLast, accPrev, j1
                success = create_batch_i_j (n1, j1, baseBatch, deltaBatch, delta_K, noise)
                if(dbg3): 
                    print("-Main: create_batch_i_j baseBatch = ",baseBatch,", deltaBatch = ",deltaBatch)
                if (success == 0):
                    break

                if(dbg): 
                    print("- j1 loop new batch: ",deltaBatch," at i = ",i1,", j = ",j1)
                    print("      batch Size = ",batchSize)
                    
                # Read the new "batch" and run Table on each measure in each row,
                # computing a score from the final MAP reached by the controller.
                if(dbg3): set_trace() ####################### i, j, n, deltaK
                success = 1
                try:
                    rows = []
                    with open(deltaBatch, 'r', newline='', encoding="utf8") as inp:
                        csv_in = csv.reader(inp, delimiter=',')
                        count = 0
                        for row in csv_in:
                            rows.append(row)
                            count += 1
                    if (count != batchSize):
                        print(print_frame())
                        print("*** j1 loop deltaBatch: count = ",count," not batchSize at i ",i1,", j ",j1," n ",n1)
                        success = 0
                    aveScore = 0
                    if(dbg15): set_trace() #####################
                    #--------- For each row of batch_i_j, advance controller across measurements
                    for k1 in range(0,batchSize):    
                        row = rows[k1]
                        rowLength = len(row)
                        if (dbg2): print("MAIN: k-loop rowLength = ",rowLength,", timeSteps = ",timeSteps)
                        mp = 0
                        if(dbg8): set_trace() #################### step thru
                        # Run Table across the k-th row
                        for m1 in range(0,rowLength):
                            owr = [ast.literal_eval(ep) for ep in row]
                            TC = TableController(i1,j1,k1,m1,mp,"bleeding.csv",owr,control_f,finalMAP)
                            if(dbg8): print("MAIN: Inner loop m1 ",m1," running advance able")
                            TC.run_model()
                            TC.advance_Table(owr, m1)   # value reached at end of row test
                            mp = TC.finalMAP
                            if(dbg3): print("MAP reached = ",mp)
                        if(dbg15 and k1 > batchSize - 2): set_trace() #################### m1, mp, k1, row[m1], rows
                        scores[k1] = min( 0.99, 1 - abs((mp - targetMAP) / targetMAP))
                        aveScore += scores[k1]
                        if(dbg2): print("scores: ",scores)
                except OSError as err:
                    print("OS error : {0}".format(err))
                    print(print_frame())
                    break
                except ValueError:
                    print("Could not convert data to an integer.")
                    print(print_frame())
                    break
                except IOError:
                    print ("File error with : ",baseBatch)
                    print(print_frame())
                    break
                except:
                    print("Unexpected error : ",sys.exc_info()[0], sys.exc_info()[1])
                    print(print_frame())
                    break
                finally:
                    inp.close()
                    if(dbg3):
                        print("end inner j1 deltaBatch = ",deltaBatch,", i = ",i1,", j = ",j1,", scores = ",scores)
                        print("  Kparms[",j1,"] = ",Kparms[j1])
                #--end inner ji loop

                
                # update accuracy parameters for i-th loop
                aveScore = aveScore / 3
                accPrev = accLast
                accLast = accuracy
                accuracy = aveScore
                # THIS accuracy SETTING NEEDS TO BE CHANGED FOR EACH CONFIGURATION:
                # Accuracy can be set be either controller or neural net.  If
                # neural net is not configured, the default will be the aveScore as below:

                if(dbg3 or dbg18): 
                    print("--- After end of j-th loop: i = ",i1,", n = ",n1)
                    print("   accuracy = ",accuracy)
                    print("   accLast = ",accLast)
                    print("   accPrev = ",accPrev)
                    print("   aveScore = ",aveScore)
                    print(" ")
                        
                if(dbg):
                    print("Finished j loop.  Kparms = ",Kparms)
                    print("    aveScore = ",aveScore)
                # create new base batch based on Kparms incremented in gradient directions
                newBatch = fileRoot+str(i1+1)+".csv"

                create_batch_i (n1, i1+1, baseBatch, newBatch, Kparms, scores, noise)
                # Next iteration of i-loop should pick the new baseBatch up.
                if(dbg15): set_trace() ##################### baseBatch, newBatch, Kparms, scores, noise

                # Re-train on the revised i1-th batch
                if(dbg): print("At i = ",i1,", n = ",n1,", newBatch = ",newBatch)

                # checking newBatch
                success = 1
                try:
                    with open(newBatch, 'r', newline='', encoding="utf8") as inp:
                        csv_in = csv.reader(inp, delimiter=',')
                        count = 0
                        for row in csv_in:
                            if(dbg11):
                                print("row ",row)
                            count += 1
                    if(dbg11): set_trace() ############## check row's
                    if (count != batchSize):
                        print(print_frame())
                        success = 0
                        print("*** After create_batch_i, reading j1 loop newBatch: count = ",count," not batchSize at i ",i1,", j ",j1," n ",n1)
                except OSError as err:
                    print("OS error : {0}".format(err))
                    print(print_frame())
                    success = 2
                except ValueError:
                    print("Could not convert data to an integer.")
                    print(print_frame())
                    success = 3
                except IOError:
                    print ("File error with : ",baseBatch)
                    print(print_frame())
                    success = 4
                except:
                    print("Unexpected error : ",sys.exc_info()[0]), sys.exc_info()[1]
                    print(print_frame())
                    success = 5
                finally:
                    inp.close()
                    if(dbg): print("newBatch read properly")
                    if(success != 1):
                        print("*** Breaking j1-loop.  success = ",success,", j1 = ",j1,", n1 = ",n1)
                        break

                if(dbg15): set_trace() #############################
                if (count != batchSize):
                    print("*** newBatch is wrong: ",count)
                else:
                    loss = TC.get_average_error()
                    '''Loss function measures the difference between the predicted label and the ground truth label. 
                    E.g., square loss is  L(y^,y)=(y^−y)^2 , hinge loss is  L(y^,y)=max{0,1 − y^ x y} ...'''
                    if (dbg):
                        print("-- ",loss)
                        print("     accuracy ",aveScore)
                        print("     After TC call, Kparms = ",Kparms)

                # Estimate presumed Gaussian noise of the controllers' accuracy
                if (i1 == 0):
                    W1 = 1
                    W2 = 0
                else:
                    W1 = 1 / float(i1)
                    W2 = 1 - W1
                Gaussian = W1 * accuracy + W2 * Gaussian

            #-- end j1-Loop over batch and delta batches
            if(dbg18): 
                print("Entering improvement: Noise level is ",Gaussian)
                print("accuracy = ",accuracy,", accLast = ",accLast,", accPrev = ",accPrev)
                    
            improve = improvement (accuracy, accLast, accPrev, Gaussian)
            
            if (improve == 0):
                countImprove += 1
                if (countImprove > 6):
                    print("*** Too many tries.  No improvement.")
                    sys.exit()
                print("-- No significant impovement in accuracy. discarding newBatch")
                file = open(baseBatch)
                lenBatch = len(file.readlines())
                if(dbg4): set_trace() ######### discarding batch
                if (lenBatch != batchSize):
                    print("*** baseBatch is wrong length : ",lenBatch, " Breaking i-loop")
                    print(print_frame())
                    break
                else:
                    newBaseBatch = baseBatch
            else: # replace old baseBatch with last batch
                file = open(newBatch)
                lenBatch = len(file.readlines())
                if(dbg4): set_trace() ######### replace old with last base batch
                if (batchSize != lenBatch):
                    print("*** newBatch is wrong length : ",lenBatch, " Breaking i-loop")
                    print(print_frame())
                    break
                else:
                    newBaseBatch = newBatch
            if(dbg15): set_trace() #####################
            print("--- At end of batch loops, newBaseBatch = ",newBaseBatch)
        #--end batch i1-loop
    #--end training n1-loop
#=========================================================
if (test):
    if(dbg): print("========== STARTING VALIDATION =========")
    '''Validation test
    Read line from RNN_training.csv data file;
    run Table(which may input a fluid losses file as an option);
    run pump simulator (basically a linear transfer function with lag);
    write line to control file;
    run "model" of body responses;
    compute error between body response and desired target MAP;
        
    As prerequisite, this depends on the parameters, K, being adequately trained
    by the training loop.  Keeping its trained node weights, the NN now serves
    to make a small correction to the Tableoutput.
    
    Run NN to make a correction to Tableoutput (trained separately for that as above);
    use accuracy to adjust step size (and other search parameter as needed).'''
    ABC = 0

