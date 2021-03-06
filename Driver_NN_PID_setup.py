#!/usr/bin/env python
# coding: utf-8

# # Driver_PID_NN_setup
# ''' Create a test file of fluid losses used for the controller
# along with 3 PID parameter outputs.  The last (13-th) is true MAP.
# This implements a series of two steps plus noise.
# Other regimens could be generated by random changes
# according to a realistic probability distribution.
# 
# The Driver_NN_PID uses a numpy 1D array with 8 features and four training fields 8, 9, 10, 11.
# The last training field is usually the MAP target.  E.g.
# dataset_train= numpy.loadtxt("YaoNN.csv", delimiter=",")
# xtrain = dataset_train [:,0:12]
# ytrain = dataset_train [:,13:14]
# dataset_test= numpy.loadtxt("YaoNN.csv", delimiter=",")
# xtest = dataset_test[:,0:12]
# ytest = dataset_test[:,13:14]
# 
# We will want to create many different test files in this form.  E.g.  
# Kashihara et Al. simulated the MAP control for acute hypotension: 
# random noise within 1 mmHg was added. An exogenous pressureperturbation 
# was introduced at a constant speed of -18 mmHg min-1 for 2 min, 
# and then maintained at-36 mmHg for the following 5 min. 
# The targetvalue of MAP control was set at the baseline MAP.
# The sampling interval was 10 s and controllers described below 
# updated an NE infusion rateevery 10 s. The infusion rate (u(t)) was
# bounded by 0 < u(t) < 6 ?g kg-1 min-1.
# Note, if one wants a 2-D array, set meas something like 
# meas = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]].
# '''

# In[1]:


#-------------------------------------------
# This first makes the fluid loss file, 
# and then a training file used for example in Driver_NN_PID.
# Currently the file is 100 lines, the first 50 of which should be successfully
# identified by the NN_PID combination.  The last 50, probably not.
# The idea is, start the training with 50 normal MAPs, so the
# PID can seek a good initial set of parameters.  Then, a 
# batch will be sent to the NN with fairly good scores, batches to the NN with higher "scores".
# Of course many other training files should be used.
# Another interesting one is with random impulse data which 
# should achieve 0.99 accuracy.  See
import random
import csv

MAP_target = 65
file_size = 100   # this might be split into batches of 20 profiles.
class SIM:
    def __init__(self, bleeding_f = "bleeding.csv", batch_f = "training.csv"):
        self.bleeding_f = bleeding_f
        self.batch_f = batch_f
        
    # -- constant bleeding rate
    def create_bleeding(self):
        try:
            outp = open(self.bleeding_f,"w+",encoding="utf8",newline='')
            csv_out = csv.writer(outp, delimiter=',')
            meas = []
            i = 0
            for i in range(0, file_size):
                loss = 9 + 2 * random.uniform(0, 1)
                meas.append(i)
                meas.append(loss)
                csv_out.writerow(meas)
            outp.close()
            print("SIM_NN completed bleeding loss file normally.")
        except:
            print("***loss simulation data file couldn't not be openned.")
        finally:
            outp.close()
            
    #-- Alternate 18 mmHg low and 0 mmHg
    #   near the MAP target.
    def create_batch_0(self):
        try:
            outp = open(self.batch_f,"w+",encoding="utf8",newline='')
            csv_out = csv.writer(outp, delimiter=',')
            # first level at 0, next at -10   
            i = 0
            good_score = 0.99
            meas = []
            for i in range(0, file_size):
                # measurments at time steps
                if (i % 2 == 0):
                    for j in range(0,8):
                        meas.append(MAP_target + 1 * random.uniform(0, 1))
                    score = good_score
                else: # bad score
                    for j in range(0,8):
                        meas.append(MAP_target - 10 + 1 * random.uniform(0, 1))
                    score = 0
                Kp = 1 + 0.1 * random.uniform(0, 1)
                Ki = 2.5 + 0.25 * random.uniform(0, 1)
                Kd = 0.5 + 0.05 * random.uniform(0, 1)
                meas.append(Kp)
                meas.append(Ki)
                meas.append(Kd)
                meas.append(score)
                csv_out.writerow(meas)
            outp.close()
            print("SIM completed file normally.")

        except OSError as err:
            print("OS error: {0}".format(err))
            success = 0
        except ValueError:
            print("Could not convert data to an integer.")
            success = 0
        except IOError:
            print ("File error with :", batch_f)
            success = 0
        except:
            print("Unexpected error:", sys.exc_info()[0])
            success = 0
            raise
        finally:
            outp.close()
            
# Check it -----------------------        
sim = SIM("bleeding.csv","training.csv")
sim.create_bleeding()
sim.create_batch_0()

