#!/usr/bin/env python
# coding: utf-8

# In[6]:


#NoiseySinusoids -- forNeuralNetwork__Params_for_PID
#https://towardsdatascience.com/time-series-forecasting-with-lstms-using-tensorflow-2-and-keras-in-python-6ceee9c6c651
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import csv
import random
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

class noisySinusoid:
    def __init__(self, out_f = "NN_3parms_sinusoids.csv"):
        self.out_f = out_f

    
    def create_bleeding(self):
        MAP_target = 65
        M = 100
        t = 0
        twoPI = 2.0 * math.pi
        wavelength = 10.0   # seconds or time steps
        del_t = twoPI / wavelength
        try:
            outp = open(self.out_f,"w+",encoding="utf8",newline='')
            csv_out = csv.writer(outp, delimiter=',')
            i = 0
            for i in range(0, M):
                meas = []
                # twelve measurments at preceeding time steps
                for j in range(0,8):
                    t += j * del_t
                    val = MAP_target + 10 * math.sin(t)+ 5 * random.uniform(0, 1)
                    meas.append(val)
                Kp = 1 * random.uniform(0, 1)
                meas.append(Kp)
                Ki = 2.1 * random.uniform(0, 1)
                meas.append(Ki)
                Kd = 0.5 * random.uniform(0, 1)
                meas.append(Kd)
                map = MAP_target - 16
                meas.append(map)
                csv_out.writerow(meas)
                i += 1           
            
            outp.close()
            print("NoisySinusoid completed normally.")
        except:
            raise("***output simulation data file couldn't not be openned.")
        finally:
            outp.close()

# Check it -----------------------        
sim = noisySinusoid("NN_3parms_sinusoids.csv")
sim.create_bleeding()


# In[5]:


# For comparisons of noise levels
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 16, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
'''generate 1,000 values from the sine function. 
But, we’ll add a little noise to it'''

time = np.arange(0, 100, 0.1)
'''numpy.arange([start, ]stop, [step, ]dtype=None)
returns evenly spaced values within a given interval'''
sin = np.sin(time) + np.random.normal(scale=0.5, size=len(time))

plt.plot(time, sin, label="sine with noise")
plt.legend()
    

