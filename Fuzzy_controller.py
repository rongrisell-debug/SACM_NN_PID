#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#------------------------------------------------------------------
# Fuzzy Controller
# This fuzzy controller, used by Kramer, was developed by Ying: 
#   Ying H. Constructing nonlinear variable gain controllers via the Takagi– Sugeno fuzzy control. 
#   IEEE Trans Fuzzy Syst. 1998;6(2):226–34. 
# Ying showed that this fuzzy controller is equivalent to a
# nonlinear proportional integral controller with variable proportional-gain and integral-gain.

# Let MAP(nT) designate MAP value at sampling time nT, where T = 5 sec was the sampling period and n = 0,1,…
# Then filter is:
#     MAPf (nT) = (MAP(nT) + MAP(nT − T) + MAP(nT − 2T) + MAP(nT − 3T))
# With the initial condition being MAP(−T) + MAP(−2T) + MAP(− 3T) = MAP(0).

# When MAPf(nT)  <  40, the fuzzy controller output is U(nT) = Umax, which was the maximum fluid volume. 
# Umax = Animal body weight (kg) × 1.429 in unit of ml/h. Whenever MAPf(nT) ≥ 40, 
# the fuzzy controller calculated Δu(nT) as follows
#     delta_u(nT) = alpha x (2.85e(nT) + 0.58r(nT))
# where e(nT) = setpoint − MAPf (nT); r(nT) = MAPf(n T − T) − MAPf(nT). 
# Which were the two input signals to the fuzzy controller. 
# Here, the set point was the target MAP desired by the user. 
# The value of alpha was calculated using the formulas in Table 2 according to the values 
# of e(nT) and r(nT) that had nine possible combinations as shown in Kramer's Fig. 5. 
# These formulas were derived from 
#     delta_u(nT) = beta x (2.85e(nT) + 0.58r(nT)).
# for this fuzzy controller configuration of Ying. 
# For this study, we experimentally determined that the following parameter values for 
# Table 2 (below) produced the best control performance: 
#   L = 16, k1 = 1, k2 = 0, k3 = 0, k4 = 0.33.  
# After computing Δu(nT), the fuzzy controller then calculated the following:
#   u(nT) = U(nT − T) + delta_u(nT) 
# where the initial condition U(−T) = 0. Then it computed the final output of the controller 
# (i.e., the input signal to the fluid pump):
#     U(nT) = u(nT), if 0 ≤ u(nT) ≤ Umax Umax, if u(nT) > Umax 0, if u(nT) < 0.
# Equ (1)
# beta = k1 4L2^2 . (1 + k2 + k3 + k4)L2 + (1 + k2 − k3 − k4)L · e(nT) +(1 − k2 + k3 − k4)L · r(nT)+(1 − k2 − k3 + k4)e(nT)r(nT)]
#
# Table 2 Formulas for computing value of ß in (1)
# Which formula to use depends on location of the controller’s two input signals (Fig. 5)
# IC No.            alpha
# 1 Expression (1) below 
# 2                 k1[(1-k2)r(nT) + (1 + k2)L]/2L 
# 3                 k1 
# 4                 k1[(1-k3)e(nT) + (1 + k3)L]/2L 
# 5                 k1k3 
# 6                 k1[(k3-k4)r(nT) + (k3 + k4)L]/2L 
# 7                 k1k4 
# 8                 k1[(k2-k4)e(nT) + (k2 + k4)L]/2L 
# 9                 k1k2
setpoint = 80
    
def Fuzzy_Controller(e, r, T = 5, L = 16, k1 = 1, k2 = 0, k3 = 0, k4 = 0.33):
    # Initial conditions
    U(− T) = 0
    
    # Nine ICs covering possible inputs of error, e, and response, r:
    # IC1
    if (e < L & e > -L & r < L & r > -L):
        beta = = k1 x 4L*L * (1 + k2 + k3 + k4) * L2 + (1 + k2 − k3 − k4) * L * e        + (1 − k2 + k3 − k4) * L * r + (1 − k2 − k3 + k4) * e * r
    # IC2
    if (e < L & e > -L & r < L & r > -L):
        beta = k1 * ((1-k2) * r + (1 + k2) x L)/(2*L)
    # I3
    if (e < L & e > -L & r < L & r > -L):
        beta = k1
    # IC4
    if (e < L & e > -L & r < L & r > -L):
        beta = k1 * ((1-k3)*e + (1 + k3)*L) / (2*L) 
    # IC5
    if (e < L & e > -L & r < L & r > -L):
        beta = k1 * k3   
    # IC6
    if (e < L & e > -L & r < L & r > -L):
        beta = k1 x ((k3-k4)*r + (k3 + k4)*L) / (2*L)  
    # IC7
    if (e < L & e > -L & r < L & r > -L):
        beta = k1 * k4    
    # IC8
    if (e < L & e > -L & r < L & r > -L):
        beta = k1 * ((k2-k4)*e + (k2 + k4)*L) * / (2*L)   
    # IC9
    if (e < L & e > -L & r < L & r > -L):
        beta = k1 * k2 
# When MAPf(nT)  <  40, the fuzzy controller output is U(nT) = Umax, which was the maximum fluid volume. 
# Umax = Animal body weight (kg) × 1.429 in unit of ml/h. Whenever MAPf(nT) ≥ 40, 
# the fuzzy controller calculated Δu(nT) as follows
#     delta_u(nT) = alpha x (2.85e(nT) + 0.58r(nT))
# where e(nT) = setpoint − MAPf (nT); r(nT) = MAPf(n T − T) − MAPf(nT). 
# Which were the two input signals to the fuzzy controller. 
    t = n * T
    if (n > 2*T):
        MAPf(t) = (MAP(t) + MAP(t − T) + MAP(t − 2*T) + MAP(t − 3*T))
    elif (n > T):
        MAPf(t) = ((MAP(t) + MAP(t − T))) / 2
    elif (n > 0):
        MAPf(t) = (MAP(1) + MAP(0)) / 2
    else:
        MAPf(t) = MAP(0) / 3
    # With the initial condition being MAP(−T) + MAP(−2T) + MAP(− 3T) = MAP(0).
    
    if (MAPf(t) < 40):
        U(t) = Umax
    else:
        e(t) = setpoint − MAPf(t);  r(nT) = MAPf(t − T) − MAPf(t)
        delta_u(nT) = alpha x (2.85 * e(t) + 0.58 * r(t))
        u(nT) = U(nT − T) + delta_u(nT) 
# where the initial condition U(−T) = 0. Then it computed the final output of the controller 
# (i.e., the input signal to the fluid pump):
        if u(nT) < 0:
            U(nT) = 0
        elif (0 < u(nT) & u(nT) ≤ Umax):
            U(nT) = Umax
        else
            U(nT) = u(nT)

