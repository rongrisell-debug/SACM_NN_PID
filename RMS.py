#!/usr/bin/env python
# coding: utf-8

# In[2]:


# RMS  --  R.M.S. performance function between measured response and controller output.
import math
class RMS:

    def __init__(self,model_f,control_f):
        self.model_f = model_f
        self.control_f = control_f
        self.rms = 0
    
    def __return_rms__(self):
        return self.rms
    
    def initialize(self):
        try:  
            count1 = 0
            with open(self.model_f, 'r', encoding = "utf8") as inp:
                if (inp.mode == 'r'):
                    model[count1] = inp.read()
                    print ("model count :",count1)
                    count1 += 1
                inp.close
            print("RMS: measured response read in correctly.")
        
            try:
                count2 = 0
                with open(self.control_f, 'r', encoding = "utf8") as inp:
                    if (inp.mode == 'r'):
                        control[count2] = inp.read()
                        print (control[count2])
                        count2 = count2 + 1
                out.close()
            
                count = min(count1, count2)
            
                for i in range(0,count):
                    sum += (control[i] - model[i]) * (control[i] - model[i])
                if (count <= 0):
                    print("***RMS: file count 0")
                else:
                    self.rms = sqrt(sum) / count
                
            except:
                raise(e_out,"***Output file",output_f," couldn't not be openned")
            finally:
                f.close
        except:
            raise("***RMS: measured response couldn't not be openned")  
        finally:
            inp.close

    print("RMS completed normally.")

#-- test ------------------------
r = RMS("control.csv", "control.csv")  # should be zero
print (r.__return_rms__())

