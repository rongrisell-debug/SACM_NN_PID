#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Kramer's table-driven controller with 6 levels
def Table_Driven_Controller(map = 0):
    if map > 40 & map < 50:
        print("Level 1 map: "+str(map))
    elif map >= 50 & map < 60:
        primt("Level 2 map: "+str(map))
    elif map >= 60 & map < 70:
        primt("Level 3 map: "+str(map))
    elif map >= 70 & map < 80:
        primt("Level 4 map: "+str(map))
    elif map >= 80 & map < 90:
        primt("Level 5 map: "+str(map))
    else
        print("Level 6 map: "str(map))
        
#-- test        
del_U = Table_Driven_Controller(55)
print(del_U)

