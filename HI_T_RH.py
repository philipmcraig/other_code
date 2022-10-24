

import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal


t = np.arange(25, 45.1, 1.0)
rh = np.arange(0, 101, 20)



def f_C(T, RH):
    return (- 8.784695 + 1.61139411*T + 2.338549*RH*0.01 - 0.14611605*T*RH*0.01 
            - 1.2308094*10**(-2)*T**2 - 1.6424828*10**(-2)*RH**2*0.01**2
            + 2.211732*10**(-3)*T**2*RH*0.01 + 7.2546*10**(-4)*T*RH**2*0.01**2
            - 3.582*10**(-6)*T**2*RH**2*0.01**2)
    
    
def adjust1(T, RH):
    return ( ((13-RH*0.01)/4) * np.sqrt((17-np.abs(T-95.0)) / 17) )

def adjust2(T, RH):
    return ( ((RH*0.01-85)/10) * ((87-T)/5) )

'''
def f_F(T, RH):
    if RH< 13:
        for j in range(len(T)):
            HI1 = []
            if 80< T[j] <=112:
                HI1.append(- 42.379 + 2.04901523*T[j] + 10.14333127*RH*0.01 - 0.22475541*T[j]*RH*0.01 
                        - 6.83783*10**(-3)*T[j]**2 - 5.481717*10**(-2)*RH**2*0.01**2
                        + 1.22874*10**(-2)*T[j]**2*RH*0.01 + 8.5282*10**(-4)*T[j]*RH**2*0.01**2
                        - 1.99*10**(-6)*T[j]**2*RH**2*0.01**2 - adjust1(T[j], RH))
                return (HI1)
            
    if RH >85:
        for j in range(len(T)):
            HI2 = []
            if 80 < T[j] <=87:
               HI2.append(- 42.379 + 2.04901523*T[j] + 10.14333127*RH*0.01 - 0.22475541*T[j]*RH*0.01 
                        - 6.83783*10**(-3)*T[j]**2 - 5.481717*10**(-2)*RH**2*0.01**2
                        + 1.22874*10**(-2)*T[j]**2*RH*0.01 + 8.5282*10**(-4)*T[j]*RH**2*0.01**2
                        - 1.99*10**(-6)*T[j]**2*RH**2*0.01**2 - adjust1(T[j], RH))
               return (HI2)
'''        
        
    
def f_F(T, RH):
    return (- 42.379 + 2.04901523*T + 10.14333127*RH*0.01 - 0.22475541*T*RH*0.01 
            - 6.83783*10**(-3)*T**2 - 5.481717*10**(-2)*RH**2*0.01**2
            + 1.22874*10**(-2)*T**2*RH*0.01 + 8.5282*10**(-4)*T*RH**2*0.01**2
            - 1.99*10**(-6)*T**2*RH**2*0.01**2)

    
def f_F2(T, RH):
    return (0.5*(T+61.0+ (T-68.0)*1.2 + RH*0.01*0.094))

    
def convert_to_f(temp):
    return (temp*1.8 + 32.0)

def convert_to_c(temp):
    return ((temp - 32.0) / 1.8)





color = ('steelblue', 'darkorange', 'g', 'rosybrown', 'blueviolet', 'r')
marker = ('.', '^', 'o', 'd', '*', 'x')


t_f = np.arange(80,113,1.0)
for i in range(len(rh)):
    plt.plot(t_f, f_F(t_f, rh[i]), label = str(rh[i]) + '% RH',
             marker = marker[i], markevery = 5)
    plt.xlabel('Temperature($^\circ$C)')
    plt.ylabel('Heat Index')
    plt.legend()
    plt.title('HI $^\circ$F)')
plt.show() 


t_f = np.arange(80,113,1)



for i in range(len(rh)):
    plt.plot(convert_to_c(t_f), convert_to_c(f_F(t_f, rh[i])), 
             label = str(rh[i]) + '% RH', marker = marker[i], markevery = 5)
    plt.xlabel('Temperature($^\circ$)')
    plt.ylabel('Heat Index')
    plt.legend()
    plt.title('HI ($^\circ$)')
plt.show()


fig, ax = plt.subplots()
for i in range(len(rh)):
    ax.plot(convert_to_c(t_f), convert_to_c(f_F(t_f, rh[i])), color = color[i],
            label = str(rh[i]) + '% RH', marker = marker[i], markevery = 5)
    lim = ax.get_ylim()
    ax.set_ylim(20, 125)
    ax2 = ax.twinx()
    ax2.plot(convert_to_c(t_f), f_F(t_f, rh[i]), color = color[i],
             marker = marker[i], markevery = 5)
    ax2.set_ylim(convert_to_f(20), convert_to_f(125))
    ax.set_xlabel('Temperature($^\circ$)')
    ax.set_ylabel('Heat Index ($^\circ$C)')
    ax2.set_ylabel('Heat Index ($^\circ$F)')
    ax.legend()
    ax.set_title('Heat Index')
plt.show() 




plt.figure(3)
stepx = 0.5; stepy = 1
x = np.arange(15, 45, stepx); y = np.arange(0, 100, stepy)
X1,Y1 = np.meshgrid(x, y)


stepx = 0.5; stepy = 1
x = convert_to_c(np.arange(60, 110, stepx)); y = np.arange(0, 100, stepy)
X1,Y1 = np.meshgrid(x, y)



stepx = 0.5; stepy = 1
x = np.arange(60, 110, stepx); y = np.arange(0, 100, stepy)
X2, Y2 = np.meshgrid(x, y)



HI_C = f_F(X1, Y1)
plt.contourf(X1, Y1, HI_C, cmap = 'YlOrRd')
plt.title('Heat Index')
plt.xlabel('Temperature ($^\circ$)C')
plt.ylabel('Humidity (%)')
plt.colorbar()
plt.show()




HI_F = f_F(X2, Y2)
plt.contourf(X2, Y2, HI_F, cmap = 'YlOrRd')
plt.title('Heat Index')
plt.xlabel('Temperature ($^\circ$F)')       
plt.ylabel('Humidity (%)')
plt.colorbar()
plt.show()




f0, f0_original = np.zeros(len(rh)), np.zeros(len(rh))

f0[:] = convert_to_f(convert_to_c(f_F(t_f[1], rh[:])))
f0_original[:] = f_F(t_f[1], rh[:])