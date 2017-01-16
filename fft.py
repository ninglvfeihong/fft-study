
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.fftpack import fft
from scipy.fftpack import ifft

def rect_(x):
    for i in x:
        if(i<-0.5):
            yield 0
        elif(i < 0.5):
            yield 1
        else:
            yield 0

def mirr(y):
    o=[y[0]];
    for i in range(1,len(y)):
        o.append(y[i]+y[len(y)-i])
    return o

def mrr(y):
    o=[]
    for i in range(len(y)):
        if(i<len(y)/2):
            o.append(y[i]);
        else:
            o.append(y[len(y)-i]);
    return o

def rect(x):
    return list(rect_(x))

# Number of samplepoints
N = 1600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = rect(x*8.1)
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

