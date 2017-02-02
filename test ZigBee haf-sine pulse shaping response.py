

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def rect(x):
    return np.array(list(rect_(x)))

T=1.0/100;
N=10000;

x = np.linspace(0.0, N*T, N)
y = 0.0*x;
for i in range(1,99):
    if np.random.rand()<0.5 :
        y = y+ rect(x-i)


yf = fft(y)
xf = np.linspace(0.0, 1.0/T,N)

#yf = yf*rect(x/4)+yf*rect((x-100)/4)
#y=ifft(yf);
yf = np.log10(np.abs(yf)**2)*10


#frequence domain
fig2d=plt.figure()
ax2 = fig2d.add_subplot(111);
ax2.plot(xf,yf);
fig2d.show();

#time domian just real part
fig2df=plt.figure()
ax2f = fig2df.add_subplot(111);
ax2f.plot(x,np.real(y));
fig2df.show();

#time domain display real and imag part
#fig3df=plt.figure()
#ax3f = fig3df.add_subplot(111,projection="3d");
#ax3f.plot(x,np.imag(y),np.real(y));
#fig3df.show();
