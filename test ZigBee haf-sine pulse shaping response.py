

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
y2 = 0.0*x;
for i in range(0,int(N*T)):
    if np.random.rand()<0.5 :
        y = y + np.sin(np.pi*(x-i))*rect(x-i-0.5)
        y2 = y2 + rect(x-i-0.5)
    else:
        y = y - np.sin(np.pi*(x-i))*rect(x-i-0.5)
        y2 = y2 - rect(x-i-0.5)

yf = fft(y)
yf2 = fft(y2)
xf = np.linspace(0.0, 1.0/T,N)

#yf = yf*rect(x/4)+yf*rect((x-100)/4)
#y=ifft(yf);
yf = np.log10(np.abs(yf)**2)*10
yf2 = np.log10(np.abs(yf2)**2)*10


#frequence domain
fig2d=plt.figure()
ax2 = fig2d.add_subplot(111);
ax2.plot(xf,yf);
ax2.plot(xf,yf2);
fig2d.show();

#time domian just real part
fig2df=plt.figure()
ax2f = fig2df.add_subplot(111);
ax2f.axis([0,100,-2,2])
ax2f.plot(x,np.real(y));
ax2f.plot(x,np.real(y2));
fig2df.show();

#time domain display real and imag part
#fig3df=plt.figure()
#ax3f = fig3df.add_subplot(111,projection="3d");
#ax3f.plot(x,np.imag(y),np.real(y));
#fig3df.show();
