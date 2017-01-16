
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
    return list(rect_(x))

# Number of samplepoints
N = 4000 #4 seconds sample
# sample spacing
T = 1.0 / 1000.0 # 1 kSPS
# sample Rate
S = 1.0/T

#lets try OFDM modulation
#Because the smapling rate is F, we get bandwidth < S/2. the signal cannot changed to faster than S/2
# typically, 1 sybol may include 1,2,4...bits, while the symbol rate should < S/2

x = np.linspace(0.0, N*T, N)

xf = np.linspace(0.0,S,N)
yf = mirr(np.sinc(xf-20)*S)

y=ifft(yf);

#frequence domain
fig2d=plt.figure()
ax2 = fig2d.add_subplot(111);
ax2.plot(xf,yf);
fig2d.show();

#time domian just real part
fig2df=plt.figure()
ax2f = fig2df.add_subplot(111);
ax2f.plot(x,y);
fig2df.show();

#time domain display real and imag part
fig3df=plt.figure()
ax3f = fig3df.add_subplot(111,projection="3d");
ax3f.plot(x,np.imag(y),np.real(y));
fig3df.show();
