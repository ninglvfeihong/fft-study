
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
N = 105 #0.105seconds sample. only one symbol. to avoid  inter-symbol interference (ISI) . Add 0.005 s guard period for one symbol
# sample spacing
T = 1.0 / 1000.0 # 1 kSPS
# sample Rate
S = 1.0/T

#lets try OFDM modulation
#Because the smapling rate is F, we get bandwidth S/2. the signal cannot changed to faster than S/2
# typically, 1 symbol may include 1,2,4...bits, while the symbol rate should < S/2
# Recap: we now have channel this channel have bandwidth S/2.
# Typically, we can define any symbol rate <S/2. such as S/10.
# But in OFDM, we dived this channel into many Subcarriers. here we define 10 subcribers
SB = 10;
# so each subchannel has bandwith S/20, here we get 50 Hz.
# Thus, typically, each subchannel can only has a symbol rate <S/20 50 Hz
# LET suppose employ AM modulation with M=4 (0,1,2,3 two bits / symbol) in each subchannels
# Let's suppose we want send a 2 in C0 and zero in other subchannels
# | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 |
# | 2  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  |

# to send such a date, we need fourier reverse transform calcuate the waveform in domain by np.sinc function

# before send data, we need to define AM symbol rate (typically < subchannel Bandwidth).
SR = S / 2 / SB / 5; #10Hz. just for easy demo, we can defind the symbol rate upto 50。here we set symbol rate is 10Hz while Bandwidth is 50Hz, we has 5 times oversampling.  

# let's think about the C0
# we the fourier tansfom rect(t) => sinc(v), to send data lets contruct waveform for time domain.
# for fourier tansform, f(at) => 1/abs(a)*g(v/a).
# because of the SymbolRate. we can get a hihg volate may last for minium 1/10 second.
# thus function of the singnal for one symbole is: rect(10t)
# thus the fourier transform shouuld be.   1/10*sinc(t/10) . In other word: 1/SR*sinc(t/SR)

# for other subchannel C1,C2,C3 ... we can easily get ransform:
# |  C0            |        C1          |        C2          |        C3          | ...|
# | 1/10*sinc(v/10)| 1/10*sinc(v/10-1), | 1/10*sinc(v/10-2), | 1/10*sinc(v/10-3), | ...|
# | 1/SR*sinc(v/SR)| 1/SR*sinc(v/SR-1), | 1/SR*sinc(v/SR-2), | 1/SR*sinc(v/SR-3), | ...|

# luckly sinc(v) is Orthogonal for every 1.0 period.
# 正交就是两函数内积为零。从之前学的量子论来说，如果一堆函数的集合中，任何一个函数除了和他的共轭函数内机不为零。。。关系，而且这个函数可以叠加成其他任何函数，这个函数集合就是个完整的态空间。
# 如cos(wt),只有w相同的两个函数积分才有值，如果不同均为零，这样就造就了t <-> w （即时域与频域的可逆转换，也就是fourier转换）。这里因为态函数是连续的，所以图形也能连续转换，这样的（cos）函数十分难得，沟通了频域与时域的可逆桥梁
# 在这里，我们不需要连续的正交。只需要C0,C1,C2 ... C9 里的sinc函数正交。就能把这些sinc函数叠加后的曲线唯一得再拆解（还原）成基本sinc的形式，切每个sinc的系数（coefficient）唯一。这样就能用sinc域的coefficient来标记一个合成波，从而传输数据。OFDM的核心。
# thus we can Add 10 sinc waveform togther and then disassemble them to get the weight of each sinc function.
# 所以，对于正交的这10个sinc波形，我们可以把这10 这样的不同幅值的sinc波形叠加起来得到一个波形，然后还能唯一还原出这10个sinc波形。
# 这样我们就能用不同sinc函数幅值来传输信息，即AM：调制sinc的的幅值。

# we define symbol in frequency domain with M=4 (0,1,2,3) here, for C0:
# |        0     |       1         |       2         |       3         |
# | 0*sinc(v/10) | 1/10*sinc(v/10) | 2/10*sinc(v/10) | 3/10*sinc(v/10) |
# | 0*sinc(v/SR) | 1/SR*sinc(v/SR) | 2/SR*sinc(v/SR) | 3/SR*sinc(v/SR) |


x = np.linspace(0.0, N*T, N)

xf = np.linspace(0.0,S,N)
yf = mirr(2/SR*np.sinc(xf/SR-9)*S)

y=ifft(yf);

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
fig3df=plt.figure()
ax3f = fig3df.add_subplot(111,projection="3d");
ax3f.plot(x,np.imag(y),np.real(y));
fig3df.show();
