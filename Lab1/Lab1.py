import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wavfile
import winsound
from mpl_toolkits.mplot3d import Axes3D
import os

pi_symbol = "\u03C0"

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))

def write_wav(y, Fs, filename):
    wavfile.write(filename, Fs, y)
    
def play_wav(y, Fs, filename="t1_16bit.wav"):
    y = fnNormalizeFloatTo16Bit(y)
    wavfile.write(filename, Fs, y)
    winsound.PlaySound(filename, winsound.SND_FILENAME)
    os.remove(filename)

def plot_graph(x=None, y=None, xlabel=None, ylabel=None, title=None, stem=True):
    plt.figure()
    plt.title(f"{title}")
    if y.ndim == 1:
        #This part is for normal signal
        if stem == True:
            plt.stem(x,y, 'r--o')
        else:
            plt.plot(x,y, 'r--o')
        print(f"n values = {np.round(y,4)}")
    else:
        #This part is for composite signal
        if stem == True:
            plt.stem(x,np.sum(y, axis=0), 'r--o')
        else:
            plt.plot(x,np.sum(y, axis=0), 'r--o')
        print(f"n values = {np.round(np.sum(y, axis=0),4)}")
        
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_equation(graph=None, A=None, F=None, Fs=None, Phi=None, sTime=None,
                  eTime=None, nSample=None, cos=True, stem=True):
    #x-axis steps (num of sampling period)
    t = np.arange(sTime, eTime, 1.0/Fs)
    
    #get amplitude on each steps
    if cos == True:
        #cos wave
        y = A*np.cos(2* np.pi * F * t + Phi)
        title = f"{A}cos(2*{pi_symbol}*({F})*t+{Phi})"
    else:
        #sin wave
        y = A*np.sin(2* np.pi * F * t + Phi)
        title = f"{A}sin(2*{pi_symbol}*({F})*t+{Phi})"
    
    if graph.lower() == "nts":
        plot_graph(x=t[0:nSample], y=y[0:nSample], xlabel='time in sec', ylabel='y[nTs]', title=title, stem=stem)
    elif graph == "n":
        plot_graph(x=np.arange(0, nSample), y=y[0:nSample], xlabel='sample index n', ylabel='y[n]', title=title, stem=stem)
    else:
        raise ValueError("graph parameter must be either 'n' or 'nts'")
    return y

def plot_complex(A=None, F=None, Fs=None, Phi=None, nSample=None, 
                 comp_exp=None, xlabel=None, ylabel=None, title=None):
    #calculate radian/sample
    w1=2*np.pi*F/Fs
    n = np.arange(0, nSample, 1)
    y = np.multiply(np.power(A, n), np.exp(1j * w1 * n + Phi))
    
    fig = plt.figure()
    #This part is for complex signal
    if comp_exp == "2dplot":
        cmap = ["r--o", "g--o"]
        y_2d = [y[0:None].real, y[0:None].imag]
        for i in range(len(y_2d)):
            plt.plot(n,y_2d[i], cmap[i])

    elif comp_exp == "polarplot":
        for x in y:
            plt.polar([0,np.angle(x)],[0,np.abs(x)],marker='o')
            
    elif comp_exp == "3dplot":
        plt.rcParams['legend.fontsize'] = 10
        ax = fig.gca(projection='3d')
        reVal = y[0:nSample].real
        imgVal = y[0:nSample].imag
        ax.plot(n,reVal, imgVal)
        ax.scatter(n,reVal,imgVal, c='k', marker='o')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel('imag')
    
    else:
        raise ValueError("comp_exp only have '2dplot', 'polarplot' and '3dplot'")

    plt.title(f"{title}")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    return y

def getDTMF(user_sel):
    freq = [697, 770, 852, 941, 1209, 1336, 1477, 1633]
    key = {
        '1': (freq[0], freq[4]),
        '2': (freq[0], freq[5]),
        '3': (freq[0], freq[6]),
        'A': (freq[0], freq[7]),
        '4': (freq[1], freq[4]),
        '5': (freq[1], freq[5]),
        '6': (freq[1], freq[6]),
        'B': (freq[1], freq[7]),
        '7': (freq[2], freq[4]),
        '8': (freq[2], freq[5]),
        '9': (freq[2], freq[6]),
        'C': (freq[2], freq[7]),
        '*': (freq[3], freq[4]),
        '0': (freq[3], freq[5]),
        '#': (freq[3], freq[6]),
        'D': (freq[3], freq[7])
        }
    
    if user_sel in key:
        return key[user_sel]
    else:
        raise ValueError("User selected invalid key")

def genDTMF(user_sel, durTone, Fs, A=None, Phi=0, nSample=None, stem=True):
    if len(user_sel) == 1:
        (f1,f2) = getDTMF(user_sel)
        t = np.arange(0, durTone, 1.0/Fs)
        y = A*( np.cos(2*np.pi*f1*t+Phi) + np.cos(2*np.pi*f2*t+Phi) )
        title = f"({A}cos(2*{pi_symbol}*{f1}*n*Ts+0) + {A}cos(2*{pi_symbol}*{f2}*n*Ts+0)"
        plot_graph(x=t[0:nSample], y=y[0:nSample], xlabel='time in sec', ylabel='y[nTs]', title=title, stem=stem)
    else:
        t = [] 
        y = []
        for key_sel in user_sel:
            temp_t = np.arange(0, durTone, 1.0/Fs)
            (f1,f2) = getDTMF(key_sel)
            temp_y = A*( np.cos(2*np.pi*f1*temp_t+Phi) + np.cos(2*np.pi*f2*temp_t+Phi) )
            t = np.concatenate((t, temp_t))
            y = np.concatenate((y, temp_y))
            title = f"({A}cos(2*{pi_symbol}*{f1}*n*Ts+0) + {A}cos(2*{pi_symbol}*{f2}*n*Ts+0)"
            plot_graph(x=temp_t[0:nSample], y=temp_y[0:nSample], xlabel='time in sec', ylabel='y[nTs]', title=title, stem=stem)
    play_wav(y, Fs=Fs, filename="DTMF.wav")
    return [t,y]

if __name__ == "__main__":
    #THINGS TO TAKE NOTE!
    #F         = Cycle/Sec
    #Fs        = Sample/Sec
    #Fs/F      = Sample/Cycle
    #Ts        = Sample Period
    #2*pi*F    = Rad/Sec
    #2*pi*F/Fs = Rad/Sample
    
# =============================================================================
#     #Part 3.1A
#     for freq in range(2000, 34000, 2000): #34k because to include last step 32khz
#         y = plot_equation(graph="n", A=0.1, F=freq, Fs=16000, Phi=0, sTime=0, eTime=1, nSample=100, cos=True, stem=True)
#         print(f"Frequence sound = {freq}")
#         play_wav(y, Fs=16000, filename=f"t1_{freq}.wav")
#     #Phenomenon: Aliasing (Fs < 2F result in Aliasing)
#     #2kHz - 8kHz no Aliasing
#     #10kHz onwards aliasing happen. when hit 16kHz, the sampled point is always at the amplitude value
#     #After 16k or nFs, Aliasing happen for sample at lower rate. (Frequency too high)
# =============================================================================

    #Part 3.1B
# =============================================================================
#     #1/1000 = 1 cycle, hence 6/1000 = 6 cycle
#     #16*6 = 1 cycle have 16 sample, hence total sample is 16*6
#     y = plot_equation(graph="nTs", A=0.1, F=1000, Fs=16000, Phi=0, sTime=-3/1000, eTime=3/1000, nSample=None, cos=True, stem=False)
#     y = plot_equation(graph="n", A=0.1, F=1000, Fs=16000, Phi=0, sTime=-3/1000, eTime=3/1000, nSample=16*6, cos=True, stem=True)
#     #i.)   y(t) continuous time where t is real number and discrete time where t is integer
#     #ii.)  y(nT) is where index n sampled during each T period, where y(n) is where n is the sample index of each rad/sample
#     #iii.) 16*6 = 96
#     #iv.)  It reconstructed perfectly. However, it has delay by 2*pi
#     y = plot_equation(graph="n", A=0.1, F=17000, Fs=16000, Phi=0, sTime=-3/1000, eTime=3/1000, nSample=16*6, cos=True, stem=True)
#     
#     #Lab
#     y = plot_equation(graph="n", A=0.5, F=1000, Fs=16000, Phi=-np.pi/2, sTime=-0.625/1000, eTime=0.625/1000, nSample=20, cos=True, stem=True)
# =============================================================================
    
# =============================================================================
#     #Part 3.1C
#     y = plot_equation(graph="n", A=0.1, F=1, Fs=16, Phi=0, sTime=0, eTime=1, nSample=16, cos=True, stem=True)
#     #Value is the same however only 1 cycle since F=1
# 
#     #Lab
#     y = plot_equation(graph="n", A=0.1, F=500, Fs=8000, Phi=0, sTime=0, eTime=1, nSample=16, cos=True, stem=True)
# =============================================================================

# =============================================================================
#     #Part 3.2
#     [t,y] = genDTMF(user_sel="1234567890C*#D",  durTone=0.25, Fs=16000, A=0.1, Phi=0, nSample=100, stem=False)
#     
#     #Lab
#     [t,y] = genDTMF(user_sel="1",  durTone=0.2, Fs=8000, A=0.1, Phi=0, nSample=100, stem=False)
# =============================================================================
    
# =============================================================================
#     #Part 3.3
#     y1 = plot_equation(graph="nTs", A=0.1, F=10, Fs=60, Phi=0, sTime=0, eTime=0.2, nSample=None, cos=True, stem=True)
#     #n values = [ 0.1   0.05 -0.05 -0.1  -0.05  0.05  0.1   0.05 -0.05 -0.1  -0.05  0.05]
#     
#     y2 = plot_equation(graph="nTs", A=0.1, F=15, Fs=60, Phi=0, sTime=0, eTime=0.2, nSample=None, cos=True, stem=True)
#     #n values = [ 0.1  0.  -0.1 -0.   0.1  0.  -0.1 -0.   0.1  0.  -0.1 -0. ]
#     
#     plot_graph(x=np.arange(0,0.2,1/60), y=np.array([y1,y2]), xlabel="Sample n", ylabel="y[nTs]", title="Composite Signal", stem=True)
#     #n values = [ 0.2   0.05 -0.15 -0.1   0.05  0.05  0.    0.05  0.05 -0.1  -0.15  0.05]
# =============================================================================
    
# =============================================================================
#     #Part 3.4 (2*pi*F/Fs)
#     #F/Fs = 1/36
#     #y have real and imag. Access by y.real or y.imag
#     #y = plot_complex(A=0.95, F=1, Fs=36, Phi=0, nSample=200, comp_exp="2dplot", xlabel="Sample n", ylabel="y[n]", title="Complex exponential (red=real) (green=imag)")
#     #y = plot_complex(A=1, F=1, Fs=36, Phi=0, nSample=200, comp_exp="polarplot", xlabel=None, ylabel=None, title="Polar plot showing phasors at n=0..N")
#     #y = plot_complex(A=0.95, F=1, Fs=36, Phi=0, nSample=200, comp_exp="3dplot", xlabel="Sample n", ylabel="y[n]", title='complex exponential phasor')
#     
#     #Lab
#     y = plot_complex(A=1.1, F=1, Fs=36, Phi=0, nSample=10, comp_exp="polarplot", xlabel=None, ylabel=None, title="Polar plot showing phasors at n=0..N")
# =============================================================================
    
# =============================================================================
#     #F/Fs = 1/18
#     y = plot_complex(A=0.95, F=1, Fs=18, Phi=0, nSample=200, comp_exp="2dplot", xlabel="Sample n", ylabel="y[n]", title="Complex exponential (red=real) (green=imag)")
#     y = plot_complex(A=0.95, F=1, Fs=18, Phi=0, nSample=200, comp_exp="polarplot", xlabel=None, ylabel=None, title="Polar plot showing phasors at n=0..N")
#     y = plot_complex(A=0.95, F=1, Fs=18, Phi=0, nSample=200, comp_exp="3dplot", xlabel="Sample n", ylabel="y[n]", title='complex exponential phasor')
# =============================================================================

# =============================================================================
#     #Part 3.5
#     #k = num of cycle, N = num of sample/cycle or the discrete period
#     k=4
#     N=16
#     y = plot_complex(A=1, F=k, Fs=N*k, Phi=0, nSample=N*k, comp_exp="2dplot", xlabel="Sample n", ylabel="y[n]", title="Complex exponential (red=real) (green=imag)")
#     y = plot_complex(A=1, F=k, Fs=N*k, Phi=0, nSample=N*k, comp_exp="polarplot", xlabel=None, ylabel=None, title="Polar plot showing phasors at n=0..N")
#     y = plot_complex(A=1, F=k, Fs=N*k, Phi=0, nSample=N*k, comp_exp="3dplot", xlabel="Sample n", ylabel="y[n]", title='complex exponential phasor')
# =============================================================================