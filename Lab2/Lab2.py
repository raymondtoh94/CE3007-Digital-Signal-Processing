import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wavfile
import scipy
import winsound
from scipy import signal
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
    
def plot_conv(y=None, xlabel="sample index n", ylabel="y[n]", title=None, nSample=None, stem=True):
    plt.figure()
    if stem == True:
        plt.stem(y[:nSample], linefmt='r--o')
    else:
        plt.plot(y[:nSample], 'r')
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    #THINGS TO TAKE NOTE!
    #F         = Cycle/Sec
    #Fs        = Sample/Sec
    #Fs/F      = Sample/Cycle
    #Ts        = Sample Period
    #2*pi*F    = Rad/Sec
    #2*pi*F/Fs = Rad/Sample
    
# =============================================================================
#     #FOR UNDERSTANDING PURPOSE!
#     x = plot_equation(graph="n", A=1, F=5, Fs=100, Phi=0, sTime=0, eTime=1, nSample=100, cos=True, stem=True)
#     h1 = [0.2,0,0] 
#     h2 = [0,0.3,0] 
#     h3 = [0,0,-0.5] 
#     y1 = np.convolve(x,h1) #replicated signal start from y1n[0] scaled by 0.2
#     y2 = np.convolve(x,h2) #replicated signal start from y2n[1] scaled by 0.3, y2n[0] is 0
#     y3 = np.convolve(x,h3) #replicated signal start from y3n[2] scaled by -0.5, y3n[0] and y3n[1] is 0
#     y = np.sum((y1,y2,y3),axis=0) #sum of all replicated signal
#     plot_conv(y=y1, title=f"1cos(0.1*{pi_symbol}*n)", nSample=100)
#     plot_conv(y=y2, title=f"1cos(0.1*{pi_symbol}*n)", nSample=100)
#     plot_conv(y=y3, title=f"1cos(0.1*{pi_symbol}*n)", nSample=100)
#     plot_conv(y=y, title=f"1cos(0.1*{pi_symbol}*n)", nSample=100)
# =============================================================================

# =============================================================================
#     #Part 1A
#     #1a) Summation of k->-infinite to infinite of x(n)*h(n-k)
#     x = plot_equation(graph="n", A=1, F=5, Fs=100, Phi=0, sTime=0, eTime=1, nSample=100, cos=True, stem=True)
#     h = [0.2,0.3,-0.5]
#     y = np.convolve(x,h)
#     plot_conv(y=y, title=f"1cos(0.1{pi_symbol}n)")
#     #Output is the summation of scaled delayed inpput impulse responses
#     
#     #1b) A scaled version of the input function
# =============================================================================
    
    #3a)
    #NOTE that if have -1 impulse, the signal flip.
    #However, human unable to differentiate the different in a 180 deg phase shift
    filename = "helloWorld_16bit.wav"
    #winsound.PlaySound(filename, winsound.SND_FILENAME)
    [Fs, sampleX_16bit] = wavfile.read(filename)
    
    #normalise value to 1 to -1, if not your ear will die without this
    sampleX = fnNormalize16BitToFloat(sampleX_16bit)
    
    #define impulse
    impulseH = np.zeros(8000)
    impulseH[1] = 1
    impulseH[4000] = 0.7
    impulseH[7900] = 0.3
    
    y = np.convolve(sampleX,impulseH)
    plot_conv(y=y, title="impulse response with echo filter", stem=False)
    write_wav(y=y, Fs=Fs, "echo_effect_helloworld.wav")
    
    #3c
    