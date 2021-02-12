import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wavfile
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

def plot_conv(y=None, xlabel="sample index n", ylabel="y[n]", title=None, nSample=None, stem=True, ylim=False):
    plt.figure()
    if stem == True:
        plt.stem(y[:nSample], linefmt='r--o')
    else:
        plt.plot(y[:nSample], 'r')
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if ylim == True:
        plt.ylim(-1, 1)
    plt.grid()
    plt.show()
    
def myFilter_Convolve(H,X):
    #H = [0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523]
    #x[n] = delta[n] - 2delta[n-15]
    #X = [[0,15], [1,-2]] -> #X = [[TIV], [Homogeneity]]
    X_TIV, X_SCALE = X
    
    length = len(H) + X_TIV[-1]
    y = np.zeros(int(length))
    
    for SCALE, TIV in enumerate(X_TIV):
        IIR = np.multiply(H,X_SCALE[SCALE])
        y[TIV:len(H)+TIV] += IIR
    
    return y

def plot_spec(y=None, Fs=None):
    sampleX_float = fnNormalize16BitToFloat(y)
    [f, t, Sxx_clean] = signal.spectrogram(sampleX_float, Fs, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    plt.figure()
    plt.pcolormesh(t, f, 10*np.log10(Sxx_clean))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('spectrogram of signal')
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
    
# =============================================================================
#     #3a)
#     #NOTE that if have -1 impulse, the signal flip.
#     #However, human unable to differentiate the different in a 180 deg phase shift
#     filename = "helloWorld_16bit.wav"
#     #winsound.PlaySound(filename, winsound.SND_FILENAME)
#     [Fs, sampleX_16bit] = wavfile.read(filename)
#     
#     #normalise value to 1 to -1, if not your ear will die without this
#     sampleX = fnNormalize16BitToFloat(sampleX_16bit)
#     
#     #define impulse
#     impulseH = np.zeros(8000)
#     impulseH[1] = 1
#     impulseH[4000] = 0.7
#     impulseH[7900] = 0.3
#     #Convolve x[n] and h[n]
#     y = np.convolve(sampleX,impulseH)
#     plot_conv(y=y, title="impulse response with echo filter", stem=False)
# =============================================================================
    
# =============================================================================
#     #3b
#     write_wav(y=y, Fs=Fs, filename="conv_helloworld.wav")
#     
#     #define n indices
#     length = len(sampleX)
#     y1 = np.zeros(int(length+7900))
#     
#     #define scaled sample
#     x1 = sampleX*0.7
#     x2 = sampleX*0.3
#     
#     #Sum of decompose sample
#     y1[1:1+len(sampleX)] = y1[1:1+length]+sampleX
#     y1[4000:4000+length] = y1[4000:4000+length]+x1
#     y1[7900:7900+length] = y1[7900:7900+length]+x2
#     plot_conv(y=y1[:-1], title="impulse response with echo filter", stem=False)
#     
#     #Write to file to check if same audio
#     write_wav(y=y, Fs=Fs, filename="decompose_helloworld.wav")
# =============================================================================

# =============================================================================
#     #4a
#     #define impulse and impulse responses
#     x = [1,0,0,0,0,0,0,0]
#     h1 = [0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523]
#     h2 = [-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523]
#     
#     #Convolve x[n] and h[n]
#     y1 = np.convolve(x, h1)
#     y2 = np.convolve(x, h2)
#     
#     plot_conv(y=y1, title="Impulse response for h1[n]", stem=True, ylim=True)
#     plot_conv(y=y2, title="Impulse response for h2[n]", stem=True, ylim=True)
# =============================================================================

# =============================================================================
#     #4b
#     h1 = [0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523]
#     h2 = [-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523]
#     X = [[0,15],[1,-2]]
#     y1 = myFilter_Convolve(h1,X)
#     y2 = myFilter_Convolve(h2,X)
#     plot_conv(y=y1, title="Impulse response for h1[n]", stem=True, ylim=False)
#     plot_conv(y=y2, title="Impulse response for h2[n]", stem=True, ylim=False)
# =============================================================================

# =============================================================================
#     #4c
#     h1 = [0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523]
#     h2 = [-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523]
#     x1 = plot_equation(graph="n", A=0.1, F=700, Fs=16000, Phi=0, sTime=0, eTime=1, nSample=50, cos=True, stem=True)
#     x2 = plot_equation(graph="n", A=0.1, F=3333, Fs=16000, Phi=0, sTime=0, eTime=1, nSample=50, cos=True, stem=True)
#     LTI_input = x1+x2
#     y1 = np.convolve(LTI_input, h1)
#     y2 = np.convolve(LTI_input, h2)
#     
#     plot_spec(y=y1, Fs=16000)
#     plot_spec(y=y2, Fs=16000)
#     play_wav(LTI_input, 16000)
#     play_wav(y1, 16000)
#     play_wav(y2, 16000)
# =============================================================================

# =============================================================================
#     #5a
#     #Read wav file
#     [Fs, sampleX_16bit] = wavfile.read("helloworld_noisy_16bit.wav")
#     
#     #normalise value to 1 to -1, if not your ear will die without this
#     sampleX = fnNormalize16BitToFloat(sampleX_16bit)
#     play_wav(sampleX, Fs)
#     plot_spec(y=sampleX, Fs=Fs)
#     
#     #B, A coefficients
#     b = [1, -0.7653668, 0.99999]
#     a = [1, 0.722744, 0.888622]
#     
#     y1 = signal.lfilter(b, a, sampleX)
#     play_wav(y1, Fs)
#     plot_spec(y=y1, Fs=Fs)
#     
#     #5c,d
#     b, a = signal.iirnotch(3000, 30, Fs)
#     y2 = signal.lfilter(b, a, sampleX)
#     play_wav(y, Fs)
#     plot_spec(y=y2, Fs=Fs)
# =============================================================================
