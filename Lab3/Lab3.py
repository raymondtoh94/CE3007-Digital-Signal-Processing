import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

omega_symbol = "\u03C9"

def plot_graph(func=None, x=None, mag=None, phase=None, xlabel=None, ylabel=None, title=None):
    plt.figure()
    
    if func == "inverse_transform":
        plt.title(f"{title}")
        plt.stem(np.abs(x))
        plt.xlabel(f"{xlabel}")
        plt.ylabel(f"{ylabel}")
        
    else:
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].stem(np.arange(0,len(mag)), mag)
        axarr[0].set_title(f"{title}")
        axarr[0].set_ylabel(f"{xlabel}")
        axarr[1].stem(np.arange(0,len(phase)), phase)
        axarr[1].set_xlabel(f"{ylabel}")
        axarr[1].set_ylabel('Phase (rad)')
    plt.show()
    plt.close()

def DTFS(x, omega=False):
    x = np.fft.fft(x, len(x))
    mag = (1/len(x))*np.abs(x)
    
    if omega:
        ylabel=omega_symbol
        phase = np.angle(x)*2*np.pi/len(x)
    else:
        ylabel="k"
        phase = np.angle(x)
    
    for i in range(len(mag)):
        if mag[i] < 1e-5:
            mag[i] = 0
            phase[i] = 0
    
    plot_graph(func="DTFS", mag=mag, phase=phase, xlabel="|Ck|", ylabel=ylabel, title="DTFS of x[n]")
    print(f"Ck mag - {np.round(mag, 3)}")
    print(f"Ck phase - {np.round(phase, 3)}")
    
    return mag, phase, x

def DFT(x):
    x = np.fft.fft(x, len(x))
    mag = np.abs(x)
    phase = np.angle(x)
    
    for i in range(len(mag)):
        if mag[i] < 1e-5:
            mag[i] = 0
            phase[i] = 0
    
    plot_graph(func="DFT", mag=mag, phase=phase, xlabel="|X[k]|", ylabel="k", title="DFT of x[n]")
    print(f"X[k] mag - {np.round(mag, 3)}")
    print(f"X[k] phase - {np.round(phase, 3)}")
    return mag, phase, x

def DTFT(x, periodic=True):
    x = np.fft.fft(x, len(x))
    
    if periodic:
        mag = 2*np.pi*(1/len(x))*np.abs(x)
    else:
        mag = len(x)*(1/len(x))*np.abs(x)
    phase = np.angle(x)
    
    for i in range(len(mag)):
        if mag[i] < 1e-5:
            mag[i] = 0
            phase[i] = 0
    
    plot_graph(func="DTFT", mag=mag, phase=phase, xlabel="|X(eʲʷ)|", ylabel=omega_symbol, title="DTFT of x[n]")
    print(f"{omega_symbol}\u2080 - 2*pi/{len(x)}")
    print(f"X(eʲʷ) mag - {np.round(mag, 3)}")
    print(f"X(eʲʷ) phase - {np.round(phase, 3)}")
    return mag, phase, x

def inverse_transform(x):
    x = np.fft.ifft(x, len(x))
    plot_graph(func="inverse_transform", x=x, xlabel="sample n", ylabel="x[n]", title="x[n]")
    print(f"x[n] - {np.round(np.abs(x),3)}")
    return x

def myDFTConvolve(x, h):
    #Zero padding
    xz = np.zeros(len(x)+len(h)-1)
    hz = np.zeros(len(xz))
    xz[0:len(x)] = x
    hz[0:len(h)] = h
    
    #Transform to frequency domain
    X = np.fft.fft(xz)
    H = np.fft.fft(hz)
    
    #Convolve in Time = Multiply in frequency
    Y = X*H
    
    y = np.real(np.fft.ifft(Y))
    return y

def check_func(y, y_check):
    assert len(y) == len(y_check)
    
    #Rounding incase of floating point error
    return all(np.round(y,5) == np.round(y_check,5))

if __name__ == "__main__":
# =============================================================================
#     #Q1
#     x = [1,1,0,0,0,0,0,0,0,0,0,0]
#     mag_DTFS, phase_DTFS, x_DTFS = DTFS(x)
#     mag_DFT, phase_DFT, x_DFT = DFT(x)
#     inverse_transform(x_DTFS)
#     inverse_transform(x_DFT)
# =============================================================================
    
# =============================================================================
#     #Q2
#     x = [1,1,0,0,0,0,0,0,0,0,0,0]
#     x1 = [0,1,1,0,0,0,0,0,0,0,0,0,]
#     x2 = [10,10,0,0,0,0,0,0,0,0,0,0]
#     mag_DTFS, phase_DTFS, x_DTFS = DTFS(x)
#     mag_DTFS1, phase_DTFS1, x_DTFS1 = DTFS(x1)
#     mag_DTFS2, phase_DTFS2, x_DTFS2 = DTFS(x2)
# =============================================================================
    
# =============================================================================
#     #Q3
#     N = 12
#     x = np.ones(N, dtype=complex)
#     xk = []
#     for k in range(N):
#         x_temp = []
#         for n in range(N):
#             x_temp.append(x[n]*np.exp(-1j*(2*np.pi/N)*n*k))
#         xk.append(x_temp)
#     
#     for idx, k in enumerate(xk):
#         plt.figure()
#         plt.title(f"x[k] for k = {idx}")
#         plt.stem(np.angle(k))
#         plt.ylabel(f"Phase")
#         plt.xlabel(f"n")
# =============================================================================
    
# =============================================================================
#     #Q4
#     truncate_len = [12,24,48,96]
#     ip_x = [1.,1.,1.,1.,1.,1.,1.]
#     
#     for i in truncate_len:
#         zero_to_append = i - len(ip_x)
#         ip_x.extend(np.zeros(zero_to_append, dtype=float))
#         DTFS(ip_x)
#         DTFS(ip_x, omega=True)
#         DTFT(ip_x, periodic=False)
# =============================================================================

# =============================================================================
#     #Q5
#     x = [1,2,3,4,4,4,4,4,4,4,0]
#     h = [1,0]
#     y = myDFTConvolve(x, h)
#     
#     #Cross check
#     y_check = fftconvolve(x,h)
#     
#     result = check_func(y, y_check)
#     print(f"Implementation vs FFTConvolve is same? {result}")
# =============================================================================

    #LAB
# =============================================================================
#     x = [0,1,2,3,0,0,0,0]
#     mag_DTFS, phase_DTFS, x_DTFS = DTFS(x)
#     mag_DTFT, phase_DTFT, x_DTFT = DTFT(x)
# =============================================================================

# =============================================================================
#     x = np.zeros(256, dtype=complex)
#     x[0] = 640
#     x[16] = 256*np.exp(1j*np.pi/3)
#     x[240] = 256*np.exp(-1j*np.pi/3)
#     y = np.round(np.fft.ifft(x),5)
#     mag = np.abs(y)
#     phase = np.angle(y)
# =============================================================================

# =============================================================================
#     x = [1,2,3,4]
#     h = [1,1]
#     N=4
#     conv = np.fft.fft(x, N)*np.fft.fft(h,N)
#     y = np.round(np.abs(np.fft.ifft(conv,N)),5)
#     
#     y_conv = fftconvolve(x,h)
# =============================================================================
