import numpy as np
import scipy.io
import matplotlib.pyplot as plt

#chirp z transform
def setGridLine(ax):
    #http://jonathansoma.com/lede/data-studio/matplotlib/adding-grid-lines-to-a-matplotlib-chart/
    # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)
    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth=1, color='grey')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth=0.8, color='#a6a6a6')
    # Turn off the display of all ticks.
    ax.tick_params(which='both', # Options for both major and minor ticks
                    top=False, # turn off top ticks
                    left=False, # turn off left ticks
                    right=False,  # turn off right ticks
                    bottom=False) # turn off bottom ticks

def chirpz(x,k,f1,f2,fs):
    m,n = x.shape;
    w = np.exp(-1j*2*np.pi*(f2-f1)/(k*fs));
    a = np.exp(1j*2*np.pi*f1/fs);
    nfft = int(np.power(2,np.ceil(np.log2(m+k-1))))
    kk = np.arange((-m+1),max(k,m),1).reshape(-1,1)
    kk2 = np.power(kk,2)/2
    ww = np.power(w,kk2)
    nn = np.arange(m).reshape(-1,1)
    aa = np.power(a,-nn)
    aa = aa*ww[(m+nn-1).ravel()]
    y = x*np.tile(aa,(1,n))
    fy = np.fft.fft(y,nfft,axis=0)
    fv = np.fft.fft(1/ww[:(k+m)],nfft,axis=0)
    fy = fy*np.tile(fv,(1,n))
    g = np.fft.ifft(fy,axis=0)    
    g=g[(m-1):(m+k-1),:]*np.tile(ww[(m-1):(m+k-1)],n)
    return g

def chirpzFreq(k,f1,f2):
    return (np.arange(k)*(f2-f1)/k+f1).reshape(-1,1)
    
if __name__ == '__main__':
    dt=0.01
    t1 = np.arange(0,10,dt)
    
    sig1 = np.exp(1j*2*np.pi*-25*t1).reshape(-1,1)
    sig2 = np.exp(1j*2*np.pi*-24.4*t1).reshape(-1,1)
    sig3 = np.exp(1j*2*np.pi*-24.7*t1).reshape(-1,1)
    h1a = sig1;
    h2a = np.hstack([sig1,sig2,sig3])
    print (h1a.shape) #(1000, 1)
    print (h2a.shape) #(1000, 3)
    
    freq1 = np.array([-25,-24.4,-24.7])
    FREQ1 ,T1 = np.meshgrid(freq1,t1)
    h2 = np.exp(1j*2*np.pi*FREQ1*T1)
    h1 = h2[:,0]
    print (h1.shape)# (1000,)
    h1 = h2[:,[0]] #tips
    print (h1.shape)#(1000, 1)
    print (h2.shape)#(1000, 3)
    
    print (np.allclose(h1a,h1))
    print (np.allclose(h2a,h2))       
        
    fs=1/dt;
    f1 = -26;
    f2 = -23;
    mFFT=1024;
    
    freq1 = chirpzFreq(mFFT,f1,f2)
    z1 = chirpz(h1,mFFT,f1,f2,fs)
    z2 = chirpz(h2,mFFT,f1,f2,fs)
    
    
    nFFT=1024*8
    freq0 =np.arange(nFFT)/nFFT*fs - fs/2
    y1 = np.fft.fft(h1,n=nFFT,axis=0)
    y1 = np.fft.fftshift(y1,axes=0)
    print (y1.shape)
    y2 = np.fft.fft(h2,n=nFFT,axis=0)
    y2 = np.fft.fftshift(y2,axes=0)
    print (y2.shape)

    fig,ax = plt.subplots(2,1)
    
    ax[0].plot(freq1,np.abs(z1[:,0]),label="chirpz")
    ax[0].plot(freq0,np.abs(y1[:,0]),label="FFT")
    
    ax[1].plot(freq1,np.abs(z2[:,0]),label="chirpz 0")
    ax[1].plot(freq1,np.abs(z2[:,1]),label="chirpz 1")
    ax[1].plot(freq1,np.abs(z2[:,2]),label="chirpz 2")
    ax[1].plot(freq0,np.abs(y2[:,0]),label="FFT 0")   
    ax[1].plot(freq0,np.abs(y2[:,1]),label="FFT 1")   
    ax[1].plot(freq0,np.abs(y2[:,2]),label="FFT 2")   
    ax[0].set_xlim(np.min(freq1),np.max(freq1))
    ax[1].set_xlim(np.min(freq1),np.max(freq1))
    setGridLine(ax[0])
    setGridLine(ax[1])
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()
    
    
    

