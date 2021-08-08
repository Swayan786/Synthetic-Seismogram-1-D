import numpy as np
import matplotlib.pyplot as plt



def syntheticSeismogram(v, rho, d):
    """
    function syntheticSeismogram()

    syntheicSeismogram generates a synthetic seismogram for a simple 1-D
    layered model.

    The wavelet options are based on (Ryan, 1994):
        Ricker:
        Ormsby:
        Klauder:
        Butterworth:

    Lindsey Heagy
    lheagy@eos.ubc.ca
    November 30, 2013
    
    v   = np.array([350, 1000, 2000]) # Velocity of each layer (m/s)
    rho = np.array([1700, 2000, 2500]) # Density of each layer (kg/m^3)
    d   = np.array([0, 100, 200]) # Position of top of each layer (m)
    """

    # Ensure that these are float numpy arrays
    v, rho, d = np.array(v, dtype=float), np.array(rho, dtype=float), np.array(d, dtype=float)

    usingT = False

    nlayer = len(v) # number of layers

    # Check that the number of layers match
    assert len(rho) == nlayer, 'Number of layer densities must match number of layer velocities'
    assert len(d)   == nlayer, 'Number of layer tops must match the number of layer velocities'

    # compute necessary parameters
    Z   = rho*v                       # acoustic impedance
    R   = np.diff(Z)/(Z[:-1] + Z[1:]) # reflection coefficients
    twttop  = 2*d[1:]/v[:-1]
    twttop  = np.cumsum(twttop)

    # create model logs
    resolution = 100
    dpth   = np.linspace(0,np.max(d)+np.max(np.diff(d)),resolution)
    nd     = len(dpth)

    rholog  = np.zeros(nd)
    vlog    = np.zeros(nd)
    zlog    = np.zeros(nd)
    rseries = np.zeros(nd)
    twti    = np.zeros(nd)

    for i in range(nlayer):
        di         = (dpth >= d[i])
        rholog[di] = rho[i]
        vlog[di]   = v[i]
        zlog[di]   = Z[i]
        if i < nlayer-1:
            di  = np.logical_and(di, dpth < d[i+1])
            ir = np.arange(resolution)[di][-1:][0] #find(di, 1, 'last' )
            if usingT:
                if i == 0:
                    rseries[ir] = R[i]
                else:
                    rseries[ir] = R[i]*np.prod(1-R[i-1]**2)
            else:
                rseries[ir] = R[i]
        if i > 0:
            twti[di] = twttop[i-1]

    t  = 2.0*dpth/vlog + twti

    # make wavelet
    # Wavelet type and Frequency (Hz):
    wavtyp = 'RICKER'
    wavf   = np.array([10])

    dtwav  = np.abs(np.min(np.diff(t)))
    twav   = np.arange(-2.0/np.min(wavf), 2.0/np.min(wavf), dtwav)

    # Get source wavelet
    wav = {'RICKER':getRicker, 'ORMSBY':getOrmsby, 'KLAUDER':getKlauder}[wavtyp](wavf,twav)

    # create synthetic seismogram
    tseis = np.arange(0,np.max(t),dtwav) + np.min(twav)
    tr    = t[np.abs(rseries) > 0]
    rseriesconv = np.zeros(len(tseis))
    for i in range(len(tr)):
        index = np.abs(tseis - tr[i]).argmin()
        rseriesconv[index] = R[i]

    seis = np.convolve(wav,rseriesconv)
    tseis = np.min(twav)+dtwav*np.arange(len(seis))
    index = np.logical_and(tseis >= 0, tseis <= np.max(t))
    tseis = tseis[index]
    seis  = seis[index]
    ##
    plt.figure(1)

    # Plot Density
    plt.subplot(151)
    plt.plot(rholog,dpth,linewidth=2)
    plt.title('Density')
    # xlim([min(rholog) max(rholog)] + [-1 1]*0.1*[max(rholog)-min(rholog)])
    # ylim([min(dpth),max(dpth)])
    # set(gca,'Ydir','reverse')
    plt.grid()

    plt.subplot(152)
    plt.plot(vlog,dpth,linewidth=2)
    plt.title('Velocity')
    # xlim([min(vlog) max(vlog)] + [-1 1]*0.1*[max(vlog)-min(vlog)])
    # ylim([min(dpth),max(dpth)])
    # set(gca,'Ydir','reverse')
    plt.grid()

    plt.subplot(153)
    plt.plot(zlog,dpth,linewidth=2)
    plt.title('Acoustic Impedance')
    # xlim([min(zlog) max(zlog)] + [-1 1]*0.1*[max(zlog)-min(zlog)])
    # ylim([min(dpth),max(dpth)])
    # set(gca,'Ydir','reverse')
    plt.grid()

    plt.subplot(154)
    plt.plot(rseries,dpth,linewidth=2) #,'marker','none'
    plt.title('Reflectivity Series');
    # set(gca,'cameraupvector',[-1, 0, 0]);
    plt.grid()
    # set(gca,'ydir','reverse');

    plt.subplot(155)
    plt.plot(t,dpth,linewidth=2);
    plt.title('Depth-Time');
    # plt.xlim([np.min(t), np.max(t)] + [-1, 1]*0.1*[np.max(t)-np.min(t)]);
    # plt.ylim([np.min(dpth),np.max(dpth)]);
    # set(gca,'Ydir','reverse');
    plt.grid()
    ##
    plt.figure(2)
    # plt.subplot(141)
    # plt.plot(dpth,t,linewidth=2);
    # title('Time-Depth');
    # ylim([min(t), max(t)] + [-1 1]*0.1*[max(t)-min(t)]);
    # xlim([min(dpth),max(dpth)]);
    # set(gca,'Ydir','reverse');
    # plt.grid()

    plt.subplot(132)
    # plt.plot(rseriesconv,tseis,linewidth=2) #,'marker','none'
    plt.title('Reflectivity Series')
    # set(gca,'cameraupvector',[-1, 0, 0])
    plt.grid()

    plt.subplot(131)
    plt.plot(wav,twav,linewidth=2)
    plt.title('Wavelet')
    plt.grid()
    # set(gca,'ydir','reverse')

    plt.subplot(133)
    plt.plot(seis,tseis,linewidth=2)
    plt.grid()
    # set(gca,'ydir','reverse')

    plt.show()


pi = np.pi
def getRicker(f,t):
    assert len(f) == 1, 'Ricker wavelet needs 1 frequency as input'
    f = f[0]
    pift = pi*f*t
    wav = (1 - 2*pift**2)*np.exp(-pift**2)
    return wav

def getOrmsby(f,t):
    assert len(f) == 4, 'Ormsby wavelet needs 4 frequencies as input'
    f = np.sort(f) #Ormsby wavelet frequencies must be in increasing order
    pif   = pi*f
    den1  = pif[3] - pif[2]
    den2  = pif[1] - pif[0]
    term1 = (pif[3]*np.sinc(pif[3]*t))**2 - (pif[2]*np.sinc(pif[2]))**2
    term2 = (pif[1]*np.sinc(pif[1]*t))**2 - (pif[0]*np.sinc(pif[0]))**2

    wav   = term1/den1 - term2/den2;
    return wav

def getKlauder(f,t,T=5.0):
    assert len(f) == 2, 'Klauder wavelet needs 2 frequencies as input'

    k  = np.diff(f)/T
    f0 = np.sum(f)/2.0
    wav = np.real(np.sin(pi*k*t*(T-t))/(pi*k*t)*np.exp(2*pi*1j*f0*t))
    return wav

if __name__ == '__main__':

    d   = [0, 100, 200] # Position of top of each layer (m)
    v   = [350, 1000, 2000] # Velocity of each layer (m/s)
    rho = [1700, 2000, 2500] # Density of each layer (kg/m^3)

    syntheticSeismogram(v, rho, d)
    
