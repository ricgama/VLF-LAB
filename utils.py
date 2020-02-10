import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.transforms import offset_copy

import cartopy.io.img_tiles as cimgt

import soundfile as sf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import math
from tqdm import tnrange, tqdm

#------------------------------------------------------------------------------------------------------------------


def distancia(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)
    
    a = math.sin(rlat1) * math.sin(rlat2) + math.cos(rlat1)*math.cos(rlat2)*math.cos(rlon2-rlon1)
    d = radius * math.acos(a)
    print('A  distância entre os dois pontos é de %.2fkm' %d)
    return d


def azimute(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination

    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)
    
    X = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1)*math.cos(rlat2)*math.cos(rlon2-rlon1)
    Y = math.sin(rlon2-rlon1)*math.cos(rlat2)
    az = math.degrees(math.atan2(Y,X))
    az =  (az+360) % 360 
    az = (az+180) % 360 # Azimute final
    print('O Azimute entre os dois pontos é de %.2f°' %az)

    return az



#------------------------------------------------------------------------------------------------------------------

# Graphing helper function
def setup_graph(title='', x_label='', y_label='',ylim=None, fig_size=None):
    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ylim!= None:  ax.set_ylim([-ylim,ylim])
        
        

#------------------------------------------------------------------------------------------------------------------

def emissores_mapa( emi_dict, limites = [-110,30,75,10]):
    # https://scitools.org.uk/cartopy/docs/latest/gallery/eyja_volcano.html
    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree())
    
    ax.set_extent(limites, ccrs.PlateCarree())
    lons = [x[1] for x in emi_dict.values()]
    lats = [x[0] for x in emi_dict.values()]
    ax.scatter(lons, lats, c='r', lw=3)
    ax.stock_img()
    ax.add_feature(cfeature.BORDERS)

    ax.coastlines()
    ax.gridlines()
    
    # Use the cartopy interface to create a matplotlib transform object
    # for the Geodetic coordinate system. We will use this along with
    # matplotlib's offset_copy function to define a coordinate system which
    # translates the text by 25 pixels to the left.
    geodetic_transform = ccrs.Geodetic()._as_mpl_transform(ax)
    text_transform = offset_copy(geodetic_transform, units='dots', x=-15)

    for name,coord in emi_dict.items():
    # Add text 15 pixels to the left of point.
        ax.text(coord[1], coord[0], name,
                verticalalignment='center', horizontalalignment='right',
                transform=text_transform,
                bbox=dict(facecolor='sandybrown', alpha=0.7, boxstyle='round'))

    plt.show()


def mapa_2pontos(origin, destination, nomes = ['Arada','DCF77'], limites = [-110,30,75,10]):
    # https://scitools.org.uk/cartopy/docs/latest/gallery/eyja_volcano.html
    fig = plt.figure(figsize=(12, 8))

    lat1, lon1 = origin
    lat2, lon2 = destination

    lons = [lon1, lon2]
    lats = [lat1, lat2]


    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree())
    
    ax.stock_img()
    ax.add_feature(cfeature.BORDERS)

    ax.coastlines()
    ax.gridlines()
    
    ax.set_extent(limites, ccrs.PlateCarree())
    ax.plot(lons, lats, c='r', lw=2, transform=ccrs.Geodetic())

    ax.scatter(lons, lats, c='r', lw=3)

    
    # Use the cartopy interface to create a matplotlib transform object
    # for the Geodetic coordinate system. We will use this along with
    # matplotlib's offset_copy function to define a coordinate system which
    # translates the text by 25 pixels to the left.
    geodetic_transform = ccrs.Geodetic()._as_mpl_transform(ax)
    text_transform = offset_copy(geodetic_transform, units='dots', x=-25)

    # Add text 25 pixels to the left of the volcano.
    ax.text(origin[1], origin[0], nomes[0],
            verticalalignment='center', horizontalalignment='right',
            transform=text_transform,
            bbox=dict(facecolor='sandybrown', alpha=0.7, boxstyle='round'))
    ax.text(destination[1], destination[0], nomes[1],
        verticalalignment='center', horizontalalignment='right',
        transform=text_transform,
        bbox=dict(facecolor='sandybrown', alpha=0.7, boxstyle='round'))
    plt.show()

#------------------------------------------------------------------------------------------------------------------


def minhascores():
    cdict = {'red': ((0.0, 1.0, 1.0),
                    (0.5, 1.0, 1.0),
                    (0.6, 0.95, 0.95),
                    (0.75,  0.098039, 0.098039),
                    (1.0,  0.6, 0.0)),
           'green': ((0.0, 1.0, 1.0),
                    (0.5, 1.0, 1.0),
                    (0.6, 0.95, 0.95),
                    (0.75,  0.098039, 0.098039),
                    (1.0,  0, 0.0)),
           'blue': ((0.0, 1.0, 1.0),
                    (0.5,  1.0, 1.0),
                    (0.6, 0.95, 0.95),
                    (0.75,  0.439216, 0.439216),
                    (1.0, 0, 1.0))}
    minhasc = LinearSegmentedColormap('my_colormap',cdict,256)
    return minhasc

def minhascoresH():
    cdict = {'red': ((0.0, 1.0, 1.0),
                   (0.75, 1.0, 1.0),
                   (0.8, 0.95, 0.95),
                   (0.9,  0.098039, 0.098039),
                   (1.0,  0.6, 0.0)),
           'green': ((0.0, 1.0, 1.0),
                    (0.75, 1.0, 1.0),
                    (0.8, 0.95, 0.95),
                    (0.9,  0.098039, 0.098039),
                    (1.0,  0, 0.0)),
           'blue': ((0.0, 1.0, 1.0),
                    (0.75,  1.0, 1.0),
                    (0.8, 0.95, 0.95),
                    (0.9,  0.439216, 0.439216),
                    (1.0, 0, 1.0))}
    minhasc = LinearSegmentedColormap('my_colormap',cdict,256)
    return minhasc

def make_spect_plot(data, sample_rate, save=False, plot_name='SAQ', nptf=16384, overlap=0.5, fmin=12000, fmax=30000):    
#==============================================================================
#  read wav file
#==============================================================================
    
    Fs = float(sample_rate)
    print( 'taxa de amostragem: %.1f amostras/segundo' %Fs) 
    Tempgrav= float(len(data))/(Fs*60)
    janela=np.hamming(nptf)

    noverlap=int(overlap*nptf)
    delta=int(nptf-noverlap)
     
    tempfft=nptf/Fs 
    print( 'tempo por fft: %.4f segundos' %tempfft) 
    reso=(Fs/(nptf))
    print( 'resolução por bin  %.2f Hz:'  %reso)
    
    f=Fs/2*np.linspace(0,1,nptf/2+1)
    
    nblocks=int((len(data)-noverlap)/delta)
    evTemp=np.zeros((nblocks,int(nptf/2)+1))
        
    for i in tnrange(0,nblocks):  
       
        Pfft =abs(np.fft.rfft(janela*data[i*delta:i*delta+nptf]))/np.sqrt(nptf*np.dot(janela,janela)) 
        mag = 20*np.log10(Pfft)
        evTemp[i,:]=mag[:]
    
    #---------Plot----------------------------------------------
 
    ondef=np.nonzero(np.logical_and(f<=fmax,f>=fmin))[0]
    f=f[ondef]/1000.
    
    temp=Tempgrav*np.linspace(0,1,nblocks) #tempo em minutos
    temp = temp[::-1]

    tmin=min(temp);
    tmax=max(temp);
    ondet=np.nonzero(np.logical_and(temp<=tmax,temp>=tmin))[0]
    
    evTemp=evTemp[ondet,:];
    evTemp=evTemp[:,ondef];
    evTemp = evTemp[::-1,:]

    media = np.mean(evTemp,0)
    
    fig = plt.figure( figsize=(14, 6))
    
    gs = GridSpec(100,100)

    ax1 = fig.add_subplot(gs[0:24,:])
    
    plt.plot(f,media,color=(0.098039, 0.098039, 0.439216))
    plt.xlim( (min(f), max(f)) ) 

    plt.ylabel('den. esp. (dB)')
    cordx=[17.2]
    cordy=[-45]

    ax2 = fig.add_subplot(gs[25:100,:])
    cmap = plt.get_cmap(minhascores())    
    
    extent = (min(f),  max(f),0,tmax-tmin)
    im = plt.imshow(evTemp, cmap=cmap,extent=extent, interpolation='nearest')

    plt.xlabel('Frequência (kHZ)')
    plt.ylabel('Tempo (m)')
    plt.axis('tight')
    
    if save:
        plt.savefig(plot_name+'.png',bbox_inches='tight',dpi=300)
    plt.show()
    return



def make_spect_plot_H(data, sample_rate,save=False, plot_name='SAQ', nptf=16384, overlap=0.9, fmin=17000, fmax=18600, tmin=0.45, tmax=0.60):    
#==============================================================================
#  read wav file
#==============================================================================

    
    Fs=float(sample_rate)
    print( 'taxa de amostragem: %.1f amostras/segundo' %Fs) 
    Tempgrav= float(len(data))/(Fs*60)
    
    janela=np.hamming(nptf)

    noverlap=int(overlap*nptf)
    delta=int(nptf-noverlap)
     
    tempfft=nptf/Fs 
    print( 'tempo por fft: %.4f segundos' %tempfft) 
    reso=(Fs/(nptf))
    print( 'resolução por bin  %.2f Hz:'  %reso)
    
    f=Fs/2*np.linspace(0,1,nptf/2+1)
    
    nblocks=int((len(data)-noverlap)/delta)
    evTemp=np.zeros((nblocks,int(nptf/2)+1))
        
    for i in tnrange(0,nblocks):  
       
        Pfft = 2*abs(np.fft.rfft(janela*data[i*delta:i*delta+nptf]))/np.sqrt(nptf*np.dot(janela,janela)) 
        mag = 20*np.log10(Pfft)
        evTemp[i,:]=mag[:]
    
    #---------Plot----------------------------------------------
    fig = plt.figure( figsize=(14, 6))

    ondef=np.nonzero(np.logical_and(f<=fmax,f>=fmin))[0]
    f=f[ondef]/1000.
    
    temp=Tempgrav*np.linspace(0,1,nblocks) #tempo em minutos
    ondet=np.nonzero(np.logical_and(temp<=tmax,temp>=tmin))[0]
    
    evTemp=evTemp[ondet,:];
    evTemp=evTemp[:,ondef];
    
    evTemp=evTemp.transpose()
    evTemp=evTemp[::-1,:]
    extent = (tmin,tmax,min(f), max(f))
        
    cmap = plt.get_cmap(minhascoresH())    
    
    plt.imshow(evTemp, cmap=cmap,extent=extent, interpolation='nearest')
    #plt.imshow(evTemp, cmap=cmap,extent=extent, interpolation='bilinear')
    #im = imshow(evTemp, cmap=cmap,extent=extent, interpolation='bicubic')
    plt.ylabel('Frequência (kHZ)')
    plt.xlabel('Tempo (m)')
    plt.axis('tight')
    #savefig(nome+'Morse.png',bbox_inches='tight',dpi=150)
    if save:
        plt.savefig(plot_name+'.png',bbox_inches='tight',dpi=300)
    plt.show()
    return


