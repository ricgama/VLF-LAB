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


def Distancia(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)
    
    a = math.sin(rlat1) * math.sin(rlat2) + math.cos(rlat1)*math.cos(rlat2)*math.cos(rlon2-rlon1)
    d = radius * math.acos(a)
    print('A  distância entre os dois pontos é de %fkm' %d)
    return d


def Azimute(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination

    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)
    
    X = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1)*math.cos(rlat2)*math.cos(rlon2-rlon1)
    Y = math.sin(rlon2-rlon1)*math.cos(rlat2)
    a = math.degrees(math.atan2(Y,X))
    print('O Azimute entre os dois pontos é de %f°' %a)

    return a



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
        
def mapa_Sinal_Hor():
    # rever!!!
    plt.figure(figsize=(8,8))

    lat2 = 58
    lat1 = 32.
    lon2 = 26.
    lon1 = -15.

    m = Basemap(resolution='l',projection='merc', llcrnrlat=lat1,urcrnrlat=lat2,llcrnrlon=lon1,urcrnrlon=lon2)

    m.drawcountries(linewidth=0.6,color=(0.098039, 0.098039, 0.439216))
    m.drawcoastlines(linewidth=0.6,color=(0.098039, 0.098039, 0.439216))
    m.drawparallels(np.arange(lat1,lat2,10.),labels=[1,0,0,0],color='black',dashes=[1,2],labelstyle='+/-',linewidth=0.2)  
    m.drawmeridians(np.arange(lon1,lon2,10.),labels=[0,0,0,1],color='black',dashes=[1,2],labelstyle='+/-',linewidth=0.2)

    lons = [-8.112222,-3.279302 ,9.008307 ]
    lats = [40.883333,54.911195 ,50.015411]
    nomes=['Arada','MSF','DCF77']

    m.drawgreatcircle(lons[0],lats[0],lons[1], lats[1], del_s=100.0, linewidth=1.5,color=(0.6 ,0,0))
    m.drawgreatcircle(lons[0],lats[0],lons[2], lats[2], del_s=100.0, linewidth=1.5,color=(0.6 ,0,0))

    x,y = m(lons,lats)

    m.plot(x,y,'o',markersize=5, color=(0.6,0,0),markeredgewidth=0.01)

    plt.text(x[0]+10000,y[0]-100000,nomes[0])
    plt.text(x[1]+10000,y[1]-100000,nomes[1])
    plt.text(x[2]+10000,y[2]-100000,nomes[2])

    plt.draw()
    return

def emissores_mapa():
    # https://scitools.org.uk/cartopy/docs/latest/gallery/eyja_volcano.html
    fig = plt.figure(figsize=(12, 8))

    lons = [13.883333,-3.278,9.733,2.567,1.250,7.617,-67.283,-98.333,27.317,-22.467,-67.183,14.436]
    lats = [66.966667,54.900,40.917,48.550,46.717,53.083,44.650,46.367,37.417,63.850,18.400,37.126]
    
    nomes=['JNX','GBZ','ICV','FTA','HWU','DHO34','NAA','NML','TBB','TFK','NAU','NSY']


    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree())
    
    ax.set_extent([-110,30,75,10], ccrs.PlateCarree())

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

    for k in range(len(nomes)):
    # Add text 15 pixels to the left of point.
        ax.text(lons[k], lats[k], nomes[k],
                verticalalignment='center', horizontalalignment='right',
                transform=text_transform,
                bbox=dict(facecolor='sandybrown', alpha=0.7, boxstyle='round'))

    plt.show()


def mapa_2pontos(origin, destination, nomes=['Arada','DCF77']):
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
    
    ax.set_extent([-110,30,75,10], ccrs.PlateCarree())
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


def make_spect_plot(fname,save=False, plot_name='SAQ',nptf=16384, overlap=0.5, fmax=23000, fmin=16000):    
#==============================================================================
#  read wav file
#==============================================================================

    sfo = sf.SoundFile(fname)
    
    Fs=float(sfo.samplerate)
    Tempgrav= float(len(sfo))/(Fs*60)
    
    janela=np.hamming(nptf)

    noverlap=int(overlap*nptf)
    delta=int(nptf-noverlap)
     
    tempfft=nptf/Fs 
    print( 'tempo por fft:', tempfft,'segundos') 
    reso=(Fs/(nptf))
    print( 'resolução por bin:', reso, 'Hz')
    
    f=Fs/2*np.linspace(0,1,nptf/2+1)
    
    nblocks=int((len(sfo)-noverlap)/delta)
    evTemp=np.zeros((nblocks,int(nptf/2)+1))
    
    data = sfo.read()
    
    for i in tnrange(0,nblocks):  
       
        Pfft =abs(np.fft.rfft(janela*data[i*delta:i*delta+nptf]))/np.sqrt(nptf*np.dot(janela,janela)) 
        mag = 20*np.log10(Pfft)
        evTemp[i,:]=mag[:]
    
    #---------Plot----------------------------------------------
    fmax=30000
    fmin=10000
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

    minhasc=minhascores()
    mapadecor=['jet','binary','Greys',minhasc]
    
    fig = plt.figure( figsize=(16, 8))
    
    gs = GridSpec(100,100)

    ax1 = fig.add_subplot(gs[0:24,:])
    
    plt.plot(f,media,color=(0.098039, 0.098039, 0.439216))
    plt.xlim( (min(f), max(f)) ) 

    plt.ylabel('den. esp. (dB)')
    cordx=[17.2]
    cordy=[-45]

    ax2 = fig.add_subplot(gs[25:100,:])
    cmap = plt.get_cmap(mapadecor[3])    
    extent = (min(f),  max(f),0,tmax-tmin)
    im = plt.imshow(evTemp, cmap=cmap,extent=extent, interpolation='nearest')

    plt.xlabel('Frequência (kHZ)')
    plt.ylabel('Tempo (m)')
    plt.axis('tight')    
    if save:
        plt.savefig(plot_name+'.png',bbox_inches='tight',dpi=300)
    plt.show()
    return



def make_SAQ_spect_plot(fname,save=False, plot_name='SAQ',nptf=16384, overlap=0.5, fmax=23000, fmin=16000):    
#==============================================================================
#  read wav file
#==============================================================================

    sfo = sf.SoundFile(fname+'.WAV')
    
    Fs=float(sfo.samplerate)
    Tempgrav= float(len(sfo))/(Fs*60)
    
    janela=np.hamming(nptf)

    noverlap=int(overlap*nptf)
    delta=int(nptf-noverlap)
     
    tempfft=nptf/Fs 
    print( 'tempo por fft:', tempfft,'segundos') 
    reso=(Fs/(nptf))
    print( 'resolução por bin:', reso, 'Hz')
    
    f=Fs/2*np.linspace(0,1,nptf/2+1)
    
    nblocks=int((len(sfo)-noverlap)/delta)
    evTemp=np.zeros((nblocks,int(nptf/2)+1))
    
    data = sfo.read()
    
    for i in tnrange(0,nblocks):  
       
        Pfft = abs(np.fft.rfft(janela*data[i*delta:i*delta+nptf]))/np.sqrt(nptf*np.dot(janela,janela)) 
        mag = 20*np.log10(Pfft)
        evTemp[i,:]=mag[:]
    
    #---------Plot----------------------------------------------
    fmax=23000
    fmin=16000
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

    minhasc=minhascores()
    mapadecor=['jet','binary','Greys',minhasc]
    
    fig = plt.figure( figsize=(12, 7))
    
    gs = GridSpec(100,100)

    ax1 = fig.add_subplot(gs[0:24,:])
    
    plt.plot(f,media,color=(0.098039, 0.098039, 0.439216))
    plt.xlim( (min(f), max(f)) ) 

    plt.ylabel('den. esp. (dB)')
    cordx=[17.2]
    cordy=[-45]
    plt.text(cordx[0], cordy[0], 'SAQ',fontsize=15,ha="center")
    plt.arrow(cordx[0], cordy[0]-0.5,0.0,-5, fc="k", ec="k",head_width=0.08, head_length=2)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2 = fig.add_subplot(gs[25:100,:])
    cmap = plt.get_cmap(mapadecor[3])    
    extent = (min(f),  max(f),0,tmax-tmin)
    im = plt.imshow(evTemp, cmap=cmap,extent=extent, interpolation='nearest')
    #im = imshow(evTemp, cmap=cmap,extent=extent, interpolation='bilinear')
    #im = imshow(evTemp, cmap=cmap,extent=extent, interpolation='bicubic')
    plt.xlabel('Frequência (kHZ)')
    plt.ylabel('Tempo (m)')
    plt.axis('tight')    
    if save:
        plt.savefig(plot_name+'.png',bbox_inches='tight',dpi=300)
    plt.show()
    return



