#PyCBC
from __future__ import print_function
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.filter.resample import resample_to_delta_t
from pycbc.filter.matchedfilter import sigma
from pycbc.psd import interpolate
import array as arr
from pycbc import catalog
import pycbc.noise
import pycbc.psd
import pylab
from pycbc.waveform import td_approximants, fd_approximants
import random
import h5py
import numpy

# Duration of time series
tlen = 128.             # 128.

# Frequency limits
flow = 10.              # 10. PSD freq limits
fhigh = 2048.           # 2048.
temp_fmin = 30.         # 20. template min freq limit should be > flow

# Seg length for whitening
seg = 4.                # 4
max_filt_seg = 4.       # 4

# Generating dependent parameters
delta_t = 0.5/fhigh    # 2.44140625e-4
delta_f = 1./tlen       # 0.0078125
flen = int(fhigh/delta_f)+1     # 262145
tsamples = int(tlen / delta_t)  # 524288

# Waveform approximant
approx = "IMRPhenomD"

# Mass Limits in Msun
mass_min = 5.       # 5.
mass_max = 95.      #95.

# SNR limits
snr_min = 5.        # 5.
snr_max = 25.       # 25.

# Duration around each side of glitch
durn = 0.5          # 0.5

# Time at which wave is to be injected into the noise
t_inject=20

# No of samples in each distribution
N_chirps = 10000       
row,col = N_chirps,2*int(fhigh)
tr_chirp_slice_arr = numpy.zeros(shape=(row,col))
   
# The color of the noise matches a PSD which you provide
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
'''
#Plotting the PSD
pylab.figure(1)
pylab.plot(psd.sample_frequencies,psd,label='LIGOZeroDetHighPower',color='red')
pylab.title('PSD of LIGO DETECTORS')
pylab.ylabel('PSD')
pylab.xlabel('Fequency (Hz)')
pylab.legend()
pylab.grid()
pylab.show()
'''
with h5py.File('Test_slice.hdf5', 'w') as f:        
    # Update attributes of data series 
    dset=f.create_dataset("TS_DataSet", (row,col)) 
    dset.attrs.create("tlen",tlen,dtype=float)
    dset.attrs.create("flow",flow,dtype=float)
    dset.attrs.create("fhigh",fhigh,dtype=float)
    dset.attrs.create("temp_fmin",temp_fmin,dtype=float)
    dset.attrs.create("seg",seg,dtype=float)
    dset.attrs.create("max_filt_seg",max_filt_seg,dtype=float)
    dset.attrs.create("delta_t",0.5/fhigh,dtype=float)
    dset.attrs.create("delta_f",1./tlen,dtype=float)
    dset.attrs.create("flen",int(fhigh/delta_f)+1,dtype=int)
    dset.attrs.create("tsamples",int(tlen/delta_t),dtype=int)
    dset.attrs.create("approx","IMRPhenomD")
    dset.attrs.create("mass_min",mass_min,dtype=float)
    dset.attrs.create("mass_max",mass_max,dtype=float)
    dset.attrs.create("snr_min",snr_min,dtype=float)
    dset.attrs.create("snr_max",snr_max,dtype=float)
    dset.attrs.create("durn",durn,dtype=float)
    dset.attrs.create("N_chirps",N_chirps,dtype=int)
    dset.attrs.create("t_inject",t_inject,dtype=float)  

    # Set i depending on number of white chirp slices (max - 10000) to be generated
    for i in range(10):
        mass1 = random.randrange(mass_min,mass_max) 
        mass2 = random.randrange(mass_min,mass_max)
        SNR = random.randrange(snr_min,snr_max)

        ind=str(i);
        mass1_atr="Time_series"+ind+" mass1";
        mass2_atr="Time_series"+ind+" mass2";
        snr_atr="Time_series"+ind+" snr";
        # Update HDF5 attributes which change in every series
        dset.attrs.create(mass1_atr,mass1,dtype=float)
        dset.attrs.create(mass2_atr,mass2,dtype=float)
        dset.attrs.create(snr_atr,SNR,dtype=float)

        #create an instance of time series pycbc class    
        time_series = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=0)
        #print('time_series1.delta_t',time_series1.delta_t)
    
        # Generate template
        wave_p,_ = get_td_waveform(approximant=approx,mass1=mass1, mass2=mass2, distance=100.,inclination=0., f_lower=temp_fmin, delta_t=delta_t)
    
        wave_p.resize(len(time_series))
        #print('wave_p after resizing:',wave_p.shape)

        # scale snr
        wave_p.data *= SNR / sigma(wave_p, psd=psd, low_frequency_cutoff=temp_fmin) 

        # Bringing the merger to t = t_inject 
        wave_p = wave_p.cyclic_time_shift(t_inject+wave_p.start_time)
        wave_p.start_time = time_series.start_time
    
        # Injecting into the time series data
        time_series = time_series.add_into(wave_p)
    
        # Whitening
        white_chirp = time_series.whiten(segment_duration=seg, max_filter_duration=max_filt_seg)
        
        # slicing 1 sec with center at t_inject for chirp
        tr_chirp_slice_arr[i,0:col] = white_chirp.crop(t_inject-durn-seg/2., white_chirp.duration-t_inject-durn+seg/2.)
    
        #print(i)
         
        # Save white chirp as slice into the dataset
        dset[i,0:col] = tr_chirp_slice_arr [i,0:col]
