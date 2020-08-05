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
tlen = 128.
# Frequency limits
flow = 10.              # PSD freq limits
fhigh = 2048.
temp_fmin = 20.         # template min freq limit
# seg length for whitening
seg = 4.
max_filt_seg = 4.
# Generating dependent parameters
delta_t = 0.5/fhigh
delta_f = 1./tlen
flen = int(fhigh/delta_f)+1
tsamples = int(tlen / delta_t)
# Waveform approximant
approx = "IMRPhenomD"
# mass Limits in Msun
mass_min = 5.
mass_max = 95.
# SNR limits
snr_min = 5.
snr_max = 25.
# duration around each side of glitch
durn = 0.5
# No of samples in each distribution
N_chirps = 10000

row,col = N_chirps,4096
tr_chirp_slice_arr = numpy.zeros(shape=(row,col))
#tr_chirp_slice_arr = [[0]*col]*row

with h5py.File('white_slice.hdf5', 'w') as f:
	mass1 = random.randrange(mass_min,mass_max)	
	mass2 = random.randrange(mass_min,mass_max)
	SNR = random.randrange(snr_min,snr_max)
	t_inject=20
	print('mass1: ',mass1,'\nmass2: ',mass2,'\nSNR: ',SNR,'\nInjection time: ',t_inject)

	#create an instance of time series pycbc class
	psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
	time_series=pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=None)
	#plotting the timeseries
	#pylab.plot(time_series.sample_times,time_series)
	#pylab.title('time_series')
	#pylab.ylabel('Strain')
	#pylab.xlabel('Time (s)')
	#pylab.legend()
	#pylab.show()

	# Generate template
	wave_p,_ = get_td_waveform(approximant=approx,
					mass1=mass1, mass2=mass2, distance=100.,
					inclination=0., f_lower=temp_fmin, delta_t=delta_t)	#mass1=? mass2=? m1=np.random.random(min to max) same for m2
	#plotting the waveform
	#pylab.plot(wave_p.sample_times, wave_p,label=approx)
	#pylab.title('waveform_template')
	#pylab.ylabel('Strain')
	#pylab.xlabel('Time (s)')
	#pylab.legend()
	#pylab.grid()
	#pylab.show()

	wave_p.resize(len(time_series))
	# scale snr
	wave_p.data *= SNR / sigma(wave_p, psd=psd, low_frequency_cutoff=temp_fmin)	
	# Bringing the merger to t = t_inject 
	wave_p = wave_p.cyclic_time_shift(t_inject+wave_p.start_time)
	wave_p.start_time = time_series.start_time

	# Injecting into the time series data
	time_series = time_series.add_into(wave_p)
	#pylab.plot(time_series.sample_times,time_series)
	#pylab.title('time_series after injection')
	#pylab.ylabel('Strain')
	#pylab.xlabel('Time (s)')
	#pylab.show()

	# Whitening
	white_chirp = time_series.whiten(segment_duration=seg, max_filter_duration=max_filt_seg)
	# slicing 1 sec with center at t_inject for chirp
	tr_chirp_slice_arr = white_chirp.crop(t_inject-durn-seg/2., white_chirp.duration-t_inject-durn+seg/2.)
	#pylab.plot(tr_chirp_slice_arr[i])
	#pylab.title('tr_chirp_slice')
	#pylab.show()
	# Save white slices to dataset
	dset=f.create_dataset("white_slice", data = tr_chirp_slice_arr) 
	dset.attrs.create("tlen",128.,dtype=float)
	dset.attrs.create("flow",10.,dtype=float)
	dset.attrs.create("fhigh",2048.,dtype=float)
	dset.attrs.create("temp_fmin",20.,dtype=float)
	dset.attrs.create("seg",4.,dtype=float)
	dset.attrs.create("max_filt_seg",4.,dtype=float)
	dset.attrs.create("delta_t",0.5/fhigh,dtype=float)
	dset.attrs.create("delta_f",1./tlen,dtype=float)
	dset.attrs.create("flen",int(fhigh/delta_f)+1,dtype=int)
	dset.attrs.create("tsamples",int(tlen/delta_t),dtype=int)
	dset.attrs.create("approx","IMRPhenomD")
	dset.attrs.create("mass_min",5.,dtype=float)
	dset.attrs.create("mass_max",95.,dtype=float)
	dset.attrs.create("mass1",mass1,dtype=float)
	dset.attrs.create("mass2",mass2,dtype=float)
	dset.attrs.create("snr_min",5.,dtype=float)
	dset.attrs.create("snr_max",25.,dtype=float)
	dset.attrs.create("SNR",SNR,dtype=float)
	dset.attrs.create("durn",0.5,dtype=float)
	dset.attrs.create("N_chirps",10000,dtype=int)
	dset.attrs.create("t_inject",t_inject,dtype=float)
