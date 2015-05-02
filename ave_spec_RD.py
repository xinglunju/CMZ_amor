import os, time
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
#from astropy import wcs
#from astropy.convolution import *
from astropy.constants import *
from astropy.units import *
#from scipy.ndimage.interpolation import map_coordinates
#from scipy.interpolate import splrep,splev
from lmfit import minimize, Parameters, report_fit

start = time.clock()
print 'Start the timer...'

def __readascii__(infile):
	"""
	Read in an ASCII file, first column is velocity/frequency,
	second column is intensity/brightness temperature.
	Return two numpy arrays.
	"""
	temp = open(infile, 'r')
	text = temp.readlines()
	spec = np.empty(len(text))
	vaxis = np.empty(len(text))
	for line in range(len(text)):
		vaxis[line] = np.float(text[line].split()[0])
		spec[line] = np.float(text[line].split()[1])
	temp.close()
	del temp

	return spec, vaxis

def __nh3_init__():
	"""
	nh3_init()
	Initialize the parameters. No input keywords.
	"""
	nh3_info = {}
	nh3_info['energy'] = [23.4, 64.9, 125., 202., 298., 412.] # Upper level energy (Kelvin)
	nh3_info['frest'] = [23.6944955, 23.7226333, 23.8701292, \
	                     24.1394163, 24.5329887, 25.0560250] # Rest frequency (GHz)
	nh3_info['c0'] = 1.89e5 # Rotational constants (MHz)
	nh3_info['b0'] = 2.98e5 # 
	nh3_info['Ijk'] = [50.0,  79.62962963,  89.4,  93.5,  95.6,  96.9]
	# Ratio of the main line intensity to all hyperfine
	nh3_info['lratio'] = [0.278, 0.222, 0.0651163, 0.0627907, 0.0302, 0.0171, 0.0115, 0.0083]
	# (1,1) inner, (1,1) outer, (2,2) inner, (2,2) outer, (3,3)-(6,6)
	# Ratio of the inner hyperfine intensity relative to the main line
	nh3_info['vsep'] = [7.756, 19.372, 16.57, 21.49, 24.23, 25.92, 26.94]
	# velocity separation between the mainline and the inner satellites (and outer satellites for (1,1)) (km/s)
	return nh3_info

def __xcorrelate__(spec, vaxis):
	"""
	__xcorrelate(spec)
	Return the channel index of the VLSR, through a cross-correlation between input spectrum
	and a model.
	"""
	voff_lines = [7.56923, 0.0, -7.47385]
	tau_wts = [0.278, 1.0, 0.278]
	vlength = len(vaxis)
	stretch = 10.0
	oldidx = np.linspace(0,vlength-1,vlength)
	newidx = np.linspace(0,vlength-1,vlength*stretch)
	temp = splrep(oldidx, vaxis)
	newvaxis = splev(newidx,temp)
	temp = splrep(oldidx, spec)
	newspec = splev(newidx,temp)
	kernel = np.zeros(vlength*10)
	sigmav = 1.0
	for i in np.arange(len(voff_lines)):
		kernel += np.exp(-(newvaxis-newvaxis[vlength*stretch/2]-voff_lines[i])**2/(2*sigmav**2))*tau_wts[i]
	lags = np.correlate(newspec,kernel,'same')
	vlsr = newvaxis[lags.argmax()]
	return vlsr

def __gauss_tau__(axis,p):
	"""
	Genenerate a Gaussian model given an axis and a set of parameters.
	Params:[peaki, tau11, peakv, sigmav]
	"""
	sx = len(axis)
	u  = ((axis-p[2])/np.abs(p[3]))**2
	f = -1.0 * p[1] * np.exp(-0.5*u)
	f = -1.0 * p[0] * np.expm1(f)
	return f

def __model_11__(params, vaxis, spec):
	"""
	Model five components of NH3 (1,1) and five components of NH3 (2,2),
	then subtract data.
	"""
	peaki = params['peaki'].value
	sigmav = params['sigmav'].value
	peakv = params['peakv'].value
	tau11 = params['tau11'].value
	peaki_s1 = params['peaki_s1'].value
	sigmav_s1 = params['sigmav_s1'].value
	peakv_s1 = params['peakv_s1'].value
	tau11_s1 = params['tau11_s1'].value
	peaki_s2 = params['peaki_s2'].value
	sigmav_s2 = params['sigmav_s2'].value
	peakv_s2 = params['peakv_s2'].value
	tau11_s2 = params['tau11_s2'].value
	peaki_s3 = params['peaki_s3'].value
	sigmav_s3 = params['sigmav_s3'].value
	peakv_s3 = params['peakv_s3'].value
	tau11_s3 = params['tau11_s3'].value
	peaki_s4 = params['peaki_s4'].value
	sigmav_s4 = params['sigmav_s4'].value
	peakv_s4 = params['peakv_s4'].value
	tau11_s4 = params['tau11_s4'].value
	
	peaki22 = params['peaki22'].value
	sigmav22 = params['sigmav22'].value
	peakv22 = params['peakv22'].value
	tau22 = params['tau22'].value
	peaki22_s1 = params['peaki22_s1'].value
	sigmav22_s1 = params['sigmav22_s1'].value
	peakv22_s1 = params['peakv22_s1'].value
	tau22_s1 = params['tau22_s1'].value
	peaki22_s2 = params['peaki22_s2'].value
	sigmav22_s2 = params['sigmav22_s2'].value
	peakv22_s2 = params['peakv22_s2'].value
	tau22_s2 = params['tau22_s2'].value
	peaki22_s3 = params['peaki22_s3'].value
	sigmav22_s3 = params['sigmav22_s3'].value
	peakv22_s3 = params['peakv22_s3'].value
	tau22_s3 = params['tau22_s3'].value
	peaki22_s4 = params['peaki22_s4'].value
	sigmav22_s4 = params['sigmav22_s4'].value
	peakv22_s4 = params['peakv22_s4'].value
	tau22_s4 = params['tau22_s4'].value

	peaki33 = params['peaki33'].value
	sigmav33 = params['sigmav33'].value
	peakv33 = params['peakv33'].value
	tau33 = params['tau33'].value

	peaki44 = params['peaki44'].value
	sigmav44 = params['sigmav44'].value
	peakv44 = params['peakv44'].value
	tau44 = params['tau44'].value

	peaki55 = params['peaki55'].value
	sigmav55 = params['sigmav55'].value
	peakv55 = params['peakv55'].value
	tau55 = params['tau55'].value

	model = __gauss_tau__(vaxis,[peaki,tau11,peakv,sigmav]) + \
			__gauss_tau__(vaxis,[peaki_s1,tau11_s1,peakv_s1,sigmav_s1]) + \
			__gauss_tau__(vaxis,[peaki_s2,tau11_s2,peakv_s2,sigmav_s2]) + \
			__gauss_tau__(vaxis,[peaki_s3,tau11_s3,peakv_s3,sigmav_s3]) + \
			__gauss_tau__(vaxis,[peaki_s4,tau11_s4,peakv_s4,sigmav_s4]) + \
			__gauss_tau__(vaxis,[peaki22,tau22,peakv22,sigmav22]) + \
			__gauss_tau__(vaxis,[peaki22_s1,tau22_s1,peakv22_s1,sigmav22_s1]) + \
			__gauss_tau__(vaxis,[peaki22_s2,tau22_s2,peakv22_s2,sigmav22_s2]) + \
			__gauss_tau__(vaxis,[peaki22_s3,tau22_s3,peakv22_s3,sigmav22_s3]) + \
			__gauss_tau__(vaxis,[peaki22_s4,tau22_s4,peakv22_s4,sigmav22_s4]) + \
			__gauss_tau__(vaxis,[peaki33,tau33,peakv33,sigmav33]) + \
			__gauss_tau__(vaxis,[peaki44,tau44,peakv44,sigmav44]) + \
			__gauss_tau__(vaxis,[peaki55,tau55,peakv55,sigmav55])
	return model - spec

clickvalue = []
def onclick(event):
	print 'The Vlsr you select: %f' % event.xdata
	clickvalue.append(event.xdata)

def fit_spec(spec1, spec2, spec3, spec4, spec5, vaxis1, vaxis2, cutoff=0.009, varyv=2, interactive=True, mode='single', RD_method='polyfit'):
	"""
	fit_spec(spec1, spec2, spec3, spec4, spec5)
	Derive the five NH3 spectra simultaneously, derive temperature and column density in the
	rotation diagram.
	"""
	if interactive:
		plt.ion()
		f = plt.figure(figsize=(14,8))
		ax = f.add_subplot(111)

	spec1 = spec1[10:-10]
	spec2 = spec2[10:-10]
	spec3 = spec3[10:-10]
	spec4 = spec4[10:-10]
	spec5 = spec5[10:-10]
	cutoff = cutoff
	vaxis1 = vaxis1[10:-10]
	vaxis2 = vaxis2[10:-10]
	spec = np.concatenate((spec1, spec2, spec3, spec4, spec5))
	#spec[np.where(spec<=cutoff)] = 0.0
	vaxis = np.concatenate((vaxis1, vaxis1+150.0, vaxis2+266.0, vaxis2+336.0, vaxis2+406.0))

	#if interactive:
		#plt.plot(vaxis, spec, 'k+', label='Original')
		#plt.legend()
		#plt.show()

	unsatisfied = True
	while unsatisfied:
		if interactive:
			f.clear()
			plt.ion()
			plt.plot(vaxis, spec, 'k-', label='Spectrum')
			cutoff_line = [cutoff] * len(vaxis)
			cutoff_line_minus = [-1.0*cutoff] * len(vaxis)
			plt.plot(vaxis, cutoff_line, 'r-')
			plt.plot(vaxis, cutoff_line_minus, 'r-')
			#plt.plot(vaxis, np.zeros(len(vaxis)), 'b-')
			plt.xlabel(r'$V_\mathrm{lsr}$ (km s$^{-1}$)', fontsize=20, labelpad=20)
			plt.ylabel(r'$T_{\nu}$ (K)', fontsize=20)
			#plt.ylim([-10,60])
			#clickvalue = []
			if mode == 'single':
				cid = f.canvas.mpl_connect('button_press_event', onclick)
				raw_input('Click on the plot to select Vlsr...')
				print clickvalue
				if len(clickvalue) >= 1:
					print 'Please select at least one velocity! The last one will be used.'
					vlsr1 = clickvalue[-1]
				elif len(clickvalue) == 0:
					vlsr1 = 0.0
				print 'Or input one velocity manually:'
				manualv = raw_input()
				manualv = manualv.split()
				if len(manualv) == 1:
					vlsr1 = np.float_(manualv)
				else:
					print 'Invalid input...'
				print 'The Vlsr is %0.2f' % vlsr1
				raw_input('Press any key to start fitting...')
				f.canvas.mpl_disconnect(cid)
				vlsr2 = 0.0
			elif mode == 'double':
				cid = f.canvas.mpl_connect('button_press_event', onclick)
				raw_input('Click on the plot to select Vlsrs...')
				print clickvalue
				if len(clickvalue) >= 2:
					print 'Please select at least two velocities! The last two will be used.'
					vlsr1,vlsr2 = clickvalue[-2],clickvalue[-1]
				elif len(clickvalue) == 1:
					vlsr1 = clickvalue[-1]
					vlsr2 = 0.0
				elif len(clickvalue) == 0:
					vlsr1,vlsr2 = 0.0,0.0
				print 'Or input two velocities manually:'
				manualv = raw_input()
				manualv = manualv.split()
				if len(manualv) == 2:
					vlsr1,vlsr2 = np.float_(manualv)
				else:
					print 'Invalid input...'
				print 'The two Vlsrs are %0.2f km/s and %0.2f km/s.' % (vlsr1,vlsr2)
				raw_input('Press any key to start fitting...')
				f.canvas.mpl_disconnect(cid)
			else:
				vlsr1,vlsr2 = 0.0,0.0
		else:
			if mode == 'single':
				if spec_low.max() >= cutoff:
					vlsr1 = __xcorrelate__(spec_low, vaxis_low)
					if vlsr1 <=82 or vlsr1 >=92:
						vlsr1 = 0.0
					if spec_low[np.abs(vaxis_low - vlsr1).argmin()] <= cutoff:
						vlsr1 = 0.0
					if spec_low[np.abs(vaxis_low - vlsr1 + 7.47385).argmin()] <= cutoff and spec_low[np.abs(vaxis_low - vlsr1 - 7.56923).argmin()] <= cutoff:
						vlsr1 = 0.0
					if spec_upp[np.abs(vaxis_low - vlsr1).argmin()] <= cutoff:
						vlsr1 = 0.0
				else:
					vlsr1 = 0.0
				vlsr2 = 0.0
			elif mode == 'double':
				vlsr1,vlsr2 = 86.0,88.0
			else:
				vlsr1,vlsr2 = 0.0,0.0


		# 43 parameters, but only 11 are indenpendent 
		params = Parameters()
		if vlsr1 != 0:
			params.add('peaki', value=5, min=0, max=20)
			params.add('tau11', value=1.0, min=0)
			if varyv > 0:
				params.add('peakv', value=vlsr1, min=vlsr1-varyv*onevpix, max=vlsr1+varyv*onevpix)
			elif varyv == 0:
				params.add('peakv', value=vlsr1, vary=False)
			params.add('sigmav', value=5.0, min=0, max=20.0)
			params.add('peaki_s1', expr="peaki*(1-exp(-tau11_s1))/(1-exp(-tau11))")
			params.add('tau11_s1', expr='tau11*0.278')
			params.add('peakv_s1', expr='peakv-7.47385')
			params.add('sigmav_s1', expr='sigmav')
			params.add('peaki_s2', expr='peaki_s1')
			params.add('tau11_s2', expr='tau11_s1')
			params.add('peakv_s2', expr='peakv+7.56923')
			params.add('sigmav_s2', expr='sigmav')
			params.add('peaki_s3', expr="peaki*(1-exp(-tau11_s3))/(1-exp(-tau11))")
			params.add('tau11_s3', expr='tau11*0.222')
			params.add('peakv_s3', expr='peakv-19.4808')
			params.add('sigmav_s3', expr='sigmav')
			params.add('peaki_s4', expr='peaki_s3')
			params.add('tau11_s4', expr='tau11_s3')
			params.add('peakv_s4', expr='peakv+19.5836')
			params.add('sigmav_s4', expr='sigmav')
			params.add('N11', value=1e15, expr='3.2788e13*peaki*sigmav*tau11/(1-exp(-tau11))')

			#params.add('peaki22', expr='peaki*(1-exp(-tau22))/(1-exp(-tau11))')
			params.add('peaki22', value=3, min=0, max=20)
			params.add('tau22', value=1.0, min=0)
			params.add('peakv22', expr='peakv+150.0')
			#params.add('sigmav22', expr='N22*2.1503e-13/peaki22/tau22*(1-exp(-tau22))')
			params.add('sigmav22', expr='sigmav')
			params.add('peaki22_s1', expr="peaki22*(1-exp(-tau22_s1))/(1-exp(-tau22))")
			params.add('tau22_s1', expr='tau22*0.0651163')
			params.add('peakv22_s1', expr='peakv22-16.2022')
			params.add('sigmav22_s1', expr='sigmav22')
			params.add('peaki22_s2', expr='peaki22_s1')
			params.add('tau22_s2', expr='tau22_s1')
			params.add('peakv22_s2', expr='peakv22+16.2117')
			params.add('sigmav22_s2', expr='sigmav22')
			params.add('peaki22_s3', expr="peaki22*(1-exp(-tau22_s3))/(1-exp(-tau22))")
			params.add('tau22_s3', expr='tau22*0.0627907')
			params.add('peakv22_s3', expr='peakv22-26.1626')
			params.add('sigmav22_s3', expr='sigmav22')
			params.add('peaki22_s4', expr='peaki22_s3')
			params.add('tau22_s4', expr='tau22_s3')
			params.add('peakv22_s4', expr='peakv22+26.1626')
			params.add('sigmav22_s4', expr='sigmav22')
			#params.add('N22', value=1e15, expr='N11*5.0/3.0*exp(-41.5/Trot)')
			params.add('N22', value=1e15, expr='1.5422e13*peaki22*sigmav22*tau22/(1-exp(-tau22))')

			#params.add('peaki33', expr='peaki22*(1-exp(-tau33))/(1-exp(-tau22))')
			params.add('peaki33', value=1, min=0, max=20)
			params.add('tau33', value=1.0, min=0)
			params.add('peakv33', expr='peakv+266.0')
			params.add('sigmav33', expr='sigmav')
			params.add('N33', value=1e14, expr='1.2135e13*peaki33*sigmav33*tau33/(1-exp(-tau33))')

			#params.add('peaki44', expr='peaki22*(1-exp(-tau44))/(1-exp(-tau22))')
			params.add('peaki44', value=1, min=0, max=20)
			params.add('tau44', value=1.0, min=0)
			params.add('peakv44', expr='peakv+336.0')
			params.add('sigmav44', expr='sigmav')
			params.add('N44', value=1e14, expr='1.0757e13*peaki44*sigmav44*tau44/(1-exp(-tau44))')

			#params.add('peaki55', expr='peaki22*(1-exp(-tau55))/(1-exp(-tau22))')
			params.add('peaki55', value=1, min=0, max=20)
			params.add('tau55', value=1.0, min=0)
			params.add('peakv55', expr='peakv+406.0')
			params.add('sigmav55', expr='sigmav')
			params.add('N55', value=1e14, expr='0.9937e13*peaki55*sigmav55*tau55/(1-exp(-tau55))')

			#params.add('Trot', value=30.0, min=0, max=200)
			#params.add('Trot', value=30.0, expr='-41.5/log(0.282*tau22/tau11)')
			#params.add('Ntot', value=1e15, expr='6.6e14*Trot/23.6944955*tau11*2.355*sigmav*0.0138*exp(23.1/Trot)*Trot**1.5')
		# another 18 parameters for the second component
		if vlsr2 != 0:
			params.add('peaki_c2', value=30,  min=0, max=50)
			#params.add('peaki_c2', value=25, vary=False)
			params.add('tau11_c2', value=1.0, min=0, max=20.0)
			if varyv > 0:
				params.add('peakv_c2', value=vlsr2, min=vlsr2-varyv*onevpix, max=vlsr2+varyv*onevpix)
			elif varyv == 0:
				params.add('peakv_c2', value=vlsr2, vary=False)
			params.add('sigmav_c2', value=1.0, min=0, max=2.0)
			params.add('peaki_s1_c2', expr="peaki_c2*(1-exp(-tau11_s1_c2))/(1-exp(-tau11_c2))")
			params.add('tau11_s1_c2', expr='tau11_c2*0.278')
			params.add('peakv_s1_c2', expr='peakv_c2-7.47385')
			params.add('sigmav_s1_c2', expr='sigmav_c2')
			params.add('peaki_s2_c2', expr='peaki_s1_c2')
			params.add('tau11_s2_c2', expr='tau11_s1_c2')
			params.add('peakv_s2_c2', expr='peakv_c2+7.56923')
			params.add('sigmav_s2_c2', expr='sigmav_c2')
			params.add('peaki_upp_c2', expr='peaki_c2*(1-exp(-tau22_c2))/(1-exp(-tau11_c2))', min=0, max=50)
			#params.add('tau22_c2', value=1.0, min=0, expr='<=tau11_c2/0.282')
			params.add('tau22_c2', value=1.0, min=0, max=10.0)
			params.add('peakv_upp_c2', expr='peakv_c2+30.0')
			#params.add('sigmav_upp_c2', value=1.0, min=0, max=5.0)
			params.add('sigmav_upp_c2', expr="sigmav_c2")
			params.add('Trot_c2', value=0.0, expr='-41.5/log(0.282*tau22_c2/tau11_c2)', min=0)
			# Add one more parameter: the total NH3 column density (column density of NH3 (1,1) * the partition function)
			params.add('Ntot_c2', value=1e15, expr='6.6e14*Trot_c2/23.6944955*tau11_c2*2.355*sigmav_c2*0.0138*exp(23.1/Trot_c2)*Trot_c2**1.5')

		# do fit, here with leastsq model
		#spec[np.where(spec<=cutoff)] = 0.0
		if vlsr1 != 0 and vlsr2 != 0:
			result = minimize(__model_11_2c__, params, args=(vaxis, spec))
		elif vlsr1 != 0 or vlsr2 != 0:
			result = minimize(__model_11__, params, args=(vaxis, spec))
		else:
			unsatisfied = False
			continue
		
		final = spec + result.residual
		#report_fit(params)

		if interactive:
			plt.plot(vaxis, final, 'r', label='Best-fitted model')
			if vlsr1 != 0 and vlsr2 != 0:
				final_c1 = __model_11__(params, vaxis, spec) + spec
				# Reconstruct the Guassian of 2nd component, using the fitting results.
				peaki_c2 = params['peaki_c2'].value
				tau11_c2 = params['tau11_c2'].value
				peakv_c2 = params['peakv_c2'].value
				sigmav_c2 = params['sigmav_c2'].value
				peaki_s1_c2 = params['peaki_s1_c2'].value
				tau11_s1_c2 = params['tau11_s1_c2'].value
				peakv_s1_c2 = params['peakv_s1_c2'].value
				sigmav_s1_c2 = params['sigmav_s1_c2'].value
				peaki_s2_c2 = params['peaki_s2_c2'].value
				tau11_s2_c2 = params['tau11_s2_c2'].value
				peakv_s2_c2 = params['peakv_s2_c2'].value
				sigmav_s2_c2 = params['sigmav_s2_c2'].value
				peaki_upp_c2 = params['peaki_upp_c2'].value
				tau22_c2 = params['tau22_c2'].value
				peakv_upp_c2 = params['peakv_upp_c2'].value
				sigmav_upp_c2 = params['sigmav_upp_c2'].value
				final_c2 = __gauss_tau__(vaxis,[peaki_c2,tau11_c2,peakv_c2,sigmav_c2]) + \
						__gauss_tau__(vaxis,[peaki_s1_c2,tau11_s1_c2,peakv_s1_c2,sigmav_s1_c2]) + \
						__gauss_tau__(vaxis,[peaki_s2_c2,tau11_s2_c2,peakv_s2_c2,sigmav_s2_c2]) + \
						__gauss_tau__(vaxis,[peaki_upp_c2,tau22_c2,peakv_upp_c2,sigmav_upp_c2])
				plt.plot(vaxis, final_c1, 'm--', label='1st component', linewidth=2)
				plt.plot(vaxis, final_c2, 'c--', label='2nd component', linewidth=2)
				plt.text(0.02, 0.9, r'1st $T_\mathrm{rot}$=%.1f$\pm$%.1f K' % (params['Trot'].value,params['Trot'].stderr), transform=ax.transAxes, color='m', fontsize=15)
				plt.text(0.02, 0.8, r'2nd $T_\mathrm{rot}$=%.1f$\pm$%.1f K' % (params['Trot_c2'].value,params['Trot_c2'].stderr), transform=ax.transAxes, color='c', fontsize=15)
			elif vlsr1 != 0 or vlsr2 != 0:
				#plt.text(0.02, 0.9, r'$T_\mathrm{rot}$=%.1f$\pm$%.1f K' % (params['Trot'].value,params['Trot'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.85, r'$\sigma_v$(1,1)=%.1f$\pm$%.1f km/s' % (params['sigmav'].value,params['sigmav'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.8, r'$\sigma_v$(2,2)=%.1f$\pm$%.1f km/s' % (params['sigmav22'].value,params['sigmav22'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.75, r'$T_m$(1,1)=%.1f$\pm$%.1f K' % (params['peaki'].value,params['peaki'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.7, r'$T_m$(2,2)=%.1f$\pm$%.1f K' % (params['peaki22'].value,params['peaki22'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.65, r'$\tau_m$(1,1)=%.1f$\pm$%.1f' % (params['tau11'].value,params['tau11'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.6, r'$\tau_m$(2,2)=%.1f$\pm$%.1f' % (params['tau22'].value,params['tau22'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.55, r'N(1,1)=%.2e cm$^{-2}$' % params['N11'].value, transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.5, r'N(2,2)=%.2e cm$^{-2}$' % params['N22'].value, transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.45, r'N(3,3)=%.2e cm$^{-2}$' % params['N33'].value, transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.4, r'N(4,4)=%.2e cm$^{-2}$' % params['N44'].value, transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.35, r'N(5,5)=%.2e cm$^{-2}$' % params['N55'].value, transform=ax.transAxes, color='r', fontsize=15)
			plt.legend()
			plt.show()
			print 'Is the fitting ok? y/n'
			yn = raw_input()
			if yn == 'y':
				unsatisfied = False 
			else:
				unsatisfied = True
			#raw_input('Press any key to continue...')
			f.clear()
		else:
			unsatisfied = False

	# Plot the rotational diagram.
	f = plt.figure(figsize=(14,8))
	ax = f.add_subplot(111)

	Trot = 10.0
	B = 298.1 # Rotational constants, in GHz
	C = 186.7
	Qrot = 168.7*np.sqrt(Trot**3/B**2/C)
	g_u = np.empty(5) # g_J*g_K*g_I
	for j in [1,2,3,4,5]:
		if j % 3 == 0:
			g_u[j-1] = (2*j+1)*2*4/8.
		else:
			g_u[j-1] = (2*j+1)*2*2/8.
	NN = np.empty(5); EE = np.empty(5); NN_err = np.empty(5)
	NN[0] = params['N11'].value; NN_err[0] = params['N11'].stderr
	NN[1] = params['N22'].value; NN_err[1] = params['N22'].stderr
	NN[2] = params['N33'].value; NN_err[2] = params['N33'].stderr
	NN[3] = params['N44'].value; NN_err[3] = params['N44'].stderr
	NN[4] = params['N55'].value; NN_err[4] = params['N55'].stderr
	for j in [1,2,3,4,5]:
		NN[j-1] /= g_u[j-1]
		NN_err[j-1] /= g_u[j-1]
	NN = np.log10(NN); NN_err = np.log10(NN_err)
	EE[0] = 23.4; EE[1] = 64.9; EE[2] = 125.0; EE[3] = 202.0; EE[4] = 298.0
	print g_u
	print NN
	print EE

	if RD_method == 'polyfit':
	# Use ployfit, simple leastsq without Yerr.
		result = np.polyfit(EE,NN,1,cov=True)
		coeff = result[0]
		print coeff
		covar = result[1]
		print covar
		slope_error = np.sqrt(-1.0*covar[0][0])
		intersect_error = np.sqrt(-1.0*covar[1][1])
		trot = -1.0 / coeff[0]
		trot_u = -1.0 / (coeff[0] + slope_error)
		trot_l = -1.0 / (coeff[0] - slope_error)
		terr = (trot_u - trot_l) / 2.
		print 'Rotation temperature: ', trot, ' and its uncertainty: ', terr
	elif RD_method == 'lsq':
	# Try leastsq method of scipy which includes Yerr.
		fitfunc = lambda p, x: p[0] + p[1] * x
		errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
		pinit = [14, -0.01]
		result = leastsq(errfunc, pinit, args=(EE, NN, NN_err), full_output=True)
		coeff = result[0]
		null = coeff[0]; coeff[0] = coeff[1]; coeff[1] = null
		covar = result[1]
		print 'Result of fitting [slope, intersect]: ', coeff
		print 'Covariance matrix: '
		print covar
		slope_error = np.sqrt(covar[1][1])
		intersect_error = np.sqrt(covar[0][0])
		trot = -1.0 / coeff[0]
		trot_u = -1.0 / (coeff[0] + slope_error)
		trot_l = -1.0 / (coeff[0] - slope_error)
		terr = (trot_u - trot_l) / 2.
		print 'Rotation temperature: ', trot, ' and its uncertainty: ', terr
	else:
		raise KeyError('Either polyfit or lsq please!')

	poly = np.poly1d(coeff)
	NN_fit = poly(EE)

	if RD_method == 'polyfit':
		plt.plot(EE, NN, 'r+', markersize=10)
	elif RD_method == 'lsq':
		EE_err = np.zeros(5)
		plt.errorbar(EE, NN, NN_err, EE_err, 'r.', markersize=10)
	plt.plot(EE, NN_fit, 'g-')
	plt.xlabel(r'E (K)', fontsize=20, labelpad=20)
	plt.ylabel(r'$N_u$/$g_u$ ($cm^{-2}$)', fontsize=20)
	plt.text(0.6, 0.8, r'T$_{rot}$=%.1f$\pm$%.1f K' % (trot,terr), transform=ax.transAxes, color='m', fontsize=20)
	#plt.savefig(source+'_trot_peak.png')
	raw_input('Press any key to finish.')
	f.clear()



nh3_info = __nh3_init__()

ave1, vaxis_11 = __readascii__('NH3_11.txt')
ave2, vaxis_22 = __readascii__('NH3_22.txt')
ave3, vaxis_33 = __readascii__('NH3_33.txt')
ave4, vaxis_44 = __readascii__('NH3_44.txt')
ave5, vaxis_55 = __readascii__('NH3_55.txt')
vaxis_11 = vaxis_11[::-1]
vaxis_33 = vaxis_33[::-1]
ave1 = ave1[::-1]
ave2 = ave2[::-1]
ave3 = ave3[::-1]
ave4 = ave4[::-1]
ave5 = ave5[::-1]
fit_spec(ave1, ave2, ave3, ave4, ave5, vaxis_11, vaxis_33, cutoff=0.5, mode='single', varyv=0, RD_method='polyfit')

elapsed = (time.clock() - start)
print 'Stop the timer...'
print 'Time used: %0.0f seconds, or %0.1f minutes.' % (elapsed, elapsed/60.)

