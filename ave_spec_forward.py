import os, time
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants, units
from lmfit import minimize, Parameters, report_fit

start = time.clock()
print 'Start the timer...'

# Define some useful constants first:
c = constants.c.cgs.value # Speed of light (cm/s)
k_B = constants.k_B.cgs.value # Boltzmann coefficient (erg/K)
h = constants.h.cgs.value # Planck constant (erg*s)


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
	nh3_info['E'] = [23.4, 64.9, 125., 202., 298., 412.] # Upper level energy (Kelvin)
	nh3_info['frest'] = [23.6944955, 23.7226333, 23.8701292, \
	                     24.1394163, 24.5329887, 25.0560250] # Rest frequency (GHz)
	nh3_info['b0'] = 298.117 # 
	nh3_info['c0'] = 186.726 # Rotational constants (GHz)
	nh3_info['mu'] = 1.468e-18 # Permanet dipole moment (esu*cm)
	nh3_info['gI'] = [2./8, 2./8, 4./8, 2./8, 2./8, 4./8]
	nh3_info['Ri'] = [[0.5,5./36,1./9], [56./135+25./108+3./20,7./135,1./20], \
					[0.8935,0,0], [0.935,0,0], [0.956,0,0], [0.969,0,0]] # Relative strength
	nh3_info['vsep'] = [[7.522,19.532], [16.210,26.163], 21.49, 24.23, 25.92, 26.94]
	# Velocity separations between hyperfine lines (km/s)

	return nh3_info

def __gauss_tau__(axis,p):
	"""
	Genenerate a Gaussian model given an axis and a set of parameters.
	p: [T, Ntot, vlsr, sigmav, J, hyper, ff]
	hyper = 0 -- main
	hyper =-1 -- left inner satellite
	hyper = 1 -- right inner satellite
	hyper =-2 -- left outer satellite
	hyper = 2 -- right outer satellite
	"""
	T = p[0]; Ntot = p[1]; vlsr = p[2]; sigmav = p[3]; J = p[4]; hyper = p[5]; ff = p[6]
	K = J
	gk = 2

	if hyper > 0:
		vlsr = vlsr + nh3_info['vsep'][J-1][hyper-1]
	elif hyper < 0:
		vlsr = vlsr - nh3_info['vsep'][J-1][abs(hyper)-1]

	phijk = 1/sqrt(2 * pi) / (sigmav * 1e5) * np.exp(-0.5 * (axis - vlsr)**2 / sigmav**2)
	Ri = nh3_info['Ri'][J-1][abs(hyper)]
	Ajk = (64 * pi**4 * (nh3_info['frest'][J-1] * 1e9)**3 * nh3_info['mu']**2 / 3 / h / c**3) * (K**2) / (J * (J + 1)) * Ri
	gjk = (2*J + 1) * gk * nh3_info['gI'][J-1]
	m = nh3_info['b0'] / nh3_info['c0']
	b0 = nh3_info['b0']
	c0 = nh3_info['c0']
	#Q = sqrt(m*pi) / 3.0 * exp(h*b0*1e9*(4-m)/12.0/k_B/T) * (k_B*T/h/b0/1e9)**1.5 * (1 + 1/90.0*(h*b0*1e9*(1-m)/k_B/T)**2)
	Q = 168.7*sqrt(T**3/b0**2/c0)
	Njk = Ntot * (gjk / Q) * exp(-1.0 * nh3_info['E'][J-1] / T)

	tau = (h * c**3 * Njk * Ajk) / (8 * pi * nh3_info['frest'][J-1]**2 * 1e18 * k_B * T) * phijk
	f = T * ff * (1 - np.exp(-1.0 * tau))

	return f

def __model_11__(params, vaxis, spec):
	"""
	Model five components of NH3 (1,1) and five components of NH3 (2,2),
	then subtract data.
	"""
	T = params['T'].value
	Ntot = params['Ntot'].value
	vlsr = params['vlsr'].value
	sigmav = params['sigmav'].value
	ff = params['ff'].value

	vlsr22 = vlsr + 150.
	vlsr33 = vlsr + 266.
	vlsr44 = vlsr + 336.
	vlsr55 = vlsr + 406.

	model = __gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,0,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,-1,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,1,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,-2,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,2,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,0,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,-1,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,1,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,-2,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,2,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr33,sigmav,3,0,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr44,sigmav,4,0,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr55,sigmav,5,0,ff])

	return model - spec

clickvalue = []
def onclick(event):
	print 'The Vlsr you select: %f' % event.xdata
	clickvalue.append(event.xdata)

def fit_spec(spec1, spec2, spec3, spec4, spec5, vaxis1, vaxis2, cutoff=0.009, varyv=2, interactive=True, mode='single', RD_method='polyfit'):
	"""
	fit_spec(spec1, spec2, spec3, spec4, spec5)
	Fit the five NH3 spectra simultaneously, derive best-fitted temperature and column density.
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

	unsatisfied = True
	while unsatisfied:
		if interactive:
			f.clear()
			plt.ion()
			plt.plot(vaxis, spec, 'k-', label='Spectrum')
			cutoff_line = [cutoff] * len(vaxis)
			cutoff_line_minus = [-1.0*cutoff] * len(vaxis)
			plt.plot(vaxis, cutoff_line, 'r--')
			plt.plot(vaxis, cutoff_line_minus, 'r--')
			plt.xlabel(r'$V_\mathrm{lsr}$ (km s$^{-1}$)', fontsize=20, labelpad=20)
			plt.ylabel(r'$T_{\nu}$ (K)', fontsize=20)
			plt.text(0.02, 0.92, sourcename, transform=ax.transAxes, color='r', fontsize=15)
			#plt.ylim([-10,60])
			#clickvalue = []
			if mode == 'single':
				cid = f.canvas.mpl_connect('button_press_event', onclick)
				raw_input('Click on the plot to select Vlsr...')
				#print clickvalue
				if len(clickvalue) >= 1:
					print 'Please select at least one velocity! The last one will be used.'
					vlsr1 = clickvalue[-1]
				elif len(clickvalue) == 0:
					vlsr1 = -9999
				print 'Or input one velocity manually:'
				manualv = raw_input()
				manualv = manualv.split()
				if len(manualv) == 1:
					vlsr1 = np.float_(manualv)
				else:
					print 'Invalid input...'
				print 'The Vlsr is %0.2f km/s' % (vlsr1)
				raw_input('Press any key to start fitting...')
				f.canvas.mpl_disconnect(cid)
				vlsr2 = -9999
			elif mode == 'double':
				cid = f.canvas.mpl_connect('button_press_event', onclick)
				raw_input('Click on the plot to select Vlsrs...')
				print clickvalue
				if len(clickvalue) >= 2:
					print 'Please select at least two velocities! The last two will be used.'
					vlsr1,vlsr2 = clickvalue[-2],clickvalue[-1]
				elif len(clickvalue) == 1:
					vlsr1 = clickvalue[-1]
					vlsr2 = -9999
				elif len(clickvalue) == 0:
					vlsr1,vlsr2 = -9999,-9999
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
				vlsr1,vlsr2 = -9999,-9999
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

		plt.text(0.02, 0.85, r'$V_\mathrm{lsr}=%.1f$ km/s' % (vlsr1), transform=ax.transAxes, color='r', fontsize=15)

		# Add 4 parameters
		params = Parameters()
		if vlsr1 != -9999:
			params.add('Ntot', value=1e15, min=0, max=1e20)
			params.add('T', value=100, min=0, max=1000)
			params.add('sigmav', value=2., min=0, max=20.)
			if varyv > 0:
				params.add('vlsr', value=vlsr1, min=vlsr1-varyv*onevpix, max=vlsr1+varyv*onevpix)
			elif varyv == 0:
				params.add('vlsr', value=vlsr1, vary=False)
			params.add('ff', value=0.5, min=0.1, max=1.0)
		# another 4 parameters for the second component
		if vlsr2 != -9999:
			params.add('Ntot_c2', value=1e15, min=0, max=1e20)
			params.add('T_c2', value=100, min=0, max=1000)
			params.add('sigmav_c2', value=5.0, min=0, max=20.0)
			if varyv > 0:
				params.add('vlsr_c2', value=vlsr1, min=vlsr1-varyv*onevpix, max=vlsr1+varyv*onevpix)
			elif varyv == 0:
				params.add('vlsr_c2', value=vlsr1, vary=False)

		# Run the non-linear minimization
		if vlsr1 != -9999 and vlsr2 != -9999:
			result = minimize(__model_11_2c__, params, args=(vaxis, spec))
		elif vlsr1 != -9999 or vlsr2 != -9999:
			result = minimize(__model_11__, params, args=(vaxis, spec))
		else:
			unsatisfied = False
			continue
		
		final = spec + result.residual
		#report_fit(params)

		if interactive:
			plt.plot(vaxis, final, 'r', label='Best-fitted model')
			if vlsr1 != -9999 and vlsr2 != -9999:
				print 'Reserved for two-component fitting.'
			elif vlsr1 != -9999 or vlsr2 != -9999:
				plt.text(0.02, 0.80, r'T$_{rot}$=%.1f($\pm$%.1f) K' % (params['T'].value,params['T'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.75, r'N$_{tot}$=%.2e($\pm$%.2e) cm$^{-2}$' % (params['Ntot'].value,params['Ntot'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.70, r'FWHM=%.2f($\pm$%.2f) km/s' % (2.355*params['sigmav'].value,2.355*params['sigmav'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.65, r'Filling factor=%.2f($\pm$%.2f)' % (params['ff'].value,params['ff'].stderr), transform=ax.transAxes, color='r', fontsize=15)
			plt.legend()
			plt.show()
			print 'Is the fitting ok? y/n'
			yn = raw_input()
			if yn == 'y':
				unsatisfied = False 
				currentT = time.strftime("%Y-%m-%d_%H:%M:%S")
				plt.savefig('NH3_fitting_'+currentT+'.png')
			else:
				unsatisfied = True
			#raw_input('Press any key to continue...')
			f.clear()
		else:
			unsatisfied = False

##############################################################################

nh3_info = __nh3_init__()

ave1, vaxis_11 = __readascii__('NH3_11.txt')
ave2, vaxis_22 = __readascii__('NH3_22.txt')
ave3, vaxis_33 = __readascii__('NH3_33.txt')
ave4, vaxis_44 = __readascii__('NH3_44.txt')
ave5, vaxis_55 = __readascii__('NH3_55.txt')
sourcename = 'Sgr C'

vaxis_11 = vaxis_11[::-1]
vaxis_33 = vaxis_33[::-1]
ave1 = ave1[::-1]
ave2 = ave2[::-1]
ave3 = ave3[::-1]
ave4 = ave4[::-1]
ave5 = ave5[::-1]

fit_spec(ave1, ave2, ave3, ave4, ave5, vaxis_11, vaxis_33, cutoff=0.5, mode='single', varyv=0)


elapsed = (time.clock() - start)
print 'Stop the timer...'
print 'Time used: %0.0f seconds, or %0.1f minutes.' % (elapsed, elapsed/60.)
