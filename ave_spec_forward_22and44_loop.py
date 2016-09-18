#####################################################################
##
#####################################################################

import os, time
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants, units
from lmfit import minimize, Parameters, report_fit
from astropy.io import fits

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

    # sigmav in the equation below must be in unit of cm/s!!!
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

def __tau__(Ntot, sigmav, T, J, hyper):
	"""
	Calculate tau
	"""
	K = J; gk = 2
	phi = 1/sqrt(2 * pi) / (sigmav * 1e5)
	Ri = nh3_info['Ri'][J-1][abs(hyper)]
	Ajk = (64 * pi**4 * (nh3_info['frest'][J-1] * 1e9)**3 * nh3_info['mu']**2 / 3 / h / c**3) * (K**2) / (J * (J + 1)) * Ri
	gjk = (2*J + 1) * gk * nh3_info['gI'][J-1]
	b0 = nh3_info['b0']
	c0 = nh3_info['c0']
	Q = 168.7*sqrt(T**3/b0**2/c0)
	Njk = Ntot * (gjk / Q) * exp(-1.0 * nh3_info['E'][J-1] / T)
	tau = (h * c**3 * Njk * Ajk) / (8 * pi * nh3_info['frest'][J-1]**2 * 1e18 * k_B * T) * phi

	return tau

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
	ff2 = params['ff2'].value
	#ff3 = params['ff3'].value

	vlsr22 = vlsr
	#vlsr33 = vlsr + 266.
	vlsr44 = vlsr + 146.
	#vlsr55 = vlsr + 336.

	model = __gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,0,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,-1,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,1,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,-2,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr22,sigmav,2,2,ff]) + \
			__gauss_tau__(vaxis,[T,Ntot,vlsr44,sigmav,4,0,ff2])
#			__gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,0,ff]) + \
#			__gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,-1,ff]) + \
#			__gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,1,ff]) + \
#			__gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,-2,ff]) + \
#			__gauss_tau__(vaxis,[T,Ntot,vlsr,sigmav,1,2,ff]) + \

	return model - spec

#__gauss_tau__(vaxis,[T,Ntot,vlsr33,sigmav,3,0,ff3]) + \

clickvalue = []
def onclick(event):
	print 'The Vlsr you select: %f' % event.xdata
	clickvalue.append(event.xdata)

def fit_spec(spec2, spec4, vaxis2, vaxis4, cutoff=0.009, varyv=2, interactive=True, mode='single'):
	"""
	fit_spec(spec2, spec4)
	Only fit NH3 (2,2) and (4,4).
	"""
	if interactive:
		plt.ion()
		f = plt.figure(figsize=(14,8))
		ax = f.add_subplot(111)

	#cutoff = cutoff
	spec = np.concatenate((spec2, spec4))
	vaxis = np.concatenate((vaxis2, vaxis4+146.0))

	unsatisfied = True
	while unsatisfied:
		if interactive:
			f.clear()
			plt.plot(vaxis, spec, 'k-', label='Spectrum')
			cutoff_line = [cutoff] * len(vaxis)
			cutoff_line_minus = [-1.0*cutoff] * len(vaxis)
			plt.plot(vaxis, cutoff_line, 'r--')
			plt.plot(vaxis, cutoff_line_minus, 'r--')
			plt.xlabel(r'$V_\mathrm{lsr}$ (km s$^{-1}$)', fontsize=20, labelpad=10)
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
				print 'The Vlsr is %0.2f km/s' % vlsr1
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
			# Non-interactive mode:
			if mode == 'single':
				v_low = spec2[50:120].argmax()+50
				p_low = spec2[v_low]
				v_high = spec4[10:80].argmax()+10
				p_high = spec4[v_high]
				if p_low >= cutoff and p_high >= cutoff:
				#if p_low >= cutoff:
					vlsr1 = vaxis2[v_low]
				else:
					vlsr1 = -9999
				vlsr2 = -9999
			elif mode == 'double':
				vlsr1,vlsr2 = 86.0,88.0
			else:
				vlsr1,vlsr2 = -9999,-9999

		if interactive:
			plt.text(0.02, 0.85, r'$V_\mathrm{lsr}=%.1f$ km/s' % vlsr1, transform=ax.transAxes, color='r', fontsize=15)

		# Add 5 parameters
		params = Parameters()
		if vlsr1 != -9999:
			params.add('Ntot', value=1e15, min=0, max=1e20)
			params.add('T', value=100, min=0, max=1000)
			params.add('sigmav', value=2.0, min=0, max=10.0)
			if varyv > 0:
				params.add('vlsr', value=vlsr1, min=vlsr1-varyv*onevpix, max=vlsr1+varyv*onevpix)
			elif varyv == 0:
				params.add('vlsr', value=vlsr1, vary=False)
			#params.add('ff', value=0.5, min=0.001, max=1.0)
			#params.add('ff2', value=0.5, min=0.001, max=1.0)
			params.add('ff', value=1.0, vary=False)
			params.add('ff2', value=1.0, vary=False)
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
			Tgas = 0.0
			print 'Break the loop'
			continue
		
		final = spec + result.residual
		#report_fit(params)
		Tgas = result.params['T'].value

		if interactive:
			plt.plot(vaxis, final, 'r', label='Best-fitted model')
			if vlsr1 != -9999 and vlsr2 != -9999:
				print 'Reserved for two-component fitting.'
			elif vlsr1 != -9999 or vlsr2 != -9999:
				plt.text(0.02, 0.80, r'T$_{rot}$=%.1f($\pm$%.1f) K' % (result.params['T'].value,result.params['T'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.75, r'N$_{tot}$=%.2e($\pm$%.2e) cm$^{-2}$' % (result.params['Ntot'].value,result.params['Ntot'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.70, r'FWHM=%.2f($\pm$%.2f) km/s' % (2.355*result.params['sigmav'].value,2.355*result.params['sigmav'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.65, r'tau(2,2,m)=%.3f' % (__tau__(result.params['Ntot'].value,result.params['sigmav'].value,result.params['T'].value,2,0)), transform=ax.transAxes, color='r', fontsize=15)
				#plt.text(0.02, 0.60, r'tau(2,2,m)=%.3f' % (__tau__(params['Ntot'].value,params['sigmav'].value,params['T'].value,2,0)), transform=ax.transAxes, color='r', fontsize=15)
				#plt.text(0.42, 0.85, r'Filling factor=%.2f($\pm$%.2f)' % (params['ff'].value,params['ff'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				#plt.text(0.42, 0.80, r'Filling factor=%.2f($\pm$%.2f)' % (params['ff2'].value,params['ff2'].stderr), transform=ax.transAxes, color='r', fontsize=15)
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
		else:
			unsatisfied = False
	
	print 'The vlsr for [%i,%i] is %.2f' % (i, j, vlsr1)
	print 'Tgas for pixel [%i,%i] is %.1f' % (i, j, Tgas)
	return Tgas

##############################################################################

nh3_info = __nh3_init__()

# Get the velocity axes:
ave2, vaxis_22 = __readascii__('ratio3_22.txt')
ave4, vaxis_44 = __readascii__('ratio3_44.txt')
vaxis_22 = vaxis_22[::-1]
vaxis_44 = vaxis_44[::-1]
onevpix = 1.0

# Read the FITS cubes.
inputim = '../NH3_22/C20kms.NH322.1kms_v2.image.fits'
img2 = fits.open(inputim)
dat2 = img2[0].data
hdr2 = img2[0].header

## The three axes loaded by pyfits are: velo, dec, ra
## Swap the axes, RA<->velo
naxis = hdr2['NAXIS']
if naxis == 4:
	data2 = np.swapaxes(dat2[0],0,2)
elif naxis == 3:
	data2 = np.swapaxes(dat2,0,2)

inputim = '../NH3_44/C20kms.NH344.1kms_v2.image.fits'
img4 = fits.open(inputim)
dat4 = img4[0].data
hdr4 = img4[0].header

## The three axes loaded by pyfits are: velo, dec, ra
## Swap the axes, RA<->velo
naxis = hdr4['NAXIS']
if naxis == 4:
	data4 = np.swapaxes(dat4[0],0,2)
elif naxis == 3:
	data4 = np.swapaxes(dat4,0,2)

bmaj = hdr2['BMAJ'] * 3600.
bmin = hdr2['BMIN'] * 3600.
Tcube = np.zeros(shape=[hdr2['NAXIS2'], hdr2['NAXIS1']])
# 1sigma rms = 0.002 Jy/beam
threesigma = 1.224e6 * 0.010 / (nh3_info['frest'][1])**2 / (bmaj * bmin)

for i in range(0,hdr2['NAXIS1']):
	for j in range(0,hdr2['NAXIS2']):
		spec2 = data2[i,j,:]
		spec2 = spec2[::-1]
		spec4 = data4[i,j,:]
		spec4 = spec4[::-1]
		# Convert spec to brightness temperatures
		spec2 = 1.224e6 * spec2 / (nh3_info['frest'][1])**2 / (bmaj * bmin)
		spec4 = 1.224e6 * spec4 / (nh3_info['frest'][3])**2 / (bmaj * bmin)
		if spec2.max() >= threesigma and spec4.max() >= threesigma:
			Tcube[j,i] = fit_spec(spec2, spec4, vaxis_22, vaxis_44, cutoff=threesigma, varyv=2, interactive=False, mode='single')


hdr2['NAXIS'] = 2; hdr2['BUNIT'] = 'K'; hdr2['BTYPE'] = 'Temperature'
#del hdr[21:37]; del hdr[31:41]
del hdr2['NAXIS3']; del hdr2['CTYPE3']; del hdr2['CRVAL3']; del hdr2['CDELT3']; del hdr2['CRPIX3']; del hdr2['CUNIT3']
# Or: hdr.remove('Key')
fits.writeto('Tmap_final.fits', Tcube, hdr2, clobber=True)

img2.close()
img4.close()

elapsed = (time.clock() - start)
print 'Stop the timer...'
print 'Time used: %0.0f seconds, or %0.1f minutes.' % (elapsed, elapsed/60.)
