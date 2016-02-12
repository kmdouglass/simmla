import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift

'''
Parameters
----------
amplitude : float
    The field amplitude carried by the Gaussian beam.
beamStd  : float
    The beam width parameter (standard deviation).
partiallyCoherent  : bool

Notes
-----
Partially coherent beam simulation from Xifeng Xiao and David Voelz,
"Wave optics simulation approach for partial spatially coherent beams."
Opt. Express 14, 6986-6992 (2006)
    
'''

def GaussianBeamWaistProfile(amplitude, beamStd):
    '''
    Returns
    -------        
    profile : numpy vectorized function
        A function describing the 1D beam profile.
        
    '''
    profile = lambda x: amplitude * np.exp(-x**2 / 2 / beamStd**2)
    profile = np.vectorize(profile)
    
    return profile

def GaussianWithDiffuser(amplitude, beamStd):
    '''A Gaussian beam passing through a telescope and rotating diffuser.
    
    '''
    pass

def GSMBeamRealization(amplitude, beamStd, cohLength, grid):
    '''Returns a single realization of the partially coherent GSM beam.
    
    '''
    # The spatial frequency grid spacing is required for normalizing the random
    # array of the phase screen.
    
    return lambda x: _applyMask(x, amplitude, beamStd, cohLength, grid.pfX)
        
def _applyMask(x, amplitude, beamStd, cohLength, pfX):
    '''Computes the random phase mask at the grid locations.
    
    '''
    dx      = x[1] - x[0] # Assumes uniform spacing between samples
    dpfX    = pfX[1] - pfX[0]
    
    # Define phase screen parameters
    sigma_f = 2.5 * cohLength
    sigma_r = np.sqrt(4 * np.pi * sigma_f**4 / cohLength**2)

    # Convolve phase screen functions
    F = ifftshift(np.exp(-np.pi**2 * sigma_f**2 * pfX**2));
    R = np.random.randn(x.size) + 1.0j * np.random.randn(x.size)
    
    # From Voelz, "Computational Fourier Optics: A MATLAB Tutorial", Chap. 9
    phaseScreen = 2 * np.pi * fftshift(ifft(F*R)) * sigma_r / (dx * np.sqrt(dpfX))
    
    # Sample the field
    fieldFunc = GaussianBeamWaistProfile(amplitude, beamStd)
    field = fieldFunc(x) * np.exp(1.0j * np.real(phaseScreen))
    
    return field