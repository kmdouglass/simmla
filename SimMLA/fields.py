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

def GSMBeamRealization(amplitude, beamStd, cohLength, grid):
    '''Returns a single realization of the partially coherent GSM beam.
    
    '''
    # The spatial frequency grid spacing is required for normalizing the random
    # array of the phase screen.
    
    #dpX = grid.pX[1] - grid.pX[0]
    dpfX = grid.pfX[1] - grid.pfX[0]
    
    return lambda x: _applyMask(x, amplitude, beamStd, cohLength, dpfX, grid.pfX)
        
def _applyMask(x, amplitude, beamStd, cohLength, dpfX, pfX):
    '''Computes the random phase mask at the grid locations.
    
    '''

    # Define phase screen functions
    dx = x[1] - x[0] # Assumes uniform spacing between samples
    sigma_f = 2.5 * cohLength
    sigma_r = np.sqrt(4 * np.pi * sigma_f**4 / cohLength**2)
    
    #f = 1 / np.sqrt(np.pi) / sigma_f * np.exp(-x**2 / sigma_f**2)

    # Convolve phase screen functions
    # F = dx * fft(ifftshift(f))
    F = ifftshift(np.exp(-np.pi**2 * sigma_f**2 * pfX**2));
    R = np.random.randn(x.size) + 1j * np.random.randn(x.size)
      
    
    #phaseScreen = fftshift(ifft(F * R)) * sigma_r / dx / np.sqrt(dpfX)
    phaseScreen = 0.01 * fftshift(ifft(F * R)) * sigma_r * dpfX * x.size / np.sqrt(dpfX)
    
    # Sample the field
    fieldFunc = GaussianBeamWaistProfile(amplitude, beamStd)
    field = fieldFunc(x) * np.exp(1j * np.real(phaseScreen))
    
    return field