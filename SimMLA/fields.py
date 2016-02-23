import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift

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
    
def GaussianBeamDefocused(amplitude, beamStd, wavelength, position):
    '''Returns the field from a Gaussian beam in an arbitrary plane.
    
    Parameters
    ----------
    amplitude  : float
        The amplitude of the input Gaussian beam.
    beamStd    : float
        The standard deviation of the beam's waist. This is related to the
        waist size through waist = sqrt(2) * beamStd.
    wavelength : float
    position   : float
        The axial position of the observation plane relative to the waist.
    
    '''
    # Compute the beam's waist from the standard deviation
    waist      = np.sqrt(2) * beamStd
    
    wavenumber = 2 * np.pi / wavelength
    
    # Compute the beam radius, curvature, and Gouy phase
    beamRad   = _wz(position, waist, wavelength)
    beamRoc   = _roc(position, waist, wavelength)
    gouyPhase = _gouyPhase(position, waist, wavelength)
    
    profile = lambda x: amplitude * np.sqrt(waist / beamRad) \
                      * np.exp(-x**2 / beamRad**2)  \
                      * np.exp(1j * wavenumber * position \
                             + 1j * wavenumber * x**2 / 2 / beamRoc \
                             - 1j * gouyPhase)
    return profile

def _wz(position, waist, wavelength):
    '''Computes the beam's radius at an arbitrary axial position.
    
    '''
    # Compute the Rayleigh range
    zR = np.pi * waist**2 / wavelength
    
    wz = waist * np.sqrt(1 + (position / zR)**2)
    return wz

def _roc(position, waist, wavelength):
    '''Computes the beam's radius of curvature at an arbitrary axial position.
    
    '''
    # Compute the Rayleigh range
    zR = np.pi * waist**2 / wavelength
    
    if position != 0:
        roc = position * (1 + (zR / position)**2)
    else:
        roc = 0
        
    return roc
    
def _gouyPhase(position, waist, wavelength):
    '''Computes the Gouy phase of the Gaussian beam.
    
    '''
    # Compute the Rayleigh range
    zR = np.pi * waist**2 / wavelength
    
    gouyPhase = np.arctan(position / zR)
    return gouyPhase

def GaussianWithDiffuser(amplitude,
                         beamStd,
                         physicalSize,
                         powerScat  = 0.01,
                         wavelength = 0.642,
                         fc         = 50000,
                         grainSize  = 40,
                         beamSize   = 100):
    '''A Gaussian beam passing through a telescope and rotating diffuser.
    
    Parameters
    ----------
    amplitude : float
        The amplitude of the **input Gaussian beam**, i.e. before the diffuser.
        The amplitude after the diffuser will be calculated from this.
    powerScat : float
        The fractional power scattered by the rotating diffuser. Must lie
        between 0 and 1.
    physicalSize  : float
        The physical size of the grid that the plane waves are defined on.
        This is used for power conservation.
    fc        : float
        The focal length of the collimating lens that collects the light coming
        from the diffuser.
        
    ''' 
    
    # Setup the point sources that model the diffuser
    numSources = np.ceil(beamSize / grainSize)
    srcAmp     = amplitude * np.sqrt(powerScat / numSources * beamStd * np.sqrt(np.pi) / physicalSize)
    srcCenters = np.arange(-numSources * grainSize / 2, numSources * (grainSize / 2) + grainSize, grainSize)

    # Return the deterministic field and scattered plane waves
    return lambda x: _applyDiffuser(x, amplitude, beamStd, srcAmp, srcCenters, powerScat, wavelength, fc)
    
def _applyDiffuser(x, amplitude, beamStd, srcAmp, srcCenters, powerScat, wavelength, fc):
    '''Computes the random plane waves coming from the diffuser.
    
    '''
    # Compute the plane waves' phases and directions
    planewaves = np.zeros(x.size)
    for ctr in range(int(srcCenters.size)):
        randomPhase = np.exp(1j * ((np.random.rand(1) * 2 * np.pi) - np.pi))
        planewaves  = planewaves + srcAmp * np.exp(1j * 2 * np.pi *srcCenters[ctr] * x / wavelength / fc) * randomPhase
        
    # Compute the carrier beam, i.e. the deterministic Gaussian
    newCarrierAmp = amplitude * np.sqrt(1 - powerScat)
    carrierBeam   = GaussianBeamWaistProfile(newCarrierAmp, beamStd)
    
    return carrierBeam(x) + planewaves
    
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
    
def diffuserMask(sigma_f, sigma_r, grid):
    '''Returns a single realization of the partially coherent GSM beam.
    
    '''
    # The spatial frequency grid spacing is required for normalizing the random
    # array of the phase screen.
    
    return lambda x: _applyDiffuserMask(x, sigma_f, sigma_r, grid.pfX)
        
def _applyDiffuserMask(x, sigma_f, sigma_r, pfX):
    '''Computes the random phase mask at the grid locations.
    
    Notes
    -----
    Partially coherent beam simulation from Xifeng Xiao and David Voelz,
    "Wave optics simulation approach for partial spatially coherent beams."
    Opt. Express 14, 6986-6992 (2006)
    
    '''
    dx      = x[1] - x[0] # Assumes uniform spacing between samples
    dpfX    = pfX[1] - pfX[0]

    # Convolve phase screen functions
    F = ifftshift(np.exp(-np.pi**2 * sigma_f**2 * pfX**2));
    R = np.random.randn(x.size) + 1.0j * np.random.randn(x.size)
    
    # From Voelz, "Computational Fourier Optics: A MATLAB Tutorial", Chap. 9
    phaseScreen = 2 * np.pi * fftshift(ifft(F*R)) * sigma_r / (dx * np.sqrt(dpfX))
    
    # Sample the field
    mask = np.exp(1.0j * np.real(phaseScreen))
    
    return mask