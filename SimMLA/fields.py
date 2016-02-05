import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift

class Gaussian1D():
    '''A perfectly coherent, 1D Gaussian beam.
    
    '''
    def __init__(self,
                 power,
                 beamStd,
                 partiallyCoherent  = False):
        '''
        Parameters
        ----------
        power   : float
            The total power carried by the Gaussian beam.
        beamStd : float
            The beam width parameter (standard deviation).
        partiallyCoherent  : bool
        
        Notes
        -----
        Partially coherent beam simulation from Xifeng Xiao and David Voelz,
        "Wave optics simulation approach for partial spatially coherent beams."
        Opt. Express 14, 6986-6992 (2006)
            
        '''
        self.power             = power
        self.beamStd           = beamStd
        self.partiallyCoherent = partiallyCoherent
        
    def __call__(self):
        '''Returns one realization of the beam.
        
        Returns
        -------
        returnGSMBeamRealization : function
        returnBeamProfile        : function
        
        '''
        if self.partiallyCoherent:
            return self._returnGSMBeamRealization
        else:
            return self._returnBeamProfile
    
    def _returnBeamProfile(self):
        '''
        Returns
        -------        
        profile : numpy vectorized function
            A function describing the 1D beam profile.
            
        '''
        profile = lambda x: np.sqrt(self.power)                 \
                          / np.sqrt(2 * np.pi) / self.beamStd   \
                          * np.exp(-x**2 / 2 / self.beamStd**2)
        profile = np.vectorize(profile)
        
        return profile
        
    def _returnGSMBeamRealization(self, cohLength):
        '''Returns a single realization of the partially coherent GSM beam.
        
        '''
        return lambda x: self._applyMask(x, cohLength)
        
    def _applyMask(self, x, cohLength):
        '''Computes the random phase mask at the grid locations.
        
        '''

        # Define phase screen functions
        dx = x[1] - x[0] # Assumes uniform spacing between samples
        sigma_f = 10 * cohLength / np.sqrt(2)
        sigma_r = 20 * np.sqrt(np.pi) * sigma_f
        
        f = 1 / np.sqrt(2 * np.pi) / sigma_f * np.exp(-x**2 / 2 / sigma_f**2)
        r = sigma_r * np.sqrt(12) * np.random.rand(x.size)

        # Convolve phase screen functions
        F = dx * fft(ifftshift(f))
        R = dx * fft(ifftshift(r))
        
        phaseScreen = fftshift(ifft(F * R)) / dx
        
        # Sample the field
        fieldFunc = self._returnBeamProfile()
        field = fieldFunc(x) * np.exp(1j * phaseScreen)
        
        return field