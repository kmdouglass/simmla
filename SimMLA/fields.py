import numpy as np

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
        GSMCoherenceLength : 500
            
        '''
        self.power   = power
        self.beamStd = beamStd
        
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
        # Spatial sampling rate
        dx = x[1] - x[0]
        
        # Define phase screen functions
        
        # Convolve phase screen functions
        
        # Sample the field
        
        # Multiply the sampled field by the phase screen