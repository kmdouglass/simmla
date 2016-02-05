import numpy as np

class Gaussian1D():
    '''A perfectly coherent, 1D Gaussian beam.
    
    '''
    def __init__(self, power, beamStd):
        self.power   = power
        self.beamStd = beamStd
        
    def __call__():
        profile = lambda x: np.sqrt(power) / np.sqrt(2 * np.pi) / beamStd \
                          * np.exp(-x**2 / 2 / beamStd**2)
        return profile
        