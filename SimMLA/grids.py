import numpy as np

# Function definitions
def isEven(x):
    return (x % 2 == 0)
    
# Class definitions
class ImproperDimensionException(Exception):
    pass    

class ImproperGridSizeException(Exception):
    pass

class Grid(object):
    def __init__(self, gridSize, physicalSize, wavelength, focalLength, dim = 2):
        '''Establishes a square grid for sampling an electromagnetic field.
        
        The grid is square with an odd number of grid locations along one
        linear dimension. It is centered at (0,0) and may be converted into
        physical units for easily sampling known field distributions.
        
        Lowercase letters are used for real units; Uppercase for Fourier
        transform units.
        
        Parameters
        ----------
        gridSize     : int
            The size of the square grid in units of grid locations.
            Must be an odd integer.
        physicalSize : float
            The full linear extent of the grid in physical units.
        wavelength   : float
            The wavelength of the EM field.
        focalLength  : float
            The focal length of the lens used to compute units of
            the Fourier transform.
        dim          : int
            The dimension of the grid (can be 1 or 2).
            
        '''
        if (not isinstance(gridSize, int)) or isEven(gridSize) or (gridSize <= 0):
            raise ImproperGridSizeException('gridSize parameter is not an odd, positive integer.')
        
        self.gridSize     = gridSize
        self.physicalSize = physicalSize
        self.wavelength   = wavelength
        self.focalLength  = focalLength
        
        coords = np.arange(-np.floor(gridSize / 2), (np.floor(gridSize / 2)) + 1)
        
        # Create the grid
        if dim == 2:
            self.x, self.y = np.meshgrid(coords, coords)
        elif dim == 1:
            self.x = coords
        else:
            raise ImproperDimensionException('dim must be an integer equal to 1 or 2.')
        
        # Setup the conversion factors
        self._gridToPhys         = self.physicalSize / (self.gridSize - 1)
        self._gridToFTGrid       = 1 / self.gridSize
        self._gridToPhysFTGrid   = 1 / self.physicalSize
    
    @property
    def px(self):
        '''Return the x-grid in physical units.
        
        '''
        return self.x * self._gridToPhys
    
    @property
    def py(self):
        '''Return the y-grid in physical units.
        
        '''
        return self.y * self._gridToPhys
    
    @property
    def X(self):
        '''Return the X-grid of the Fourier transform.
        
        '''
        return self.x * self._gridToFTGrid
    
    @property
    def Y(self):
        '''Return the Y-grid of the Fourier transform.
        
        '''
        return self.y * self._gridToFTGrid
    
    @property
    def pX(self):
        '''Return the X-grid of the Fourier transform in physical units.
        
        Units are wavelength * (focal length) / (physical size).
        '''
        return self.x * self.wavelength * self.focalLength * self._gridToPhysFTGrid
    
    @property
    def pY(self):
        '''Return the Y-grid of the Fourier transform in physical units.
        
        Units are wavelength * (focal length) / (physical size).
        '''
        return self.y * self.wavelength * self.focalLength * self._gridToPhysFTGrid
        
    @property
    def pfX(self):
        '''Return the X-grid of the Fourier transform in spatial frequencies.
        
        This is spatial frequency fx = x' / (wavelength * focalLength).
        '''
        return self.x * self._gridToPhysFTGrid
    
    @property
    def pfY(self):
        '''Return the X-grid of the Fourier transform in spatial frequencies.
        
        This is spatial frequency fy = y' / (wavelength * focalLength).
        '''
        return self.y * self._gridToPhysFTGrid
        
class GridArray(Grid):
    '''An array of grids on a fixed coordinate system.
    
    The grid array facilitates building a square array of non-overlapping square grids
    by managing the placement of the subgrids on a fixed coordinate system.
    
    '''
    def __init__(self, numSubgrids, subgridSize, physicalSize, wavelength, focalLength, dim = 2):
        '''Builds an array of grids all lying on a common coordinate system.
        
        Parameters
        ----------
        numSubgrids : int
            The number of subgrids in the full coordinate system. Must be odd.
        subGridSize : int
            The linear size of a square subgrid. Must be odd.
        physicalSize : float
            The size of the full grid in physical units.
        wavelength : float
            The wavelength of the light for computing the grid of the Fourier transform.
        focalLength : float
            The focal length of the lens for computing the grid of the Fourier transform.
        dim          : int
            The dimension of the grid (can be 1 or 2).
       
        ''' 
        if (not isinstance(numSubgrids, int)) or isEven(numSubgrids) or (numSubgrids <= 0):
            raise ImproperGridSizeException('numSubgrids parameter is not an odd, positive integer.')
            
        if (not isinstance(subgridSize, int)) or isEven(subgridSize) or (subgridSize <= 0):
            raise ImproperGridSizeException('subgridSize parameter is not an odd, positive integer.')

        self.numSubgrids = numSubgrids
        self.subgridSize = subgridSize    
        
        # Build the common coordinate system
        gridSize = numSubgrids * subgridSize
        super(GridArray, self).__init__(gridSize, physicalSize, wavelength, focalLength, dim = dim)
        
        # Set the centers of the subgrids
        self.subgridCenters = subgridSize * np.arange(-np.floor(numSubgrids / 2), np.floor(numSubgrids / 2) + 1)
        self.subgridx, self.subgridy = np.meshgrid(self.subgridCenters, self.subgridCenters)
        
    def rect(self, fieldIn, xInd):
        '''Samples an input 1D field on the given subgrid.
        
        Parameters
        ----------
        fieldIn : function
            A 2D, real or complex valued function defining an input field distribution.
        xInd    : int
            x-index (column) of the subgrid.
        
        Returns
        -------
        fieldSample : 1D array of complex
            The input field on the given subgrid.
            
        '''
        xCenter = self.subgridCenters[xInd]
        
        # Sample the field onto the grid
        fieldSample = fieldIn(self.px)
        
        # Create a mask centered on the specified subgrid
        sgHalfSize = np.floor(self.subgridSize / 2)
        mask       = np.logical_and(self.x >= (xCenter - sgHalfSize), (self.x <= xCenter + sgHalfSize))

        # Return the sampled and masked field
        return fieldSample * mask.astype(int)
    
    def rect2(self, fieldIn, xInd, yInd):
        '''Samples an input 2D field at the given subgrid.
        
        Parameters
        ----------
        fieldIn : function
            A 2D, real or complex valued function defining an input field distribution.
        xInd    : int
            x-index (column) of the subgrid.
        yInd    : int
            y-index (row) of the subgrid.
        
        Returns
        -------
        fieldSample : 2D array of complex
            The input field at the given subgrid.
            
        '''
        xCenter, yCenter = self.subgridCenters[xInd], self.subgridCenters[yInd]
        
        # Sample the field onto the grid
        fieldSample = fieldIn(self.px, self.py)
        
        # Create a mask centered on the specified subgrid
        sgHalfSize = np.floor(self.subgridSize / 2)
        maskX      = np.logical_and(self.x >= (xCenter - sgHalfSize), (self.x <= xCenter + sgHalfSize))
        maskY      = np.logical_and(self.y >= (yCenter - sgHalfSize), (self.y <= yCenter + sgHalfSize))
        mask       = np.logical_and(maskX, maskY)

        # Return the sampled and masked field
        return fieldSample * mask.astype(int)