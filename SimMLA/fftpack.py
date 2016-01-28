from scipy.fftpack     import fft2
from scipy.fftpack     import fftshift
from scipy.interpolate import RectBivariateSpline

def fftSubgrid(uIn, grid):
    '''Computes the 2D FFT of individual subgrids.
    
    fftSubgrid computes the 2D fast Fourier transform of a discretized field in
    each subgrid of a GridArray. This function models the field in the focal plane
    of a single lenslet array using scalar diffraction theory.
        
    Parameters
    ----------
    uIn  : function
        A 2D, real or complex valued function defining an input field distribution.
    grid : GridArray
        The grid array for sampling the field.
    
    Returns
    -------
    interpMag   : array of scipy.interpolate.RectBivariateSpline
    interpPhase : array of scipy.interpolate.RectBivariateSpline
    '''
    # Create arrays to hold the interpolations
    interpMag   = []
    interpPhase = []
    
    for subgridX in range(grid.numSubgrids):
        for subgridY in range(grid.numSubgrids):
    
            # Sample the field at the grid's real locations
            fieldSample = grid.rect(uIn, subgridX, subgridY)

            # Shift the sample to the center of the coordinate system
            shiftX, shiftY = int(grid.subgridCenters[subgridX]), int(grid.subgridCenters[subgridY])
            fieldSample    = np.roll(np.roll(fieldSample, -shiftX, axis=1), -shiftY, axis = 0)

            # Compute the Fourier transform with appropriate scaling to conserve energy
            scalingFactor = ((grid.physicalSize / (grid.gridSize - 1)) ** 2) / (grid.wavelength * grid.focalLength)
            F             = scalingFactor * fftshift(fft2(fieldSample))

            # Shift the grid coordinates back to the original location
            newGridX = grid.pX + (shiftX * grid.physicalSize / grid.gridSize)
            newGridY = grid.pY + (shiftY * grid.physicalSize / grid.gridSize)

            # Find the transform's magnitude and phase for interpolation
            mag   = np.abs(F)
            phase = np.angle(F)

            # Reduce the grid to 1D arrays to use more efficient RectBivariateSpline
            xx = np.sort(np.unique(newGridX))
            yy = np.sort(np.unique(newGridY))

            # Interpolate the transform
            # NOTE: RectBivariateSpline associates X with rows and Y with columns!
            interpMag.append(RectBivariateSpline(yy, xx, mag))
            interpPhase.append(RectBivariateSpline(yy, xx, phase))
    
    return interpMag, interpPhase