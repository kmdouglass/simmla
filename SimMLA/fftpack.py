# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Experimental Biophysics, 2016
# See the LICENSE.docx file for more details.

import numpy           as np
from scipy.fftpack     import fft, fft2, ifft
from scipy.fftpack     import fftshift, ifftshift
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

def fftSubgrid(uIn, grid, clip = True):
    '''Computes the 1D FFT of individual subgrids.
    
    fftSubgrid computes the 1D fast Fourier transform of a discretized field in
    each subgrid of a GridArray. This function models the field in the focal plane
    of a single lenslet array using scalar diffraction theory.
        
    Parameters
    ----------
    uIn  : function
        A 1D, real or complex valued function defining an input field
        distribution.
    grid : GridArray
        The grid array for sampling the field.
    clip : bool
        Should the field be clipped in size to the same extent as the initial
        lens aperture? Setting this to False will sample the transformed field
        across the entire computational grid. Setting it to True sets the field
        outside of the aperture to zero.
    
    Returns
    -------
    interpMag   : array of scipy.interpolate.RectBivariateSpline
    interpPhase : array of scipy.interpolate.RectBivariateSpline
    '''
    # Create arrays to hold the interpolations
    interpMag   = []
    interpPhase = []
    
    for subgridX in range(grid.numSubgrids):
        # Sample the field at the grid's real locations
        fieldSample = grid.rect(uIn, subgridX)
        
        # Propagate the field to the MLA
        # fftPropagate(fieldSample, grid, 200e-3)

        # Shift the sample to the center of the coordinate system
        shiftX      = int(grid.subgridCenters[subgridX])
        fieldSample = np.roll(fieldSample, -shiftX)

        # Compute the Fourier transform with appropriate scaling to conserve energy
        scalingFactor = (grid.physicalSize / (grid.gridSize - 1)) \
                      / np.sqrt(grid.wavelength * grid.focalLength)
        F             = scalingFactor * fftshift(fft(ifftshift(fieldSample)))
        
        # Set the field to zero outside of the extent of a single subgrid
        F[np.logical_or(grid.x < -np.floor(grid.subgridSize / 2), grid.x > np.floor(grid.subgridSize / 2))] = 0

        # Shift the grid coordinates back to the original location
        newGridX = grid.pX + (shiftX * grid.physicalSize / grid.gridSize)

        # Find the transform's magnitude and phase for interpolation
        mag   = np.abs(F)
        phase = np.angle(F)
        
        # Interpolate the transform
        # kind = 'linear' SHOULD NOT BE USED. This is because it will introduce
        # artifacts when the phase jumps from zero to +/- pi by interpolating
        # between the jumps.
        interpMag.append(interp1d(newGridX,
                                  mag,
                                  kind         = 'nearest',
                                  bounds_error = False,
                                  fill_value   = 0.0))
        interpPhase.append(interp1d(newGridX,
                                    phase,
                                    kind         = 'nearest',
                                    bounds_error = False,
                                    fill_value   = 0.0))           
            
    return interpMag, interpPhase

def fft2Subgrid(uIn, grid):
    '''Computes the 2D FFT of individual subgrids.
    
    fft2Subgrid computes the 2D fast Fourier transform of a discretized field in
    each subgrid of a GridArray. This function models the field in the focal plane
    of a single lenslet array using scalar diffraction theory.
        
    Parameters
    ----------
    uIn  : function
        A 2D, real or complex valued function defining an input field
        distribution.
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
            fieldSample = grid.rect2(uIn, subgridX, subgridY)

            # Shift the sample to the center of the coordinate system
            shiftX, shiftY = int(grid.subgridCenters[subgridX]), int(grid.subgridCenters[subgridY])
            fieldSample    = np.roll(np.roll(fieldSample, -shiftX, axis=1), -shiftY, axis = 0)

            # Compute the Fourier transform with appropriate scaling to conserve energy
            scalingFactor = ((grid.physicalSize / (grid.gridSize - 1)) ** 2) \
                          / (grid.wavelength * grid.focalLength)
            F             = scalingFactor * fftshift(fft2(ifftshift(fieldSample)))

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
    
def fftPropagate(field, grid, propDistance):
    '''Propagates a sampled 1D field along the optical axis.
    
    fftPropagate propagates a sampled 1D field a distance L by computing the
    field's angular spectrum, multiplying each spectral component by the
    propagation kernel exp(j * 2 * pi * fx * L / wavelength), and then
    recomibining the propagated spectral components. The angular spectrum is
    computed using a FFT.
    
    Parameters
    ----------
    field        : 1D array of complex
        The sampled field to propagate.
    grid         : Grid
        The grid on which the sampled field lies.
    propDistance : float
        The distance to propagate the field in the same physical units as the
        grid.
    
    '''
    scalingFactor = (grid.physicalSize / (grid.gridSize - 1))
    F             = scalingFactor * fftshift(fft(ifftshift(field)))
    
    # Compute the z-component of the wavevector
    # Adding 0j ensures that numpy.sqrt returns complex numbers
    kz = 2 * np.pi * np.sqrt(1 - (grid.pfX * grid.wavelength)**2 + 0j) / grid.wavelength
    
    # Propagate the field's spectral components
    Fprop = F * np.exp(1j * kz * propDistance)
    
    # Recombine the spectral components
    fieldProp = fftshift(ifft(ifftshift(Fprop))) / scalingFactor
    
    return fieldProp
