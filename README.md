# SimMLA

Fourier optics simulation code for super-resolution microscopes
utilizing dual microlens arrays.

# License Information

© All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
Switzerland, Laboratory of Experimental Biophysics, 2016-2017
See the LICENSE.docx file for more details.

# Citing this Work

If you use SimMLA in your research, please cite the following paper in
which this work was used:

> K. M. Douglass, C. Sieben, A. Archetti, A. Lambert, and S. Manley, " Super-resolution imaging of multiple cells by optimized flat-field epi-illumination," Nature Photonics 10, 705-708, doi:10.1038/nphoton.2016.200 (2016)

<a href="http://www.nature.com/nphoton/journal/v10/n11/full/nphoton.2016.200.html">http://www.nature.com/nphoton/journal/v10/n11/full/nphoton.2016.200.html</a>

## Author

- Kyle M. Douglass, kyle.m.douglass at gmail.com

# Installation

SimMLA uses Python 3.5 and a few scientific libraries associated with
it. The easiest way to install these libraries is through the
[Anaconda package manager](https://www.continuum.io/downloads).

After installing Anaconda, update the package manager in either the
conda prompt or terminal with the command

`conda update conda`.

Once conda is updated, use the prompt/terminal to navigate to the
folder containing the SimMLA directory. Enter the command

`pip install -e SimMLA`

to install SimMLA in development mode.

In case there are dependency issues, you can try installing a conda
environment known to work with SimMLA. To do this, navigate to the
SimMLA parent directory and run the command

`conda env create -f environment.yml`

# Directions

SimMLA contains three modules that may be used in any Python 3.5
library:

1. **fftpack** - Convenience routines for fast Fourier transforms
2. **fields** - Used to generate coherent and partially coherent beams
3. **grids**  - Discrete grids for sampling fields

Examples of how to use the code may be found in the `tests` directory.
Jupyter notebooks for generating the data in the publication's figures
are in the `publication_data` directory.
