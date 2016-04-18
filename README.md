# SimMLA

Fourier optics simulation code for super-resolution microscopes
utilizing microlens arrays.

# License Information

© All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
Switzerland, Laboratory of Experimental Biophysics, 2016
See the LICENSE.docx file for more details.

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

# Directions

SimMLA contains three modules that may be used in any Python 3.5
library:

1. **fftpack** - Convenience routines for fast Fourier transforms
2. **fields** - Used to generate coherent and partially coherent beams
3. **grids**  - Discrete grids for sampling fields

Examples of how to use the code may be found in the `tests` directory.
Jupyter notebooks for generating the data in the publication's figures
are in the `publication_data` directory.
