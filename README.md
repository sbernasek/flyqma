FlyEye Clones
=============

**FlyEye Clones** is part of the **NU FlyEye** platform for studying gene expression in the developing *Drosophila* eye. The package is focused on comparing expression between distinct clonal subpopulations within the eye epithelium.

Expression patterns are typically identified by comparing the fluorescence of target gene reporters between groups of cells. Clones helps quantify these differences in reporter expression. Quantification entails measuring fluorescence intensities by analyzing experimentally collected images of fixed eye discs. Such measurements may then used to identify and compare distinct subpopulations within the eye field.

The initial release is primarily limited to basic tools required to replicate [our study](https://github.com/sebastianbernasek/pnt_yan_ratio) of Pnt and Yan expression during retinal patterning in *Drosophila*. Given confocal microscopy data, the clones package facilitates:

  - **Segmentation.** Detect cell nuclei within a microscope image

  - **Measurement.** Quantify reporter expression levels

  - **Bleedthrough Control.** Correct for fluorescence bleedthrough

  - **Annotation.** Identify distinct subpopulations of cells

  - **Cell Selection.** Manually select cells for comparison

  - **Comparison.** Compare expression levels between subpopulations


Please refer to the [documentation](https://sebastianbernasek.github.io/clones/index.html#) page for tips on getting started with analyzing your data.


Dependencies
============

 - Python 3.6+

Required:

 - [PyTables](https://www.pytables.org/usersguide/installation.html)
 - [Scipy](https://www.scipy.org/)
 - [Pandas](https://pandas.pydata.org/)
 - [PIL](https://pillow.readthedocs.io/en/5.2.x/)
 - [TiffFile](https://pypi.org/project/tifffile/)
 - [Matplotlib](https://matplotlib.org/)
 - [Seaborn](https://seaborn.pydata.org/)
 - [Scikit-Image](https://scikit-image.org/)
 - [Scikit-Learn](http://scikit-learn.org/stable/)
 - [Statsmodels](https://www.statsmodels.org/stable/index.html)
 - [NetworkX](https://networkx.github.io/)
 - [Infomap](https://mapequation.github.io/infomap/)


Installation
============

After downloading the [latest distribution](https://github.com/sebastianbernasek/clones/archive/v0.1-beta.tar.gz), the simplest method is to install via ``pip``:

    pip install clones-0.1.0-beta.tar.gz


Examples
========

For an example of a complete project utilizing the entire **NU FlyEye** platform, please refer to the [code](https://github.com/sebastianbernasek/pnt_yan_ratio) used to generate the figures in our manuscript.


Authors
=======

[Sebastian Bernasek](https://github.com/sebastianbernasek)
