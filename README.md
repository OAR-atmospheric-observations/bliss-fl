# clamps_fuzzyPBLh
Multi-instrument fuzzy logic PBL height detection algorithm using CLAMPS Doppler wind lidar and thermodynamic (in this case AERI/MWR) retrieval profiles.

## Authors
Dr. Elizabeth Smith, NOAA NSSL (elizabeth.smith@noaa.gov)

Dr. Jacob Carlin, OU CIWRO / NSSL (jacob.carlin@noaa.gov; @jtcarlin)

## Background
This algorithm was developed from the basis set forth in Bonin et al. (2018):

*Doppler Lidar Observations of the Mixing Height in Indianapolis Using an Automated Composite Fuzzy Logic Approach
Journal of Atmospheric and Oceanic Technology, 35(3), 473-490. https://doi.org/10.1175/JTECH-D-17-0159.1*

While the algorithm was developed for Doppler wind lidar and AERI retrieval profiles, it could likey be adapted for other combinations of boundary layer wind and thermodynamic profile observations.

The CLAMPS platform is maintained by a collaborative team at the University of Oklahoma and the NOAA National Severe Storms Laboratory. You can learn more about CLAMPS, the instruments onboard, contacts, and more at https://apps.nssl.noaa.gov/CLAMPS/

The authors acknowledge Dr. Tim Bonin (MIT Lincoln Labs), Tyler Bell (OU CIWRO / NSSL) and Joshua Gebauer (OU CIWRO/ NSSL) for contributions and insights.

## Dependent Packages

- datetime
- matplotlib
- netCDF4
- numpy
- pandas
- scipy
- siphon -- could skip/remove if working with local files only; https://www.unidata.ucar.edu/software/siphon/; can install via conda forge
- suntime -- pip3: https://pypi.org/project/suntime/

## Output

### Optional Figures 
If `plot_me=True`, several figures are generated. If `show_me=True`, they will plot to the console, otherwise they will save to the user provided plot save location. If option `inv_check=True` an optional extra pair of plots will be generated showing sunset/rise time dependent inversion weighting function and the inversion height. 

### Optional netCDF writeout
If `write_me=True` the resulting PBL height estimates will be written out to a netcdf file and saved to the user provides save location. 

## Algorithm Flowchart
![Fuzzy_flowchart](https://github.com/eeeeelizzzzz/clamps_fuzzyPBLh/assets/47791747/f9fd32c2-5d4e-4c4f-94ea-1994c1094a86)



 
