#--- Step One - Generate the 1D Spectra
python ./Espectra_Lou.py hrrr w &
python ./Espectra_Lou.py fv3 w &
python ./Espectra_Lou.py GFDL w &
python ./Espectra_Lou.py gsl w &
python ./Espectra_Lou.py NAM w &
wait


#--- Step Two - Plot the Spectra
python plot1D_spectra.py w ./
