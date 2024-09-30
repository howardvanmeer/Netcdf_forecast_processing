
""" This script, developed by Howard van Meer (howard.vanmeer@wur.nl or vanmeer.howard@inta.gob.ar), is intended to process the new CDS beta seasonal daily forecasts available at: https://cds-beta.climate.copernicus.eu/datasets/seasonal-original-single-levels?tab=overview. 
The provided NETCDF files are CF 1.7 compliant and in order to be used in CDO they require conversion and dimension transposition using this script, as they cannot be directly used in CDO in their current format as of August 2024. """

import cdsapi
import netCDF4
from netCDF4 import num2date, Dataset
import numpy as np
import xarray as xr
import os
import pandas as pd
import datetime
import matplotlib
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
import matplotlib.dates as mdates
from IPython.display import Image
# Prevent minus value error
matplotlib.rcParams['axes.unicode_minus'] = False

originating_centre = 'ECMWF'
os.chdir("C:/MyData/Forecasts/")
working_dir= 'C:/MyData/Forecasts/'
files_to_convert = 'C:/MyData/Forecasts/' + str(originating_centre.upper()+'/')
files_processed= 'C:/MyData/Forecasts/' + '/Processed/'+str(originating_centre.upper()+'/')

temp_variables =['tmax','tmin','tmean','tdew']
region_name = 'Chaco' 
#Years must coincide with the years in directory where forecasts are stored
start_year = 2021 # Download start year
end_year   = 2023 # Download end year
years      = range(start_year, end_year + 1)

#Define months to correctly calculate leadtime starting with 1 in all months
start_month = 1   # Download start month
end_month   = 12   # Download end month
months      = range(start_month, end_month + 1) 
print("Start")
start_time = datetime.datetime.now()
for filename in os.listdir(files_to_convert):
    print(filename)
    ds = xr.open_dataset(files_to_convert + filename,decode_cf=True)
    #Be sure to have at least version 3.9 of Python and latest xarray version to avoid problemas with sumation
    ds['forecast_period'] = ds['forecast_reference_time']+ds['forecast_period']
    #Rename coordinates as indicated below
    ds= ds.rename({'forecast_period': 'time','number': 'ensamble'})
    del ds['forecast_reference_time']
    ds= ds.squeeze('forecast_reference_time')
    #Revert order of dimensions time must be first
    ds= ds.transpose('time','ensamble','latitude','longitude')
    #Substract one day to set to 1st day of month and send time to index
    ds['time'] = ds.time.to_index() - pd.Timedelta(days=1)
    #Start with ensamble 1 to 51
    ds['ensamble'] = ds['ensamble']+1
    #Make data xarray with differences between next value
    dsdiff = ds.diff(dim="time", label="upper")
    #Make xarray leaving original 1st value en fill all other with differences
    dummy= xr.concat([ds.isel(time=[0]), dsdiff], dim="time")
    #Drop all variables except "tp" and "ssrd"
    dummy= dummy.drop_vars(["mx2t24",'mn2t24','t2m','d2m','u10','v10'])
    dummy= dummy.rename({'tp':'precipitation','ssrd':'rad'})
    #Merge original xarray with differences xarray 
    ds= xr.merge([ds, dummy])
    #Rename following variables
    ds= ds.rename({'mx2t24': 'tmax','mn2t24': 'tmin','t2m':'tmean','tp':'total_precipitation','d2m':'tdew','ssrd':'totalrad'})
    #Convert temp- and precipitation variables to Celsius and mm respectively
    ds[temp_variables]= ds[temp_variables]- 273.15
    print("Still going strong")
    ds['precipitation']= ds['precipitation']* 1000
    ds['total_precipitation']= ds['total_precipitation']* 1000
    #Assign units and add description to variable   
    ds['tmax'] = ds['tmax'].assign_attrs(
        units="DegC", description="Max temperature")
    ds['tmin'] = ds['tmin'].assign_attrs(
        units="DegC", description="Min temperature")
    ds['tmean'] = ds['tmean'].assign_attrs(
        units="DegC", description="Mean temperature")
    ds['total_precipitation'] = ds['total_precipitation'].assign_attrs(
        units="mm", description="Total cumulative rainfall")
    ds['precipitation'] = ds['precipitation'].assign_attrs(
        units="mm day**-1", description="Daily rainfall")
    #Calculate avg wind speed based on horizonzal and vertical wind component
    ds['wnd'] = ((ds.u10* ds.u10)+ (ds.v10* ds.v10))**0.5
    #Convert average wind speed form 10 to 2 m according to FAO Irrigation and Drainage Paper 56 (Allen et al, 1998)
    #np.log gives natural log as stated in formula given by Allen
    ds['wnd'] = (ds['wnd']*4.87) / np.log(67.8*10-5.42)
    ds['wnd'] = ds['wnd'].assign_attrs(
        units="m s**-1", long_name="Average wind speed at 2 m")
    #Calculate RH with Tdew and Tmean as described by American Society of Meteorology
    ds['rh'] = 100-((5*(ds['tmean']-ds['tdew'])))
    ds['rh'] = ds['rh'].assign_attrs(
        units="%", long_name="Relative humidity")
    del ds.attrs['GRIB_edition']
    del ds.attrs['GRIB_centre']
    del ds.attrs['GRIB_centreDescription']
    del ds.attrs['GRIB_subCentre']
    del ds.attrs['history']
    del ds.attrs['institution']
    ds.attrs['Conventions']= 'CF-1.7 (modified to be compatible with tools like CDO and NCO that work with CF-1.6)'
    ds.attrs['Institution']= 'ECMWF, Wageningen University (WUR), Instituto Nacional de Tecnología Agropecuaria (INTA)'
    ds.attrs['Region']= 'Chaco Region, Argentina AOI 24S°-31°S and 59W°-65°W'
    actual_time= datetime.datetime.now()
    ds.attrs['Processing date']= actual_time.strftime("%d/%m/%Y %H:%M:%S")
    ds.attrs['Contact']= 'howard.vanmeer@wur.nl ; vanmeer.howard@inta.gob.ar'
    ds.attrs['History']= 'Developed by Howard van Meer and Iwan Supit (WUR) based on ECMWF new format https://cds-beta.climate.copernicus.eu/datasets/seasonal-original-single-levels?tab=overview Days have been shifted one day and daily differences have been calculated for precipitation and solar radiation. Average wind speed is calculated by using horizontal and vertical wind component. Lead time was incorporated to enable the comparison of forecast performance starting from lead time 1 across all years and months'
     
    #Substract one day from valid_time 
    ds['valid_time'] = ds.valid_time - np.timedelta64(1, 'D')
    #Calculate leadtime and add to coordinates of array and set to 1 for all xarrays throughout timeperiod every file begins with leadtime=1
    for years in range(start_year, end_year+1):
        for months in range(start_month, end_month+1):
            ds['leadtime']=(ds['time'].dt.month - months + 1 + 12 * (ds['time'].dt.year - years))
    arr = np.array(ds['leadtime'])
    #Make array with occurences of every unique value
    arr = (np.unique(arr, return_inverse = True)[1])+1
    ds['leadtime']= xr.DataArray(arr, dims= ('time'))
    ds['leadtime'] = ds['leadtime'].assign_attrs(units='months',long_name="Leadtime")
    ds.set_coords(("leadtime"))
    #Convert solar radiation to MJ per day
    ds['rad']= ds['rad']/ 1000000
    ds['rad'] = ds['rad'].assign_attrs(
        units="MJ day**-1", long_name="Solar radiation")
    ds['totalrad']= ds['totalrad']/ 1000000
    ds['totalrad'] = ds['totalrad'].assign_attrs(
        units="MJ m**-2", long_name="Cumalative solar radiation")
    ds= ds.drop_vars(["u10","v10"]) 
    del dsdiff
    del dummy
    del arr
    #Exclude extension nc from filename 
    ds.to_netcdf(files_processed+filename.rsplit(".",1)[0] +'_'+'Processed'+'.nc')
    print('Successful')
no_files= (len([iq for iq in os.scandir(files_to_convert)]))
end_time = datetime.datetime.now()
elapsed_time= end_time-start_time
print(str (no_files)+ ' files have been succesfully processed in ' + str(elapsed_time))
