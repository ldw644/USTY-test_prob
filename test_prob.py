##########################################
# This script is the wrapper program for #
# running parzeroBond.py - displaying    #
# forward rates, par bond, zero bond,    #
# annuity curves, and monthly returns    #
# with plots/animations and tables.      #
# It reads data from stored .csv or      #
# pickle files and generates various  #
# visualizations and reports.            #
##########################################

'''
Overview
-------------
This script loads fitted forward rate data from .csv or pickle files,
generates par bond, zero bond, annuity, and monthly returns tables, exports reports,
and creates various plots and animations by running parzeroBond.py.

Requirements
-------------
- Forward rate data stored in .csv or pickle files.
- Various input parameters for generating plots and tables.

Functionality
-------------
1. Reads fitted forward rate data from stored .csv or pickle files.
2. Generates Par Bond/Zero Bond/Annuity tables.
3. Generates monthly returns tables: total return, yield return, yield excess return.
4. Produces visual reports in text and PDF formats with the above tables.
5. Generates plots
    - Forward rates for all curve types.
    - Par Bond/Zero Bond/Annuity curve, and with actual yields as scatters.
    - Python vs. FORTRAN curves.
6. Creates animations from the plots.

Usage
-------------
1. Define the input parameters, including file paths and options for plots and tables.
2. Run the script to generate the desired visualizations and reports.
'''

#%% Import python packages
# magic %reset resets by erasing variables, etc. 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default='svg'
pio.renderers.default='browser'
import importlib as imp
import time as time



#%% Import py files

OUTPUT_DIR = 'output'

# Import the modules
import DateFunctions_1 as dates
import discfact as df

imp.reload(dates)
imp.reload(df)



#%%  Current Graphing function for comparing two curve types

#############################################
### Your task:
### Modify (and create new) functions to compare py & fort curves
#############################################

def plot_fwdrate_wrapper(output_dir, estfile, curve_df, curvetypes, plot_points_yr, taxflag=False, sqrtscale=False, yield_to_worst=True, 
                         start_date=None, end_date=None,pltshow=False):
        """Plot and export to png in a created folder multiple forward curve plots by curvetype provided."""

        # curve_df = pd.read_pickle(output_dir+'/'+estfile+'_curve.pkl')
        if start_date is not None and end_date is not None:
            curve_df = curve_df.loc[(curve_df.index.get_level_values('quotedate_ind') >= start_date) & 
                                    (curve_df.index.get_level_values('quotedate_ind') <= end_date)]

        curve_df = find_max_min_fwd_rate(curve_df)
        curve_points_day = plot_points_yr * 365.25

        curvetypes_str = "_".join(curvetypes)
        path = f"{output_dir}/{estfile}/fwd_rates_{curvetypes_str}"
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' created")
        except OSError as error:
            print(f"Creation of the directory {path} failed due to: {error}")

        #if not taxflag:
        if (sqrtscale):
            sqrt_plot_points_yr = np.sqrt(plot_points_yr)

        # Loop through dates to create plots
        for date in (curve_df.index.get_level_values('quotedate_ind').unique()):  # This references the 'quotedate_ind' index. Maybe there's a more elgant way?
            julian_date = dates.YMDtoJulian(date)   # get the julian 
            plot_points = julian_date + curve_points_day
            curves_all = curve_df.xs(date,level=1)
            
            # for curvetype in curves_all.index.get_level_values('type_ind'):  # loop over curve types
            for curvetype in curvetypes:
                xcurvedf = curves_all.loc[curvetype]  # all the elements of the saved curve
                curve = xcurvedf[0:4]  # select out the specific curve (this date)
                y_max = xcurvedf['max_5yr']
                y_min = xcurvedf['min_5yr']
                if not(yield_to_worst):
                    yvols = np.round(xcurvedf['yvols'],decimals=4)
                term1 = df.discFact(plot_points + 1, curve)
                term2 = df.discFact(plot_points, curve)
                result = -365 * np.log(term1 / term2)
                if (sqrtscale):
                    plt.plot(sqrt_plot_points_yr, 100*result,label=f'{curvetype} - {date}')
                else:
                    plt.plot(plot_points_yr, 100*result,label=f'{curvetype} - {date}')
            
            plt.ylim(y_min*100 - 0.8 * abs(y_min*100), y_max*100 + 0.1 * abs(y_max*100))

        # sqrt root ticks and labeling
            if (sqrtscale):
                x1 = max(plot_points_yr)
                if (x1 > 20):
                    plt.xticks(ticks=np.sqrt([0.25,1,2,5,10,20,30]).tolist(),labels=['0.25','1','2','5','10','20','30'])
                elif (x1 > 10):
                    plt.xticks(ticks=np.sqrt([0.25,1,2,5,10,20]).tolist(),labels=['0.25','1','2','5','10','20'])
                elif (x1 >5):
                    plt.xticks(ticks=np.sqrt([0.25,1,2,5,7,10]).tolist(),labels=['0.25','1','2','5','7','10'])
                else :
                    plt.xticks(ticks=np.sqrt([0.25,1,2,3,5]).tolist(),labels=['0.25','1','2','3','5'])
                plt.xlabel('Maturity (Years, SqrRt Scale)')            
            else:
                plt.xlabel('Maturity (Years)')            

            plt.title(f'Forward Rates for {date}, vol={yvols}' if not yield_to_worst else f'Forward Rates for {date}, Yield-to-Worst')

            # plt.ylim(y_min*100 - 0.1 * abs(y_min*100), y_max*100 + 0.1 * abs(y_max*100))

            plt.ylabel('Rate')
            plt.legend()
            plt.grid(True)
            full_path = f'{output_dir}/{estfile}/fwd_rates_{curvetypes_str}'
            os.makedirs(full_path, exist_ok=True)
            plt.savefig(f'{full_path}/{date}_fwd_rate_{curvetypes_str}.png')
            if pltshow:
                plt.show()
            plt.close()


# Utility function for finding max and min during 5 yr period

def find_max_min_fwd_rate(curve_df):

    curve_df = curve_df.reset_index()
    curve_df['quotedate_ind'] = curve_df['quotedate_ind'].astype(int).astype(str)
    curve_df['year'] = pd.to_datetime(curve_df['quotedate_ind'], format='%Y%m%d').dt.year
    curve_df['5_year_bin'] = (curve_df['year'] // 5) * 5

    # Extract the max and min from the rates
    curve_df['max_rate'] = curve_df['rates'].apply(lambda x: max(x))
    curve_df['min_rate'] = curve_df['rates'].apply(lambda x: min(x))
    
    # Calculate 5-year rolling max and min for each ctype group
    # curve_df['max_5yr'] = curve_df.groupby('type_ind')['max_rate'].transform(lambda x: x.rolling(window=5, min_periods=1).max())
    # curve_df['min_5yr'] = curve_df.groupby('type_ind')['min_rate'].transform(lambda x: x.rolling(window=5, min_periods=1).min())
    curve_df['max_5yr'] = curve_df.groupby(['type_ind', '5_year_bin'])['max_rate'].transform('max')
    curve_df['min_5yr'] = curve_df.groupby(['type_ind', '5_year_bin'])['min_rate'].transform('min')

    curve_df = curve_df.drop(['year'], axis=1)
    curve_df['quotedate_ind'] = curve_df['quotedate_ind'].astype(int)
    curve_df.set_index(['type_ind', 'quotedate_ind'], inplace=True, drop=True)

    return curve_df



#%% Main script - Define user inputs and create plots 
### Be consistent with the inputs in calcFwds.py when running each estimation

# Define paths
# BASE_PATH = "C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024"
# BASE_PATH = "/Users/tcoleman/tom/yields/New2024/progs"
# OUTPUT_DIR = os.path.join(BASE_PATH, 'curve_utils/output')
# OUTPUT_DIR = os.path.join(BASE_PATH, 'FORTRAN2024/results_15fwds')


### Option-related inputs - THESE SHOULD BE TAKEN FROM THE CURVE FILE
yield_to_worst = False # if True, ignore yvolsflg, yvols, opt_type; if False - w/ opt & est calls
yvolsflg = False  # if True - est yvols; if False - must give a reasonale yvols and will not estimate yvols
yvols = 0.35  # set the reasonable start for option vol, LnY, NormY, LnP should be around 0.35, 0.01, 0.06 - Too small will fail; or as the starting value for yvols estimation
opt_type="LnY"  # The value of opt_type must be LnY, NormY, SqrtY or LnP


## Other inputs
tax = False   # if False - no tax; if True - estimate all bonds with their taxability
calltype = 0  # 0 to keep all bonds, 1 for callable bonds, 2 for non-callable bonds
curvetypes =  ['pwcf','pwtf']  # ['pwcf', 'pwlz', 'pwtf'] 
wgttype = 1
lam1 = 1
lam2 = 2
sqrtscale = True
durscale = False
twostep = False
parmflag = True
padj = False
padjparm = 0
fortran = False  # True - if for displaying the FORTRAN results, else defaults to Python
pltshow = False  # Flag to control whether to show plots on interactive screen


## Breaks
base_breaks = np.array([round(7/365.25,4), round(14/365.25,4), round(35/365.25,4), 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])  # 0.0833,
base_breaks = np.array([round(7/365.25,4), round(14/365.25,4), round(21/365.25,4), round(28/365.25,4),
                        round(35/365.25,4), 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
table_breaks_yr = np.array([0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])  

## Plot points
curve_points_yr1 = np.arange(.01,4,.01)
curve_points_yr2 = np.arange(4.01,32,.01)
curve_points_yr3 = np.arange(0,32,.01)

plot_points_yr = np.arange(.01,30,.01)
plot_points_yr = np.arange(0,32,.01)  # np.arange(.01,4,.01), np.arange(4.01,32,.01)
plot_points_yr = np.round(np.arange(.01,30,.01)*365.25) / 365.25
plot_points_yr = np.concatenate((np.arange(0, 2, 1/365.25), 
                                 np.arange(2, 5, 0.01), 
                                 np.arange(5, 30, 0.02)))

## Additional report inputs
# ctype = 'pwtf'  # or False, for displaying in the report
ctype = 'pwcf'  # or False, for displaying in the report
tax = False
padjparm = None
date = 20240924  # The date when the curve estimation is run, not when the reports are made




#%%  
#############################################
### Your task:
### fix (and define) paths for module imports
#############################################

# Import fwd curve df
estfile = 'test2000YTW_forb'
df_curve = pd.read_pickle(OUTPUT_DIR+'/'+estfile+'/'+estfile+'_curve.pkl')

## Start and end date
start_date = 20000101
end_date =   20010101


curvetypes1 = ['pwcf']
curvetypes2 = ['pwtf']
curvetypes1 = ['pwcf', 'pwtf']

for curvetypes in [curvetypes1]:  # , curvetypes2, curvetypes3
    plot_fwdrate_wrapper(OUTPUT_DIR, estfile, df_curve, curvetypes, plot_points_yr, False, sqrtscale, yield_to_worst, start_date, end_date,pltshow=pltshow)
    image_folder = os.path.join(OUTPUT_DIR, estfile, f'fwd_rates_{"_".join(curvetypes)}')



#%% Read in both python curve and fortran curve

#############################################
### Your task:
### merge the 2 dataframes and create a new index
### use .pkl
#############################################

estpythfile = 'test2000YTW_forb'
estfortdir = 'fortran15'
estfortfile = 'pycurve198606_present'

df_curve_python = pd.read_pickle(OUTPUT_DIR+'/'+estpythfile+'/'+estpythfile+'_curve.pkl')
df_curve_fortran = pd.read_pickle(OUTPUT_DIR+'/'+estfortdir+'/'+estfortfile+'_curve.pkl')



#%% Now merge the two dataframes and, modify (and make a new version for) the 'plot_fwdrate_wrapper' and create the new graphs

# The gaol is 
#  1) Merge the two dataframes (python curves and fortran curves) with a new index to spcify fortran vs python curve
#  2) modify (and make a new function) for 'plot_fwdrate_wrapper' which will plot a fortran and python curve on the same graph
#  3) Write out the new graphs, but to a new directory under 'output' so that you do not overwrite the original graphs






