UST Yields Project - Test Problem for Research Assistant Candidate
==================

# About this test problem

We are looking for candidates with strong programming skills and a basic understanding of financial concepts. In this test, you will be required to modify the test_prob.py file, generate reports, and visualize the forward curve results.


# Quick Start

To quickest way to run code in this repo is to use the following steps. 

## Step 1: Set up your Python environment
Open a terminal and navigate to the root directory of the project and create a conda environment using the following command:
```
conda create -n blank python=3.12
conda activate blank
```
and then install the dependencies with pip:
```
pip install -r requirements.txt
```

## Step 2: Understand the current code
This test provides a portion of our project's code, which compares two types of forward curves: pwtf and pwcf. The dataset contains curves from the year 2000. The code loops through each month's unique quote date, plotting both the pwtf and pwcf curves on a single graph for comparison.

Specifically, the test_prob.py script uses the dataset output/test2000YTW_forb/test2000YTW_forb_curve.pkl and calls functions from DateFunctions_1.py and discfact.py to generate a series of images, which will be saved in the output/test2000YTW_forb/fwd_rates_pwcf_pwtf folder.

Tips: You can open test_prob.py and run each cell to generate images, which will be saved to the fwd_rates_pwcf_pwtf.


# Task for the Test Problem
Your task consists of three main objectives:

1. Merge the DataFrames
Merge the two DataFrames containing Python and Fortran forward curves with a new index specifying whether the curve is Python or Fortran.
The data files to merge are located at:
output/fortran15/pycurve198606_present_curve.pkl
output/test2000YTW_forb/test2000YTW_forb_curve.pkl
You will need to create new csv and pkl files with a new index.

2. Modify and Create New Functions
Modify (and create a new version of) the plot_fwdrate_wrapper and find_max_min_fwd_rate functions so that both the Fortran and Python curves are plotted on the same graph.

3. Save New Graphs
Store the new graphs to a new directory under the output folder to avoid overwriting the original images.


# Folder Structure and Explanation

1. output
This folder contains the datasets. One subfolder contains the Fortran data, and the other contains the Python data. Please use the .pkl files for programming. Although .csv files are provided for your reference, they are not suitable for the task as they cannot store the full data structure.
As part of the task, you will create a new folder under output to store the new graphs.

2. DateFunctions_1.py & discfact.py
These files contain functions that are necessary for the code to run correctly. No modifications are needed, as the functions are already correct.

3. test_prob.py
This is the only file you need to modify.
Currently, it compares two curve types (pwtf and pwcf). Your task is to adjust it to compare Python and Fortran curves.

4. requirements.txt
This file lists the required packages. Follow the steps in Step 2 to install them all at once.