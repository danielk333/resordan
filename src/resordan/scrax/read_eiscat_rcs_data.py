#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:17:25 2024

@author: stia
"""

import os
import numpy as np

# Read list of data dictionaries per pass
def read_data_per_pass(dir_path):

    # Check if the given path is a valid directory
    if not os.path.isdir(dir_path):
        print("Invalid directory path.")
        return
    
    # Define list of data dictionaries per pass
    pass_list = [] # empty list

    # Recursively walk through directory and subdirectories
    for root, _, files in os.walk(dir_path):
        for filename in files:
            # Check if the file has a .pickle extension
            if filename.endswith('.pickle'):
                file_path = os.path.join(root, filename)
                try:
                    # Open the pickle file
                    with open(file_path, 'rb') as file:
                        # Read pass data (as dictionary)
                        data = np.load(file, allow_pickle=True)
                        # Append to list of dictionaries
                        pass_list.append(data)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
    
    return pass_list

# Read list of RCS data arrays per pass
def read_rcs_per_pass(dir_path):
    
    # Check if the given path is a valid directory
    if not os.path.isdir(dir_path):
        print("Invalid directory path.")
        return

    # Read list of data dictionaries per pass
    pass_list = read_data_per_pass(dir_path)

    # Defined list of RCS data arrays per pass    
    rcs_data = [] # empty list

    # Extract RCS data array
    for this_pass in pass_list:
        # Append to list
        rcs_data.append(this_pass["rcs_data"])
    
    return rcs_data

# Read list of maximum dimensions per pass
def read_maxdim_per_pass(dir_path):
    
    # Check if the given path is a valid directory
    if not os.path.isdir(dir_path):
        print("Invalid directory path.")
        return

    # Read list of passes
    pass_list = read_data_per_pass(dir_path)
    
    # Define list of maximum dimensions per pass
    max_dim = [] # empty list

    # Extract maximum dimension value
    for this_pass in pass_list:
        # Append to list
        max_dim.append(this_pass["xSectMax"])
    
    return max_dim

# Read list of minimum dimensions per pass
def read_mindim_per_pass(dir_path):
    
    # Check if the given path is a valid directory
    if not os.path.isdir(dir_path):
        print("Invalid directory path.")
        return

    # Read list of passes
    pass_list = read_data_per_pass(dir_path)

    # Define list of minimum dimensions
    min_dim = [] # empty list

    # Extract minimum dimension value
    for this_pass in pass_list:
        # Append to list
        min_dim.append(this_pass["xSectMin"])
    
    return min_dim

# Read list of average dimensions per pass
def read_avgdim_per_pass(dir_path):
    
    # Check if the given path is a valid directory
    if not os.path.isdir(dir_path):
        print("Invalid directory path.")
        return

    # Read list of passes
    pass_list = read_data_per_pass(dir_path)
    
    # Define list of average dimensions
    avg_dim = [] # empty list
    
    # Extract average dimension value
    for this_pass in pass_list:
        # Append to list
        avg_dim.append(this_pass["xSectAvg"])
    
    return avg_dim

def read_satno_per_pass(dir_path):
    
    # Check if the given path is a valid directory
    if not os.path.isdir(dir_path):
        print("Invalid directory path.")
        return

    # Read list of passes
    pass_list = read_data_per_pass(dir_path)
    
    # Define list of satellite ID numbers
    satno = [] # empty list
    
    # Extract satellite ID number
    for this_pass in pass_list:
        # Append to list
        satno.append(this_pass["satno"])
    
    return satno

# Usage example
dir_path = "/home/AD.NORCERESEARCH.NO/stia/dev/git/resordan/tests/data"  # directory path
sd_objects = read_data_per_pass(dir_path)
rcs_data = read_rcs_per_pass(dir_path)
max_dim = read_maxdim_per_pass(dir_path)
