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


# Usage example
dir_path = "/home/AD.NORCERESEARCH.NO/stia/dev/git/resordan/tests/data"  # directory path
sd_objects = read_data_per_pass(dir_path)
