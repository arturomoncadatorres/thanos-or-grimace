# -*- coding: utf-8 -*-
"""
purplefunctions.py
Auxiliary functions of the thanos-or-grimace project.

Created on Wed Oct 30 22:57:55 2019
@author: Arturo Moncada-Torres
arturomoncadatorres@gmail.com
"""


#%% Preliminaries
import os
import random
import numpy as np
import pathlib
import shutil


#%%
def structure_images(path_data, val_prop=0.2, verbose=True):
        
    # Make sure path_data is a pathlib.Path.
    if isinstance(path_data, str):
        path_data = pathlib.Path(path_data)
    
    # Create paths for images.
    path_source = path_data/'images'
    path_training = path_data/'training'
    path_validation = path_data/'validation'
    
    # Get classes of interest.
    classes = list()
    [classes.append(str(x).split(os.sep)[-1]) for x in path_source.iterdir() if x.is_dir()]
    
    # Create directories.
    for path in [path_training, path_validation]:
        for class_ in classes:
            
            # If the directory exists, delete it and its contents
            # (each time we run this we want a new set of images)
            if (path/class_).exists():
                shutil.rmtree(path/class_)    
                if verbose:
                    print("Deleted directory (and its contents) " + str(path/class_))
                    
            # Now, create new directory to be populated by training/validation files.
            (path/class_).mkdir(parents=True)
            if verbose:
                print("Created directory " + str(path/class_))
                
                
    # Get files for each class.
    classes_files = dict.fromkeys(classes)
    for class_ in classes:
        class_files = list()
        for item in (path_source/class_).glob('*'):
            class_files.append(item)
        classes_files[class_] = class_files


    # Class balancing.
    # We want classes to be balanced. Thus we will undersample the
    # overrepresented class to match its number.

    # Count number of samples in each class.
    classes_samples = dict.fromkeys(classes)
    for class_ in classes:
        classes_samples[class_] = len(classes_files[class_])
        
    # Find minimum number of instances.
    min_instances = classes_samples[min(classes_samples, key=classes_samples.get)]

    # Undersampling.
    # Notice that random.sample samples without replacement already.
    for class_ in classes:
        classes_files[class_] = random.sample(classes_files[class_], min_instances)
    

    # Split data (i.e., files) into training and validation sets.
    # Notice that every time we run this code, we will get different set of images.
    for class_ in classes:
        random.shuffle(classes_files[class_])
        train_files = classes_files[class_][int(np.floor(len(classes_files[class_])*val_prop)):]
        val_files = classes_files[class_][0:int(np.floor(len(classes_files[class_])*val_prop))]
        
        for index, file in enumerate(train_files):
            if file.exists():
                path_destination_full = str(path_training/class_/(str(index).zfill(3) + '.png'))
                shutil.copyfile(file, path_destination_full)
                if verbose:
                    print("Copied " + str(file) + " to " + str(path_destination_full))

        for index, file in enumerate(val_files):
            if file.exists():
                path_destination_full = str(path_validation/class_/(str(index).zfill(3) + '.png'))
                shutil.copyfile(file, path_destination_full)
                if verbose:
                    print("Copied " + str(file) + " to " + str(path_destination_full))   
                    
    return None