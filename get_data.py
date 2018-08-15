"""
Function to collect data for gp.py
returns dictionary with key as tuple of eps24 and radius, and value as conductivity
"""
import os

def main():
    res = {}

    def readfile(filename):
        """
        extracts parameter information from conductivity.data file
        filename:str
            path to conductivity.data file
        """
        with open(str(filename), 'r') as fin:
            for _ in range(4): #skip some lines
                fin.readline()
            eps24 = float(fin.readline())
            eps34 = float(fin.readline())
            for _ in range(6):
                fin.readline()
            radius = float(fin.readline())
            conductivity = float(fin.readline())
            res[(eps24, radius, eps34)] = conductivity

    home_dir = "/Users/Arthur/Desktop/MIT/"

    
    for item in os.listdir(home_dir):
        if item == 'Param10_II' or item == 'eps24_II' or item == 'eps34': ### Three experiment directories to investigate
            new_dir = os.path.join(home_dir, item)
            
            for folder in os.listdir(new_dir):
                
                if folder == 'Results': ### Results from Simulations
                    results = os.path.join(new_dir,folder)
                    for folder in os.listdir(results):
                        if folder.startswith('run'): ###Individual sub directory for simulation
                            dest_file = results + '/' + folder + '/conductivity.data'
                            filename = os.path.join(new_dir,dest_file)
                           
                            readfile(filename)
    return res