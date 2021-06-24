# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:27:22 2020

@author: j.white

Code to Anomise player / Match Data create a lookup table of anon names and replaces player 
name with number e.g. "A.Bloggs" replaced with "Player_6"
The filename is also anonymised

"""
import glob2
import anonymise_data

# Have the main calls to the functions that will process the data
if __name__ == "__main__":
    # Get filenames of the player data
    filenames = glob2.glob("Team_1A/20190910.MD/20190910_1st_Half/*.csv")
    print(filenames)
    
    playernames = "Team_1A/20190910.MD/20190910_Player_IDs.xlsx"
    

    name_map = anonymise_data.load_lookup_table(playernames)
    anonymise_data.load_files(filenames,name_map)
    