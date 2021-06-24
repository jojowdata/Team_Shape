# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:40:10 2020

@author: j.white

Anonymise the player name data within each file and filename
"""
import numpy as np
import pandas as pd


def load_lookup_table(lookup_table):
    # Look up the number for the player and replace all mention of it in file
    df_lookup = pd.read_excel(lookup_table)
    
    # Create an empty dictionarty for the look up table
    name_map = {}
    
    # Fill the dictionary by looping over each row and give a name a number
    for row in df_lookup.iloc:
        # formatting the name
        # Assign the name to the dictionary
        name_map[process_name(row["TeamA_Player"])] = row["TeamA_ID"]
        name_map[process_name(row["TeamB_Player"])] = row["TeamB_ID"]
    # return the filled dictionary
    return name_map

def process_name(name):   
    # Check to see name is in format L.Surname
    name_pieces = name.lower().split()
    if len(name_pieces)!=2:
        return name
    # returns first letter of first name and all of surname
    return f"{name_pieces[0][0]}. {name_pieces[1]}"

def load_files(filenames,name_map):
    print(filenames)
    # Create a loop over the list of file names with an index
    for filename in filenames:          
        # For each file load it
        df = pd.read_csv(filename)   
        # Get player name
        player_name = df["Player Display Name"][0].lower()
        new_name = name_map[player_name]
        
        # Assign the new name back to the file player display 
        df["Player Display Name"] = new_name
        
        
        # Create the file name to be new name
        new_filename = f"Team1_Anon/20190910.MD/20190910_1st_Half/{new_name}_1st_Half.csv" 
        df.to_csv(new_filename,index=False)
        
    
    