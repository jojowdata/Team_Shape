# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:14:18 2020

@author: j.white
This code imports the data from cvs files (player location data), it then 
formats and processes the data to remove columns not required and 
duplicate information.

"""
# Import libraries
import numpy as np
import pandas as pd
import pyproj as proj # For gps to cartisean conversion
import matplotlib.pyplot as plt # for plotting
from datetime import timedelta
from pyproj import Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import TransverseMercatorConversion

# Constants for time format
LONG_TIME = "%H:%M:%S.%f"
SHORT_TIME = "%M:%S.%f"
LONG_TIME_LENGTH = 10
SHORT_TIME_LENGTH = 7
PITCH_LENGTH =106
PITCH_WIDTH  =68


# Function to process and clean the data from the player file
# Import the csv files one at a time and process before adding it to the 
# dataframe 

def load_merge_data(filenames):
    
    # Create an empty dataframe to hold the merged data
    df_master = None
    
    # Create a loop over the list of file names with an index
    for (i, filename) in enumerate(filenames):            
        # For each file load it
        df = pd.read_csv(filename)   
        # Strip column names of whitespace
        df = df.rename(columns=lambda col_name: col_name.strip())
    
        # Keep necessary columns only
        df = df[["Player Display Name", "Time","Lat","Lon"]]
        # Drop duplicate rows
        df = df.drop_duplicates()
        
        
        # Parsing the time column
        if len(df["Time"][0]) == LONG_TIME_LENGTH:
            # Time is longtime format
            df["Time"] = pd.to_datetime(df["Time"],format = LONG_TIME)
         
        else:
            # Parse as a short time datetime
            df["Time"] = pd.to_datetime(df["Time"],format = SHORT_TIME)
            # Format to include hours
            delta_time = df["Time"] - df["Time"].shift(1)
            
            # Check whether delta time is negative (returns boolean series) 
            neg_time = delta_time < timedelta()
            # Calculate the hour value for each row
            hours = neg_time.cumsum()
            hours  = pd.to_timedelta(hours,"hour")
            df["Time"] = df["Time"] + hours
            
    
        # Make player indexes for the column names except "Time" e.g Player_Display_Name_0
        df = df.rename(columns=lambda col_name: f"{col_name}_{i}" if col_name!= "Time" else col_name)
        df.to_csv('Testdata12')
        
        # Merge with master Dataframe by timestamp
        if df_master is None:
            df_master = df
        else:
            # Merge the existing master df with the new cleaned frame with index as a suffix
            df_master = pd.merge(df_master, df, on ="Time",how='outer')
            df_master = df_master.drop_duplicates('Time')
    df_master.to_csv('testmaster.csv')        
   
    # Return Master Dataframe
    return df_master
  
# Function to convert locations from GPS co-ordinates to Cartisean x,y
def process_location(df):

    # Create an array for the pitch co-ordinates (in Iraq)
    
    corner_points = np.array([
        [35.2642021,44.99991],    # Top right
        [35.26443,45.0004044],    # Bottom right
        [35.264576,45.0000877],   # Bottom left
        [35.2643471,44.99959]     # Top left
        ])

    lat_corner = corner_points[:,0]
    lon_corner = corner_points[:,1]
    
    plt.scatter(lat_corner,lon_corner)
    plt.title('Corners Lat Long')
    plt.show()
    
    # Process pitch corners using pyproj, create an instance of proj, 
    # proj_crs is used to convert to cartisean    
    
    # Transform the pitch co-ordinates to cartesian  
    src_proj = proj.Proj('epsg:4326') # WGS84
    
    dst_proj = ProjectedCRS(
        conversion=TransverseMercatorConversion(
            latitude_natural_origin= lat_corner[0],
            longitude_natural_origin = lon_corner[0]
        )
    )
    
    transformer = Transformer.from_proj(src_proj,dst_proj)
    # Transform the corner points from GPS to x,y                                                                                      
    xx_corner, yy_corner = transformer.transform(lat_corner, lon_corner)    
    
    # PLot the x, y co-ordinates
    plt.scatter(xx_corner,yy_corner)
    plt.scatter(xx_corner[0],yy_corner[0],c='r') #bottom left
    plt.scatter(xx_corner[1],yy_corner[1],c='g') #bottom left
    plt.title('Corners metres')
    plt.show()
    # Create array
    corners_projected = np.array([
        [xx_corner[0],yy_corner[0]],  # Top right
        [xx_corner[1],yy_corner[1]],  # Bottom right
        [xx_corner[2],yy_corner[2]],  # Bottom Left
        [xx_corner[3],yy_corner[3]],  # Top left
        ])
    # Create transformation matrix      
    rectification_matrix = np.linalg.inv([
        [corners_projected[1][0], corners_projected[3][0]],
        [corners_projected[1][1], corners_projected[3][1]]
        ])
           
        
    #  Put the points into single array and apply transfomration matrix
    #   and then split again
    x,y = np.dot(rectification_matrix, np.vstack((xx_corner,yy_corner)))
    
    plt.scatter(x,y)
    plt.scatter(x[0],y[0],c='r') #bottom left
    plt.scatter(x[1],y[1], c='g') #top left
    plt.title('Corners rectified x,y in metres')
    plt.show()

    # Main x,y co-ordinates
    # Pull out the Latitude points into a numpy array
    latitudes = df[[col for col in df if col.startswith("Lat")]].values
    longitudes = df[[col for col in df if col.startswith("Lon")]].values
    
    # Get the number of players 
    number_players = latitudes.shape[1]
    
    latitudes = latitudes.flatten()
    longitudes = longitudes.flatten()
    
    # Select the data that is not NaN
    indices = np.argwhere(~np.isnan(latitudes))
    indices = np.argwhere(~np.isnan(longitudes))
    
    # Remove single dimensional entries from the array    
    not_nan_lat = latitudes[indices].squeeze()
    not_nan_lon = longitudes[indices].squeeze()

    # Grab the number of players in the match
    print(number_players)
    #print(not_nan_lat.shape)
    #print(not_nan_lon.shape)

    # Transform the points from GPS to x,y                                                                                      
    xx, yy = transformer.transform(not_nan_lat, not_nan_lon)    
    # PLot the x, y co-ordinates
    plt.scatter(xx,yy)
    plt.title('all points projected')
    plt.show()

    # Change rotation using corner point transformation matrix
    x,y = np.dot(rectification_matrix, np.vstack((xx,yy)))
    
    plt.scatter(x,y)
    plt.title('all points rectified')
    plt.show()
    # Plot out the cartisean vs pca points to see if major difference
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Comparing points with projection and with rectified points')
    ax1.scatter(xx, yy)
    ax2.scatter(x, y)
    
    # Re scale for pitch size
    x = (x-0.5)*PITCH_LENGTH
    y = (y-0.5)*PITCH_WIDTH
    
    plt.scatter(x,y)
    plt.title('all points rectified')
    plt.show()
    
    # Reinstate the data in the array 
    # Merge x,y's back into dataframe
    # Need to reshape the array using 1 ensures error kicked out if reminder
    x = np.expand_dims(x,axis=1)
    y = np.expand_dims(y, axis=1) 
    
    # Reinsert processed data with our NaN'so now we have 2 arrays with 
    # new data x and y 
    latitudes [indices] = x
    longitudes [indices] = y
       
    x = np.reshape(latitudes,(-1,number_players))
    # Change back to a dataframe
    df_x = pd.DataFrame(x)
    # rename cols in the df_x
    df_x.columns = [f'X_{i}' for i in range(number_players)]
   
    y = np.reshape(longitudes,(-1,number_players))
    
    df_y = pd.DataFrame(y)
    df_y.columns = [f'Y_{i}' for i in range(number_players)]
    
    # Take the original dataframe passed in and remove columns with
    # the Lat and long in them and replace with the updated cartisean points
    df.drop(df.columns[df.columns.str.contains('Lat')], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('Lon')], axis=1, inplace=True)
    df = df.reset_index(drop=True)
    print('Print head of df') 
   
    # Add om the x,y, columns to the dataframe
    df = pd.concat([df,df_x,df_y], axis=1)    

    # Merge points back into dataframe 
    df.to_csv('Testdata14')
                                                                                                                            
    return (df,number_players)
     





