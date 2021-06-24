# -*- coding: utf-8 -*-
"""
Created on Tue Sep # 1 16:08:49 2020

@author: j.white

This program has been written in fulflment of Higher Diploma in Data Analytics 
in with conjunction with STATSports by Joanne White. 

This code calls functions that import and wrangle data (csv files).  From this 
they are manipluated using various libaries to show the position of players
on the pitch but also the team shape using convex hull theory.
"""

# Import libaries 
import glob2
import dataprocessing
import formation
import unique_cluster
 
# Constants
TIME_INTERVAL = .2 #Seconds therefore skip x rows, to use all dataset set to 1 / DATA_CAPTURE_RATE
DATA_CAPTURE_RATE = 5 # Samples per second recorded by Sonar device

# Have the main calls to the functions that will process the data
if __name__ == "__main__":
    
    # Get filenames of the player data
    filenames = glob2.glob("Team1_Anon/20190910.MD/20190910_1st_Half/TeamB/*.csv")
    team = 'Away Team'
    #direction of play right to left 0 = no, 1=yes
    right_to_left = 0
    # Load files
    df_processed = dataprocessing.load_merge_data(filenames)
    # Only keep rows equal to the time interval
    skip_rows = int(TIME_INTERVAL*DATA_CAPTURE_RATE)
    # select data by rows
    df_processed = df_processed.iloc[::skip_rows, :]
    # Convert Lat and long (GPS) to Cartisean x.y
    df_main, number_players= dataprocessing.process_location(df_processed) 
    # Create convex hull and visualizations
    ####################convex_hull.create_convex_hull(df_main,number_players)
    
    # Load_tagged_events
    event_filenames = glob2.glob("Team_1A/20190910.MD/20190910_1st_Half_Events/TeamB/*.csv")
    
    team_arr = []
    offense_defense = []
    # Create a loop over the list of file names with an index (offensive and defensive files)
    for (i, filename) in enumerate(event_filenames):
        # Load in the event data
        df_events = formation.load_tagged_events(filename)
        #print(df_events.head)
    
        # Process tagged team possession and reset time
        df_events,possession_type = formation.process_tagged_events(df_main,number_players,df_events,team)   
        
        # Take tagged time stamps and make into 2 min blobs
        df_2min_events = formation.create_event_segment(df_events)   
        
        # Pull all timestamps for each segment together and add segment number
        df_segment_data = formation.process_segment_id(df_2min_events, df_main)
        
        list_seg = df_segment_data["Segment_ID"].unique()

        # Build array to hold if formations offensive or defensive
        offense_defense.extend([i]*len(list_seg))
        print(offense_defense)
        
        # Process each segment - calculate the centre of mass and the team formation and return 
        # mean postion for player in each segment and the covariance for the players positions 
        # for all segments
        overall_formation_arr, player_covariance_matrices = formation.segment_pre_process(df_segment_data,number_players,possession_type,team,right_to_left)
        
        #gather the formations into an array
        team_arr.append((overall_formation_arr, player_covariance_matrices)) 
        
    # Calculates average formation for offense and defense and puts on the same page in pdf
    formation.plot_average_shape(team_arr,right_to_left,team)
    
    # Calculate the formation clustering
    unique_cluster.cluster_formation(team_arr,offense_defense,right_to_left,team)
    
    #fig,ax = formation.plot_ave_formation(fig,ax,area)
    print("the end")
        
