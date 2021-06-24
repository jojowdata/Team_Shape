# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:54:50 2020

@author: j.white

Looking at the Formations of the players using 
- Load tagged event data for the match for POC this is offensive possession by Team A vs Team B
- Process event tags, by syncing the tagged events timestamp with the gps timestamp.

find the centroid of a formation in at a timestamp (average distance to the 1st,2nd and 3rd
   nearest),  this is  a test for github 2
"""
# Import libaries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for plotting
import plot_pitch
from datetime import timedelta
from matplotlib.backends.backend_pdf import PdfPages
import cv2
# Removes warning message "rcParam `figure.max_open_warning`"
plt.rc('figure', max_open_warning = 0)


# Constants for time format
LONG_TIME = "%H:%M:%S.%f"
SHORT_TIME = "%M:%S.%f"
EVENT_TIME = "%H:%M:%S %p"
EVENT_LONG_TIME = "%H:%M:%S.%f"
DURATION_TIME = "%M:%S"
LONG_TIME_LENGTH = 10
SHORT_TIME_LENGTH = 7
EVENT_TIME_LENGTH = 10


"""
Process the tagged events and extract the player data for these
"""
# Pull in tagged team possession 
def load_tagged_events(filename):
    
    print("Print filename")
    print(filename)            
    # For event segments file, load it
    df = pd.read_csv(filename)   
    # Strip column names of whitespace
    df = df.rename(columns=lambda col_name: col_name.strip())
    
   
    return(df)
    
    
# Take tagged event time stamps & sync with match gps timestamp
# Extract the player data for these time stamps
# if length of is less than 5 seconds discard
def process_tagged_events(df_match, number_players,df_events,team_name):
    
    # Find out if offense or defensive segments and set all the data labels for this 
    if df_events['Drill Title'][1] == "Offense":
        #set all the varibles to offense
        possession = "In possession"
        
    else:
        # team is in defense
        possession = "Out of possession"
        
    
    # Loop through all columns except Drill Title
    for col in df_events.columns: 
       if col != "Drill Title" :
            # Reset timestamp for the events
            # Pull in the starttimestamp for match gps and compare to event time 
            match_starttime = df_match['Time'].iloc[0]
            #print(match_starttime)
            event_time = df_events[col].iloc[0]

            # Format the time in the event tag file, Drill STart and End Time
            if len(df_events[col][0]) == EVENT_TIME_LENGTH:
                # Time is eventtime format
                df_events[col] = pd.to_datetime(df_events[col],format = EVENT_TIME)
                df_events[col] = pd.to_datetime(df_events[col],format = EVENT_LONG_TIME)    

                delta_time = df_events[col].iloc[0] - df_match['Time'].iloc[0]
                print(delta_time)
                # change time of eventtag by hours and minutes
                df_events[col] = df_events[col] - timedelta(hours=1, minutes=0) 
                     
            else:
                # Parse as a short time datetime
                #df_events[col] = pd.to_datetime(df_events[col],format = SHORT_TIME)
                # Format to include hours
                #delta_time = df_events[col] - df_events[col].shift(1)
                    
                # Check whether delta time is negative (returns boolean series)
                #neg_time = delta_time < timedelta()
                # Calculate the hour value for each row
               # hours = neg_time.cumsum()
                #hours  = pd.to_timedelta(hours,"hour")
                #df_events[col] = df_events[col] + hours
                continue
                
            match_starttime = df_match['Time'].iloc[0]
            #print(match_starttime)
            event_time = df_events[col].iloc[0]
           
       else:
            continue
        
    # Create the event tag file name to be new name
    new_filename = "Team1_Anon/20190910.MD/EventTags_1st_HalfD" + team_name + ".csv"
    df_events.to_csv(new_filename,index=False)

    return(df_events,possession)


def create_event_segment(df_events):  
    # for each row in event tag file find out the duration f the event
    df_events['Duration']  =  (df_events['Drill End Time'] -  df_events['Drill Start Time'])
    # Chnge the duration into seconds
    df_events['Duration'] = df_events['Duration'] / np.timedelta64(1,"s")
   
    # remove event <= to 5 secs 
    df_events = df_events.drop(df_events[df_events.Duration <= 5].index)
    
  
    # Loop over rows and add duration together until = 120
    # When duration is equal 120 then give the time segments the index segment i 
    segment_time  = 0
    segment_num = 1
    segment = []
    for row in df_events.iloc:
        if segment_time <= 120:
            segment_time += row['Duration']
        else:
        # Set the segment number 
            segment_num += 1
            segment_time = 0
            #print ('segent reset to 0')
        segment.append(segment_num)
    
    # Creat new column
    df_events['Segment_ID'] =segment
    
    
    return df_events


def process_segment_id(df_segment_times,df_main):
    # Outputs Df that has all the data for each segment identified
    # Pull timestamps for segment 
    list_seg = df_segment_times["Segment_ID"].unique()
    df_segment_all = pd.DataFrame()
    # Put DF of x,y together for this segement
    for i in list_seg:
        df_seg = df_segment_times[df_segment_times["Segment_ID"] == i]
        
        # Pull the timestamps from df_main to new df_segment_test
        for row in df_seg.iloc:
            start = row["Drill Start Time"]
            end = row["Drill End Time"]
            # Take the start and duration and pull those records to new df
            df_segment_data = df_main.loc[df_main['Time'].between(start,end, inclusive=True)]

            df_segment_data['Segment_ID'] =i
            df_segment_test = df_segment_data
            # add segment data frame to existing segment df
            df_segment_all = df_segment_all.append(df_segment_data)
            
    return(df_segment_all)


def segment_pre_process(df,num_players,possession_type, team_name,direction_of_play):
    #set-up pdf 
    pdf = PdfPages(team_name + possession_type +' team_formations3.pdf')
    # PDF for overall
    pdf2 = PdfPages(team_name + possession_type +' team_overall.pdf')
    # PDF convex hull
    pdf3 = PdfPages(team_name +  possession_type + ' team_convexhull.pdf')
    
    # Extract the segment and preprocess, other players
    list_seg = df["Segment_ID"].unique()
    
    # Calculate the co-variance matrices of player observation for segments 
    # INitialise array to hold covariance matrices
    player_covariance_matrices = np.zeros([len(list_seg),num_players,2,2])    
    
    # PDF for overall
    overall_formation_arr = np.zeros([len(list_seg),num_players,2])
    #for fig in list_seg:  
    for seg_id in list_seg:
        # Get all data for a time stamp            
        df_seg = df[df['Segment_ID'] == seg_id]
        # In this preprocess complete
        # 1. find vectors for each player for each Timestamp (TS) 
        # 2. find distance for each player for each TS
        # 3. Calculate ave. vector for each player pair over time
        # 4. Calculate ave. distance for each player over time
            
        # Build the array for player positions over time this feeds into covarinace calculations
        player_positions = np.zeros([df_seg.shape[0],num_players,2])
        for (time_index, row) in enumerate(df_seg.iloc):
            for i in range(num_players):    
                player_positions[time_index,i,0] = row[f'X_{i}']
                player_positions[time_index,i,1] = row[f'Y_{i}']
                
        # Calculate covariance matrices for the segment
        for i in range (num_players):
            player_covariance_matrices [seg_id-1, i] =np.cov(player_positions[:,i].T)
        
        # 1.Find the vectors for each player for each TS output=array 
        vector_arr = np.zeros([df_seg.shape[0],num_players, num_players,2])
        # Calculate Vector for every timestamp
        for (time_index, row) in enumerate(player_positions):
             # Calculate vector for every player
             for i in range(num_players):
                 for j in range(num_players):                  
                     # find the vector between the points and put into array
                     vector_arr[time_index, i,j] = row[i] - row[j]
                          
        # 2 Find the distance between each player for each timestamp
        distance_arr = np.linalg.norm(vector_arr,axis=3)
                
        # 3. Find the average vector for each player pair (e.g.player1-2) over time
        ave_vector = np.mean(vector_arr, axis = 0)
                
        # 4. Calculate ave. distance for each player over time
        ave_distance = np.mean(distance_arr, axis = 0)
                
        # Find the centre of mass /the player who is in the densest part of the team formation
        # Is defined as the 3rd nearest neighbour on average to player
        # Sort on distance to the end player using average distance
        distance_sorted = np.sort(distance_arr,axis=2)
        ##2distance_sorted = np.sort(ave_distance,axis=0)
                    
        # Select out the 3rd nearest neighbour for each player
        distance_3rdNN = distance_sorted[:,:,3]
             ##2distance_3rdNN = distance_sorted[:,3]
                    
        # Calculate the average 3rd NN distance for each player
        avg_distance_3rdNN = np.mean(distance_3rdNN,axis=0)
                    
        # Shortest distance is the player in the densest part of the formation
        player_cof = np.argmin(avg_distance_3rdNN)
            
        # Loop to get formation
        # map out the team formation for the segment with cof set as player_cof
        # player number of cof is the index so we take this as the starting point of 
        # the formation
        
        # create empty array for formation
        player_formation_arr = np.zeros([num_players,2])
        #plot_player_formation_arr = np.zeros([num_players,3])
        placed_player_indices =[]
        current_player = player_cof
        
        # Addin the cof player to the indices so they are removed from team formation calculations
        placed_player_indices.append(current_player)
        if seg_id == 1:
            num_in_team = (num_players -1)
             
        # Calculate Team Formation - for the number of players loop round
        for _ in range(num_in_team):         
            # Get the ave distances for the cof player
            ab = np.array(ave_distance[:,current_player])
            # sort Order of closest to current player
            next_player_sort = np.argsort(ab)
                    
            next_player_index = 1
                    
            # Check player until we find one that has not been used
            while next_player_sort[next_player_index] in placed_player_indices:
                next_player_index+=1
                        
            next_player = next_player_sort[next_player_index]
                        
            # get the vector for that player
            next_player_vector  = ave_vector[current_player, next_player]
            # add player and vector to array=
            player_formation_arr [next_player] = player_formation_arr[current_player] + next_player_vector
                        
            current_player = next_player
            placed_player_indices.append(current_player)
                    
        # find the average formation position of each player and set that to be centre            
        player_formation_arr = player_formation_arr - np.mean(player_formation_arr, axis=0, keepdims=True)
        
        #PDF for overall
        overall_formation_arr[(seg_id-1)] = player_formation_arr
        
        # Map the team formation for the segment
        fig,ax = plot_pitch.plot_frame(player_formation_arr)
        ax.text( player_formation_arr[2,0],player_formation_arr[2,1],str('Right full back'), fontsize=10, color='b')
        ax.text( player_formation_arr[3,0],player_formation_arr[3,1],str('Left full back'), fontsize=10, color='b')
        ax.text( player_formation_arr[9,0],player_formation_arr[9,1],str('Centre Forward'), fontsize=10, color='b')
  
        ax.set_title(team_name +' '+ possession_type + " formation for segment" + str(seg_id))
        if direction_of_play == 1:
            ax.set_xlabel('← Direction of play')
        else:
            ax.set_xlabel('--> Direction of play')
        plt.show()
        pdf.savefig(fig)
        plt.clf()    
        
        # Call the convex hull for the segment
        fig2,ax,area = convex_hull_average(player_formation_arr,num_players,seg_id,possession_type,direction_of_play)
        pdf3.savefig(fig2)
        plt.clf()      
    pdf3.close()
    pdf.close()
    
    # Create an overall pdf for segments -1 = no. of segmentsxplayers, 2 = x,y   
    fig1,ax = plot_pitch.plot_frame(overall_formation_arr.reshape([-1,2]))
    ax.set_title(team_name +' '+ possession_type + " formation" )
    if direction_of_play == 1:
        ax.set_xlabel('← Direction of play')
    else:
        ax.set_xlabel('--> Direction of play')
    pdf2.savefig (fig1)
    plt.clf()      
    pdf2.close()   
       
    return overall_formation_arr, player_covariance_matrices

def convex_hull_average(arr,number_players,seg_id,possession_type,direction_of_play):
    fig,ax = plot_pitch.plot_pitch() # plot the pitch
    defpoints = []
    
    # plot the pitch        
    defpoints = np.array(arr,dtype=np.float32)                        
    # create the convex hull
    hull= cv2.convexHull(defpoints).squeeze()
    #hull = ConvexHull(defpoints)
    output = np.mean(defpoints, axis=0)
    area = cv2.contourArea(hull)
            
    # COnvex hull put it above the pitch in layer form
    ax.fill(hull[:,0],hull[:,1],'b',alpha=0.3, zorder=10)    
    ax.scatter(defpoints[:,0],defpoints[:,1], c='y',zorder=20, label=f"{possession_type} = {area:.0f} m2")
    # Add the centre of mass arrow to the plot origin is 0,0
    ax.annotate("", xy=(output[0],output[1]), xytext=(0,0),alpha=0.3, arrowprops=dict(alpha=0.3,width=0.5,headlength=4.0,headwidth=4.0,color="black"),annotation_clip=False)
     
    ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.00), shadow=True, ncol=2)
    ax.set_title(possession_type + " Convex hull - Segment" + str(seg_id) )
    if direction_of_play == 1:
        ax.set_xlabel('← Direction of play')
    else:
        ax.set_xlabel('--> Direction of play')
    
    return fig,ax,area

# Calculates average foramtion for offense and defense and puts on the same page
def plot_average_shape(formations,direction_of_play,team_name):
    pdf_ave = PdfPages('Ave_Team_formations.pdf')
    # Pull out each formation and processing
    fig,ax = plot_pitch.plot_pitch() # plot the pitch
    j = 0
    
    # Change shape of the data 
    reshape_overall_formation  =np.zeros([2,10,2])
    # Loop over formations and extract the average formation data and not covariance
    for (i, (formation_position, _)) in enumerate(formations):
        # Set reshape ave formation to be ave
        reshape_overall_formation[i] = np.mean(formation_position, axis=0)
        
    
    for i in reshape_overall_formation:
           
        i= np.float32(i)    
        
        # create the convex hull
        hull1= cv2.convexHull(i).squeeze()
        area = cv2.contourArea(hull1)
        if j==0 :
            possession_type = "Out of possession"
            # COnvex hull put it above the pitch in layer form
            ax.fill(hull1[:,0],hull1[:,1],'b',alpha=0.3, zorder=10)    
            ax.scatter(i[:,0],i[:,1], c='#1E3082',edgecolors= 'b',s=100,zorder=20, label=f"{possession_type} = {area:.0f} m2")
        else:
            possession_type = "In possession"
            ax.fill(hull1[:,0],hull1[:,1],c='#1E3082',fill= False,hatch='\\',alpha=0.5, zorder=10)  
            ax.scatter(i[:,0],i[:,1], c='#F0FAF3', edgecolors= 'b',s=100, zorder=20, label=f"{possession_type} = {area:.0f} m2")
           
        ax.legend(loc='upper left', bbox_to_anchor=(0.40, 1.00), shadow=True, ncol=2)
        ax.set_title(team_name + " formation in and out of possession"  )
        if direction_of_play == 1:
            ax.set_xlabel('← Direction of play')
        else:
            ax.set_xlabel('--> Direction of play')
        j+=1
    # Close the pdf file
    pdf_ave.savefig(fig)
    plt.clf()      
    pdf_ave.close()

    
    return

    




        
