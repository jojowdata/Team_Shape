# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:58:08 2020

# This code takes in player location data and produces a 
# convex hull for these points and the centre of the hull mass.  
# It displays the player positions in a moment of time with the 
# calculated convex hull amd centre of mass overlayed.
# This moment is saved as part of an MP4 file.

This code was based on code from Laurie Shaw at friends of tracking 
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Viz.py

@author: j.white
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import plot_pitch
import matplotlib.animation as animation

fps = 5 # realtime
def create_convex_hull(df,number_players):
    
    # Set figure and movie settings
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='STATSport tracking data clip')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fname = 'video.mp4' # path and filename
    fig,ax = plot_pitch.plot_pitch() # plot the pitch
    # Generate movie
    print("Generating movie...",end='')
    with writer.saving(fig, fname, 100):
        # For each timestamp calcuate the convex hull and centre of mass and
        # output onto pitch plot
        for row in df.iloc:
            defpoints = []
            figobjs = []
        
            for i in range(number_players):
                if row[f'X_{i}'] != np.nan:
                    defpoints.append(row[[f'X_{i}',f'Y_{i}']].values)
                    
            # plot the pitch        
            defpoints = np.array(defpoints,dtype=np.float32)                        
            # create the convex hull
            hull= cv2.convexHull(defpoints).squeeze()
            output = np.mean(defpoints, axis=0)
            
            # COnvex hull put it above the pitch in layer form
            objs = ax.fill(hull[:,0],hull[:,1],'b',alpha=0.3, zorder=10)    
            figobjs.extend(objs)
            objs = ax.scatter(defpoints[:,0],defpoints[:,1], c='y',zorder=20)
            figobjs.append(objs)
            # Add the centre of mass arrow to the plot origin is 0,0
            objs = ax.annotate("", xy=(output[0],output[1]), xytext=(0,0),alpha=0.3, arrowprops=dict(alpha=0.3,width=0.5,headlength=4.0,headwidth=4.0,color="black"),annotation_clip=False)
            figobjs.append(objs)
            #write from to the file
            writer.grab_frame()
            # Delete all axis objects (other than pitch lines) in preperation for next frame
            
            for figobj in figobjs:
                figobj.remove()
    print("done")
    plt.clf()
    plt.close(fig)    
   
    