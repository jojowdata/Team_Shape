# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 08:01:46 2020

@author: j.white
"""

# Import libaries 
import numpy as np
import scipy.optimize
import scipy.spatial
from scipy.cluster.hierarchy import dendrogram, linkage, ward
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plot_pitch
## For plotting the ellipses on ave clusters
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
    
def calculate_wasserstein_dist(f1_mean,f2_mean,co_var1, co_var2, epsilon=1e-4):
    '''
    This function calculates the Wasserstein Distance between two obsesrvation formations using the 
    bivarite distribution (means and co-variance matrix).
    
    Using code from Wasserstein Distance and Optimal Transport Map of Gaussian Processes by https://github.com/VersElectronics/WGPOT
    
    
    input: Player position average for the formations and bi-variate distribution for the two formations 
    
    output: a single number that is the distance or metric on how similar the two formations being compared are
    fyi: equation in shaws paper 
    W(μ1,μ2)2=||m1−m2||2+trace(C1+C2−2(C1/22C1C1/22)1/2)
    '''
    
    scv2 = scipy.linalg.sqrtm(co_var2)
    first_term = np.dot(scv2, co_var1)
    scv2_cv1_scv2 = np.dot(first_term, scv2)
    cov_dist = np.trace(co_var1) + np.trace(co_var2) - 2 * np.trace(scipy.linalg.sqrtm(scv2_cv1_scv2))
    if cov_dist > -epsilon and cov_dist < 0:
        cov_dist = 0   
    l2norm = (np.sum(np.square(abs(f1_mean - f2_mean))))
    cost = np.real(np.sqrt(l2norm + cov_dist))

    return cost


def minimum_cost_formation (f1,covar1,f2,covar2):
    '''
    This function uses the pair formation comparison of WD to build up a cost table of similarity.  It calculates the
    distance between players in 2 formations which is the cost.  The function loops over the data and for each comparison it
    creates a 10x10 cost matrix.  The Hungarian  Method is applied to the table and the minimum cost.
    
    Using code snippets from https://stackoverflow.com/questions/64245389/pairwise-wasserstein-distance-on-2-arrays

    input: player data for the segment
    
    output: minimum cost bewtween to formations
    '''
    # Create bucket for mincost for these formations
    min_cost = np.inf
    p = 0
    # factors = l,k for re-scalingand comparing
    for l in np.linspace(0.8, 1.2,3):
        for k in np.linspace(0.8, 1.2,3):
            # Need to apply scaling factors to formations
            f1_scaled = f1*l
            covar1_scaled = np.dot(k,covar1)
            f2_scaled = f2*k
            # Something to do with Matrices?? not the same as f1
            covar2_scaled = covar2*k
            # Create cost table
            cost =np.zeros([len(f1),len(f2)])
            #for each player in the first formations 
            for (i, (playeri_mean,playeri_covar)) in enumerate(zip(f1_scaled,covar1_scaled)):
                #For each player on the second formation
                for (j, (playerj_mean,playerj_covar)) in enumerate(zip(f2_scaled,covar2_scaled)):
                    #assign values to player pair cost matrix
                    cost[i,j] =  calculate_wasserstein_dist(playeri_mean,playerj_mean, playeri_covar, playerj_covar)
                    p+= 1
                    
            # Apply Hungarian method
            # This is using the Hungerian Method to assign closest player 
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
            curr_cost = cost[row_ind, col_ind].sum()
            if curr_cost < min_cost:
                min_cost = curr_cost
    return min_cost
                    
def cluster_ahc (seg_cost_matrix):
    '''
    This function takes the cost matric for all formations in segments and intputs them in AHC
    '''
    
    Z = ward(seg_cost_matrix)
    cluster = fcluster(Z, 3 , criterion='maxclust') 
    plt.figure()
    dn = dendrogram(Z,show_leaf_counts=True,no_labels=False)
    plt.title("Clustering of Segment Formations" )
    plt.ylabel("Distance (Dissimlarity)")
    plt.xlabel("Segment formations Number")
    
    return cluster



def cluster_formation(all_seg_data, offense_defense,direction_of_play,team_name):
    
    '''
    This function calculates a number that represents the cost between formations for example if there are 7 segments then 
    the cost table will be 7x7 were there is a cost (minimum) associated between each formation.
    
    input: data for all the segments
    output: cost for each comparion of formation    
    '''
    # Merge the formations into a single list 
    formations = []

    for (means, covars) in all_seg_data:
        formations.extend(zip(means,covars))
    
   
    # Initilise cost array
    cost_table = np.zeros([len(formations),len(formations)])
    
     
    # Loop over formations and extract the average formation data and covariance for each segment
    for (i, (formation_pos1, covar1)) in enumerate(formations):
        for (j, (formation_pos2, covar2)) in enumerate(formations):
        # Set reshape ave formation to be ave
        #reshape_overall_formation[i] = np.mean(formation_position, axis=0) 

            # Call Minimum_Cost Function
            cost_table[i,j] = minimum_cost_formation(formation_pos1,covar1,formation_pos2,covar2)
                
     # AHC using the cost-table
    clusters = cluster_ahc(cost_table)
    
    # Plot the clusters onto pdf
    pdf_cluster = PdfPages('Unique_formations.pdf')
    for i in range (clusters.max()):
        fig,ax = plot_pitch.plot_pitch() # plot the pitch
        clusterindices = np.where(clusters == i+1)[0]
        # Average fomration positions set to same size as the formation array
        average_cluster_points = np.zeros(formations[0][0].shape)
        # Covariance of the fomration positions set to same size as the formation array
        average_cluster_covariance = np.zeros(formations[0][1].shape)
        # get average and covariance for the cluster
        
        # Calculate the means of the cluster given
        for index in clusterindices:
            mean,covariance = formations[index]
            average_cluster_points += mean
            average_cluster_covariance += covariance
            
        average_cluster_points = average_cluster_points / len(clusterindices)
        average_cluster_covariance = average_cluster_covariance/len(clusterindices)     
        
        #plot the cluster
        ax.scatter(average_cluster_points[:,0],average_cluster_points[:,1])
        ax.set_title(team_name + " Cluster " + str(i+1)  )
        # Loop each player and draw the cofidence interval
        for (cov,mean) in zip(average_cluster_covariance,average_cluster_points) :
            confidence_ellipse(cov,mean,ax,edgecolor='red')
            
        # Add the % of offense and defense using the segments formation in a cluster
        # Pull all the segments in a cluster
        #for index in clusterindices:
        
        defense_formation = 0
        offense_formation = 0
        
        for value in (clusterindices):
        # Are they offense or defense             
        # for k in enumerate(clusterindices):
            if (offense_defense[value]) == 0:
                defense_formation += 1
            else:
                offense_formation += 1
                
        # Calculate the %
        if direction_of_play == 1:
            direction = ('← Direction of play')
        else:
            direction = ('--> Direction of play')
        plt.xlabel('Defensive:' f"{((defense_formation/(defense_formation+offense_formation))*100):.2f}%"
                   + '  Offense:' + f"{((offense_formation/(defense_formation+offense_formation))*100):.2f}%"
                   + f' ({defense_formation + offense_formation} formations total)'
                   + '                           ' +  str(direction),
                   horizontalalignment='left', x=0.0)
            
        pdf_cluster.savefig(fig)
    
    plt.clf()      
    pdf_cluster.close()
    
    
    return

def confidence_ellipse(cov, mean, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of covariances and means this 
    is completed using an example from https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    cov: array-like, shape (2,2 )
        Input data.
    
    mean: array-like shape(2,)

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='#BC96E6',alpha=0.3, zorder=10, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


