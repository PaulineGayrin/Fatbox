#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MODULE Structural analysis

# This file contains a series of function to analyse the structure of the faults.

"""

import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os

# Fatbox
path_file=Path(__file__).absolute()
path_folder=path_file.parent
#print(path_folder)
os.chdir(path_folder)

import preprocessing
import metrics
import plots
import utils 
import edits


def strike_edges (G):
    """
    Compute the azimuth of each edge of graph G.
    
    Azimuth 0 is the North, clockwise increasing until 180° (=South).

    Parameters
    ----------
    G : nx.graph
        Graph

    Returns
    -------
    Graph G with included strike of each edge in G.edges[edge]['strike']

    """

    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    for edge in G.edges:
      x1 = G.nodes[edge[0]]['pos'][0]
      x2 = G.nodes[edge[1]]['pos'][0]
      y1 = G.nodes[edge[0]]['pos'][1]
      y2 = G.nodes[edge[1]]['pos'][1]

      if (x2-x1)<0:
        strike= 180 - math.degrees(math.atan2((x2-x1),(y2-y1)))
      else:
        strike= 180 - math.degrees(math.atan2((x2-x1),(y2-y1)))

      if strike<=180:
        G.edges[edge]['strike']=strike
      else:
        G.edges[edge]['strike']=strike-180
    return G



#################

def calcul_coordinates (strike_rad,midpoint_x,midpoint_y,n):
    """

    Compute the coordinates of the points D and E located at distance n
    on the perpendicular line crossing the middle of the segment.
    
    Look at the documentation to see explanations in schemes.

    Parameters
    ----------
    strike_rad : float
        Azimuth of the segment. This segment can be an edge 
        or the average or median between several edges.
    midpoint_x : flaot
        x coordinate of the middle of the segment
    midpoint_y : float
        y coordinate of the middle of the segment
    n : integer
        distance from middle to D and middle to E

    Returns
    -------
    xD : float
        x coordinate of D
    yD : float
        y coordinate of D
    xE : float
        x coordinate of E
    yE : float
        y coordinate of E

    """
    
    #le beau magnifique qui est vrai
    
    if strike_rad > (np.pi/2):
        #D
        xD= midpoint_x - (np.sin(strike_rad-(np.pi/2))* n)
        yD= midpoint_y + (np.cos(strike_rad-(np.pi/2))* n)
        #E
        xE= midpoint_x + (np.sin(strike_rad-(np.pi/2))* n)
        yE= midpoint_y - (np.cos(strike_rad-(np.pi/2))* n)
            
    else:    
        #D
        xD=midpoint_x - np.sin((np.pi/2)-strike_rad) * n
        yD=midpoint_y - np.cos((np.pi/2)-strike_rad) * n
        #E
        xE=midpoint_x + np.sin((np.pi/2)-strike_rad) * n
        yE=midpoint_y + np.cos((np.pi/2)-strike_rad) * n
              
    return xD,yD,xE,yE

#################

def filter_out_edges (G, img_dem,d):
    """
    Remove the edges for which the cross section would be out of the DEM.

    Parameters
    ----------
    G : nx.graph
        Graph
    img_dem : np.array
        Digital Elevation Model of the area
    d : int
        maximum distance from center of the fault to left or 
        right extremity of cross-section. In pixels.

    Returns
    -------
    G : nx.graph
        Graph without the edges for which the cross section 
        would be out of the DEM.

    """
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert len(img_dem.shape)==2 , "The shape of the DEM array is not 2D"

    # boundaries of the box
    sy_dem=np.size(img_dem,0)
    sx_dem=np.size(img_dem,1)

    list_labels=metrics.get_component_labels(G)

    k=0

    nb_nodes_removed=0 
    nb_removed_edge=0

    for label in list_labels:
        nodes = [node for node in G if G.nodes[node]['component']==label]
        K=nx.subgraph(G,nodes)
        list_remove_nodes=[]

        for edge in K.edges:
            remove_edge=False
            # Cross section on center of segment with d from center to each side

            # Extract grid coord and altitudes of cross-section
            x0, y0 = K.nodes[edge[0]]['pos']
            x1, y1 = K.nodes[edge[1]]['pos']
            strike=K.edges[edge]['strike']

            ################################
            # compute coordinates and gather altitudes of the profile
            # perpendicular to the segment, in its middle M, with length d on each side

            #compute the coordinates of the middlepoint of each segment
            midpoint_x=(x0+x1)/2
            midpoint_y=(y0+y1)/2

            #loop cross section.
            #Pull apart, pixel by pixel, from M, to M + d and M - d
            for n in range (1,d+1):

                xD,yD,xE,yE=calcul_coordinates(np.radians(strike), midpoint_x, midpoint_y, n)

                # ROUND OF COORDINATES
                #goal of those 4 lines is to have the coord of the pixel where the points are
                #it have to be an int because img_dem[x,y] can take only int

                xD=math.floor(xD+0.5)
                yD=math.floor(yD+0.5)
                xE=math.floor(xE+0.5)
                yE=math.floor(yE+0.5)

                if xD >sx_dem-1 or xE>sx_dem-1 or xD<0 or xE<0 or yD>sy_dem-1 or yD<0 or yE>sy_dem-1 or yE<0:
                    remove_edge=True
                elif img_dem[yD,xD]==0 or img_dem[yE,xE]==0:
                    remove_edge=True


            xD,yD,xE,yE=calcul_coordinates(np.radians(strike), midpoint_x, midpoint_y, d)

            k=k+1 #k is the num of edge (same as index in list_all_edges)

            if remove_edge==True:
                #u,v=edge
                G.remove_edge(edge[0],edge[1])
                nb_removed_edge=nb_removed_edge+1

                if (edge[0] not in list_remove_nodes) and (edge[1] not in list_remove_nodes):
                    list_remove_nodes.append(edge[0])
                    list_remove_nodes.append(edge[1])


        for node in list_remove_nodes:
            G.remove_node(node)
            nb_nodes_removed=nb_nodes_removed+1 ##

    print('Number of edges removed : '+ str(nb_removed_edge))
    print('Number of nodes removed : '+ str(nb_nodes_removed))


    return G



### 

def write_slip_to_displacement(G, dim):
    """ Write slip to displacment
    
    Used for numerical modeling.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    dim : int
        Dimension of graph
    
    Returns
    -------  
    G : nx.graph
        Graph      
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    if dim == 2:
        for node in G:
            G.nodes[node]['heave'] = G.nodes[node]['slip_x']
            G.nodes[node]['throw'] = G.nodes[node]['slip_z']
            G.nodes[node]['displacement'] = G.nodes[node]['slip']

    if dim == 3:
        for node in G:
            G.nodes[node]['heave'] = G.nodes[node]['slip_x']
            G.nodes[node]['lateral'] = G.nodes[node]['slip_y']
            G.nodes[node]['throw'] = G.nodes[node]['slip_z']
            G.nodes[node]['displacement'] = G.nodes[node]['slip']
    return G


def assign_displacement(G, points, dim):
    """ Assign displacements from network
    
    Used for numerical modeling.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    dim : int
        Dimension of graph
    
    Returns
    -------  
    G : nx.graph
        Graph      
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    if dim == 2:
        for node in G:
            for point in points:
                if node == point[0]:
                    G.nodes[node]['heave'] = point[3]
                    G.nodes[node]['throw'] = point[4]
                    G.nodes[node]['displacement'] = point[5]
    if dim == 3:
        for node in G:
            for point in points:
                if node == point[0]:
                    G.nodes[node]['heave'] = point[3]
                    G.nodes[node]['lateral'] = point[4]
                    G.nodes[node]['throw'] = point[5]
                    G.nodes[node]['displacement'] = point[6]
    return G


def get_slip_rate(G, dim):
    """ Get slip rate from network
    
    Parameters
    ----------
    G : nx.graph
        Graph
    dim : int
        Dimension of graph
    
    Returns
    -------  
    points : array
        Float    
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    if dim == 2:
        points = np.zeros((len(list(G)), 6))
        for n, node in enumerate(G):
            points[n, 0] = node
            points[n, 1] = G.nodes[node]['pos'][0]
            points[n, 2] = G.nodes[node]['pos'][1]
            points[n, 3] = G.nodes[node]['slip_rate_x']
            points[n, 4] = G.nodes[node]['slip_rate_z']
            points[n, 5] = G.nodes[node]['slip_rate']
    if dim == 3:
        points = np.zeros((len(list(G)), 7))
        for n, node in enumerate(G):
            points[n, 0] = node
            points[n, 1] = G.nodes[node]['pos'][0]
            points[n, 2] = G.nodes[node]['pos'][1]
            points[n, 3] = G.nodes[node]['slip_rate_x']
            points[n, 4] = G.nodes[node]['slip_rate_y']
            points[n, 5] = G.nodes[node]['slip_rate_z']
            points[n, 6] = G.nodes[node]['slip_rate']
    return points



def get_displacement(G, dim):
    """ Get displacments from network
    
    Parameters
    ----------
    G : nx.graph
        Graph
    dim : int
        Dimension of graph
    
    Returns
    -------  
    points : array
        Float     
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    if dim == 2:
        points = np.zeros((len(list(G)), 6))
        for n, node in enumerate(G):
            points[n, 0] = node
            points[n, 1] = G.nodes[node]['pos'][0]
            points[n, 2] = G.nodes[node]['pos'][1]
            points[n, 3] = G.nodes[node]['heave']
            points[n, 4] = G.nodes[node]['throw']
            points[n, 5] = G.nodes[node]['displacement']
    if dim == 3:
        points = np.zeros((len(list(G)), 7))
        for n, node in enumerate(G):
            points[n, 0] = node
            points[n, 1] = G.nodes[node]['pos'][0]
            points[n, 2] = G.nodes[node]['pos'][1]
            points[n, 3] = G.nodes[node]['heave']
            points[n, 4] = G.nodes[node]['lateral']
            points[n, 5] = G.nodes[node]['throw']
            points[n, 6] = G.nodes[node]['displacement']
    return points



def dip(x0, z0, x1, z1):
    """ Compute dip between two points: (x0, z0) (x1, z1)
    
    Used for numerical modeling.
    
    Parameters
    ----------
    x0 : float
        X-coordinate of point 0
    x1 : float
        X-coordinate of point 1
    z0 : float
        Z-coordinate of point 0
    z1 : float
        Z-coordinate of point 1
        
    Returns
    -------  
    value : float
        Dip between points
    """

    # Assertions

    
    # Calculation
    if (x0 - x1) == 0:
        value = 90
    else:
        value = math.degrees(math.atan((z0 - z1)/(x0 - x1)))
        if value == -0:
            value = 0

    return value






def calculate_dip(G, non):
    """ Compute dip of fault network
    
    Used for numerical modeling.
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    non: int
        Number of neighbors
        
    Returns
    -------  
    G : nx.graph
        Graph containing edges with 'strike' attribute
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    
    
    for node in tqdm(G):
    
        
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=non)
        
        
        neighbors = sorted(neighbors.items())
        
        first = neighbors[0][0]
        last = neighbors[-1][0]
        
        # print(node)
        # print(neighbors)
        # print(first, last)

        
        
        x1 = G.nodes[first]['pos'][0]
        y1 = G.nodes[first]['pos'][1]
           
        x2 = G.nodes[last]['pos'][0]
        y2 = G.nodes[last]['pos'][1]
          
        
        G.nodes[node]['dip'] = dip(x1, y1, x2, y2)

    return G




def calculate_diff_dip(G, non):
    """ Compute dip difference between nodes of fault network
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    non: int
        Number of neighbors
        
    Returns
    -------  
    G : nx.graph
        Graph containing edges with 'strike' attribute
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'

    
    for node in G:
    
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=non)
        dips = [G.nodes[node]['dip'] for node in neighbors.keys()]
        G.nodes[node]['max_diff'] = np.max(np.diff(dips))
    
    return G












