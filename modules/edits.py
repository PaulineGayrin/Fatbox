#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MODULE edits

# This file contains a series of function to edit fault networks (graphs). 
# This includes functions for: 
# (1) nodes
# (2) edges
# (3) components (i.e. connected nodes)
# (4) the whole network 

"""

# Packages
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
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
import structural_analysis


#******************************************************************************
# (1) NODE EDITS
#******************************************************************************

def scale(G, fx, fy):
    """ Scale coordinates of nodes (x,y) of graph by factor (fx, fy)
    
    Parameters
    ----------
    G : nx.graph
        Graph
    fx : float
    fy : float
    
    Returns
    -------  
    G : nx.graph
        Graph
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"    
    assert isinstance(fx, int) or isinstance(fx, float), "fx is neither int nor float"
    assert isinstance(fy, int) or isinstance(fy, float), "fx is neither int nor float"
    
    # Scaling
    for node in G:
        G.nodes[node]['x'] = G.nodes[node]['x'][0]*fx
        G.nodes[node]['y'] = G.nodes[node]['y'][1]*fy
        
    return G



#####  REMOVE  #####


def remove_triangles(G):
    """ Remove triangles from network
    
    Remove longest edge of the triangular shape

    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    H : nx.graph
        Graph without triangles. 
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    G = metrics.compute_edge_length(G)

    # Find triangles through edges
    triangles = []
    for node in tqdm(G, desc='Find triangles'): #print progression of algo
        for n in G.neighbors(node):
            for nn in G.neighbors(n):
                for nnn in G.neighbors(nn):
                    if node == nnn:
                        triangles.append((node, n, nn))

    triangles = set(tuple(sorted(t)) for t in triangles)

    # Remove triangles
    H = G.copy()

    for t in tqdm(triangles, desc='Remove triangles'):
        # Find longest edge
        length = 0
        for edge in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]:
            if G.edges[edge]['length'] > length:
                length = G.edges[edge]['length']
                longest_edge = edge

        # Remove longest edge
        if longest_edge in list(H.edges):
            H.remove_edge(*longest_edge)

    return H



def remove_node_alone (G): 
    """ Remove the nodes that have zero connection in graph G

    Parameters
    ----------
    G : nx.Graph
        Graph to process

    -------
    Returns
    -------
    H : nx.Graph
        Graph G without node that were not connected to any other.

    """
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    H=G.copy()

    for node in G:
           if H.degree[node] == 0: # degree=0 means node not connected = alone
               H.remove_node(node)
    return H



def remove_cycles(G):
    """ Remove cycles from network
    
    Remove y-nodes that are organised in cycle.

    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    G : nx.graph
        Graph without the cycles
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    nodes_to_remove = set()

    # Find cycles
    cycles = nx.cycle_basis(G)

    for n, cycle in enumerate(cycles):

        # Find y-nodes (i.e. node with 3 edges)
        y_nodes = [node for node in cycle if G.degree(node) == 3]

        # If cycle has only one y-node, remove it (except the y-node itself)
        if len(y_nodes) == 1:
            #print('Cycle ' + str(n) + ' has only one y-node. Remove it')
            for node in cycle:
                if node not in y_nodes:
                    nodes_to_remove.add(node)

    G.remove_nodes_from(nodes_to_remove)

    return G



def remove_cycles_hard(G):
    """ Remove cycles from network
    
    Remove y-nodes that are organised in cycle.
    Hard version (take less precaution) of edits.remove_cycles

    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    G : nx.graph
        Graph without the cycles
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    nodes_to_remove = set()

    # Find cycles
    cycles = nx.cycle_basis(G)

    for n, cycle in enumerate(cycles):

        # Find y-nodes (i.e. node with 3 edges)
        y_nodes = [node for node in cycle if G.degree(node) == 3]

        # If cycle has only one y-node, remove it (except the y-node itself)
        print('Cycle ' + str(n) + ' has only one y-node. Remove it')
        for node in cycle:
            if node not in y_nodes:
                nodes_to_remove.add(node)

    G.remove_nodes_from(nodes_to_remove)

    return G


def remove_triple_junctions(G):
    """ Remove triple junction from network
    
    Triple junctions are nodes with 3 edges .

    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    H : nx.graph
        Graph
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    if 'edges' not in G.nodes[list(G.nodes)[0]]: #compute the attribute 'edges' in case it was not already done
        G=metrics.count_edges(G)

    # Calculation
    H = G.copy()
    for node in G:
        if H.nodes[node]['edges'] == 3:
            H.remove_node(node)
    return H


def remove_empty_labels (G):
    """ Remove label that content a single node and no edges
    
    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    H : nx.graph
        Graph without the empty labels

    """

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    
    # Calculation
    H=G.copy()
    list_labels=metrics.get_component_labels(H)
    for label in list_labels:
        nodes = [node for node in G if G.nodes[node]['component']==label]
        K=nx.subgraph(H,nodes)

        if len(list(K.nodes))<=1:
            #print(label)
            remove_component(H, label)

    return H


def remove_nodes(G, attribute, sign, threshold):
    """ Remove node with attribute below/above/at certain value
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute
    sign : str 
        '<' to remove attribute below threshold
        '=' to remove attribute equal to threshold
        '>' to remove attribute above threshold

    threshold : float
        Value
    
    Returns
    -------  
    G : nx.graph
        Graph modified.

    """  
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert (sign in ['<', '=', '>']), \
    'Sign not available. Check the writing of the sign argument'  

    assert (attribute in G.nodes[list(G.nodes(0))[0][0]]) == True , \
        'The attribute '+attribute+' is not in the nodes of the Graph. Check that you have already computed the attribute.'

    assert isinstance(threshold,float), "Argument threshold is not a float."

    # Computation
    removals = []
    
    if sign == '>':        
        for node in G:
            if G.nodes[node][attribute] > threshold:
                removals.append(node) 
                
    elif sign == '<':
        for node in G:
            if G.nodes[node][attribute] < threshold:
                removals.append(node)         
    
    elif sign == '=':
        for node in G:
            if G.nodes[node][attribute] == threshold:
                removals.append(node)         
    
    G.remove_nodes_from(removals)
    
    return G




def remove_below(G, attribute, value):
    """ Write nan in node attribute if this node attribute is below a certain value.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute
    value : float
        Value threshold
    
    Returns
    -------  
    G : nx.graph
        Graph modified
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(value,float), "Argument value is not a float."
    assert (attribute in G.nodes[list(G.nodes(0))[0][0]]) == True , \
        'The attribute '+attribute+' is not in the nodes of the Graph. Check that you have already computed the attribute.'

    # Calculation 
    for node in G.nodes:
        if G.nodes[node][attribute] > value:
            G.nodes[node][attribute] = float('nan')
    return G



#####   FIND   #####


def find_neighbor_except(G, neighbor, node):
    """ Find a neighbor of node expect for the one given
    
    Parameters
    ----------
    G : nx.graph
        Graph
    neighbor : int
        Neighbor to avoid
    node : int
        Node
    
    Returns
    -------  
    neighbor : int
        Neighbor
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(node,int), 'Argument node is not an integer'
    assert isinstance(neighbor,int), 'Argument neighbor is not an integer'


    # Calculation
    if len(list(G.neighbors(neighbor))) != 2:
        return neighbor
    else:
        for nn in G.neighbors(neighbor):
            if nn != node:
                return nn



def find_new_neighbors(G, neighbors, origins):
    """ Find new neighbors of node except origins
    
    Parameters
    ----------
    G : nx.graph
        Graph
    neighbors : list
        Neighbors to avoid
    origins : list
        Origins to use
    
    Returns
    -------  
    new_neighbor : list
        Neighbors
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation 
    new_neighbors = [None]*3
    for k in range(3):
        new_neighbors[k] = find_neighbor_except(G, neighbors[k], origins[k])
        
    return new_neighbors


def find_tips (K):
    """ Find the 2 tips (extremities) of a subgraph.
    
    Parameters
    ----------
    K : nx.Graph
        K is a subgraph containing only one component.

    Returns
    -------
    tip : list
        list with the node of the two extremities of the edge
        
    """
    assert isinstance(K, nx.Graph), "K is not a NetworkX graph"
    
    tip=[]
    for n in K.degree: #n = (key, nb_of_link)
        if n[1]==1: 
            tip.append(n[0]) #tip content key of tips of this subgraph
    return tip


def find_next_node (K, node, already_view_list):
    """ Find the next node along the component in the subgraph K.
    
    Fonction used in single_shape_nodes_tip_to_tip.
    
    Parameters
    ----------
    G : nx.Graph
        Graph
    node : int
        node here
    already_view_list : list
        list of previous nodes (along the component)

    Returns
    -------
    next_node : int
        next node along the component (=next after the node in argument)

    """
    # Assertions
    
    assert isinstance(K, nx.Graph), "K is not a NetworkX graph"
    assert isinstance(node, int), "node is not a integer"

    # Computation
    for edge in K.edges:
        if (node in edge) and  ((edge[0] not in already_view_list) or (edge[1] not in already_view_list)):
            for  k in edge:
                if k != node: next_node=k
    return next_node


def closest_node(node, nodes):
    """ Find the closest node of argument "node' in nodes list.
    
    Parameters
    ----------
    node : int
        Node
    nodes : list
        Nodes
    
    Returns
    -------  
    value : int
        Closest node
    """ 
    # Computation
    assert isinstance(node, int), "Argument 'node' is not an integer."

    nodes = np.asarray(nodes)
    dist = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist)



##### Edits of nodes #####



def split_triple_junctions(G, dos, split='minimum', threshold = 20, plot=False):
   """ 
    This function splits up triple junctions (or y-nodes) based on their 
    orientation, so that the two branches most closely aligned remain
    connected and the splay is cut off.
    
    Parameters
    ----------
    G : nx.Graph
        Graph
        
    dos : int
        depth of search or the number of nodes used to determine the 
        orientation of each branch
        
    split : mode of split of branches, optional
        'minimum' Default mode.
        If angle between red and green branch is smallest, remove blue branch.
        'threshold'
        Split y-node based on difference in angles based on threshold.
        remove branch if difference below threshold
        'custom'
        If all difference of angle very similar, split all.
        'all'
        Split all the branches

    threshold : int, optional
        Specify if you use  the mode of split 'threshold'
        The default is 20.
        
    plot : bool, optional
        The default is False.

    Returns
    -------
    G : nx.Graph
        Graph modified.

    """
    
   # Assertions
   
   assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
   assert isinstance(dos, int), "Argument 'dos' is not a integer"
   
   assert (split in ['minimum', 'threshold', 'custom', 'all']), \
   'Split mode not available. Check the writing of the split argument.'   
    
   # Computation
   count = 0 
        
   for node in G:
        if G.degree[node] == 3:     # Find y-node
            
            count = count+1         # Increase counter
            
            true_neighbors = list(G.neighbors(node))    # Find 1st neighbors
            
            # Set up branches, i.e. list of nodes belonging to each
            # (rgb refers to the colors when plotting)
            branch_r = [node, true_neighbors[0]]
            branch_g = [node, true_neighbors[1]]
            branch_b = [node, true_neighbors[2]]
            
            # Set up set of eadges belonging to each branch                        
            edges_r = {(node, true_neighbors[0])}
            edges_g = {(node, true_neighbors[1])}
            edges_b = {(node, true_neighbors[2])}
            
            # Extract branches - it starts with the y-node (i.e. node) as
            # origins and the true neighbors as neighbors. Then we search
            # for the neighbors' neighbors (i.e. new_neighbors) ignoring
            # the origins. Next the neighbors become the origins and the
            # new neighbors become the neighbors, so that the search can go
            # one level deeper.
            origins = [node]*3
            neighbors = true_neighbors
            
            for n in range(dos):                     
                new_neighbors = find_new_neighbors(G, neighbors, origins)
                origins   = neighbors
                neighbors = new_neighbors
                
                # Add new neigbors to branch, if they're not already are
                if new_neighbors[0] not in branch_r:                
                    branch_r.append(new_neighbors[0])
                    
                if new_neighbors[1] not in branch_g:                
                    branch_g.append(new_neighbors[1])  
                    
                if new_neighbors[2] not in branch_b:                
                    branch_b.append(new_neighbors[2])
                    
    
                # Add new edges to set of edges, unless they're self-edges
                if origins[0] != neighbors[0]:
                    edges_r.add((origins[0], neighbors[0]))
                if origins[1] != neighbors[1]:
                    edges_g.add((origins[1], neighbors[1]))
                if origins[2] != neighbors[2]:
                    edges_b.add((origins[2], neighbors[2]))
                
                
            # Plot y-nodes with edges
            if plot:
                
                plt.figure(figsize=(12,12))
            
                nx.draw_networkx_nodes(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       nodelist=[node],
                                       node_color="yellow")
                    
                nx.draw_networkx_nodes(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       nodelist=branch_r,
                                       node_color="red")
                
                nx.draw_networkx_edges(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       edgelist=edges_r,
                                       edge_color="red")
                
                nx.draw_networkx_nodes(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       nodelist=branch_g,
                                       node_color="green")
                
                nx.draw_networkx_edges(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       edgelist=edges_g,
                                       edge_color="green")
                
                nx.draw_networkx_nodes(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       nodelist=branch_b,
                                       node_color="blue")
                
                nx.draw_networkx_edges(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       edgelist=edges_b,
                                       edge_color="blue")
                
                plt.axis('equal')
                
                            
            
            
            # Calculate slope of each branch
            def slope(G, nodes):
                
                x = [G.nodes[node]['pos'][0] for node in nodes]
                y = [G.nodes[node]['pos'][1] for node in nodes]
                
                dx = x[0]-x[-1]
                dy = y[0]-y[-1]
                
                # If point cloud is vertical
                if dx < 1e-10 or abs(dy/dx) > 8:
                    slope = 1e16
                    x_pred = np.ones_like(x)*np.mean(x)
                    y_pred = y                   
                    
                # If point cloud is 'normal', fit linear function
                else:
                    slope = dy/dx
                    intercept = y[0]-dy/dx*x[0]
                    x_pred = x
                    y_pred = [slope*xn + intercept for xn in x_pred]
                  
                return x_pred, y_pred, slope
                
            
            x_r, y_r, slope_r = slope(G, branch_r)
            x_g, y_g, slope_g = slope(G, branch_g) 
            x_b, y_b, slope_b = slope(G, branch_b)
            
                            
            # Convert slopes to angles              
            angle_r = np.degrees(np.arctan(slope_r))
            angle_g = np.degrees(np.arctan(slope_g))
            angle_b = np.degrees(np.arctan(slope_b))
            
    
            
            # Plot linear approximation to check angles
            if plot:  
                plt.plot(x_r, y_r, 'r', linewidth=2) 
                plt.plot(x_g, y_g, 'g', linewidth=2)
                plt.plot(x_b, y_b, 'b', linewidth=2)
            
            
            # Calculate differences in angles (from -180 to +180 degrees)
            def difference_between_angles(a0,a1):
                return abs((((2*a1-2*a0+540)%360)-180)/2)
            
            diff_rg = difference_between_angles(angle_r, angle_g)
            diff_rb = difference_between_angles(angle_r, angle_b)
            diff_gb = difference_between_angles(angle_g, angle_b)
            
            
            # Plot angles and differences as figure title
            if plot:
                plt.suptitle('Angles: Red: '     + str(round(angle_r)) +
                          ', Green: ' + str(round(angle_g)) +
                          ', Blue: '  + str(round(angle_b)))
                
                plt.title(' Minimum difference: RG: ' + str(round(diff_rg)) +
                          ', RB: ' + str(round(diff_rb)) +
                          ', GB: ' + str(round(diff_gb)))
            
            
            
            if split=='minimum':
                # Split y-node based on difference in angles
                # If angle between red and green branch is smallest, remove 
                # blue branch.
                if diff_rg < diff_rb and diff_rg < diff_gb:
                    if plot:
                        nx.draw_networkx_edges(G, 
                                    pos = nx.get_node_attributes(G, 'pos'),
                                    edgelist=[(node, true_neighbors[2])],
                                    edge_color="black",
                                    width=10)
                    G.remove_edge(node, true_neighbors[2])
                    
                # If angle between red and blue branch is lowest, remove green
                # branch.
                elif diff_rb < diff_rg and diff_rb < diff_gb:
                    if plot:
                        nx.draw_networkx_edges(G, 
                                    pos = nx.get_node_attributes(G, 'pos'),
                                    edgelist=[(node, true_neighbors[1])],
                                    edge_color="black",
                                    width=10)
                    G.remove_edge(node, true_neighbors[1])
                    
                # If angle between green and blue brach is smallest (and all 
                # equal cases), remove red branch.
                else:
                    if plot:
                        nx.draw_networkx_edges(G, 
                                pos = nx.get_node_attributes(G, 'pos'),
                                edgelist=[(node, true_neighbors[0])],
                                edge_color="black",
                                width=10)
                    G.remove_edge(node, true_neighbors[0])
                
                
                
                
                
                
            
            if split=='threshold':
                # Split y-node based on difference in angles based on threshold
                                
                if diff_rg < threshold:
                    if plot:
                        nx.draw_networkx_edges(G, 
                                            pos = nx.get_node_attributes(G, 'pos'),
                                            edgelist=[(node, true_neighbors[2])],
                                            edge_color="black",
                                            width=10)
                    G.remove_edge(node, true_neighbors[2])
                        
        
                if diff_rb < threshold:
                    if plot:
                        nx.draw_networkx_edges(G, 
                                            pos = nx.get_node_attributes(G, 'pos'),
                                            edgelist=[(node, true_neighbors[1])],
                                            edge_color="black",
                                            width=10)
                    G.remove_edge(node, true_neighbors[1])
                    
        
                if diff_gb < threshold:
                    if plot:
                        nx.draw_networkx_edges(G, 
                                            pos = nx.get_node_attributes(G, 'pos'),
                                            edgelist=[(node, true_neighbors[0])],
                                            edge_color="black",
                                            width=10)
                    G.remove_edge(node, true_neighbors[0])
                    
                    
    
                    
    
            if split=='custom':
                # If all very similar, split all
                if diff_rg < threshold and diff_rb < threshold and diff_gb < threshold:
                    if plot:
                        nx.draw_networkx_edges(G, 
                                            pos = nx.get_node_attributes(G, 'pos'),
                                            edgelist=[(node, true_neighbors[0]), (node, true_neighbors[1]), (node, true_neighbors[2])],
                                            edge_color="black",
                                            width=10)
                    G.remove_edge(node, true_neighbors[0])
                    G.remove_edge(node, true_neighbors[1])
                    G.remove_edge(node, true_neighbors[2])
                
                else:
                # Split y-node based on difference in angles
                # If angle between red and green branch is smallest, remove 
                # blue branch.
                    if diff_rg < diff_rb and diff_rg < diff_gb:
                        if plot:
                            nx.draw_networkx_edges(G, 
                                        pos = nx.get_node_attributes(G, 'pos'),
                                        edgelist=[(node, true_neighbors[2])],
                                        edge_color="black",
                                        width=10)
                        G.remove_edge(node, true_neighbors[2])
                        
                    # If angle between red and blue branch is lowest, remove green
                    # branch.
                    elif diff_rb < diff_rg and diff_rb < diff_gb:
                        if plot:
                            nx.draw_networkx_edges(G, 
                                        pos = nx.get_node_attributes(G, 'pos'),
                                        edgelist=[(node, true_neighbors[1])],
                                        edge_color="black",
                                        width=10)
                        G.remove_edge(node, true_neighbors[1])
                        
                    # If angle between green and blue brach is smallest (and all 
                    # equal cases), remove red branch.
                    else:
                        if plot:
                            nx.draw_networkx_edges(G, 
                                    pos = nx.get_node_attributes(G, 'pos'),
                                    edgelist=[(node, true_neighbors[0])],
                                    edge_color="black",
                                    width=10)
                        G.remove_edge(node, true_neighbors[0])                
              
                
              
                
              
            if split=='all':
                G.remove_edge(node, true_neighbors[0])
                G.remove_edge(node, true_neighbors[1])
          
            
          
            
            if plot:
                plt.show()
                
            # # Stop criteria to look at certain junctions
            # if count==29:
            #     break
        
    
   return G
    



def distance_between_points(p0, p1):
    """ Euclidian distance between two points (x0, y0), (x1, y1)
    
    Function used in edits.min_dist
    
    Parameters
    ----------
    p0 : tuple
        Point 0
    p1 : tuple
        Point 1
    
    Returns
    -------  
    distance : value, float
        Distance
    """    
    
    return math.sqrt((p0[1] - p1[1])**2+(p0[0] - p1[0])**2)



def min_dist(point, G):
    """ Compute minimum distance between a point and graph G.
    
    Parameters
    ----------
    
    point : tuple
        coordinates (x,y) of the point
    G : nx.graph
        Graph
    
    Returns
    -------  
    threshold : value
        Minimum distance 
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation 
    threshold = 1e8
    for node in G:
        if distance_between_points(point, G.nodes[node]['pos']) < threshold:
            threshold = distance_between_points(point, G.nodes[node]['pos'])
            
    return threshold



#******************************************************************************
# (2) EDGE METRICS
#******************************************************************************

def remove_self_edge(G):
    """ Remove self edges, e.g. (1,1), (3,3)
    
    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    G : nx.graph
        Graph modified
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    removals = []
    for edge in G.edges:
        if edge[0] == edge[1]:
            removals.append(edge)
            
    for edge in removals:
            G.remove_edge(*edge)
            #the * in the argument means that the function unpack the tuple edge
    return G





#******************************************************************************
# (3) COMPONENT EDITS
#******************************************************************************

def label_components(G):
    """ Label components of the network
    
    If the label exist already, the function will rewrite them.
    This is particulary usefull after filtering, when component have been deleted.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    G : nx.graph
        Graph modified
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation   
    for label, cc in enumerate(sorted(nx.connected_components(G))):
        for n in cc:
            G.nodes[n]['component'] = label
            
    return G




def select_components(G, components):
    """ Select component(s) from network
    Create a subgraph containing the given components and their attributes.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    components : list
        Label(s) of the component(s) to write in subgraph.
    
    Returns
    -------  
    K : nx.graph
        Subgraph that have been made.
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation    
    K = G.copy()
    if type(components) != list:
        selected_nodes = [n[0] for n in K.nodes(data=True) if n[1]['component'] == components]
    else:
        selected_nodes = [n[0] for n in K.nodes(data=True) if n[1]['component'] in components]
    K = K.subgraph(selected_nodes)
    
    return K




def remove_component(G, component):
    """ Remove a component from the network G.
    Return the modified network.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    component : int
        label of G to remove
    
    Returns
    -------  
    G : nx.graph
        Graph modified
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    for cc in sorted(nx.connected_components(G)):
        for n in cc:
            if G.nodes[n]['component'] == component:
                G.remove_node(n)
    return G




def remove_small_components(G, minimum_size=10):
    """ Remove the component below the minimum size
    
    The component size is the number of nodes which belongs to this component.
    
    This function remove the components of G, that have less than 'minimum_size' nodes.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    minimum_size : int
        Minimum size threshold for components
    
    Returns
    -------  
    G : nx.graph
        Graph modified
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(minimum_size, int), "Argument 'minimum_size' is not a integer"


    # Calculation
    for cc in sorted(nx.connected_components(G)):
        if len(cc) < minimum_size:
            for n in cc:
                G.remove_node(n)
                
    return G




def connect_components(G, cc0, cc1, relabel=True):
    """ Connect two components in network
    
    Parameters
    ----------
    G : nx.graph
        Graph
    cc0 : list
        Component 0
    cc1 : list
        Component 1
    relabel : bolean
        Whether to relabel components afterwards or not
    
    Returns
    -------  
    G : nx.graph
        Graph modified
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation 
    
    if 'edges' not in G.nodes[list(cc0)[0]]: #compute the attribute 'edges' in case it was not already done
        G=metrics.count_edges(G)
    
    edge0 = []
    for node in cc0:
        if G.nodes[node]['edges'] == 1: #find tip of cc0
            edge0.append(node)

    edge1 = []
    for node in cc1:
        if G.nodes[node]['edges'] == 1: #find tip of cc1
            edge1.append(node)

    value = 1000000

    for e0 in edge0:
        for e1 in edge1:
            distance = metrics.distance_between_nodes(G, e0, e1)
            if distance < value:
                value = distance
                ep0 = e0
                ep1 = e1

    G.add_edge(ep0, ep1)

    if relabel is True:
        for node in cc1:
            label = G.nodes[node]['component']

        for node in cc0:
            G.nodes[node]['component'] = label

    return G





def min_dist_comp(G, cc0, cc1):
    """ Compute minimum distance between two components
    
    Unit is pixels.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    cc0 : list
        Component 0
    cc1 : list
        Component 1
        
    Returns
    -------  
    threshold : float
        Minimum distance (unit pixels)
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    threshold = 1e6
    for n0 in cc0:
        for n1 in cc1:
            distance = metrics.distance_between_nodes(G, n0, n1)
            if distance < threshold:
                threshold = distance
                
    return threshold





def connect_close_components(G, value):
    """ Connect components which are closer than value
    
    Parameters
    ----------
    G : nx.graph
        Graph
    value : float
        Minimum distance in pixels
        
    Returns
    -------  
    H : nx.graph
        Graph modified
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(value, float), "Argument 'value' is not a float"

    # Calculation
    H=G.copy()
    for cc0 in sorted(nx.connected_components(G)):
        for cc1 in sorted(nx.connected_components(G)):
            if min_dist_comp(G, cc0, cc1) < value and (cc0 != cc1):
                H = connect_components(H, cc0, cc1)
                
    return H






def max_dist_comp(G, cc0, cc1):
    """ Compute maximum distance between components
    
    Parameters
    ----------
    G : nx.graph
        Graph
    cc0 : list
        Component 0
    cc1 : list
        Component 1
        
    Returns
    -------  
    threshold : float
        Maximum distance measured (unit pixels)
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    threshold = 0
    for n0 in cc0:
        for n1 in cc1:
            distance = metrics.distance_between_nodes(G, n0, n1)
            if distance > threshold:
                threshold = distance
    return threshold





def similarity_between_components(G, H):
    """ Similarity between components
    
    Average of the distance between nodes of the two graphs.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    H : nx.graph
        Graph
        
    Returns
    -------  
    value : float
        Similarity
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(H, nx.Graph), "H is not a NetworkX graph"

    # Calculation
    # Determine short and long graph
    if len(G.nodes) > len(H.nodes):
        short = H
        long = G
    else:
        short = G
        long = H

    N = len(long.nodes)
    distance = np.zeros((N))

    for n in range(N):
        distance[n] = min_dist(long.nodes[random.choice(list(long))]['pos'], short)

    return np.average(distance)





def assign_components(G, components):
    """ Assign a specified label to component attribute
    
    Parameters
    ----------
    G : nx.graph
        Graph
    components : list
        Components
        
    Returns
    -------  
    G : nx.graph
        Graph
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    for n, cc in enumerate(sorted(nx.connected_components(G))):
        for node in cc:
            G.nodes[node]['component'] = components[n]
    return G





def common_components(G, H):
    """ Common components
    
    Parameters
    ----------
    G : nx.graph
        Graph
    H : nx.graph
        Graph
        
    Returns
    -------  
    list : list
        List of common components
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(H, nx.Graph), "H is not a NetworkX graph"
    
    # Calculation
    C_G = metrics.get_component_labels(G)
    C_H = metrics.get_component_labels(H)
    
    return list(set(C_G) & set(C_H))





def unique_components(G_0, G_1):
    """ Unique components
    
    Parameters
    ----------
    G_0 : nx.graph
        Graph
    G_1 : nx.graph
        Graph
        
    Returns
    -------  
    list : list
        List of unique components
    """      
    
    # Assertions
    assert isinstance(G_0, nx.Graph), "G_0 is not a NetworkX graph"
    assert isinstance(G_1, nx.Graph), "G_1 is not a NetworkX graph"
    
    # Calculation   
    G_0_components = set(metrics.get_component_labels(G_0))
    G_1_components = set(metrics.get_component_labels(G_1))

    return ([item for item in G_0_components if item not in G_1_components],
            [item for item in G_1_components if item not in G_0_components])





def comp_to_fault(G):
    """ Define node attribute 'fault' as same as node 'component'

    Parameters
    ----------
    G : nx.graph
        Graph
        
    Returns
    -------
    G : nx.graph
        Graph modified
    """
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Computation
    for node in G:
        G.nodes[node]['fault'] = G.nodes[node]['component']
    return G


#******************************************************************************
# (4) NETWORK EDITS
# A couple of functions to edit the network
#******************************************************************************

def expand_network(G, relabel=True, vertical_shift=960, distance=5):
    """ Connect two components in network
    
    Parameters
    ----------
    G : nx.graph
        Graph
    relabel : bolean
        Relabel components
    vertical_shift : int
        Vertical shift applied to network
    distance : int
        Distance from edge
    
    Returns
    -------  
    G : nx.graph
        Graph
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation   
    def minimum_y(G, nodes):

        minimum = 1000000

        for node in nodes:
            y = G.nodes[node]['pos'][1]
            if y < minimum:
                minimum = y

        return minimum

    def maximum_y(G, nodes):

        maximum = -1000000

        for node in nodes:
            y = G.nodes[node]['pos'][1]
            if y > maximum:
                maximum = y

        return maximum

    def add_y(G, nodes, value):

        for node in nodes:
            G.nodes[node]['pos'] = (
                G.nodes[node]['pos'][0],
                G.nodes[node]['pos'][1] + value
            )

    def within_reach(G, cc0, cc1, value):

        min_dist = 10000000

        for n0 in cc0:
            for n1 in cc1:
                distance = metrics.distance_between_nodes(G, n0, n1)
                if distance < min_dist:
                    min_dist = distance

        if min_dist < value:
            return True
        else:
            return False

        return G

    # Expand network upwards
    for n_cc, cc in enumerate(sorted(nx.connected_components(G))):
        #n_cc = component index (~label); cc = set of nodes of this component

        # Component reaches lower, but not upper boundary
        if minimum_y(G, cc) == 0 and maximum_y(G, cc) != 959:

            # Move component up
            print('Move component ' + str(n_cc) + ' up')
            add_y(G, cc, vertical_shift)

            # Connect to other component
            for n_cc_other, cc_other in enumerate(
                sorted(nx.connected_components(G))
            ):

                # if it is within reach
                if within_reach(G, cc, cc_other, distance) and cc != cc_other:
                    print('Connect component %i to %i' % (n_cc, n_cc_other))
                    G = connect_components(G, cc, cc_other, relabel)

    # Expand network downwards
    for n_cc, cc in enumerate(sorted(nx.connected_components(G))):

        # Component reaches upper, but not lower boundary
        if minimum_y(G, cc) != 0 and maximum_y(G, cc) == 959:

            # Move component down
            print('Move component ' + str(n_cc) + ' down')
            add_y(G, cc, -vertical_shift)

            # Connect to other component
            for n_cc_other, cc_other in enumerate(
                sorted(nx.connected_components(G))
            ):

                # if it is within reach
                if within_reach(G, cc, cc_other, distance) and cc != cc_other:
                    print('Connect component %i to %i' % (n_cc, n_cc_other))
                    G = connect_components(G, cc, cc_other, relabel)

    return G





def simplify(G, degree, remember=False):
    """ Simplify network to a degree
    
    This function allows to describe a component with less nodes.
    It is useful to decrease the computation time because the variable take
    less space of RAM. This smooth the component.
    
    It remove one node out of 'degree'.
    For example degree=3 means that it keep one node ouf of 3 on the component.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    degree : int
        Degree of simplification
    remember : bool
        Whether the function has to remember which node has been deleted.
    
    Returns
    -------  
    If remember == False, it returns H ie. the sumplified Graph (nx.graph)
    If remember == True, it returns H, list_removed. H is the simplified Graph
    list_removed is the list including the nodes removed during the algorithm
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(degree, int), "Argument 'degree' is not a integer"

    # Computation
    if remember==True:
        list_removed=[]
    H = G.copy()
    for _ in range(degree):
        for n, node in enumerate(list(nx.dfs_preorder_nodes(H))):
            # print(H.degree(node))
            if H.degree(node) == 2 and (n % 2) == 0:
                edges = list(H.edges(node))
                H.add_edge(edges[0][1], edges[1][1])
                if remember==True:
                    list_removed.append(node)
                H.remove_node(node)
    if remember==True:
        return H,list_removed
    return H


def split_graph_by_polarity(G):
    """ Split network by polarity
    
    G_0 get the nodes with polarity==0
    G_1 get the nodes with polarity!=0
    
    Parameters
    ----------
    G : nx.graph
        Graph

    Returns
    -------  
    G_0 : nx.graph
        Graph
    G_1 : nx.graph
        Graph
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    G_0 = G.copy()
    G_1 = G.copy()
    for node in G.nodes:
        if G.nodes[node]['polarity'] == 0:
            G_1.remove_node(node)
        else:
            G_0.remove_node(node)
            
    return G_0, G_1




def similarity_between_graphs(G, H, normalize=True):
    """ Similarity between components of network
    
    Compare the position of the nodes between G and H.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    H : nx.graph
        Graph
    normalize : bolean
        Normalize similarity matrix
    
    Returns
    -------  
    matrix : np.array
        Similarity matrix
    components_G : list
        Int
    components_H : list
        Int        
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(H, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    components_G = sorted(metrics.get_component_labels(G))  # components undefined!?
    components_H = sorted(metrics.get_component_labels(H))

    matrix = np.zeros((len(components_G), len(components_H)))

    for n, c_G in enumerate(components_G):
        cc_G = select_components(G, components=c_G)

        for m, c_H in enumerate(components_H):
            cc_H = select_components(H, components=c_H)

            matrix[n, m] = similarity_between_components(cc_G, cc_H)

    if normalize:
        minimum = np.min(matrix)
        maximum = np.max(matrix)

        matrix = (matrix-minimum)/(maximum-minimum)

    return matrix, components_G, components_H



def similarity_to_connection(matrix, rows, columns, threshold):
    """ Convert similarity to connections
    
    Compute connections from similarity

    Parameters
    ----------
    matrix : np.array
        Similarity matrix
    rows : list
        Components 0
    columns : list
        Components 1
    threshold : float
        Threshold for connections
        distance in pixel
    
    Returns
    -------  
    connections : list (tuples)
        Connections between components       
    """     

    # Calculation
    connections = []
    for col in range(matrix.shape[0]):
        for row in range(matrix.shape[1]):
            if matrix[col, row] < threshold:
                connections.append([columns[row], rows[col]])
                
    return connections





def relabel(G, connections, count):
    """ Relabel components of network based on connections
    
    Parameters
    ----------
    G : nx.graph
        Graph
    connections : list (tuples)
        Connections between components  
    count : int
        Maximum label
    
    Returns
    -------  
    G : nx.graph
        Graph
    count : int
        Maximum label
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    sources = np.zeros((len(connections)), int)
    targets = np.zeros((len(connections)), int)

    for n, connection in enumerate(connections):
        sources[n] = connection[0]
        targets[n] = connection[1]

    highest_index = max(np.max(sources), count)

    components_old = metrics.get_component_labels(G)
    components_new = [None] * len(components_old)

    for n in range(len(components_old)):
        component = components_old[n]
        if component in sources:
            index = np.where(sources == component)[0][0]
            components_new[n] = targets[index]
        else:
            components_new[n] = highest_index + 1
            highest_index += 1

    G = assign_components(G, components_new)

    return G, count



def combine_graphs(G, H):
    """ Combine two graphs
    
    Parameters
    ----------
    G : nx.graph
        Graph
    H : nx.graph
        Graph
    
    Returns
    -------  
    F : nx.graph
        Graph result from the combination of G and H   
    """     

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(H, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    highest_node = max(list(G.nodes)) + 1
    nodes_new = [node + highest_node for node in H.nodes]

    mapping = dict(zip(H.nodes, nodes_new))
    H = nx.relabel_nodes(H, mapping)

    F = nx.compose(G, H)

    return F


def single_shape_nodes_tip_to_tip (G, plot=False,axs=[],background=None):
    """ Get the list of nodes and nodes coordinates, ordered tip to tip of the component.
    

    Parameters
    ----------
    K : nx.Graph
        K is a subgraph containing only one component.
    plot : bool, optional
        True if you want to plot the structure tip to tip
        The default is False.
    axs : matplotlib axis, optional        
        The default is [].
    background : 2D-numpy array, optional
        The default is None.

    Returns
    -------
    list_node : list
        list of ordered nodes tip to tip
    list_node_coord : list
        list of tuple of coordinates of the ordered nodes.

    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "K is not a NetworkX graph"
    
    list_comp=[]
    for node in G.nodes: 
        list_comp.append(G.nodes[node]['component'])
        
    def checkList(list):
        return len(set(list)) == 1
    
    assert checkList(list_comp)==True,'The graph K is composed of more than one label.'
    
    assert len(background.shape)==2 , "The shape of the background array is not 2D"
        
    
    # Computation
    list_tip=find_tips(G)
    already_view=[]
    node_here=list_tip[0]
    list_node=[node_here]
    list_node_coord=[metrics.coord_node(G,node_here)]
    next_node=node_here
    i=0
    
    if plot==True:
        if axs == []:
            fig, axs = plt.subplots(1, 1, figsize=(12,12)) #init figure
        axs.imshow(background,cmap=cm.gray_r,alpha=0.7)
    
    while G.degree(next_node) != 1 or i ==0:
        already_view.append(node_here)
        next_node=find_next_node(G,node_here,already_view)
        node_here=next_node
        list_node.append(node_here)
        list_node_coord.append(metrics.coord_node(G,node_here))

        if plot==True:           
            x,y=metrics.coord_node(G,node_here)
            axs.plot(x,y,'bo', ms=1.2)
            axs.text(x,y,str(i))

        i=i+1
        
    return list_node, list_node_coord


def cut_U_bend_curvature (J,ignoring,curvature_threshold,path_way=0):
    """  Cut the component that form a U shape
    
    Parameters
    ----------
    J : nx.Graph
        Graph 
    ignoring : int
        number of node to ignore between the nodes forming the angle.
    curvature_threshold : float
        angle below which the function cut the U shape
    path_way : binary 0 or 1.
        The default is 0.
        Component path direction from a tip to the other A to B or B to A. 
        Better to always check on both directions.

    Returns
    -------
    J : nx.Graph 
        Graph without the points at the bottom of the U.
    list_removed : list
        list of nodes that have been removed.

    """
    # Assertions
    assert isinstance(J, nx.Graph), "J is not a NetworkX graph"

    list_removed=[]

    list_compo_long=metrics.get_component_labels(J)
    list_compo=[]
    [list_compo.append(x) for x in list_compo_long if x not in list_compo]


    for label in list_compo:
        nodes = [node for node in J if J.nodes[node]['component']==label]
        K=nx.subgraph(J,nodes)
        endpoints = [node for node in K if K.degree[node]==1]

        if len(endpoints)==2: 
            if (ignoring==0 and len(K.nodes)>=3) or (ignoring==1 and len(K.nodes)>=5) or (ignoring ==2 and len(K.nodes)>=7):
                
                if path_way == 0:
                    path0 = nx.shortest_simple_paths(K, source=endpoints[0], target=endpoints[1])
                    path_list0 = list(path0)[0]
            
                elif path_way == 1:
                    path0 = nx.shortest_simple_paths(K, source=endpoints[1], target=endpoints[0])
                    path_list0 = list(path0)[0]
                else: return 'invalid path direction, the argument is 0 or 1'

                if ignoring!=0:
                    nb_angles=(len(path_list0)-1) - 2 -2*ignoring
                else:
                    nb_angles=len(path_list0)-3
        
                angles=np.zeros((nb_angles,1))
                bend=np.zeros((nb_angles,4))
        
                for n in range(nb_angles):
        
                    node0 = path_list0[n]
                    node1 = path_list0[n+1 + ignoring]
                    node2 = path_list0[n+2 + 2*ignoring]
        

                    length_ed0  = metrics.distance_between_nodes(K, node0, node1)
                    length_ed1  = metrics.distance_between_nodes(K, node1, node2)
        
                    top_v=metrics.distance_between_nodes(K, node0, node2)
                    alpha = np.rad2deg ( np.arccos ( (np.square(length_ed0) + np.square(length_ed1) - np.square(top_v))
                                          / (2*length_ed0*length_ed1) ) )
                    # This formula is the application of the Al-Kashi theorem (law of cosine)

                    angles[n,0]=alpha
        
                    bend[n,0]=node0
                    bend[n,1]=node1
                    bend[n,2]=node2
                    bend[n,3]=alpha  
   
                for n in range(nb_angles):
                    if bend[n,3]<curvature_threshold:
                        list_removed.append(int(bend[n,1]))
                        J.remove_node(int(bend[n,1]))
                        
    return J,list_removed


def loop_cut_U_global (I, img_dem, curvature_threshold=110, plot=False):
    """ Cut the U shape in Graph. 
    Ignore first one point, then 2, in both directions of components.

    Parameters
    ----------
    I : TYPE
        DESCRIPTION.
    img_dem : TYPE
        DESCRIPTION.
    curvature_threshold : TYPE, optional
        DESCRIPTION. The default is 110.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    J : TYPE
        DESCRIPTION.

    """
    J=I.copy()

    ignoring=1
    J,list_removed1=cut_U_bend_curvature(J,ignoring=1,curvature_threshold=curvature_threshold, path_way=0)

    if plot==True:
        fig, axs = plt.subplots(1, 1, figsize=(12,12), num ='ignoring=1, path_way=0')
        axs.imshow(img_dem,cmap='gray_r')
        plots.plot_components(J,  label=True, ax=axs,node_size=0.5)
        for i in range(len(list_removed1)):
            x1, y1 = I.nodes[list_removed1[i]]['pos']
            axs.plot(x1, y1, 'ro')


    ignoring=1
    J,list_removed2=cut_U_bend_curvature(J,ignoring=1,curvature_threshold=curvature_threshold, path_way=1)

    if plot==True:
        fig, axs = plt.subplots(1, 1, figsize=(12,12), num ='ignoring=1, path_way=1')
        axs.imshow(img_dem,cmap='gray_r')
        plots.plot_components(J,  label=True, ax=axs,node_size=0.5)

        for i in range(len(list_removed2)):
            x1, y1 = I.nodes[list_removed2[i]]['pos']
            axs.plot(x1, y1, 'ro')


    ignoring=2 #
    J,list_removed3=cut_U_bend_curvature(J,ignoring=2,curvature_threshold=curvature_threshold, path_way=0)

    if plot==True:
        fig, axs = plt.subplots(1, 1, figsize=(12,12), num ='ignoring=2 , path_way=0')
        axs.imshow(img_dem,cmap='gray_r')
        plots.plot_components(J,  label=True, ax=axs,node_size=0.5)

        for i in range(len(list_removed3)):
            x1, y1 = I.nodes[list_removed3[i]]['pos']
            axs.plot(x1, y1, 'ro')


    ignoring=2 #
    J,list_removed4=cut_U_bend_curvature(J,ignoring=2,curvature_threshold=curvature_threshold, path_way=1)

    if plot==True:
        fig, axs = plt.subplots(1, 1, figsize=(12,12), num ='ignoring=2, path_way=1')
        axs.imshow(img_dem,cmap='gray_r')
        plots.plot_components(J,  label=True, ax=axs,node_size=0.5)
        for i in range(len(list_removed4)):
            x1, y1 = I.nodes[list_removed4[i]]['pos']
            axs.plot(x1, y1, 'ro')

    return J

