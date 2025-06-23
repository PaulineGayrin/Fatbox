#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MODULE plots

# This file contains a series of function to plot fault networks (graphs). 
# This includes functions for 
# (1) various needs of the plotting functions
# (2) plotting arrays
# (3) plotting networks
# (4) plotting structural analysis

"""


# Packages
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from PIL import Image
from pathlib import Path
import os

# Fatbox
path_file=Path(__file__).absolute()
path_folder=path_file.parent
#print(path_folder)
os.chdir(path_folder)

import preprocessing
import metrics
import edits
import utils 
import structural_analysis


#******************************************************************************
# (1) Useful functions
# A couple of helper functions
#******************************************************************************

cmap = colors.ListedColormap(
    [
        '#ffffffff', '#64b845ff', '#9dcd39ff',
        '#efe81fff', '#f68c1bff', '#f01b23ff'
    ]
)


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks

    Parameters
    ----------
    nlabels : int
        Number of labels (size of colormap)
    type : str, optional
        'bright' for strong colors, 'soft' for pastel colors
        The default is 'bright'.
    first_color_black : bool, optional
        Option to use first color as black, True or False. 
        The default is True.
    last_color_black : bool, optional
        Option to use last color as black
        The default is False.
    verbose : bool, optional
        Prints the number of labels and shows the colormap.
        The default is True.

    Returns
    -------
    random_colormap : colormap
       colormap for matplotlib

    """

    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


def get_node_colors(G, attribute,  node_palette=None, return_palette=False):
    """ Get node attribute colors for plotting.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute name
    node_palette: str or None
        Name of the seaborn palette of your choice.
    
    Returns
    -------  
    array : array
        Node colors
    """

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    

    # Calculation
    n_comp  = 100000
    palette = sns.color_palette(node_palette, n_comp)
    palette = np.array(palette)
    
    # Shuffle
    perm = np.arange(palette.shape[0])
    np.random.seed(42)
    np.random.shuffle(perm)
    palette = palette[perm]
    
    node_color = np.zeros((len(G), 3))

    for n, node in enumerate(G):
        color = palette[G.nodes[node][attribute]]
        node_color[n, 0] = color[0]
        node_color[n, 1] = color[1]
        node_color[n, 2] = color[2]

    if return_palette:
        return node_color, palette
    else:
        return node_color





def get_edge_colors(G, attribute, return_palette=False):
    """ Get edge attribute colors for plotting 
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute name
    
    Returns
    -------  
    array : array
        Node colors
    """

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    n_comp = 10000
    palette = sns.color_palette('husl', n_comp)
    palette = np.array(palette)
    
    # Shuffle
    perm = np.arange(palette.shape[0])
    np.random.seed(42)
    np.random.shuffle(perm)
    palette = palette[perm]
    

    for n, edge in enumerate(G.edges):
        color = palette[G.edges[edge][attribute]]
        edge_color[n, 0] = color[0]
        edge_color[n, 1] = color[1]
        edge_color[n, 2] = color[2]

    if return_palette:
        return edge_color, palette
    else:
        return edge_color




def plot_matrix(matrix, rows, columns, threshold):
    """ Plot similarity matrix
    
    Parameters
    ----------
    matrix : np.array
        Matrix to plot
    rows : np.array
        Rows
    columns : np.array
        Columns
    threshold : float
        Threshold
        
    Returns
    -------  
    """
    
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(matrix, 'Blues_r')

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticklabels(columns)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(rows)

    ax.set_xlim(-0.5, matrix.shape[1]-0.5)
    ax.set_ylim(-0.5, matrix.shape[0]-0.5)

    # Loop over data dimensions and create text annotations.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] < threshold:
                ax.text(
                    j, i, round(matrix[i, j], 3),
                    ha="center", va="center", color="r"
                )
            else:
                ax.text(
                    j, i, round(matrix[i, j], 3),
                    ha="center", va="center", color="k"
                )





def plot_connections(matrix, rows, columns):
    """ Plot connections
    
    Parameters
    ----------
    matrix : np.array
        Matrix to plot
    rows : np.array
        Rows
    columns : np.array
        Columns
        
    Returns
    -------  
    None.
    """
    
    # Plotting
    for n in range(100):
        threshold = n/100
        connections = edits.similarity_to_connection(
            matrix, rows, columns, threshold)
        plt.scatter(threshold, len(connections), c='red')
        plt.xlabel('Threshold')
        plt.ylabel('Number of connections')
        
        
#******************************************************************************
# (2) Array plotting
# A couple of functions to visualize arrays
#******************************************************************************

def plot_overlay(label, image, **kwargs):
    """ Plot a label array onto of image
    
    Parameters
    ----------
    label : np.array
        Label
    image : np.array
        Image
    
    Returns
    ------- 
    None. Plot generated.
    
    """

    label = (label-np.min(label))/(np.max(label)-np.min(label))

    label_rgb = np.zeros((label.shape[0], label.shape[1], 4), 'uint8')
    label_rgb[:, :, 0] = 255 - 255*label
    label_rgb[:, :, 1] = 255 - 255*label
    label_rgb[:, :, 2] = 255 - 255*label
    label_rgb[:, :, 3] = 255*label

    overlay = Image.fromarray(label_rgb, mode='RGBA')

    image = (image-np.min(image))/(np.max(image)-np.min(image))

    background = Image.fromarray(np.uint8(cmap(image)*255))

    background.paste(overlay, (0, 0), overlay)

    plt.figure()
    plt.imshow(background, **kwargs)




def plot_comparison(data_sets, **kwargs):
    """ Plot a couple of images for comparison
    
    Parameters
    ----------
    data_sets : list of np.array
        List of data sets       
    
    Returns
    -------  
    None.
    
    """
    count = len(data_sets)

    fig, axs = plt.subplots(count, 1, figsize=(12, 12))
    
    for n, data in enumerate(data_sets):
        axs[n].imshow(data, **kwargs)
        





def plot_threshold(data, threshold, value, **kwargs):
    """
    Plot threshold array next to data.

    Parameters
    ----------
    data : array
    threshold : int
    value : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    # First plot
    p0 = axs[0].imshow(data)

    # Color bar locator
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cb0 = fig.colorbar(p0, ax=axs[0], cax=cax)
    cb0.ax.plot([-1, 1], [value]*2, 'r')

    # Second plot
    p1 = axs[1].imshow(threshold, **kwargs)

    # Color bar locator
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cb0 = fig.colorbar(p1, ax=axs[1], cax=cax)






#******************************************************************************
# (3) Network plotting
# A couple of functions to visulize networks
#******************************************************************************

def plot(G, **kwargs):
    """ Draw network G
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis
    color : str
        Color of network
    with_labels : bolean
        Whether to plot labels
    
    Returns
    -------  
    None. Plot the network.
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            **kwargs)

def plot_components(G, label=True, **kwargs):
    """ Plot network components
    Color by component.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis
    node_size : float
        Size of network nodes
    label : bolean
        Whether to plot labels
    node_palette : str or None
        Name of the seaborn palette of your choice.

    Returns
    -------  
    None. Return the plot
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting    
    #node_color, palette = get_node_colors(G, 'component', kwargs['node_palette'], return_palette=True)
    node_color, palette = get_node_colors(G, 'component', return_palette=True)
    if 'ax' in kwargs: 
        ax=kwargs['ax']
        nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_color=node_color,
            node_size=kwargs['node_size'],ax=ax)
        ax.axis('on')  # turns on axis
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    else: 
        nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_color=node_color,
            node_size=kwargs['node_size'])
        
    if label is True:

        for cc in sorted(nx.connected_components(G)):
            # Calculate centre
            x_avg = 0
            y_avg = 0

            for n in cc:
                y_avg = y_avg + G.nodes[n]['pos'][0]
                x_avg = x_avg + G.nodes[n]['pos'][1]

            N = len(cc)
            y_avg = y_avg/N
            x_avg = x_avg/N

            # Scale color map
            label = G.nodes[n]['component']
            if 'ax' in kwargs:
                ax.text(y_avg, x_avg, label, fontsize=15,
                    color=palette[G.nodes[n]['component']])


def plot_faults(G, label=True, **kwargs):
    """ Plot network faults
    Color by faults.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis
    node_size : float
        Size of network nodes
    label : bolean
        Whether to plot labels
    filename : str
        Save figure with this name
    
    Returns
    -------  
    None. Return the plot
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    node_color, palette = get_node_colors(G, 'fault', return_palette=True)

    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_color=node_color,
            **kwargs)

    ax=kwargs['ax']
    if label is True:
        
        labels = metrics.get_fault_labels(G)
        
        for l in labels:
            fault = metrics.get_fault(G, l)
            
            # Calculate centre
            x_avg = 0
            y_avg = 0

            for n in fault:
                y_avg = y_avg + G.nodes[n]['pos'][0]
                x_avg = x_avg + G.nodes[n]['pos'][1]

            N = len(fault.nodes)
            y_avg = y_avg/N
            x_avg = x_avg/N

            # Scale color map
            label = G.nodes[n]['fault']

            ax.text(y_avg, x_avg, label, fontsize=15,
                    color=palette[G.nodes[n]['fault']])

    ax.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True, top=False, labelleft=True, labelbottom=True, labeltop=False)    




def plot_attribute(G, attribute, colorbar=True, **kwargs):
    """ Plot network node attribute
    Color nodes by node attribute.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute used for plotting
    ax : plt axis
        Axis
    vmin : float
        Minium value
    vmax : float
        Maximum value
    node_size : float
        Size of network nodes        
    filename : str
        Save figure with this name
    
    Returns
    -------  
    None. Return the plot.
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    assert (attribute in G.nodes[list(G.nodes(0))[0][0]]) == True , \
    'The attribute '+attribute+' is not in the nodes of the Graph. Check that you have already computed the attribute.'

    # Plotting
    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_color=np.array([G.nodes[node][attribute] for node in G]),
            node_size=0.75,
            **kwargs)

    ax = kwargs['ax']
    ax.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    if 'cmap' in kwargs:
      cmap = kwargs['cmap']
    else:
      cmap = plt.get_cmap('viridis')

    if 'vmin' in kwargs:
      vmin = kwargs['vmin']
    else:
        vmin = metrics.compute_node_values(G, attribute, 'min')

    if 'vmax' in kwargs:
      vmax = kwargs['vmax']
    else:
        vmax = metrics.compute_node_values(G, attribute, 'max')

    if colorbar:
      sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
      sm.set_array([])

      cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
      cbar.ax.set_ylabel(attribute, rotation=270)







def plot_edge_attribute(G, attribute, **kwargs):
    """ Plot network edge attribute
    Color edges by edge attribute.
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute used for plotting
    ax : plt axis
        Axis      
    filename : str
        Save figure with this name
    
    Returns
    -------  
    None. Return the plot.
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    assert (attribute in G.edges[list(G.edges(0))[0]]) == True , \
    'The attribute '+attribute+' is not in the edges of the Graph. Check that you have already computed the attribute.'

    # Plotting
    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_size=0.001,
            **kwargs)

    nx.draw_networkx_edges(G,
                           pos=nx.get_node_attributes(G, 'pos'),
                           edge_color=np.array([G.edges[edge][attribute] for edge in G.edges]),
                           **kwargs)
    ax = kwargs['ax']
    ax.axis('on')

    # Colorbar
    cmap = plt.cm.twilight_shifted
    vmax = metrics.compute_edge_values(G, attribute, 'max')
    vmin = metrics.compute_edge_values(G, attribute, 'min')

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(attribute, rotation=270)


def plot_edge_attribute_other_version(G, attribute, ax=[]):
    """ Plot the attribute of G with a colorbar.
    The colors are unique to each attribute to show better the properties.
    
    Function optimal for DEM processing.
    
    Parameters
    ----------
    G : nx.Graph
        Graph
    attribute : str
        'strike' or 'extension' or 'throw' or 'displacement' or 'natural dip'
    ax : matplotlib.axis, optional
        Axis. 
        The default is [].

    Returns
    -------
    None. Return the plot.
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    # assert (attribute in G.nodes[list(G.nodes(0))[0][0]]) == True , \
    #     'The attribute '+attribute+' is not in the nodes of the Graph. Check that you have already computed the attribute.'

    
    if ax==[]:
        fig, ax = plt.subplots()


    # Each possible attribute get a title and a colormap
    if attribute=='strike':
        cmap=plt.cm.twilight
        vmin = metrics.compute_edge_values(G, attribute, 'min')
        title = 'Strike [°]'

    elif attribute=='extension':
        cmap=plt.cm.Blues
        vmin = metrics.compute_edge_values(G, attribute, 'min_but_0')
        # remplacer cette fonction par le 2e plus petit
        title = 'Extension [m]'

    elif attribute=='throw':
        cmap=plt.cm.YlOrBr
        vmin = metrics.compute_edge_values(G, attribute, 'min_but_0')
        title = 'Throw [m]'

    elif attribute=='displacement':
        cmap=plt.cm.YlGn
        vmin = metrics.compute_edge_values(G, attribute, 'min')
        title = 'Displacement [m]'

    elif attribute=='natural_dip':
        cmap=plt.cm.Spectral
        vmin = metrics.compute_edge_values(G, attribute, 'min')
        title = 'Dip measured [°]'
        
    
    # Draw the network with colors 
        
    # for edge in G.edges:
    #     print(G.edges[edge])
    #     print(edge)
        
    #     print(G.edges[edge][attribute])
    # #
    nx.draw_networkx_edges(G,
                           pos = nx.get_node_attributes(G, 'pos'),
                           edge_color = np.array([G.edges[edge][attribute] for edge in G.edges]),
                           edge_cmap=cmap,
                           ax=ax)

    vmax = metrics.compute_edge_values(G, attribute, 'max')


    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04,ax=ax)
    cbar.ax.set_ylabel(title, rotation=270,labelpad=15)






def cross_plot(G, var0, var1, **kwargs):
    """ Cross-plot two network (node) properties
    
    Parameters
    ----------
    G : nx.graph
        Graph
    var0 : str
        Attribute to plot as x-axis
    var0 : str
        Attribute to plot as y-axis  
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    x = np.zeros(len(G.nodes))
    z = np.zeros(len(G.nodes))

    if var0 == 'x':
        for n, node in enumerate(G):
            x[n] = G.nodes[node]['pos'][1]
            z[n] = G.nodes[node][var1]

    if var0 == 'z':
        for n, node in enumerate(G):
            x[n] = G.nodes[node]['pos'][0]
            z[n] = G.nodes[node][var1]

    if var1 == 'x':
        for n, node in enumerate(G):
            x[n] = G.nodes[node][var0]
            z[n] = G.nodes[node]['pos'][1]

    if var1 == 'z':
        for n, node in enumerate(G):
            x[n] = G.nodes[node][var0]
            z[n] = G.nodes[node]['pos'][0]

    plt.plot(x, z, '.', **kwargs)





def plot_compare_graphs(G, H, **kwargs):
    """ Plot two graphs for comparison
    
    Parameters
    ----------
    G : nx.graph
        Graph
    H : nx.graph
        Graph 
    
    Returns
    -------  
    None. Return the plots.
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    assert isinstance(H, nx.Graph), "H is not a NetworkX graph" 

    # Plotting
    fig, ax = plt.subplots(2, 1)
    plot_components(G, ax=ax[0], **kwargs)
    plot_components(H, ax=ax[1], **kwargs)









#******************************************************************************
# (3) Fault evolution plots
# A couple of functions to visualize fault network properties
#******************************************************************************

def plot_rose_diagram(G, ax=[], title='Rose plot', color=False):
    """
    Plot the circular histogram of the azimuth of the edges in the fault network.

    The rose plot shows a statistical view on the strike parameter.
    The edges strike are gathered in an histogram with their
    edges length as weight. 

    Parameters
    ----------
    G : nx.Graph
    ax : plt axis
        Axis
    title : str
        Title of the plot
    color : bool, optional, default is False.
        whether to color the histogram 
        
    Returns
    -------
    None. Return the plot.
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    assert ('strike' in G.edges[list(G.edges)[0]]) == True , \
    'The attribute strike is not in the edges of the Graph. Check that you have already computed the strike.'
    assert ('length' in G.edges[list(G.edges)[0]]) == True , \
    'The attribute length is not in the edges of the Graph. Check that you have already computed the length.'

    
    # Getting the data
    strikes = np.zeros(len(G.edges))
    lengths = np.zeros(len(G.edges))


    for n, edge in enumerate(G.edges):
        strikes[n] = G.edges[edge]['strike']
        lengths[n] = G.edges[edge]['length']


    # ROSE PLOT
    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(strikes, bin_edges, weights = lengths)
    number_of_strikes[0] += number_of_strikes[-1]
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])


    cmap = plt.cm.BrBG(np.concatenate((np.linspace(0, 1, 18), np.linspace(0, 1, 18)), axis=0))

    if color==True:
        c=cmap
        
        # Possibility of Orange Blue color as Thilo used
        # top = cm.get_cmap('Oranges_r', 128) # r means reversed version
        # bottom = cm.get_cmap('Blues', 128)# combine it all
        # newcolors = np.vstack((top(np.linspace(0, 1, 128)),
        #                         bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
        # orange_blue = ListedColormap(newcolors, name='OrangeBlue')
        
        # cmap = orange_blue(np.concatenate((np.linspace(0, 1, 18), np.linspace(0, 1, 18)), axis=0))
        
    else:
        c=".8"

    if ax==[]:
        fig = plt.figure(figsize=(8,8))

        ax = fig.add_subplot(111, projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))

    ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves,
            width=np.deg2rad(10), bottom=0.0, color=c, edgecolor='k')
    #ax.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight= 'black')
    ax.set_title(title, y=1.10, fontsize=15)
    
    return ax


def bar_plot(attribute, faults, times, steps=[], ax=[]):
    """ Bar plot of fault network attribute
    Used for numerical models
    
    Parameters
    ----------
    attribute : np.array
        Attribute to plot
    faults : np.array
        Fault labels
    times : np.array
        Times used for plotting
    
    Returns
    -------  
    None.
    """
    
    # Plotting
    colors = utils.get_colors

    if ax == []:
        fig, ax = plt.subplots()

    if steps == []:
        steps = range(attribute.shape[1])

    for n, step in enumerate(steps):
        bottom = 0
        for m, fault in enumerate(faults[:, step]):
            if np.isfinite(fault):
                a = attribute[m, step]
                ax.bar(n, a, 1, bottom=bottom, alpha=0.75,
                       edgecolor='white', color=colors[int(fault), :])
                bottom += a
            else:
                break





def stack_plot(attribute, faults, times, steps=[], ax=[]):
    """ Stack plot of fault network attribute
    Used for numerical models
    
    Parameters
    ----------
    attribute : np.array
        Attribute to plot
    faults : np.array
        Fault labels
    times : np.array
        Times used for plotting
    
    Returns
    -------  
    None.
    """
    
    # Plotting
    colors = utils.get_colors()

    if ax == []:
        fig, ax = plt.subplots()

    if steps == []:
        steps = range(attribute.shape[1])

    max_fault = int(np.nanmax(faults))

    x = np.arange(len(steps))
    y = np.zeros((max_fault, len(steps)))

    for N in range(max_fault):
        for n in steps:
            row = faults[:, n]
            if N in faults[:, n]:
                index = np.where(row == N)[0][0]
                y[N, n] = attribute[index, n]

    ax.stackplot(x, y, fc=colors[:max_fault, :],
                 alpha=0.75, edgecolor='white', linewidth=0.5)






def plot_width(G, ax, width, tips=True, plot=False):
    """ Plot edge width of fault network
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis
    width : np.array
        Width of network edges
    tips : bolean
        Plot tips
    plot : False
        Plot helper functions
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    pos = nx.get_node_attributes(G, 'pos')

    n_comp = 10000

    sns.color_palette(None, 2*n_comp)

    colors = get_node_colors(G, 'fault')

    def get_points(u):

        u0 = np.array(pos[u[0]])
        u1 = np.array(pos[u[1]])

        u_vec = u0-u1

        u_perp = np.array([-u_vec[1], u_vec[0]])
        u_perp = u_perp/np.linalg.norm(u_perp)

        u0a = u0 - u_perp*width[u[0]]
        u0b = u0 + u_perp*width[u[0]]

        u1a = u1 - u_perp*width[u[1]]
        u1b = u1 + u_perp*width[u[1]]

        return u0a, u0b, u1a, u1b

    def get_intersect(a1, a2, b1, b2):
        """
        Returns the point of intersection of the lines passing through a2,a1
        and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1, a2, b1, b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return (float('inf'), float('inf'))
        return np.array([x/z, y/z])

    def clockwiseangle_and_distance(origin, point):
        refvec = [0, 1]
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod = normalized[0]*refvec[0] + \
            normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - \
            refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to
        # subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance
        # should come first.
        return angle, lenvector

    def get_edges(G, node):
        neighbors = list(G.neighbors(node))
        pts = [G.nodes[neighbor]['pos'] for neighbor in neighbors]
        pts, neighbors = zip(
            *sorted(
                zip(pts, neighbors),
                key=lambda x: clockwiseangle_and_distance(
                    G.nodes[node]['pos'], x[0])
                )
            )
        edges = [(node, neighbor) for neighbor in neighbors]
        return edges

    for node, color in zip(G, colors):
        if tips is True and G.degree(node) == 1:

            edge = get_edges(G, node)[0]

            node0 = np.array(pos[edge[0]])
            node1 = np.array(pos[edge[1]])

            vec = node0-node1
            vec_perp = np.array([-vec[1], vec[0]])
            vec_perp = vec_perp/np.linalg.norm(vec_perp)

            vec_pos = node0 + vec_perp*width[edge[0]]
            vec_neg = node0 - vec_perp*width[edge[0]]

            stack = np.vstack((vec_pos,
                               node0+vec,
                               vec_neg,
                               vec_pos))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 2:

            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[1][1], points[1][3]))
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))

            stack = np.vstack((points[0][3], intersects[1], points[1][2],
                               points[1][3], intersects[0], points[0][2]))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 3:

            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))
            intersects.append(get_intersect(
                points[1][1], points[1][3], points[2][0], points[2][2]))
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[2][1], points[2][3]))

            stack = np.vstack((points[0][3], intersects[0], points[1][2],
                               points[1][3], intersects[1], points[2][2],
                               points[2][3], intersects[2], points[0][2]))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 4:
            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))
            intersects.append(get_intersect(
                points[1][1], points[1][3], points[2][0], points[2][2]))
            intersects.append(get_intersect(
                points[2][1], points[2][3], points[3][0], points[3][2]))
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[3][1], points[3][3]))

            stack = np.vstack((points[0][3], intersects[0], points[1][2],
                               points[1][3], intersects[1], points[2][2],
                               points[2][3], intersects[2], points[3][2],
                               points[3][3], intersects[3], points[0][2]))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 5:
            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))
            intersects.append(get_intersect(
                points[1][1], points[1][3], points[2][0], points[2][2]))
            intersects.append(get_intersect(
                points[2][1], points[2][3], points[3][0], points[3][2]))
            intersects.append(get_intersect(
                points[3][1], points[3][3], points[4][0], points[4][2]))
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[4][1], points[4][3]))

            stack = np.vstack((points[0][3], intersects[0], points[1][2],
                               points[1][3], intersects[1], points[2][2],
                               points[2][3], intersects[2], points[3][2],
                               points[3][3], intersects[3], points[4][2],
                               points[4][3], intersects[4], points[0][2]))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

    ax.axis('equal')
    plt.show()







#******************************************************************************
# (4) Plot of analysis of structure
# A couple of functions to visualize the structures
#******************************************************************************


def plot_map_edges_in_label (list_labels, G, background, axs=[], plot_hillshade=False):
    """
    Plot only the specified labels of the Graph G over the background. 

    Number of nodes written next to the node.
    Number of edge written next to middle of the edge.
    
    By modifying the code possible to write any attribute next to middle of edge.
    
    Parameters
    ----------
    list_labels: list
        labels to plot

    G : nx.Graph
        Graph
    background : 2D numpy array
        background of the plot. Ex: DEM, strain rate.

    plot_hillshade: bool
        When True, plot the hillshade of the background (ex: useful for DEM).

    Returns
    -------
    None. Return the plot.

    """
    
    # Assertions 
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(list_labels, list), "list_labels is not a list"
    for label in list_labels:
        assert label in metrics.get_component_labels(G),'A label of the list given is not a label of G.'


    if plot_hillshade==True:
        assert len(background.shape)==2 , "The shape of the background array is not 2D"

        if axs == []:
            fig, axs = plt.subplots(1, 1, figsize=(12,12)) #init figure
            
        import earthpy as et
        import earthpy.spatial as es
        import earthpy.plot as ep
        hillshade_map = es.hillshade(background)
        ep.plot_bands(hillshade_map, cbar=False, ax=axs,alpha=0.7)

    else: 
        if axs == []:
            fig, axs = plt.subplots(1, 1, figsize=(12,12)) #init figure
            axs.imshow(background)#,cmap=cm.gray_r)
        else: axs.imshow(background,cmap=cm.gray_r)

    for label in list_labels:     

        nodes = [node for node in G if G.nodes[node]['component']==label]
        K=nx.subgraph(G,nodes)
        list_all_edges=list(K.edges)

        plot_components(K,False,ax=axs, node_size=3)
        node_color = get_node_colors(K, 'component')
        
        #Plot the first node just to have the nodes in the legend
        x0, y0 = K.nodes[list(K.edges)[0][0]]['pos']
        plt_node,=axs.plot(x0,y0,color=node_color[0],marker='o',markersize=3)
        plt_node.set_label("Nodes (+ number of the node)")
        
        # Plot the edges
        for edge in K.edges:
            num_edge=list_all_edges.index(edge)
            x0, y0 = K.nodes[edge[0]]['pos']
            x1, y1 = K.nodes[edge[1]]['pos']

            midpoint_x=(x0+x1)/2
            midpoint_y=(y0+y1)/2
            plt_mid,=axs.plot(midpoint_x,midpoint_y,'bo',markersize=3)
            plt_mid.set_label("Middle of the edges (+ number of edge)")
            #['dip_direction']


            # If need to check the strike of each edge: uncomment the following line
            # and comment the lines that plot text bellow
            #plot the strike of edge in the middle of the edge
            #axs.text(midpoint_x,midpoint_y,str(round(K.edges[edge]['strike'],0)))

            #plot number of edge next to center of edge
            axs.text(midpoint_x,midpoint_y,str(num_edge))
            
            #plot number of node next to the node
            #axs.text(x0, y0 , str(edge[0]),c=node_color[0]) ##
            #axs.text(x1, y1 , str(edge[1]),c=node_color[0])
            
            #plot polarity next to node
            #axs.text(x0, y0,str(K.nodes[edge[0]]['polarity']))
            #axs.text(x1, y1,str(K.nodes[edge[1]]['polarity']))
            
            #axs.text(midpoint_x,midpoint_y,str(K.edges[edge]['dip_direction'])) #MODIF

    #axs.set_ylim(axs.get_ylim()[::-1]) #MODIF
    
    axs.legend(handles=[plt_mid,plt_node])
    plt.show()

    return axs



def length_histo_computation(G, list_labels,resolution):
    """
    Compute array with labels and length of specified labels.
    
    
    Parameters
    ----------
    G : nx.graph
        Graph

    list_labels : list
        list with the index of the labels to plot
    resolution: int 
        physical unit meters
        horizontal resolution of the pixels to plot the length in meters
    
    Returns
    -------
    length_per_label : np.array
        array with column 0: label
                   column 1: total length of the component

    """

    # Assertions    
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(list_labels, list), "list_labels is not a list"
    for label in list_labels:
        assert label in metrics.get_component_labels(G),'A label of the list given is not a label of G.'

    
    length_per_label=np.zeros((len(list_labels),2))
    i=0
    for label in list_labels:
        length_per_label[i,0]=label
        tot_length=0
        
        nodes = [node for node in G if G.nodes[node]['component']==label]
        K=nx.subgraph(G,nodes)
        
        for edge in K.edges:
            #tot_length=tot_length+K.edges[edge]['length']*resolution
            tot_length=tot_length+K.edges[edge]['length']
        tot_length=tot_length*resolution
        length_per_label[i,1]=tot_length
        i=i+1

    return length_per_label



def plot_length_histogram (G, list_labels, resolution, title='Length histogram'):
    """
    Plot the histogram of the specified labels of graph G.
    Needs edge_length inn pixels.

    Parameters
    ----------

     G : nx.graph
         Graph
    list_labels : list
        list with the index of the labels to plot
    resolution: int 
        physical unit meters
        horizontal resolution of the pixels to plot the length in meters
    title : str
        title of the figure.
                
    Returns
    -------
    None.

    """
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(list_labels, list), "list_labels is not a list"
    for label in list_labels:
        assert label in metrics.get_component_labels(G),'A label of the list given is not a label of G.'

    
    
    length_per_label=length_histo_computation(G, list_labels,resolution)
    
    fig1, axs = plt.subplots(1, 1, figsize=(12,12),num=title)
    n=plt.hist(length_per_label[:,1],60) #last argument is the number of categories in the histogram
    n1=n[1][0:-1]
    axs.set_xlabel('Fault length [m]')
    axs.set_ylabel('Frequency')


    #plt.scatter(n1,n[0])
    fig2, axs = plt.subplots(1, 1, figsize=(12,12))
    axs.plot(n1,n[0],marker='o',markersize=3)
    #axs.set_xlim(200,60000)
    axs.set_xlabel('Fault length [m]')
    axs.set_ylabel('Frequency')
    axs.set_title(title, y=1.10, fontsize=15)
    
    return fig1,fig2


def cross_section_here (label, edges_to_cross, G, img_dem,  resolution, d=7, plot=False, plot_map=False):
    """ Virtually draw a cross-section at given edge. Possible to plot this section.
    
    Parameters
    ----------
    label : int
        label to which belongs the edges
    edges_to_cross : list or 'all'
        list of the edges where to do the cross-sections
        when 'all', all the edges of the label will be computed
    G : nx.Graph
        Graph
    img_dem : np.array 
        altitudes array
    resolution : int  
        physical unit meters
        horizontal resolution of the pixels
    d : int
        distance (unit pixel) from center to each side of the section
    plot : bool, optional
        Whether to plot the cross-section. 
        The default is False.
    plot_map : bool, optional
        Whether to plot the location of the cross-section from top, with elevation background. 
        The default is False.

    Returns
    -------
    coord_section : np.array
        coordinates of the section (=horizontal axis of section)
    
    alt_section : np.array
        altitudes of the section (=vertical axis of section)

    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(label, int), "label is not an integer"
    assert label in metrics.get_component_labels(G),'The label specified is not a label of G.'
    
    
    nodes = [node for node in G if G.nodes[node]['component']==label]
    K=nx.subgraph(G,nodes)
    K=structural_analysis.strike_edges(K)
    K=metrics.compute_edge_length(K)

    list_all_edges=list(K.edges)
    
    
    ######
   # d=7
    ######

    # initialisation of the profile 
    prof=np.zeros((2*d+1,1))
    for dist in range (0,2*d+1):
          prof[dist,0]= dist * resolution 

    if edges_to_cross=='all':
        list_edges=list_all_edges
    else:
        list_edges=[]
        for i in edges_to_cross:
            list_edges.append(list_all_edges[i])


    coord_section=[]
    alt_section=[]
    extremities=np.zeros((len(list(G.edges)),4))
    k=0

    if plot_map==True: ##
        axs=plot_map_edges_in_label(label, G, img_dem) ##

    for edge in list_edges:
        num_edge=list_all_edges.index(edge)
        # Cross section on center of segment with d from center to each side

        # Extract grid coord and altitudes of cross-section

        x0, y0 = K.nodes[edge[0]]['pos']
        x1, y1 = K.nodes[edge[1]]['pos']

        strike=K.edges[edge]['strike']

        #initialisations of beginning of loop
        all_x=np.zeros([2*d+1,1]) #x coordinates of profile
        all_y=np.zeros([2*d+1,1]) #y coordinates of profile
        alt=np.zeros([2*d+1,1])

        ################################
        # compute coordinates and gather altitudes of the profile
        # perpendicular to the segment, in its middle M, with length d on each side

        #compute the coordinates of the middlepoint of each segment

        midpoint_x=(x0+x1)/2
        midpoint_y=(y0+y1)/2

        all_x[d,0]=midpoint_x
        all_y[d,0]=midpoint_y
        midpoint_xx=math.floor(midpoint_x+0.5)
        midpoint_yy=math.floor(midpoint_y+0.5)
        alt[d,0]=img_dem[midpoint_yy,midpoint_xx]

        #loop cross section. Pull apart from M, to M + d and M - d
        for n in range (1,d+1):

            xD,yD,xE,yE=structural_analysis.calcul_coordinates(np.radians(strike), midpoint_x, midpoint_y, n)

            all_x[d-n,0]=xD
            all_y[d-n,0]=yD
            all_x[d+n,0]=xE
            all_y[d+n,0]=yE

            # ROUND OF COORDINATES
            #goal of those 4 lines is to have the coord of the pixel where the points are
            #it have to be an int because img_dem[x,y] can take only int

            xD=math.floor(xD+0.5)
            yD=math.floor(yD+0.5)
            xE=math.floor(xE+0.5)
            yE=math.floor(yE+0.5)

            alt[d-n,0]=img_dem[yD,xD]
            alt[d+n,0]=img_dem[yE,xE]

        coord_section.append((all_x,all_y))#exact
        alt_section.append((alt))

        if plot_map==True:
            xD,yD,xE,yE=structural_analysis.calcul_coordinates(np.radians(strike), midpoint_x, midpoint_y, d)

            extremities[k,0]=xD
            extremities[k,1]=yD
            extremities[k,2]=xE
            extremities[k,3]=yE

            axs.plot((xE,xD),(yE,yD))

        k=k+1
        if plot==True:

            #extension=G.edges[edge]['extension']
            #throw=G.edges[edge]['throw']
            #displacement=G.edges[edge]['displacement']

            #print(throw,extension,displacement)

            fig, ax = plt.subplots(1, 1, figsize=(12,12),num=str(num_edge))
            ax.plot(prof,alt)
            ax.plot(prof[d],alt[d],marker='d',color='m')#middle
            #ax.axis('equal')
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Elevation [m]')

            xD,yD,xE,yE=structural_analysis.calcul_coordinates(np.radians(strike), midpoint_x, midpoint_y, d)

            #ax.plot((prof[d]-(1/2*extension), prof[d]+(1/2)*extension), (alt[d]-1/2*throw,alt[d]-1/2*throw),c='m', label='extension')#line dip

            # to print which direction the fault is dipping
            # if img_dem[math.floor(yD+0.5),math.floor(xD+0.5)]<img_dem[math.floor(yE+0.5),math.floor(xE+0.5)]: #pend vers ouest
            #     ax.plot((prof[d]+(1/2)*extension, prof[d]+(1/2)*extension), (alt[d]-(1/2)*throw,alt[d]+(1/2)*throw), label='throw')#line di
            #     ax.plot((prof[d]-1/2*extension,prof[d]+(1/2)*extension), (alt[d]-1/2*throw, alt[d]+(1/2)*throw), label='displacement')
            #     #print("west") 
            # else:
            #     ax.plot((prof[d]-(1/2)*extension, prof[d]-(1/2)*extension), (alt[d]-(1/2)*throw,alt[d]+(1/2)*throw),label='throw')
            #     ax.plot((prof[d]+1/2*extension,prof[d]-(1/2)*extension), (alt[d]-1/2*throw, alt[d]+(1/2)*throw), label='displacement')
            #     #print('east')
            #ax.plot((prof[d], prof[d]), (alt[d]-(1/2)*throw,alt[d]+(1/2)*throw), label='throw')#line dip
            #ax.plot((prof[d]-(1/2)*extension, prof[d]-(1/2)*extension), (alt[d]-(1/2)*throw,alt[d]+(1/2)*throw), label='throw')#line dip

            #ax.plot((prof[d]-1/2*extension,prof[d]+(1/2)*extension), (alt[d]-1/2*throw, alt[d]+(1/2)*throw), label='displacement')

            ax.legend(loc='best')


    return coord_section,alt_section


def cross_section_here_hem (label, edges_to_cross, G, img_dem, crop_hem, resolution, d=7, plot=False):
    """ Virtually draw a cross-section at given edge and get error at each point.
    Possible to plot this section and the errors on elevation.

    Parameters
    ----------
    label : int
        label of G to which belongs the edges to cross
    edges_to_cross : list of int or 'all'
        list of the edges where to do the cross-sections
        when 'all', all the edges of the label will be computed
    G : nx.Graph
        Graph
    img_dem : np.array (2D)
        Elevation map
    crop_hem : np.array (2D)
        Hight Error Map
    resolution : int  
        physical unit meters
        horizontal resolution of the pixels
    d : int
        distance from center to extremity of cross-section.
    plot : bool, optional
        Whether to plot the cross-section. 
        The default is False.

    Returns
    -------
    coord_section : np.array
        coordinates of the section (=horizontal axis of section)
    
    alt_section : np.array
        altitudes of the section (=vertical axis of section)
    error_section : np.array
        elevation error at each point

    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert isinstance(label, int), "label is not an integer"
    assert label in metrics.get_component_labels(G),'The label specified is not a label of G.'


    nodes = [node for node in G if G.nodes[node]['component']==label]
    K=nx.subgraph(G,nodes)
    K=structural_analysis.strike_edges(K)
    K=metrics.compute_edge_length(K)

    # Initialisation profile
    list_all_edges=list(K.edges)

    prof=np.zeros((2*d+1,1))
    for dist in range (0,2*d+1):
          prof[dist,0]= dist * resolution

    if edges_to_cross=='all':
        list_edges=list_all_edges

    else:
        list_edges=[]
        for i in edges_to_cross:
            print(i)
            list_edges.append(list_all_edges[i])

    coord_section=[]
    alt_section=[]

    error_section=[]


    for edge in list_edges:
        num_edge=list_all_edges.index(edge)
        # Cross section on center of segment with d from center to each side

        # Extract grid coord and altitudes of cross-section

        x0, y0 = K.nodes[edge[0]]['pos']
        x1, y1 = K.nodes[edge[1]]['pos']

        strike=K.edges[edge]['strike']

        #initialisations of beginning of loop
        all_x=np.zeros([2*d+1,1]) #x coordinates of profile
        all_y=np.zeros([2*d+1,1]) #y coordinates of profile
        alt=np.zeros([2*d+1,1])
        err=np.zeros([2*d+1,1])

        ################################
        # compute coordinates and gather altitudes of the profile
        # perpendicular to the segment, in its middle M, with length d on each side

        #compute the coordinates of the middlepoint of each segment

        midpoint_x=(x0+x1)/2
        midpoint_y=(y0+y1)/2

        all_x[d,0]=midpoint_x
        all_y[d,0]=midpoint_y
        midpoint_xx=math.floor(midpoint_x+0.5)
        midpoint_yy=math.floor(midpoint_y+0.5)
        alt[d,0]=img_dem[midpoint_yy,midpoint_xx]

        err[d,0]=crop_hem[midpoint_yy,midpoint_xx]

        #loop cross section. Pull apart from M, to M + d and M - d
        for n in range (1,d+1):

            xD,yD,xE,yE=structural_analysis.calcul_coordinates(np.radians(strike), midpoint_x, midpoint_y, n)

            all_x[d-n,0]=xD
            all_y[d-n,0]=yD
            all_x[d+n,0]=xE
            all_y[d+n,0]=yE

            # ROUND OF COORDINATES
            #goal of those 4 lines is to have the coord of the pixel where the points are
            #it have to be an int because img_dem[x,y] can take only int

            xD=math.floor(xD+0.5)
            yD=math.floor(yD+0.5)
            xE=math.floor(xE+0.5)
            yE=math.floor(yE+0.5)

            alt[d-n,0]=img_dem[yD,xD]
            alt[d+n,0]=img_dem[yE,xE]


            err[d+n,0]=crop_hem[yE,xE]
            err[d-n,0]=crop_hem[yD,xD]

        coord_section.append((all_x,all_y))#exact
        alt_section.append((alt))

        error_section.append((err))

        if plot==True:
            fig, ax = plt.subplots(1, 1, figsize=(12,12),num=str(num_edge))
            #ax.plot(prof,alt)

            for ii in range(0,err.shape[0]):
                ax.errorbar(prof[ii], alt[ii], yerr=err[ii][0],color='red')

            ax.scatter(prof,alt, s=110)
            ax.plot(prof[d],alt[d],marker='d', markersize=10,color='c',label='point A')#middle
            #ax.axis('equal')
            #ax.set_xlabel('Distance [m]',fontsize="large")
            #ax.set_ylabel('Elevation [m]',fontsize="large")
            ax.set_xlim(0,max(prof))
            ax.errorbar(prof[ii], alt[ii], yerr=err[ii][0],color='red', label="error bar")

            ax.legend(loc='best')

    return coord_section, alt_section, error_section

