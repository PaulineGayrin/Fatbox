## Welcome on Fatbox v2.0, the Fault analysis toolbox

Fatbox the fault analysis toolbox is an open-source Python toolbox. It integrates semi-automated fault extraction with automated geometric and kinematic analysis of fault networks. The library includes about 150 Python functions for mapping and analyzing faults from various datasets, such as topography and strain data. Fatbox is versatile, well-documented, and can be adapted to the specific needs of your project.

The tutorials demonstrate the application of Fatbox through three practical cases:
- DEM Tutorials: The first tuto shows the semi-automated mapping and structural analysis of a small area in the East African Rift. A second tutorial explains how to import a manually created map, extract a fault network, and automatically analyze the faults using the same functions.
- Numerical Model Tutorials: Using a basic rift forward model provided by Dr. D. Neuharth, these tutorials illustrate how to map, analyze, and track active faults over time during modeled rift extension, using strain data.
- Analogue Model Tutorials: These provide a step-by-step guide on how to map and analyze faults from elevation and PIV data. This section also covers fault tracking as the model evolves.

The tutorials are available as Jupyter notebooks (.ipynb) to help you get started.

Contributions are welcome via Pull Requests. To ask questions or provide feedback, please open an Issue so everyone can benefit from your experience.

Have fun and enjoy your scientific discoveries!

Fatbox is a project initiated by Pauline Gayrin and Dr. Thilo Wrona under the supervision of Prof. Dr. Sascha Brune.
Contact: Pauline Gayrin  -> gayrin@gfz.de

## Getting started

### Modules
Fatbox functions are grouped in 6 different Python scripts that follow a typical sequential workflow. 
The 6 scripts of the library are accessible in the folder */modules*.
1. *preprocessing.py* - Prepare the dataset for fault network extraction.
2. *edits.py* - Extract the fault network from the dataset and edit the network and its sub-networks.
3. *metrics.py* - Compute various metrics of the fault network, such as length of the edges 
4. *plots.py* - Visualize the fault network and results of the analysis.
5. *utils.py* - Various low-level helper functions.
6. *structural_analysis.py* - Measure the geometric properties of the faults.

In the script, to see the documentation of a function, type in the console
*module.function._doc_*   eg. *plots.plot_components._doc_* to get in line docstring
or 
*help(module.function)*   eg. *help(plots.plot_components)* to get paragraph docstring

### Tutorials

The 3 tutorial folders illustrate the main applications of Fatbox:
- *analog_modelling*

- *numerical_modelling*

- *digital_elevation_models*

For each of them, you have a Jupyter python notebook where the workflow is detailed step by step. My advice: check this if you're a beginner coder and/or don't know Fatbox. The Notebooks can be executed in Google Colab or lacally on Jupyter notebook.

### Installation

I recommend coupling your computer with your github account to always get the latest version of Fatbox and have access offline.
You can then open the jupyter notebook and python scripts locally.

#### Install using Google Collab
If you want to look at the notebooks without prior Python installation on your computer, although it is much less versatile, you can use Google Collab. (Requirement: a Google account)
Create a new folder on your Google drive.
Download fatbox at this location
Download the Colab tutorials at the same location. 

#### Install locally using conda
**Possibility 1: Use Anaconda** 
Download Anaconda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)

Then in the anaconda prompt 
Create new environement with basic packages of anaconda
*conda create --name fatbox_env anaconda*

**Possibility 2: Use Miniforge** 
Download and install instructions [here](https://github.com/conda-forge/miniforge)

Next in the miniforge prompt
*conda create -c conda-forge -n fatbox_env*

**Then**
Activate environment: *conda activate fatbox_env*

Install the packages from the file *requirements.txt* (located in Fatbox folder)
*conda install --yes --file requirements.txt* 

If the installation of earthpy bug, execute in the prompt directly 
*conda install conda-forge::earthpy* then *conda install conda-forge::git* and *pip install vtk* and *conda install conda-forge::tqdm* *conda install conda-forge::seaborn*

Install opencv 
*pip install opencv-python*
Note: never install opencv using conda, it crash the environment

On Windows, Cv algorithms is usually difficult to install, thankfully the solution is easy. 
Error you might get: Failed building wheel for cv-algorithms
Solution 
Go on [Uli Koehler Github] (https://github.com/ulikoehler/cv_algorithms/tree/master)
Download the zip file (green button <> Code)
At the end of the import in the tutorials, and in preprocessing:
Uncomment the following line. 
*sys.path.append("C:\\Users\\your_directory\\cv_algorithms-master\\cv_algorithms-master\\cv_algorithms")*
Write the directory of the library cv_algorithms you just downloaded instead of *your_directory*. The total path is the directory where the _init_.py file of cv_algorithm is located.

## License

Creative Commons Attribution 4.0 International

## Acknowledgment
A huge thanks to Thilo Wrona for the collaboration and support on the developpment of Fatbox. Special thanks to Nicolas Molnar and Derek Neuharth for the contribution to the analogue and numerical tutorials and access to their model's data. Thank to Sascha Brune for the supervision and support. The developper thank Baptiste Bordet for his huge support along the way and help on debugging. Thank to Tim Hake, who contributed to a former version of the fault extraction workflow.

## Citation

If you use this project in your research or wish to refer to the results of the tutorials.
Gayrin, P., Wrona, T., & Brune, S. (2025). Fatbox, the fault analysis toolbox (1.1). GFZ Helmholtz Centre for Geosciences. https://doi.org/10.5281/zenodo.15716080
