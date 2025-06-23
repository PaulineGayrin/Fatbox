## Welcome on Fatbox v2.0, the Fault analysis toolbox

Fatbox, the fault analysis toolbox is an open-source Python library that integrates semi-automated fault extraction with automated geometric and kinematic analysis of fault networks. Fatbox gathers about 150 Python functions to map and analyze faults from various datasets, typically topography and strain data. The library is versatile, documented and can be adapt the specific needs of you project.

The tutorials show the application of Fatbox to 3 cases:
- The DEM tutorials show the semi automated mapping and the structural analysis on a little area of the East African Rift. A second tutorial explain how to import a mapping made by hand, extract as network and analyse the faults automatically with the same functions.
- The numerical model tutorials use a basic rift forward model, provided by D. Neuharth. Using the strain data, we show how to map, analyse and track active faults through time, during the rift extension modeled. 
- In the analogue models tuto, we explain step by step how to map and analyse the faults from elevation and PIV data. This part show also the fault tracking as the model evolve.

The tutorials are available in .py for local adaptation and .ipynb to get started, withe plenty of comments.

Contributions are welcome using Pull request.
To ask questions or give feedback send an Issue so everyone can learn from your experience.

I wish you a lot of fun and good science!

Fatbox is a project initiated by Pauline Gayrin and Dr. Thilo Wrona under the supervision of Prof. Dr. Sascha Brune.

Contact: Pauline Gayrin  -> PaulineGayrin@protonmail.com

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

- *topography*

For each of them, you have a Jupyter python notebook where the workflow is detailed step by step. My advice: check this if you're a beginner coder and/or don't know Fatbox.
And an executable .py python script, with same content as the notebooks but less explaination. The scripts are designed to give you a good basis to adapt to your data. 

### Install

I recommend coupling your computer with your github account to always get the latest version of Fatbox and have access offline.
You can then open the jupyter notebook and python scripts locally.

#### Install using Google Collab
If you want to look at the notebooks without prior Python installation on your computer, although it is much less versatile, you can use Google Collab. (Requirement: a Google account)
Create a new folder on your Google drive.
Download fatbox at this location
Download the Colab tutorials at the same location. 

#### Install locally using conda
Requirement: download Anaconda
https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html

In the anaconda prompt (Windows) or in Terminal (Mac OS)
Create new environement with basic packages of anaconda
*conda create --name fatbox_env anaconda*

Activate environement: *conda activate fatbox_env*

Install the packages from the file *requirements.txt*

Install opencv 
*pip install opencv-python*
Note: never install opencv using conda, it crash the environment

On Windows, Cv algorithms is usually difficult to install, thankfully the solution is easy. 
Error you might get: Failed building wheel for cv-algorithms
Solution 
Go to https://github.com/ulikoehler/cv_algorithms/tree/master
Download the zip file (green button <> Code)
At the end of the import in the tutorials, and in preprocessing:
Uncomment the following line. 
*sys.path.append("C:\\Users\\your_directory\\cv_algorithms-master\\cv_algorithms-master\\cv_algorithms")*
Write the directory of the library cv_algorithms you just downloaded instead of *your_directory*. The total path is the directory where the _init_.py file of cv_algorithm is located.

### License

Creative Commons Attribution 4.0 International

## Citation

If you use this project in your research or wish to refer to the results of the tutorials.
Gayrin, P., Wrona, T., & Brune, S. (2025). Fatbox, the fault analysis toolbox (1.1). GFZ Helmholtz Centre for Geosciences. https://doi.org/10.5281/zenodo.15716080
