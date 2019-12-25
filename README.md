## DataViewer README
---

### INTRO
This tool is for dataset visualization.

1. 2dViewer : designed for HDF5 dataset visualization
2. 3dViewer : designed for 3D volume dataset visualization

### SETUP
The programs are based on Anaconda environment.

1. Install Anaconda 3
2. Run 'make.sh' (for Linux and MacOS user), or create conda environment using 'environment.yaml' by yourself

### PLOT

1. View HDF5 dataset
```bash
python ./2dViewer/viewer.py
```

2. View 3D volume
```bash
python ./3dViewer/viewer.py
```
---

### DOC

#### (1) 3dViewer

The first thing you need to do is to choose data file , then click "Import Selected File" button. You can choose different plot types (The select list named "Plot") as you want.

**[Movie Maker]**

Click "Rorate and Movie" button to rotate the scene or create a gif movie. 

* To make a movie, you should set the total length first and click "Make movie" (There is a progress bar at the left-bottom corner of the pop window). 
* Just click "Rotate" if you only want a rotation view of the model.

**[Data Format]**

* Now **support ".npy"(numpy), ".mat"(matlab) and ".bin"(binary) files**. Please store your matrix into file in these formats.
	 
* "scalar" plot means your data contains a scalar field, which requires that the input should have 3 dimensions.
	* shape=(Nx,Ny,Nz)
* "vector" plot means your data contains a vector field, which requires that the input should have 2 dimensions, with 6 rows and Num_of_points columns.
	* s=[X,Y,Z,Vx,Vy,Vz] , shape=(6, Num_of_points)
* "points plot" means your data contains a set of points (with or without intensity) in 3D space, and the input should have 2 dimensions, with 3 or 4 rows and Num_of_points columns.
	* s=[X,Y,Z] or s=[X,Y,Z,Intensity] , shape=(3 or 4, Num_of_points)

#### (2) 2dViewer

Just **drag** HDF5 file into the file list and double click dataset name to plot.

There are several parameters to adjust the visualization.