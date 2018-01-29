## <center>3dView Help Doc</center>

### 1) How To Use It ?
----
This tool is for viewing 3D fields which stores in a 3D matrix.

#### SETUP
Support Linux and MacOS.

**Need Anaconda2 installed first**. After unzip then switdh to "3dView" directory and run "./INSTALL"

(It is better to check that your qt package in Anaconda is **pyqt4**, not *pyqt5*, or there might be some problems while installing.)

```
unzip -v 3dView.zip
cd 3dView
./INSTALL
```

#### PLOT
Open terminal, at anywhere run :

```
3dplot -t [scalar|vector] -f [Your data file]
```

For simplicity, you can just run :

```
3dplot.scalar [your data file]
3dplot.vector [your data file]
```

#### GUI
It is highly recomended to use GUI for straight-forward operation. Now GUI supports scalar fields, vector fields and points plot. To use it, just open terminal and run :

```
3dplot.gui
[or]
3dplot -t gui
```

The first thing you need to do is to import data file , then click "Import Selected File" button. You can choose different plot types (The select list named "Plot") as you want.

#### Movie Maker
Click "Rorate and Movie" button to rotate the scene or create a gif movie. 

* To make a movie, you should set the total length first and click "Make movie" (There is a progress bar at the left-bottom corner of the pop window). 
* Just click "Rotate" if you only want a rotation view of the model.

----
### 2) About Data Format : 
* Now **support ".npy"(numpy), ".mat"(matlab) and ".bin"(binary) files**. Please store your matrix into file in these formats.
	 
* "scalar" plot means your data contains a scalar field, which requires that the matrix should be in 3 dimension.
	* shape=(Nx,Ny,Nz)
* "vector" plot means your data contains a vector field, which requires that the matrix should be in 6 dimension.
	* s=[X,Y,Z,U,V,W] , shape=(6,Nd,Nd,Nd)
* "points plot" means your data contains a set of points (with or without intensity) in 3D space, and the input matrix should be in 3 or 4 dimension. 
	* s=[X,Y,Z] or s=[X,Y,Z,Intensity] , shape=(3 or 4,Nd,Nd,Nd)