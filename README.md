### README
----
This tool is for viewing 3D fields which stores in a 3D matrix.

#### SETUP
Support Linux and MacOS.

**Need Anaconda2 installed first**. After unzip then switdh to "3dView" directory and run "./INSTALL"

(It is better to check that your qt package in Anaconda is *pyqt4*, not *pyqt5*)
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
Now we offered a GUI for straight-forward operation. Just open terminal and run :

```
3dplot.gui
[or]
3dplot -t gui
```

To use the gui, you have to import data file first, then click "Import Selected File" button and choose plot type("Plot" select list).

#### Movie Maker
Click "Make rorate movie" button to create a gif movie. All you have to do is to set the total length of your movie. There is a progress bar at the left-bottom corner of the pop window.

Note : 
<li> Now **only support ".npy" file**. Please store your matrix into a numpy file.
<li> Option "scalar" means your data is a scalar field, which requires that the matrix should be in 3 dimension shape=(x,y,z). 
<li> Option "vector" means your data is a vector field, which requires that the matrix should be in 6 dimension (s=[X,Y,Z,U,V,W], shape=(6,x,y,z))