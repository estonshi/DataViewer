### README
----
This tool is for viewing 3D fields which stores in a 3D matrix.

#### SETUP
Support Linux and MacOS.

**Need Anaconda2 installed first**. After unzip then switdh to "3dView" directory and run "./INSTALL"

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

Note : 
<li> Now **only support ".npy" file**. Please store your matrix into a numpy file.
<li> Option "scalar" means your data is a scalar field, which requires that the matrix should be in 3 dimension shape=(x,y,z). 
<li> Option "vector" means your data is a vector field, which requires that the matrix should be in 6 dimension (s=[X,Y,Z,U,V,W], shape=(6,x,y,z))