import numpy as np

#from scpy2.tvtk import fix_mayavi_bugs


if __name__ == '__main__':
	import sys
	try:
		filepath = sys.argv[1]
		s = np.load(filepath)
		s = s.astype(float)
		s.shape[2]
	except:
		print('Input error. Scalar file should be a 3-dimensional matrix.')
		sys.exit()

	from mayavi import mlab
	from fix_mayavi_bugs import *
	fix_mayavi_bugs()

	surface = mlab.contour3d(s, contours=4, transparent=True)
	surface.contour.maximum_contour = s.max()
	surface.contour.number_of_contours = 10
	surface.actor.property.opacity = 0.4
	mlab.figure()

	mlab.pipeline.volume(mlab.pipeline.scalar_field(s))
	mlab.figure()

	field = mlab.pipeline.scalar_field(s)
	mlab.pipeline.volume(field, vmin=1.5, vmax=10)
	cut = mlab.pipeline.scalar_cut_plane(field.children[0], plane_orientation="y_axes")
	cut.enable_contours = True
	cut.contour.number_of_contours = 40
	mlab.gcf().scene.background = (0.8, 0.8, 0.8)
	mlab.show()
