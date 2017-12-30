# -*- coding: utf-8 -*-
import os
import numpy as np


def draw(s, t=[1,1,1]):

	from mayavi import mlab
	import fix_mayavi_bugs
	fix_mayavi_bugs.fix_mayavi_bugs()

	x,y,z,u,v,w = s[0],s[1],s[2],s[3],s[4],s[5]

	"""
	mlab.figure(size=(600, 600))
	mlab.quiver3d(x, y, z, u, v, w)
	"""
	if t[0]==1:
		mlab.figure(size=(600, 600))
		vectors = mlab.quiver3d(x, y, z, u, v, w)
		vectors.glyph.mask_input_points = True
		vectors.glyph.mask_points.on_ratio = 10
		vectors.glyph.glyph.scale_factor = 5.0

	if t[1]==1:
		mlab.figure(size=(600, 600))
		src = mlab.pipeline.vector_field(x, y, z, u, v, w)
		mlab.pipeline.vector_cut_plane(src, mask_points=2, scale_factor=5)
		magnitude = mlab.pipeline.extract_vector_norm(src)
		surface = mlab.pipeline.iso_surface(magnitude)
		surface.actor.property.opacity = 0.3
		mlab.gcf().scene.background = (0.8, 0.8, 0.8)

	if t[2]==1:
		mlab.figure(size=(600, 600))
		mlab.flow(x, y, z, u, v, w)

	mlab.show()

if __name__ == '__main__':
	import sys
	try:
		filepath = sys.argv[1]
		s = np.load(filepath)
		s = s.astype(float)
		s[5]
	except:
		print('Input error. Vector file should be a 6-dimensional matrix.')
		sys.exit()

	draw(s,[1,1,1])
	"""
	from mayavi import mlab
	from fix_mayavi_bugs import *
	fix_mayavi_bugs()

	mlab.figure(size=(600, 600))

	mlab.quiver3d(x, y, z, u, v, w)

	mlab.figure(size=(600, 600))
	vectors = mlab.quiver3d(x, y, z, u, v, w)
	vectors.glyph.mask_input_points = True
	vectors.glyph.mask_points.on_ratio = 20
	vectors.glyph.glyph.scale_factor = 5.0

	mlab.figure(size=(600, 600))
	src = mlab.pipeline.vector_field(x, y, z, u, v, w)
	mlab.pipeline.vector_cut_plane(src, mask_points=2, scale_factor=5)

	magnitude = mlab.pipeline.extract_vector_norm(src)
	surface = mlab.pipeline.iso_surface(magnitude)
	surface.actor.property.opacity = 0.3
	mlab.gcf().scene.background = (0.8, 0.8, 0.8)

	mlab.figure(size=(600, 600))
	mlab.flow(x, y, z, u, v, w)

	mlab.show()
	"""
