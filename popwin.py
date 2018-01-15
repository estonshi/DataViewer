import numpy as np
from traits.api import *
from traitsui.api import *

import fix_mayavi_bugs

fix_mayavi_bugs.fix_mayavi_bugs()

class movie_settings(HasTraits):
	father = None
	# define flag to control gui
	flag = False
	now = CFloat
	# define input text
	total_time = Int
	# define current percent
	curr = Property(depends_on='now')
	# define button
	start = Button('Start')
	# define view
	view = View(
		Group(
			Group(
				Item('total_time', label='Movie length (second):')
			),
			Group(
				Item('start', show_label=False)
			),
			orientation = 'vertical',
			show_border = True
		),
		title='Movie Maker',
		kind='live',
		resizable=True,
        statusbar=[StatusItem(name='curr')]
		)

	def set_now(self, nowtime):
		self.now = nowtime

	# events
	def _get_curr(self):
		pre = 100*self.now/(self.total_time+1e-5)
		return ('Progress: %d %% recorded.' % int(pre))

	def _start_fired(self):
		if self.total_time <= 0:
			message('Movie length should not <0 !')
			pass
		else:
			self.father.flag = 1

ctrl = None

def init(fatherr):
	global ctrl
	ctrl = movie_settings()
	ctrl.father = fatherr
	ctrl.edit_traits()

def set_now(time):
	global ctrl
	if time=='finished':
		ctrl.set_now(float(ctrl.total_time))
	else:
		ctrl.set_now(time)

def get_total_time():
	global ctrl
	return ctrl.total_time

if __name__ == '__main__':
	init(None)
	ctrl.close()
