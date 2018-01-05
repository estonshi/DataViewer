#ecoding=utf8
import numpy as np
import sys

from traits.api import *
from traitsui.api import *
from traitsui.message import message
from tvtk.pyface.scene_editor import SceneEditor 
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi import mlab
import fix_mayavi_bugs

fix_mayavi_bugs.fix_mayavi_bugs()

class FieldViewer(HasTraits):

    # define data var
    s = None
    # define plot button
    plotbutton = Button("Import Selected File")
    rotatebutton = Button("Rotate")
    # define mayavi scene
    scene = Instance(MlabSceneModel, ()) 
    # define a file trait to view:
    file_name = File
    # init plot type => self._plotbutton_fired()
    plot_types = ['scalar', 'vector']
    plot_type = None
    # plot scence => self.plot() && self.*
    plot_scene_scalar = ['counter','cut plane','volume']
    plot_scene_vector = ['quiver','cut plane','Lorentz attractor trajectory']
    plot_scene = List(['None','None','None'])
    select = Str


    """
    # define enum list
    enum_list = ['3d scalar filed','3d vector filed']
    # define plot type 
    select = Enum(*enum_list)
    """
    G1 = VGroup(
                Item('file_name', style='simple', label='Open'),
                Item('file_name', style='custom', label=''),
                Item('_'),
                #Item('file_name', style='readonly', label='File name'),
                Item('plotbutton'), show_labels=False
            )

    G2 = VGroup(
            HGroup(
                Item('scene', 
                    editor=SceneEditor(scene_class=MayaviScene), 
                    resizable=True,
                    height=600,
                    width=600,
                    label='Scene'),
                show_labels=False
                ), 
            HGroup(
                Group(
                    Item('select', 
                        editor=EnumEditor(name='object.plot_scene'),
                        tooltip='Display functions',
                        label='Plot')
                    ),
                Group(
                    Item('rotatebutton', show_label=False)
                    )
                )
            )
    
    view = View(
        HSplit(
            G1
            ,
            G2
        ),
        width = 900, resizable=True, title="3dplot GUI"
    )

    def _select_changed(self):
        self.plot()
      
    def _plotbutton_fired(self):
        try:
            s = np.load(self.file_name)
            s.astype(float)
            if len(s.shape)==3:
                self.plot_type = self.plot_types[0]
                self.plot_scene = self.plot_scene_scalar
            elif s.shape[0]==6:
                self.plot_type = self.plot_types[1]
                self.plot_scene = self.plot_scene_vector
            else:
                np.zeros('Woop!')
            self.s = s
            self.plot()
        except:
            message("I can't handle your file!")
            pass

    def plot(self):
        s = self.s
        if self.plot_type == self.plot_types[0]:
            if self.select == self.plot_scene[0]:
                self.plot_scalar_scene_1(s)
            elif self.select == self.plot_scene[1]:
                self.plot_scalar_scene_2(s)
            elif self.select == self.plot_scene[2]:
                self.plot_scalar_scene_3(s)
            else :
                pass
        elif self.plot_type == self.plot_types[1]:
            if self.select == self.plot_scene[0]:
                self.plot_vector_scene_1(s)
            elif self.select == self.plot_scene[1]:
                self.plot_vector_scene_2(s)
            elif self.select == self.plot_scene[2]:
                self.plot_vector_scene_3(s)
            else:
                pass
        else:
            pass

    def plot_scalar_scene_1(self, s):
        self.scene.mlab.clf()
        g = self.scene.mlab.contour3d(s, contours=10, transparent=True)
        g.actor.property.opacity = 0.4
        self.g = g

    def plot_scalar_scene_2(self, s):
        self.scene.mlab.clf()
        field = self.scene.mlab.pipeline.scalar_field(s)
        self.scene.mlab.pipeline.volume(field, vmin=s.min(), vmax=s.max())
        cut = self.scene.mlab.pipeline.scalar_cut_plane(field.children[0], plane_orientation="y_axes")
        cut.enable_contours = True
        cut.contour.number_of_contours = 20
        self.scene.mlab.gcf().scene.background = (0.8, 0.8, 0.8)
        self.g = field
      
    def plot_scalar_scene_3(self, s):
        self.scene.mlab.clf()
        v = self.scene.mlab.pipeline.volume(self.scene.mlab.pipeline.scalar_field(s))
        self.g = v

    def plot_vector_scene_1(self, s):
        self.scene.mlab.clf()
        x,y,z,u,v,w = s[0],s[1],s[2],s[3],s[4],s[5]
        vectors = self.scene.mlab.quiver3d(x, y, z, u, v, w)
        vectors.glyph.mask_input_points = True
        vectors.glyph.mask_points.on_ratio = 10
        vectors.glyph.glyph.scale_factor = 5.0
        self.g = vectors

    def plot_vector_scene_2(self, s):
        self.scene.mlab.clf()
        x,y,z,u,v,w = s[0],s[1],s[2],s[3],s[4],s[5]
        src = self.scene.mlab.pipeline.vector_field(x, y, z, u, v, w)
        src_cut = self.scene.mlab.pipeline.vector_cut_plane(src, mask_points=2, scale_factor=5)
        src = self.scene.mlab.pipeline.vector_field(x, y, z, u, v, w)
        magnitude = self.scene.mlab.pipeline.extract_vector_norm(src)
        surface = self.scene.mlab.pipeline.iso_surface(magnitude)
        surface.actor.property.opacity = 0.3
        self.scene.mlab.gcf().scene.background = (0.8, 0.8, 0.8)
        self.g = src_cut

    def plot_vector_scene_3(self, s):
        self.scene.mlab.clf()
        x,y,z,u,v,w = s[0],s[1],s[2],s[3],s[4],s[5]
        f = self.scene.mlab.flow(x, y, z, u, v, w)
        self.g = f

    def _rotatebutton_fired(self):
        @mlab.animate
        def anim():
            f = mlab.gcf()
            while 1:
                f.scene.camera.azimuth(1)
                f.scene.render()
                yield

        a = anim()


if __name__ == '__main__':
    app = FieldViewer()
    app.configure_traits()
###1###