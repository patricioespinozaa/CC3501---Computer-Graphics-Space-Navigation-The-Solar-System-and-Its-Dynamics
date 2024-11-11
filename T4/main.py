# Librerias de python
from pyglet.window import Window, key
from pyglet.gl import *
from pyglet.app import run
from pyglet import math
from pyglet import clock
import trimesh as tm
import sys, os
import numpy as np

#Librerias del curso
sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
from utils.helpers import mesh_from_file, init_pipeline
from utils.camera import FreeCamera
from utils.scene_graph import SceneGraph
from utils.drawables import Texture, PointLight, DirectionalLight, SpotLight, Material

class Controller(Window):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time = 0

class MyCam(FreeCamera):
    def __init__(self, position=np.array([0, 0, 0]), camera_type="perspective"):
        super().__init__(position, camera_type)
        self.direction = np.array([0,0,0])
        self.speed = 2

    def time_update(self, dt):
        self.update()
        dir = self.direction[0]*self.forward + self.direction[1]*self.right
        dir_norm = np.linalg.norm(dir)
        if dir_norm:
            dir /= dir_norm
        self.position += dir*self.speed*dt
        self.focus = self.position + self.forward

def models_from_file(path, pipeline):
    geom = tm.load(path)
    meshes = []
    if isinstance(geom, tm.Scene):
        for m in geom.geometry.values():
            meshes.append(m)
    else:
        meshes = [geom]

    models = []
    for m in meshes:
        m.apply_scale(2.0 / m.scale)
        m.apply_translation([-m.centroid[0], 0, -m.centroid[2]])
        vlist = tm.rendering.mesh_to_vertexlist(m)
        models.append(Model(vlist[4][1], vlist[3], pipeline))

    return models

class Model:
    def __init__(self, vertices, indices, pipeline) -> None:
        self.pipeline = pipeline
        
        self._buffer = pipeline.vertex_list_indexed(len(vertices)//3, GL_TRIANGLES, indices)
        self._buffer.position = vertices

    def draw(self, mode):
        self._buffer.draw(mode)

if __name__ == "__main__":

    #Controller / window
    controller = Controller(800,600,"Tarea 4")
    controller.set_exclusive_mouse(True)

    #Cámara
    cam = MyCam([2,2,6])

    #Para localizar archivos, fijese como se usa en el pipeline de ejemplo
    root = os.path.dirname(__file__)

    # Ejemplo de pipeline, con el clásico Phong shader visto en clases
    phong_pipeline = init_pipeline(root + "/shaders/phong.vert", root + "/shaders/phong.frag")
    
    # Shaders para cada planeta
    mercury_color_pipeline = init_pipeline(root + "/shaders/color.vert", root + "/shaders/color.frag")                           # (a)
    venus_flat_pipeline = init_pipeline(root + "/shaders/flat.vert", root + "/shaders/flat.frag")                                # (b)
    jupiter_phong_pipeline = init_pipeline(root + "/shaders/phong.vert", root + "/shaders/phong.frag")                           # (c)
    mars_toon_pipeline = init_pipeline(root + "/shaders/toon.vert", root + "/shaders/toon.frag")                                 # (d)
    earth_textured_pipeline = init_pipeline(root + "/shaders/textured.vert", root + "/shaders/textured.frag")                    # (e)

    #grafo para contener la escena    
    world = SceneGraph(cam)
    
    # Nave
    nave_pipeline = init_pipeline(root + "/shaders/phong.vert", root + "/shaders/phong.frag")
    nave = mesh_from_file(root + "/nave.obj")[0]["mesh"]
    world.add_node("nave", mesh=nave, pipeline=nave_pipeline, rotation=[0, np.pi/2, 0], material=Material(diffuse=[0,1,0]))
 
    # Mesh Sphere
    sphere = mesh_from_file(root + "/sphere.obj")[0]["mesh"]
    sphere.init_gpu_data(phong_pipeline)
    
    # Sol
    sun_pipeline = init_pipeline(root + "/shaders/color.vert", root + "/shaders/color.frag")  
    world.add_node("sun_to_root")
    world.add_node("sun_base", attach_to="sun_to_root", mesh=sphere, pipeline=sun_pipeline, rotation=[0, 0, 0], material=Material(), scale=[5, 5, 5], position=[0,0,0], color=[1,1,0])

    # Planetas

    # Mercurio 
    world.add_node("mercury_to_sun", attach_to="sun_to_root")
    world.add_node("mercury_base", attach_to="mercury_to_sun", mesh=sphere, pipeline=mercury_color_pipeline, rotation=[0, 0, 0], material=Material(), scale=[1, 1, 1], position=[0.2,0,0], color=[1,0,0])

    # Venus
    world.add_node("venus_to_sun", attach_to="sun_to_root")
    world.add_node("venus_base", attach_to="venus_to_sun", mesh=sphere, pipeline=venus_flat_pipeline, rotation=[0, 0, 0], material=Material(diffuse=[0,1,0], ambient=[1,1,1], shininess=20), scale=[2, 2, 2], position=[1,0,0])

    # Tierra
    earth_textured_pipeline = init_pipeline(root + "/shaders/textured.vert", root + "/shaders/textured.frag") 
    earth_text = Texture(root + "/assets/earth.jpg")
    world.add_node("earth_to_sun", attach_to="sun_to_root")
    world.add_node("earth_base", attach_to="earth_to_sun", mesh=sphere, pipeline=earth_textured_pipeline, rotation=[0, 0, 0], texture=earth_text, material=Material(shininess=100), scale=[4, 4, 4], position=[3,0,0])

    # Marte
    world.add_node("mars_to_sun", attach_to="sun_to_root")
    world.add_node("mars_base", attach_to="mars_to_sun", mesh=sphere, pipeline=mars_toon_pipeline, rotation=[0, 0, 0], material=Material(diffuse=[0,0,1]), scale=[3, 3, 3], position=[5,0,0])

    # Jupiter
    world.add_node("jupiter_to_sun", attach_to="sun_to_root")
    world.add_node("jupiter_base", attach_to="jupiter_to_sun", mesh=sphere, pipeline=jupiter_phong_pipeline, rotation=[0, 0, 0], material=Material(diffuse=[0.5,0,0.5]), scale=[6, 6, 6], position=[7,0,0])
    
    # Tierra
    #earth_text = Texture(root + "/assets/earth.jpg")
    #world.add_node("earth_to_sun", attach_to="sun_to_root")
    #world.add_node("earth_base", attach_to="earth_to_sun", mesh=sphere, pipeline=earth_textured_pipeline, rotation=[0, 0, 0], texture=earth_text, material=Material(shininess=100), scale=[4, 4, 4], position=[3,0,0])

    # Luces puntuales
    # Afectados por la luz (union de pipelines)
    afectados = [nave_pipeline, sun_pipeline, mercury_color_pipeline, venus_flat_pipeline, earth_textured_pipeline, mars_toon_pipeline, jupiter_phong_pipeline]
    
    # Luces puntuales
    world.add_node("sun_point_light", attach_to="sun_base",light=PointLight(), pipeline=afectados, position=[0, 0, 0])
    world.add_node("nave_light", attach_to="nave", light=PointLight(), pipeline=afectados)
    
    # Luz direccional
    world.add_node("directional_light", light=DirectionalLight(), pipeline=afectados, rotation=[-np.pi/4, 0, 0])

    @controller.event
    def on_draw():
        controller.clear()
        glClearColor(0.1, 0.1, 0.1, 1)
        glEnable(GL_DEPTH_TEST)

        world.draw()

    @controller.event
    def on_key_press(symbol, modifiers):
        if symbol == key.SPACE: controller.light_mode = not controller.light_mode
        if symbol == key.W:
            cam.direction[0] = 1
        if symbol == key.S:
            cam.direction[0] = -1

        if symbol == key.A:
            cam.direction[1] = 1
        if symbol == key.D:
            cam.direction[1] = -1


    @controller.event
    def on_key_release(symbol, modifiers):
        if symbol == key.W or symbol == key.S:
            cam.direction[0] = 0

        if symbol == key.A or symbol == key.D:
            cam.direction[1] = 0

    @controller.event
    def on_mouse_motion(x, y, dx, dy):
        cam.yaw += dx * .001
        cam.pitch += dy * .001
        cam.pitch = math.clamp(cam.pitch, -(np.pi/2 - 0.01), np.pi/2 - 0.01)

        world["nave"]["rotation"] = [0, -cam.yaw, cam.pitch]


    @controller.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        controller.light_distance += scroll_y*.01

    def update(dt):
        world.update()
        cam.time_update(dt)

        world["nave"]["position"] = cam.position + cam.forward*2 + [0, -1.5, 0]

        #======== Movimiento de planetas no lo olvide!! ============
        t = controller.time  # Tiempo acumulado
        
        world["mercury_base"]["position"] = [5 * np.cos(t), 0, 5 * np.sin(t)]
        world["venus_base"]["position"] = [8 * np.cos(t * 0.7), 0, 8 * np.sin(t * 0.7)]
        world["earth_base"]["position"] = [10 * np.cos(t * 0.5), 0, 10 * np.sin(t * 0.5)]
        world["mars_base"]["position"] = [12 * np.cos(t * 0.3), 0, 12 * np.sin(t * 0.3)]
        world["jupiter_base"]["position"] = [15 * np.cos(t * 0.2), 0, 15 * np.sin(t * 0.2)]
        #============================================================
        
        world.update()

        controller.time += dt

    clock.schedule_interval(update,1/60)
    run()

