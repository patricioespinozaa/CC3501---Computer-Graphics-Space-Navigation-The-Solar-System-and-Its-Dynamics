import trimesh as tm
import pyglet
import numpy as np
from pyglet.gl import *
import os
import sys
#sys.path.append(os.path.dirname((os.path.dirname(__file__))))
sys.path.append(r'C:\Users\pbast\Desktop\Grafica\CC3501')
import grafica.transformations as tr
from pyglet.math import Mat4, Vec3, clamp
from pyglet.window import Window, key

WIDTH = 1000
HEIGHT = 1000

# Tarea 2
# Curso: CC3501
# Sección: 1
# Nombre: Patricio Espinoza A.

#Controller
class Controller(pyglet.window.Window):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time = 0.0
        self.fov = 90
        super().set_exclusive_mouse(True)

window = Controller(WIDTH, HEIGHT, "Tarea 2")

#Para los contorles
keys = pyglet.window.key.KeyStateHandler()
window.push_handlers(keys)
window.set_exclusive_mouse(True)

#Defina aquí una clase "Ship" para la nave
# Combinar implementación de camara y de model
class Ship:
    def __init__(self, size, vertices, indices, pipeline) -> None: # Igual a Model
        # Cualidades del objeto de la nave
        self.color = np.zeros(3, dtype=np.float32)
        self.position = np.zeros(3, dtype=np.float32)
        self.scale = np.ones(3, dtype=np.float32)
        self.rotation = np.zeros(3, dtype=np.float32)
        self._buffer = pipeline.vertex_list_indexed(size, GL_TRIANGLES, indices)
        self._buffer.position = vertices
        self.angle = 0.0
        self.rotation_angle = 0.0

        # Movimiento basado en la camara
        self.yaw = 0
        self.pitch = 0
        self.speed = 0
        self.sensitivity = 0.01
        self.front = np.array([0, 0, -1], dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.direction = np.zeros(2)
    
    def model(self):
        translation = Mat4.from_translation(Vec3(*self.position))           # Tralacion
        scale = Mat4.from_scale(Vec3(*self.scale))                          # Scale
        rotation = Mat4.from_rotation(self.rotation_angle, Vec3(0, 1, 0))   # Rotacion
        additional_rotation = Mat4.from_rotation(np.pi / 2, Vec3(0, 1, 0))  # 90 grados rotados ya que el obj no estaba bien orientado
        return translation @ additional_rotation @ rotation @ scale
    
    def update(self, dt):
        # Actualizar la direccion de la nave basada en yaw y pitch
        self.front[0] = np.cos(self.yaw) * np.cos(self.pitch)
        self.front[1] = np.sin(self.pitch)
        self.front[2] = np.sin(self.yaw) * np.cos(self.pitch)
        self.front /= np.linalg.norm(self.front)

        dir = self.direction[0]*self.front + self.direction[1]*np.cross(self.up, self.front)
        dir_norm = np.linalg.norm(dir)
        if dir_norm:
            dir /= dir_norm

        # Actualizar la posicion de la nave en funcion de la velocidad
        self.position += self.front * self.speed * dt

        # Calcular el angulo de rotacion de la nave en base a donde mira el mouse
        self.rotation[0] = np.arctan2(self.front[1], self.front[2])
        self.rotation[1] = np.arctan2(self.front[0], self.front[2])

        # Asignar el nuevo angulo de rotacion de la nave
        self.rotation_angle = np.arctan2(self.front[0], self.front[2])

    def handle_mouse_movement(self, dx, dy):
        # Actualizar yaw y pitch de la nave de acuerdo a la sensibilidad del mouse
        self.yaw += dx * self.sensitivity
        self.pitch += dy * self.sensitivity  # Mouse hacia arriba apunta hacia arriba, Mouse hacia abajo apunta hacia abajo

        # Evitar que el pitch se pase al girar hacia arriba o abajo
        self.pitch = np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        
        # Usar update para actualizar la dirección de la nave
        self.update(1/60)

    # Matriz de vista de la nave
    def view(self):
        return Mat4.look_at(Vec3(*self.position), Vec3(*(self.position + self.front)), Vec3(*self.up))

    # Permite dibujar la nave
    def draw(self):
        self._buffer.draw(GL_TRIANGLES)

#Defina aquí una clase "Camara", basada en la del Aux4
class Camara:
    def __init__(self, distance=10, height=5, initial_offset=(0, 2, 0)) -> None:  # Recibe la distancia y altura de la camara respecto a la nave
        self.distance = distance                                                  # Distancia fija de la cam respecto a la nave
        self.height = height                                                      # Altura fija de la cam respecto a la nave
        self.position = np.zeros(3, dtype=np.float32)
        self.initial_offset = np.array(initial_offset, dtype=np.float32)          # Offset para ajustar la posición inicial de la cam y que quede detras de la nave

    def update(self, target_position, target_front):
        # Calcular la posición de la cámara basada en la nave y su dirección frontal
        direction = -target_front                                                 # La cam mira a la direccion opuesta de la nave

        # CActualiza la posicion de la camara
        self.position = target_position + direction * self.distance + np.array([0, self.height, 0])
        self.front = target_front

    # Matriz de vista de la camara
    def view(self):
        return Mat4.look_at(Vec3(*self.position), Vec3(*(self.position + self.front)), Vec3(0, 1, 0))


#Defina aquí una clase "Model" para el resto de los objetos, basada en GameModel
class Model:
    def __init__(self, size, vertices, indices, pipeline, orbit_radius=0, orbit_speed=0, rotation_speed=0) -> None:
        # Parametros ajustables de los planetas, sus orbitas, rotaciones y respectivas velocidades
        self.color = np.zeros(3, dtype=np.float32)
        self.position = np.zeros(3, dtype=np.float32)
        self.scale = np.ones(3, dtype=np.float32)
        self.rotation = np.zeros(3, dtype=np.float32)
        self.orbit_radius = orbit_radius
        self.orbit_speed = orbit_speed
        self.rotation_speed = rotation_speed
        self._buffer = pipeline.vertex_list_indexed(size, GL_TRIANGLES, indices)
        self._buffer.position = vertices
        self.angle = 0.0
        self.rotation_angle = 0.0

    def model(self):
        # Se calcula la matriz de transformacion de los planetas para simular su orbita y rotacion
        orbit_x = np.cos(self.angle) * self.orbit_radius
        orbit_z = np.sin(self.angle) * self.orbit_radius

        translation = Mat4.from_translation(Vec3(orbit_x, 0, orbit_z))
        rotation_y = Mat4.from_rotation(self.rotation_angle, Vec3(0, 1, 0))
        
        scale = Mat4.from_scale(Vec3(*self.scale))
        return translation @ rotation_y @ scale

    def update(self, dt):
        # Se actualiza el angulo de orbita y rotacion de los planetas
        self.angle += self.orbit_speed * dt
        if self.angle > 2 * np.pi:                                          # Evitar que el angulo de orbita de vueltas
            self.angle -= 2 * np.pi

        self.rotation_angle += self.rotation_speed * dt
        if self.rotation_angle > 2 * np.pi:                                 # Evitar que el angulo de rotacion de vueltas
            self.rotation_angle -= 2 * np.pi

    # Permite dibujar los planetas
    def draw(self):
        self._buffer.draw(GL_TRIANGLES)

# IMPORTAR MODELOS DE PLANETAS (Sacado del Aux4)
def models_from_file(path, pipeline):
    geom = tm.load(path)
    meshes = [geom] if not isinstance(geom, tm.Scene) else list(geom.geometry.values())

    models = []
    for m in meshes:
        m.apply_scale(2.0 / m.scale)
        m.apply_translation([-m.centroid[0], 0, -m.centroid[2]])
        vlist = tm.rendering.mesh_to_vertexlist(m)
        model = Model(vlist[0], vlist[4][1], vlist[3], pipeline)
        models.append(model)

    return models

# IMPORTAR MODELO DE NAVE (Igual a models from file pero cambiando la clase Model por Ship)
def ship_model_from_file(path, pipeline):
    geom = tm.load(path)
    meshes = [geom] if not isinstance(geom, tm.Scene) else list(geom.geometry.values())

    models = []
    for m in meshes:
        m.apply_scale(2.0 / m.scale)
        m.apply_translation([-m.centroid[0], 0, -m.centroid[2]])
        vlist = tm.rendering.mesh_to_vertexlist(m)
        models.append(Ship(vlist[0], vlist[4][1], vlist[3], pipeline))

    return models

#shaders
if __name__ == "__main__":
    vertex_source = """
#version 330

in vec3 position;

uniform mat4 model;
uniform mat4 transform;
uniform mat4 view = mat4(1.0);
uniform mat4 projection = mat4(1.0);

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0f);
}
    """

    fragment_source = """
#version 330

uniform vec3 color;  
out vec4 outColor;

void main() {
    outColor = vec4(color, 1.0f);  
}
    """

    #Creación del pipeline
    vert_program = pyglet.graphics.shader.Shader(vertex_source, "vertex")
    frag_program = pyglet.graphics.shader.Shader(fragment_source, "fragment")
    pipeline = pyglet.graphics.shader.ShaderProgram(vert_program, frag_program)

    #Defina sus objetos y la cámara

    # Tamaños de los planetas
    # Sol: 10
    # Mercurio: 1.5
    # Venus: 2
    # Tierra: 3
    # Marte: 2.5
    # Jupiter: 5
    # Saturno: 4.5
    # Urano: 4
    # Neptuno: 3.5

    # Colores de los planetas
    # Mercurio: [0.627, 0.322, 0.176]
    # Venus: [1, 0.647, 0]
    # Tierra: [0, 1, 0]
    # Marte: [1, 0, 0]
    # Jupiter: [0.545, 0.271, 0.075]
    # Saturno: [0.824, 0.706, 0.549]
    # Urano: [0.5, 0.8, 1.0]
    # Neptuno: [0, 0, 1]

    sun = models_from_file(__file__ + "/../Objs/sun.obj", pipeline)[0]
    sun.color = [1.0, 1.0, 0.0]
    sun.position = [-1, 0, -1]
    sun.scale = [10] * 3

    Mercurio = models_from_file(__file__ + "/../Objs/planet.obj", pipeline)[0]
    Mercurio.color = [0.5, 0.5, 0.5]
    Mercurio.scale = [1.5] * 3
    Mercurio.orbit_radius = 11
    Mercurio.orbit_speed = 1
    Mercurio.rotation_speed = 0.2

    Venus = models_from_file(__file__ + "/../Objs/planet.obj", pipeline)[0]
    Venus.color = [1, 0.647, 0]
    Venus.scale = [2] * 3
    Venus.orbit_radius = 15
    Venus.orbit_speed = 0.8
    Venus.rotation_speed = 0.5

    Tierra = models_from_file(__file__ + "/../Objs/planet.obj", pipeline)[0]
    Tierra.position = [22, 0, 0]
    Tierra.color = [0, 1, 0]
    Tierra.scale = [3] * 3
    Tierra.orbit_radius = 22
    Tierra.orbit_speed = 0.6
    Tierra.rotation_speed = 0.8

    # BONO 
    # MAS PLANETAS
    Marte = models_from_file(__file__ + "/../Objs/planet.obj", pipeline)[0]
    Marte.color = [1, 0, 0] 
    Marte.scale = [2.5] * 3
    Marte.orbit_radius = 30
    Marte.orbit_speed = 0.4
    Marte.rotation_speed = 1

    Jupiter = models_from_file(__file__ + "/../Objs/planet.obj", pipeline)[0]
    Jupiter.color = [0.545, 0.271, 0.075]
    Jupiter.scale = [5] * 3
    Jupiter.orbit_radius = 40
    Jupiter.orbit_speed = 0.3
    Jupiter.rotation_speed = 0.6

    Saturno = models_from_file(__file__ + "/../Objs/planet.obj", pipeline)[0]
    Saturno.color = [0.824, 0.706, 0.549]
    Saturno.scale = [4.5] * 3
    Saturno.orbit_radius = 50
    Saturno.orbit_speed = 0.3
    Saturno.rotation_speed = 0.4

    Urano = models_from_file(__file__ + "/../Objs/planet.obj", pipeline)[0]
    Urano.color = [0.5, 0.8, 1.0]
    Urano.scale = [4] * 3
    Urano.orbit_radius = 60
    Urano.orbit_speed = 0.2
    Urano.rotation_speed = 0.35

    Neptuno = models_from_file(__file__ + "/../Objs/planet.obj", pipeline)[0]
    Neptuno.color = [0, 0, 1]
    Neptuno.scale = [3.5] * 3
    Neptuno.orbit_radius = 65
    Neptuno.orbit_speed = 0.15
    Neptuno.rotation_speed = 0.3

    # Nave
    ship = ship_model_from_file(__file__ + "/../Objs/tie_fighter.obj", pipeline)[0]
    ship.scale = [1] * 3
    ship.color = [0.5, 0.5, 0.5]
    ship.position = [-40, 0, 0]

    # Escena
    scene = [ship, sun, Mercurio, Venus, Tierra, Marte, Jupiter, Saturno, Urano, Neptuno]

    # Camara
    cam = Camara(distance=15, height=5, initial_offset=(5, -5, -10))

    @window.event
    def on_draw():
        
        glClearColor(0.1, 0.1, 0.1, 1) 
        
        window.clear() 

        pipeline.use() 
        
        #Dibuje sus objetos (De Aux4)
        with pipeline:
            pipeline["view"] = cam.view()
            pipeline["projection"] = Mat4.perspective_projection(window.aspect_ratio, .1, 100, window.fov)

            # Dibujar el sol, planetas y nave
            for m in scene:
                pipeline["color"] = m.color
                pipeline["model"] = m.model()
                m.draw()

    def update(dt):
        #Pasa el tiempo
        window.time += dt
        
        # Actualizar la dirección de la nave (Ship) mediante yaw y pitch
        ship.update(dt)

        #Actualice la posición de la cámara
        cam.update(ship.position, ship.front)

        #Actualice los planetas para que giren
        for planeta in scene:
            planeta.update(dt)


    #Control mouse
    @window.event
    def on_mouse_motion(x, y, dx, dy):

        # Generar el movimiento de la nave con el mouse
        ship.handle_mouse_movement(dx, dy)
        
        #Actualizar la vista de la camara que sigue a la nave
        cam.update(ship.position, ship.front)


    @window.event
    def on_key_press(symbol, modifiers):
        # Se utiliza solo 'W' y 'S', con estas teclas se avanza/retrocede la nave
        if symbol == key.W:
            ship.speed = 10.0
        if symbol == key.S:
            ship.speed = -10.0

    @window.event
    def on_key_release(symbol, modifiers):
        # Se utiliza solo 'W' y 'S', con estas teclas se detiene la nave
        if symbol in [key.W, key.S]:
            ship.speed = 0

    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()