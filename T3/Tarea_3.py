import pyglet
from pyglet.gl import *
from pyglet.graphics.shader import Shader, ShaderProgram
import numpy as np

from grafica.scene_graph import SceneGraph
from grafica.camera import Camera, OrbitCamera, FreeCamera
from grafica.helpers import mesh_from_file
from grafica.drawables import Model

import os
# Auxiliar 6
from grafica import shapes
from grafica.drawables import Texture, Model

# Ship
import random
from pyglet.math import Mat4, Vec3, clamp
import trimesh as tm

# Camara
import grafica.transformations as tr

# BONUS
import random

# Tarea 3
# Nombre: Patricio Espinoza A.
# Seccion: 1

# Controlador
class Controller(pyglet.window.Window):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time = 0
        self.sensitivity = 0.1
        # Auxiliar 6: Agregar atributos para manejar la luz
        self.light_mode = True
        self.light_dir = np.zeros(2)
        self.light_color = np.ones(3)
        self.light_distance = 1

# Clase Ship, trabajada en tareas anteriores
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
    def draw(self, mode=None, cull_face=None):
        self._buffer.draw(GL_TRIANGLES)

# IMPORTAR MODELO DE NAVE, al igual que en tareas anteriores
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

# ==== CAMARAS ==== #
# Se implementan las distintas camaras pero a excepción de la camara que apunta al sol el resto falla en seguir correactamente a los objetivos 
# despues de t=0 (las camaras apuntan siempre a la posicion inicial).

# Basada en la camara de camera.py
class OrbitCamera(Camera):
    def __init__(self, distance, focus=np.array([0, 0, 0], dtype=np.float32), camera_type="perspective"):
        super().__init__(camera_type)
        self.distance = distance
        self.phi = 0
        self.theta = np.pi / 2
        self.focus = focus  # El punto donde la cámara estará centrada
        self.update()

    def update(self):
        # Asegura que theta esté en el rango adecuado
        if self.theta > np.pi:
            self.theta = np.pi - 0.0001
        elif self.theta < 0:
            self.theta = 0.0001

        # Actualizar la posición de la cámara en función de la distancia, phi y theta
        self.position[0] = self.distance * np.sin(self.theta) * np.sin(self.phi) + self.focus[0]
        self.position[1] = self.distance * np.cos(self.theta) + self.focus[1]
        self.position[2] = self.distance * np.sin(self.theta) * np.cos(self.phi) + self.focus[2]

    def get_view(self):
        #print(self.position, self.focus)
        lookAt_matrix = tr.lookAt(self.position, self.focus, np.array([0, 1, 0], dtype=np.float32))
        return np.reshape(lookAt_matrix, (16, 1), order="F")
    
# Basada en la camara de camera.py
class FreeCamera(Camera):
    def __init__(self, position = [0, 0, 0], focus=None, camera_type="perspective"):
        super().__init__(camera_type)
        self.position = np.array(position, dtype=np.float32)
        self.focus = focus if focus is not None else self.position + np.array([0, 0, -1], dtype=np.float32)
        self.pitch = 0
        self.yaw = 0
        self.forward = np.array([0, 0, -1], dtype=np.float32)
        self.right = np.array([1, 0, 0], dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.update()

    def update(self):
        # Limitar angulo
        self.pitch = np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        # Calcular la dirección de forward
        self.forward[0] = np.cos(self.yaw) * np.cos(self.pitch)
        self.forward[1] = np.sin(self.pitch)
        self.forward[2] = np.sin(self.yaw) * np.cos(self.pitch)
        self.forward = self.forward / np.linalg.norm(self.forward)

        # Asignar direccion por defecto para evitar errores (solucion parche)
        if np.linalg.norm(self.forward) < 1e-6:
            self.forward = np.array([0, 0, -1], dtype=np.float32)  

        # Actualizar la posición del focus hacia donde la cámara está mirando
        self.focus = self.position + self.forward

    def get_view(self):
        # Asegurar una dirección para evitar errores (solucion parche)
        if np.allclose(self.position, self.focus, atol=1e-6): 
            self.focus = self.position + np.array([0, 0, -1], dtype=np.float32)  
        # Mirar al focus designado
        lookAt_matrix = tr.lookAt(self.position, self.focus, np.array([0, 1, 0], dtype=np.float32))
        return np.reshape(lookAt_matrix, (16, 1), order="F")


# Gestor de las distintas camaras, permitiendo que se cambie entre ellas y actualizando la camara activa
class CameraController:
    def __init__(self, world):
        self.world = world  # Necesitamos acceso al grafo de escena para encontrar posiciones
        self.orbit_camera_sun = OrbitCamera(distance=20, focus=np.array([0, 0, 0], dtype=np.float32), camera_type="orthographic")
        self.orbit_camera_earth = OrbitCamera(distance=10, focus=np.array([5, 0, 0], dtype=np.float32), camera_type="perspective")
        self.orbit_camera_saturn = OrbitCamera(distance=15, focus=np.array([10, 0, 0], dtype=np.float32), camera_type="perspective")
        self.orbit_camera_binary = OrbitCamera(distance=18, focus=np.array([13, 0, 0], dtype=np.float32), camera_type="perspective")
        self.free_camera_mars = FreeCamera(position=[7.2, 0.5, 0], focus=np.array([7, 0, 0], dtype=np.float32))
        self.current_camera = self.orbit_camera_sun  # Cámara por defecto

    def switch_camera(self, key):
        # Cambio de camaras dependiendo del input (tecla)
        if key == pyglet.window.key.SPACE:
            self.current_camera = self.orbit_camera_sun
            self.current_camera.focus = self.world.find_position("sun_base")                # Seguir el Sol
        elif key == pyglet.window.key.T:
            self.current_camera = self.orbit_camera_earth
            self.current_camera.focus = self.world.find_position("earth_base")              # Seguir la Tierra
        elif key == pyglet.window.key.S:
            self.current_camera = self.orbit_camera_saturn
            self.current_camera.focus = self.world.find_position("saturn_base")             # Seguir Saturno
        elif key == pyglet.window.key.U:
            self.current_camera = self.orbit_camera_binary
            self.current_camera.focus = self.world.find_position("binary_center_base")      # Seguir el centro binario
        elif key == pyglet.window.key.M:
            self.current_camera = self.free_camera_mars
            # Seguir la nave en Marte
            self.current_camera.position = self.world.find_position("ship_base") + np.array([0, 1, 0])  # Añadir altura respecto a la nave
            self.current_camera.focus = self.world.find_position("ship_base")

    def update(self, world):
        # Recibe world para updatear las componentes de posicion en cada camara
        self.world = world
        self.current_camera.update()

    # Funciones referenciado a cada camara
    def get_view(self):
        return self.current_camera.get_view()

    def get_projection(self):
        return self.current_camera.get_projection()

    # Movimiento del moouse
    def on_mouse_motion(self, dx, dy):
        if isinstance(self.current_camera, OrbitCamera):
            self.current_camera.phi += dx * 0.001
            self.current_camera.theta += dy * 0.001
        elif isinstance(self.current_camera, FreeCamera):
            self.current_camera.yaw += dx * 0.001
            self.current_camera.pitch += dy * 0.001
            self.current_camera.pitch = np.clip(self.current_camera.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        self.current_camera.update()

# ==== TEXTURAS ==== #
# 1. FUNCION CREATE SPHERE
def create_sphere(definition):
    # coordenadas de posición
    positions = np.zeros((definition+1)*(definition+1)*3, dtype=np.float32) 
    # coordenadas de texturas
    uv = np.zeros((definition+1)*(definition+1)*2, dtype=np.float32)
    dtheta = 2*np.pi / definition
    dphi = np.pi / definition
    r = 1.0

    for i in range(definition+1):
        for j in range(definition+1):
            idx = 3*(i*(definition+1) + j)
            tidx = 2*(i*(definition+1) + j)
            theta = j*dtheta
            phi = i*dphi
            positions[idx:idx+3] = [np.sin(phi)*np.cos(theta)*r, np.cos(phi)*r, np.sin(phi)*np.sin(theta)*r]
            uv[tidx:tidx+2] = [j/definition, i/definition]

    indices = np.zeros(6*definition*definition, dtype=np.int32)
    for i in range(definition):
        for j in range(definition):
            idx = 6*(i*definition + j)
            # t0
            indices[idx:idx+3] = [i*(definition+1) + j, i*(definition+1) + j+1, (i+1)*(definition+1) + j]
            # t1
            indices[idx+3:idx+6] = [i*(definition+1) + j+1, (i+1)*(definition+1) + j+1, (i+1)*(definition+1) + j]

    return Model(positions, uv, None, indices)

def generate_ring(definition):
    # coordenadas de posición
    positions = np.zeros((definition)*3*2, dtype=np.float32) 
    # coordenadas de texturas
    uv = np.zeros((definition)*2*2, dtype=np.float32)
    dtheta = 2*np.pi / definition
    r1 = 0.5
    r2 = 1.0

    for i in range(definition):
        idx = 3*i
        tidx = 2*i
        theta = i*dtheta
        positions[idx:idx+3] = [np.cos(theta)*r2, np.sin(theta)*r2, 0.0]
        if i%2==0:
            uv[tidx:tidx+2] = [1, 1]
        else:
            uv[tidx:tidx+2] = [1, 0]

    for i in range(definition):
        idx = 3*(i+definition)
        tidx = 2*(i+definition)
        theta = i*dtheta
        positions[idx:idx+3] = [np.cos(theta)*r1, np.sin(theta)*r1, 0.0]
        if i%2==0:
            uv[tidx:tidx+2] = [0, 1]
        else:
            uv[tidx:tidx+2] = [0, 0]

    indices = np.zeros(6*definition, dtype=np.int32)
    for i in range(definition-1):
        idx = 6*i
        # t0
        indices[idx:idx+3] = [i, i+1, i+definition]
        # t1
        indices[idx+3:idx+6] = [i+1, i+definition+1, i+definition]
   
    # Completamos el anillo
    # indices[3*definition:] = [definition, definition - 1, 0]
    idx = 6*(definition-1)
    indices[idx:idx+3] = [definition-1, 0, 2*definition-1]
    indices[idx+3:idx+6] = [2*definition-1, definition, 0]

    return Model(positions, uv, None, indices)

# ==== TEXTURAS ==== #
# 3. MODIFCAR EL SHADER PARA QUE ACEPTE LAS COORDENADAS DE LAS TEXTURAS
if __name__ == "__main__":
    vert_source = """
#version 330
in vec3 position;
in vec2 texCoord;

uniform mat4 u_model;
uniform mat4 u_view = mat4(1.0);
uniform mat4 u_projection = mat4(1.0);

out vec2 fragTexCoord;

void main() {
    fragTexCoord = texCoord;
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0f);
}
    """
    frag_source = """
#version 330

in vec2 fragTexCoord;
out vec4 outColor;

uniform sampler2D u_texture;

void main() {
    outColor = texture(u_texture, fragTexCoord);
}
    """

    pipeline = ShaderProgram(Shader(vert_source, "vertex"), Shader(frag_source, "fragment"))

    cam = OrbitCamera(distance=20, focus=np.array([0, 0, 0], dtype=np.float32), camera_type="orthographic")
    cam.width = 800
    cam.height = 800

    # Crear el grafo de escena (SceneGraph) con la cam base
    world = SceneGraph(cam)

    # Pasamos el grafo de la escena al controlador de la camara
    cam_controller = CameraController(world)  

    # window config
    window = Controller(cam_controller.current_camera.width, cam_controller.current_camera.height, "Tarea 3")
    window.set_exclusive_mouse(True)
    
    # Mesh que se usarana para los planetas y anillos en el grafo de escena
    sphere = create_sphere(36)
    sphere.init_gpu_data(pipeline)

    ring = generate_ring(36)
    ring.init_gpu_data(pipeline)

    # Cargar el modelo 3D de la nave desde el archivo .obj
    ship_mesh = ship_model_from_file(__file__ + "/../Objs/tie_fighter.obj", pipeline)[0]
    ship_mesh.color = [0.5, 0.5, 0.5]

    # ==== TEXTURAS ==== #
    # 2. USAR CLASE TEXTURE PARA INICIALIZAR CADA PLANETA CON SU TEXTURA CORRESPONDIENTE
    # ==== GRAFOS DE ESCENA ==== #
    # 1. TODA LA ESCENA PRINCIPAL EN GRAFO DE ESCENA, USAR CLASE SCENEGRAPH
    world = SceneGraph(cam_controller)
    # 2. LA ESCENA SE COMPONE DE: SOL, MERCURIO, VENUS, TIERRA, LUNA, MARTE, JUPITER, SATURNO, URANO, NEPTUNO
    # Sol
    world.add_node("sun_to_root")
    world.add_node("sun_base", attach_to="sun_to_root", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/sun.jpg"), scale=[1.5, 1.5, 1.5])
    # Mercurio
    # Orbita irregular, aumentando y disminuyendo periodicamente el radio -> se utiliza coseno
    world.add_node("mercury_to_sun", attach_to="sun_to_root")
    world.add_node("mercury_base", attach_to="mercury_to_sun", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/mercury.jpg"), scale=[.3, .3, .3], position=[2,0,0])
    # Venus
    # Orbita irregular, aumentando y disminuyendo periodicamente el radio -> se utiliza coseno
    world.add_node("venus_to_sun", attach_to="sun_to_root")
    world.add_node("venus_base", attach_to="venus_to_sun", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/venus.jpg"), scale=[.5, .5, .5], position=[3,0,0])  
    # Tierra
    world.add_node("earth_to_sun", attach_to="sun_to_root")
    world.add_node("earth_base", attach_to="earth_to_sun", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/earth.jpg"), scale=[.4, .4, .4], position=[5,0,0])
    # Luna
    world.add_node("moon_to_earth", attach_to="earth_base")
    world.add_node("moon_base", attach_to="moon_to_earth", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/moon.jpg"), scale=[.15, .15, .15], position=[5.2,0,0])
    # Marte
    world.add_node("mars_to_sun", attach_to="sun_to_root")
    world.add_node("mars_base", attach_to="mars_to_sun", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/mars.jpg"), scale=[.3, .3, .3], position=[7,0,0])
    
    # Marte: Nave 
    world.add_node("ship_to_mars", attach_to="mars_base")
    world.add_node("ship_base", attach_to="ship_to_mars", mesh=ship_mesh, pipeline=pipeline, scale=[1.5, 1.5, 1.5], position=[7.2,0,0])

    # Jupiter
    world.add_node("jupiter_to_sun", attach_to="sun_to_root")
    world.add_node("jupiter_base", attach_to="jupiter_to_sun", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/jupiter.jpg"), scale=[.7, .7, .7], position=[8,0,0])
    # Saturno
    world.add_node("saturn_to_sun", attach_to="sun_to_root")
    world.add_node("saturn_base",attach_to="saturn_to_sun", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/saturn.jpg"), scale=[.8, .8, .8], position=[10,0,0])
    world.add_node("saturn_ring", attach_to="saturn_base", mesh=ring, pipeline=pipeline, texture=Texture(__file__+"/../assets/saturn_ring.png"), scale=[2, 2, 2], rotation=[np.pi/2, 0, 0], cull_face=False)
    
    ### Sistema binario para Urano y Neptuno ###

    # Nodo comun
    world.add_node("binary_center_to_sun", attach_to="sun_to_root")
    world.add_node("binary_center_base", attach_to="binary_center_to_sun", position=[13, 0, 0])

    # Neptuno y Urano orbitan alrededor del nodo comun
    # Urano
    world.add_node("uranus_to_binary", attach_to="binary_center_base")
    world.add_node("uranus_base", attach_to="uranus_to_binary", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/uranus.jpg"), scale=[.6, .6, .6])
    # Neptuno
    world.add_node("neptune_to_binary", attach_to="binary_center_base")
    world.add_node("neptune_base", attach_to="neptune_to_binary", mesh=sphere, pipeline=pipeline, texture=Texture(__file__+"/../assets/neptune.jpg"), scale=[.6, .6, .6])

    # BONUS 
    # Parámetros del cinturón de asteroides
    num_asteroids = 100
    asteroid_min_radius = 7.5  # Distancia mínima (entre Marte y Júpiter)
    asteroid_max_radius = 8.5  # Distancia máxima (justo antes de Júpiter)
    asteroid_size = 0.05       # Tamaño de los asteroides

    # Agregar asteroides al grafo de escena
    for i in range(num_asteroids):
        # Crea un nodo para cada asteroide
        world.add_node(f"asteroid_{i}_to_sun", attach_to="sun_to_root")
        # Posición aleatoria en un anillo entre las órbitas de Marte y Júpiter
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(asteroid_min_radius, asteroid_max_radius)
        position = [radius * np.cos(angle), 0, radius * np.sin(angle)]
        
        # Escala aleatoria para variar el tamaño de los asteroides
        scale = [asteroid_size, asteroid_size, asteroid_size]
        
        # Agrega el asteroide con textura si es necesario
        world.add_node(f"asteroid_{i}_base", attach_to=f"asteroid_{i}_to_sun", mesh=sphere, pipeline=pipeline, 
                    texture=Texture(__file__+"/../assets/extra_moon4.jpg"), position=position, scale=scale)

    @window.event
    def on_draw():
        window.clear()
        glClearColor(.1,.1,.1,1)
        # 3D
        glEnable(GL_DEPTH_TEST)
        # Transparencia
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable( GL_BLEND );
        # Usar la cámara actual
        # -> Configurar la cámara actual antes de dibujar
        current_view = cam_controller.get_view()
        current_projection = cam_controller.get_projection()
        with pipeline:
            world.draw()

        glDisable(GL_DEPTH_TEST)

    @window.event
    def on_key_press(symbol, modifiers):
        # Cambiar cámara según la tecla presionada
        cam_controller.switch_camera(symbol)

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        # Rotar la cámara activa con el mouse
        cam_controller.on_mouse_motion(dx, dy)

    def update(dt):
        # ==== GRAFO DE ESCENA ==== #
        # 3. CARACTERISTICAS DE ROTACION Y TRASLACION
        # -> Velocidad de traslacion es proporcional al radio de esta
        # -> Velocidad de rotacion es proporcional al tamaño del planeta

        def vel_translation(radius):
            return 1.0 / radius         # Velocidad inversamente proporcional al radio (mas cerca, mas rápido)
        
        def vel_rotation(size):
            return 1.0 / size           # Velocidad inversamente proporcional al tamaño (mas grande, mas lento)
    
        # Traslación de los planetas alrededor del sol
        world["sun_to_root"]["rotation"][1] = -window.time
        world["mercury_to_sun"]["rotation"][1] = window.time * vel_translation(2.5)  # Radio de Mercurio
        world["venus_to_sun"]["rotation"][1] = window.time * vel_translation(3.5)    # Radio de Venus
        world["earth_to_sun"]["rotation"][1] = window.time * vel_translation(5.0)    # Radio de la Tierra
        world["moon_to_earth"]["rotation"][1] = -window.time * vel_translation(5.2)  # Orbita de la Luna
        world["mars_to_sun"]["rotation"][1] = window.time * vel_translation(7.0)     # Radio de Marte
        world["jupiter_to_sun"]["rotation"][1] = window.time * vel_translation(8.0)  # Radio de Júpiter
        world["saturn_to_sun"]["rotation"][1] = window.time * vel_translation(10.0)  # Radio de Saturno
        world["saturn_ring"]["rotation"][1] = window.time * vel_translation(10.0)    # Radio del anillo de Saturno

        # Rotacion de los planetas sobre su propio eje
        world["mercury_to_sun"]["rotation"][0] = window.time * vel_rotation(0.3)     # Tamaño de Mercurio
        world["venus_to_sun"]["rotation"][0] = window.time * vel_rotation(0.5)       # Tamaño de Venus
        world["earth_to_sun"]["rotation"][0] = window.time * vel_rotation(0.4)       # Tamaño de la Tierra
        world["moon_to_earth"]["rotation"][0] = window.time * vel_rotation(0.15)     # Tamaño de la Luna
        world["mars_to_sun"]["rotation"][0] = window.time * vel_rotation(0.3)        # Tamaño de Marte
        world["jupiter_to_sun"]["rotation"][0] = window.time * vel_rotation(0.7)     # Tamaño de Júpiter
        world["saturn_to_sun"]["rotation"][0] = window.time * vel_rotation(0.8)      # Tamaño de Saturno

        # ==== GRAFO DE ESCENA ==== #
        # 4.2 En Marte debe existir una nave que se mueva de manera errática sobre su superficie a una altura
        # constante con respecto al planeta.
        # -> Posiciones aleatorias para la nave (x, z) en la superficie de Marte
        radius_mars = 0.4                                                                                                 # Radio alrededor de Marte donde la nave se moverá
        speed_ship = 2.0                  
                                                                                        # Velocidad de la nave
        world["ship_to_mars"]["position"][0] = radius_mars * np.cos(window.time * speed_ship + random.uniform(-0.2, 0.2)) # Posiciones aleatorias para la nave (x, z) en la superficie de Marte
        world["ship_to_mars"]["position"][1] = 0.5                                                                        # Altura constante sobre Marte
        world["ship_to_mars"]["position"][2] = radius_mars * np.sin(window.time * speed_ship + random.uniform(-0.2, 0.2)) # Posiciones aleatorias para la nave (x, z) en la superficie de Marte
        world["ship_to_mars"]["rotation"][1] = window.time * 1.5                                                          # Rotación de la nave para que esté "mirando" en la dirección de movimiento

        # ==== GRAFO DE ESCENA ==== #
        # 4.1 Mercurio y Venus deben tener órbitas irregulares, aumentando y disminuyendo periódicamente el radio de sus órbitas.
        # Orbita irregular de mercurio y venus usando coseno 
        world["mercury_base"]["position"][0] = 2.5 + 0.5 * np.cos(window.time * 4)
        world["venus_base"]["position"][0] = 3.5 + 0.5 * np.cos(window.time * 3.5)

        # 4.3 Neptuno y Urano deben orbitar entre ellos con respecto a un centro común de manera similar a un sistema binario.
        # Y este centro común órbita con respecto a El Sol.
        # -> Sistema binario: Urano y Neptuno
        radius_binary = 0.8                     # Radio de la orbita de Neptuno y Urano alrededor del centro comun
        speed_binary = 1.0                      # Velocidad de rotación de Neptuno y Urano alrededor del centro comun

        # -> Urano orbita alrededor del nodo comun
        world["uranus_to_binary"]["position"][0] = radius_binary * np.cos(window.time * speed_binary)
        world["uranus_to_binary"]["position"][2] = radius_binary * np.sin(window.time * speed_binary)
        # -> Neptuno orbita alrededor del nodo comun
        world["neptune_to_binary"]["position"][0] = -radius_binary * np.cos(window.time * speed_binary)
        world["neptune_to_binary"]["position"][2] = -radius_binary * np.sin(window.time * speed_binary)
        # -> Nodo comun orbita alrededor del sol
        world["binary_center_to_sun"]["rotation"][1] = window.time * 0.5  

        # 4.4 Sol pulsa cambiando su tamaño
        pulse_rate = 5                              # Velocidad de pulso del sol
        world["sun_base"]["scale"] = [1.5 + 0.1 * np.cos(window.time * pulse_rate), 1.5 + 0.1 * np.cos(window.time * pulse_rate), 1.5 + 0.1 * np.cos(window.time * pulse_rate)]


        # BONUS
        # Rotación de los asteroides alrededor del sol
        # Rotación de los asteroides
        asteroid_rotation_speed = 0.1               # Velocidad de rotación de los asteroides

        for i in range(num_asteroids):
            # Los asteroides rotan alrededor del Sol, similar a los planetas
            world[f"asteroid_{i}_to_sun"]["rotation"][1] = window.time * asteroid_rotation_speed
            
        # Actualizar
        world.update()

        # Actualizar camara
        # Si la camara activa es de tipo OrbitCamera, hacemos update de su focus para seguir el nuevo objetivo
        if isinstance(cam_controller.current_camera, OrbitCamera):
            if cam_controller.current_camera == cam_controller.orbit_camera_sun:
                node_to_follow = "sun_base"
            elif cam_controller.current_camera == cam_controller.orbit_camera_earth:
                node_to_follow = "earth_base"
            elif cam_controller.current_camera == cam_controller.orbit_camera_saturn:
                node_to_follow = "saturn_base"
            elif cam_controller.current_camera == cam_controller.orbit_camera_binary:
                node_to_follow = "binary_center_base"
        
            # Ajustar el focus de la camara actual
            cam_controller.current_camera.focus = world.find_position(node_to_follow)
        
        # Si la camara activa FreeCamera (sigue la nave en Marte)
        if isinstance(cam_controller.current_camera, FreeCamera):
            # Ajustar la pos y focus de la cam segun la de la nave
            cam_controller.current_camera.position = world.find_position("ship_base") + np.array([0, 1, 0])
            cam_controller.current_camera.focus = world.find_position("ship_base")

        # Cam update
        cam_controller.update(world)

        window.time+=dt

    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()