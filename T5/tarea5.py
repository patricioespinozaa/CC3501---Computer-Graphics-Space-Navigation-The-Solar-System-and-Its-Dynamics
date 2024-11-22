from networkx.algorithms.bipartite import collaboration_weighted_projected_graph
import numpy as np
import os
from pyglet import window, gl, app, clock

import trimesh as tm
from utils import helpers, scene_graph, drawables, camera
from utils.colliders import Sphere as ColliderSphere, CollisionManager

                                    ########################### 2.1 SIMULACION FISICA ###########################
### Clase para los atributos de cada particula ###
# Posee • Masa • Radio • Posición • Velocidad
class Particle:
    def __init__(self, name, mass, radius, position, velocity, color=[1.0, 1.0, 1.0]):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=np.float32) 
        self.velocity = np.array(velocity, dtype=np.float32)
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.color = color
    
    def update_position(self, dt):
        self.position += self.velocity * dt

### Fuerza gravitacional ###
# Uso del metodo de Newton mediante el metodo de Verlet
def compute_gravitational_force(p1, p2, G=1.0):
    distance_vector = p2.position - p1.position                                        # Calculo de la distancia entre las partículas p1 y p2
    distance = np.linalg.norm(distance_vector)
    
    if distance < 1e-5:                                                                # Evitar la división por cero si las partículas están muy cerca
        return np.array([0.0, 0.0, 0.0])
    
    force_magnitude = G * (p1.mass * p2.mass) / (distance ** 2)                        # Calculo de la magnitud de la fuerza gravitacional
    force_vector = (force_magnitude / distance) * distance_vector                      # Direccionar la fuerza en función del vector de distancia normalizado
    return force_vector

# Calcula las aceleraciones de las partículas en función de las fuerzas gravitacionales
def compute_accelerations(particles, sun):
    for i, p1 in enumerate(particles):
        total_force = np.array([0.0, 0.0, 0.0])                                         # Inicializar la fuerza neta

        for j, p2 in enumerate(particles):                                              # Fuerza gravitacional de otras partículas
            if i != j:
                total_force += compute_gravitational_force(p1, p2, G=GRAVITY)

        total_force += compute_gravitational_force(p1, sun, G=GRAVITY)                  # Fuerza gravitacional del Sol
        p1.acceleration = total_force / p1.mass                                         # Actualizar la aceleración

                                    ########################### 2.2 MANEJO DE COLISIONES ###########################
# Manejo de colisiones entre partículas 2.2.1
# Simula un choque entre dos particulas, donde ambas son consumidas para crear una nueva
def handle_particle_collisions(world, collision_manager):
    global particles

    new_particles = []
    to_remove = set()                                                               # Evitar procesar dos veces las partículas eliminadas

    for p1 in particles:
        collisions = collision_manager.check_collision(p1.name)                     # Obtener las partículas con las que colisiona p1
        for p2_name in collisions:
            p2 = next((p for p in particles if p.name == p2_name), None)

            if p2 and p1.name not in to_remove and p2.name not in to_remove:
                # Crear una nueva partícula combinada
                new_mass = p1.mass + p2.mass                                        # m = m_i + m_j
                new_radius = p1.radius + p2.radius                                  # r = r_i + r_j
                new_position = (p1.position + p2.position) / 2                      # p = (p_i + p_j)/2
                new_velocity = (p1.velocity + p2.velocity) / 2                      # v = v_i + v_j
                new_color = (np.array(p1.color) + np.array(p2.color)) / 2

                new_particle = Particle(
                    name=f"particle_{len(particles) + len(new_particles)}",         # Nombre único para la nueva partícula
                    mass=new_mass,                                                  # Crear la nueva partícula asignando los valores calculados       
                    radius=new_radius,
                    position=new_position,
                    velocity=new_velocity,
                    color=new_color,
                )
                new_particles.append(new_particle)                                  # Añadir al listado de nuevas particulas

                # Eliminar nodos de la escena y colisionadores
                world.remove_node(p1.name)
                world.remove_node(p2.name)
                collision_manager.colliders = [
                    c for c in collision_manager.colliders if c.name != p1.name and c.name != p2.name
                ]

                # Marcar las partículas para eliminación
                to_remove.add(p1.name)
                to_remove.add(p2.name)

    particles = [p for p in particles if p.name not in to_remove]                    # Actualizar la lista de partículas eliminando las marcadas

    # Agregar las nuevas partículas al grafo y a collision_manager
    for particle in new_particles:
        particles.append(particle)                                                   # Añadir las nuevas partículas a la lista de partículas
        world.add_node(                                                              # Añadirlas como nodos del grafo de escena
            particle.name,
            attach_to=sun.name,
            mesh=sphere2,  
            pipeline=phong_pipeline,  
            material=drawables.Material(
                diffuse=particle.color.tolist(),
                specular=[0.5, 0.5, 0.5],
                shininess=16.0,
            ),
            scale=np.full(3, particle.radius),
            position=particle.position,
        )
        collider = ColliderSphere(particle.name, particle.radius)                     # Crear un collider para la nueva partícula
        collider.set_position(particle.position)                                      # Establecer la posición del collider
        collision_manager.add_collider(collider)                                      # Añadir el collider al collision_manager

# Manejo de colisiones entre partículas y el Sol 2.2.2
# Refleja la velocidad de la partícula respecto a la normal de la colisión y evita que la partícula quede dentro del Sol
def handle_sun_collisions(world, collision_manager):
    sun_collider = collision_manager[sun.name]                                        # Obtener el collider del Sol

    for particle in particles:
        particle_collider = collision_manager[particle.name]

        if particle_collider and particle_collider.detect_collision(sun_collider):    # Verificar si hay colisión con el Sol
            initial_speed = round(np.linalg.norm(particle.velocity))

            # Reflejar la velocidad respecto a la normal
            collision_normal = (particle.position - sun.position)
            collision_normal /= np.linalg.norm(collision_normal)
            particle.velocity -= 2 * np.dot(particle.velocity, collision_normal) * collision_normal
     
            speed_after_collision = round(np.linalg.norm(particle.velocity))
            #print(f"Colisión con el Sol ¿misma velocidad?: {initial_speed==speed_after_collision}")     # Chequear que tienen la misma velocidad, round para evitar pequeñas diferencias decimales

            # Asegurarse de que la partícula no quede dentro del Sol
            overlap = particle.radius + sun.radius - np.linalg.norm(particle.position - sun.position)
            particle.position += collision_normal * overlap

            # Actualizar la posición del collider
            collision_manager.set_position(particle.name, particle.position)
            world[particle.name]["position"] = particle.position.tolist()


########################### CODIGO BASE ###########################
class Controller(window.Window):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input = np.zeros(3)
        self.speed = 5

class Sphere(drawables.Model):
    def __init__(self, resolution=20):
        position_data = np.zeros(6 * resolution * resolution)
        index_data = np.zeros(6 * (2 * resolution - 1) * (resolution - 1))
        normal_data = np.zeros(6 * resolution * resolution)

        delta_phi = np.pi / (resolution - 1)
        delta_theta = 2 * np.pi / (resolution - 1)

        vcount = 0
        for i in range(2 * resolution):
            for j in range(resolution):
                phi = j * delta_phi
                theta = i * delta_theta
                position_data[vcount : vcount + 3] = [
                    np.cos(theta) * np.sin(phi),
                    np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                ]
                normal_data[vcount : vcount + 3] = [
                    np.cos(theta) * np.sin(phi),
                    np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                ]
                vcount += 3

        icount = 0
        for i in range(2 * resolution - 1):
            for j in range(resolution - 1):
                current = i * resolution + j
                index_data[icount : icount + 3] = [
                    current + 1,
                    current,
                    current + resolution,
                ]
                index_data[icount + 3 : icount + 6] = [
                    current + resolution,
                    current + resolution + 1,
                    current + 1,
                ]
                icount += 6

        super().__init__(position_data, None, normal_data, index_data)
########################### ----------- ###########################

# Sacado de la tarea 4 para importar un modelo
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
        
        self._buffer = pipeline.vertex_list_indexed(len(vertices)//3, gl.GL_TRIANGLES, indices)
        self._buffer.position = vertices

    def draw(self, mode):
        self._buffer.draw(mode)


# Variables iniciales
SUN_MASS = 20
SUN_RADIUS = 1.0
GRAVITY = 2

if __name__ == "__main__":
    controller = Controller(1000, 1000, "Tarea 5")

    ### CODIGO BASE ###
    cam = camera.OrbitCamera(10, camera.Camera.ORTHOGRAPHIC)
    cam.width = controller.width
    cam.height = controller.height

    world = scene_graph.SceneGraph(cam)

    shaders_folder = os.path.join(os.path.dirname(__file__), "shaders")
    pipeline = helpers.init_pipeline(
        shaders_folder + "/color_mesh_lit.vert", shaders_folder + "/color_mesh_lit.frag"
    )
    lpipeline = helpers.init_pipeline(
        shaders_folder + "/color_mesh.vert", shaders_folder + "/color_mesh.frag"
    )

    # Mesh Sphere
    mesh = Sphere(36)
    mesh.init_gpu_data(pipeline)
    ### ------------ ###

    ### Creacion de particulas y sol en la simulacion ###
    # Aqui opte por usar el mesh de la tarea 4 al igual que el phong_pipeline trabajado en ella 
    # ya que el Sphere(36) me daba problemas con la iluminacion

    # Mesh Sphere
    phong_pipeline = helpers.init_pipeline(shaders_folder + "/phong.vert", shaders_folder + "/phong.frag")
    root = os.path.dirname(__file__)
    sphere2 = helpers.mesh_from_file(root + "/sphere.obj")[0]["mesh"]
    sphere2.init_gpu_data(phong_pipeline)

    ### SOL ###
    # Posee posicion, masa y radio, pero no velocidad
    sun = Particle(name="sun", mass=SUN_MASS, radius=SUN_RADIUS, position=[0, 0, 0], velocity=[0, 0, 0])
    world.add_node("sun_to_root")
    world.add_node(
        sun.name,
        attach_to="sun_to_root",
        mesh=sphere2,
        pipeline=phong_pipeline,
        material=drawables.Material(diffuse=[1.0, 0.8, 0.3]),
        scale=np.full(3, sun.radius),
        color=[1.0, 0.8, 0.3],
    )

                                ########################### 2.2 MANEJO DE COLISIONES ###########################
    # Crear un CollisionManager para manejar los colliders
    collision_manager = CollisionManager()

    ### COLISIONES CON EL SOL ###
    sun_collider = ColliderSphere(sun.name, sun.radius)                                         # Crear un collider para el Sol
    sun_collider.set_position(sun.position)                                                     # Establecer la posición del collider
    collision_manager.add_collider(sun_collider)                                                # Añadir el collider al collision_manager

    # Lista de particulas 
    particles = []
    # Crear 10 partículas con masa 1 y radio 0.1
    nro_particulas = 15
    for i in range(nro_particulas): 
        # 2.1 Simulacion Fisica (inicializacion de particulas con sus variables)
        name = f"particle_{i}"                                                                  # Identificador
        radius =min(0.1 + i*0.005, SUN_RADIUS-0.25)                                             # Radio menor al radio del sol
        mass = radius / 0.1                                                                     # Masa proporcional al radio
        position = np.random.uniform(-5, 5, 3).astype(np.float32)                               # Estado inicial aleatorio
        #velocity = np.random.uniform(-1, 1, 3)                                                  # Velocidad inicial aleatoria
        velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)                                 # Velocidad inicial nula para testeo de gravedad
        acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float32)                              # Aceleracion inicial nula
        color = np.random.uniform(0, 1, 3)                                                      # Color aleatorio
        particle = Particle(name, mass, radius, position, velocity, color)
        particles.append(particle)
        
        # 2.2 Manejo de colisiones
        collider = ColliderSphere(name, radius)                                                 # Crear collider para la particula
        collider.set_position(position)                                                         # Establecer posicion
        collision_manager.add_collider(collider)                                                # Añadir collider a collision_manager

        # Agregar cada partícula como un nodo en el grafo de escena
        world.add_node(
            particle.name,
            attach_to=sun.name,  
            mesh=sphere2,
            pipeline=phong_pipeline,  
            material= drawables.Material(
                diffuse=color.tolist(),
                specular=[0.5, 0.5, 0.5],
                shininess=16.0
            ),
            scale=np.full(3, particle.radius),
            position=position,
        )

    # Luz direccional para iluminar las partículas
    world.add_node("directional_light", light=drawables.DirectionalLight(), pipeline=phong_pipeline, rotation=[-np.pi/4, 0, 0])

    @controller.event
    def on_draw():
        controller.clear()
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        world.draw()

    @controller.event
    def on_key_press(symbol, modifiers):
        if symbol == window.key.W:
            controller.input[1] = 1
        if symbol == window.key.S:
            controller.input[1] = -1

        if symbol == window.key.A:
            controller.input[0] = 1
        if symbol == window.key.D:
            controller.input[0] = -1

    @controller.event
    def on_key_release(symbol, modifiers):
        if symbol == window.key.W or symbol == window.key.S:
            controller.input[1] = 0

        if symbol == window.key.A or symbol == window.key.D:
            controller.input[0] = 0

    def update(dt):
        cam.phi += controller.input[0] * controller.speed * dt
        cam.theta += controller.input[1] * controller.speed * dt

        compute_accelerations(particles, sun)                                          # Calcular aceleraciones iniciales

        for p in particles:
            # Actualizar posición con Verlet
            p.position += p.velocity * dt + 0.5 * p.acceleration * (dt ** 2)

            # Actualizar la posición en el CollisionManager
            collision_manager.set_position(p.name, p.position)
            world[p.name]["position"] = p.position.tolist()

        # Calcular nuevas aceleraciones después del movimiento
        compute_accelerations(particles, sun)

        # Actualizar velocidades usando las nuevas aceleraciones
        for p in particles:
            p.velocity += 0.5 * (p.acceleration + (p.acceleration)) * dt

        # Manejar colisiones
        handle_particle_collisions(world, collision_manager)
        handle_sun_collisions(world, collision_manager)

        #### 2.3 RENDIMIENTO ####
        fps = 1 / dt                                                                       # Calcular los FPS
        print(f"FPS: {fps:.2f}")

        world.update()
        cam.update()

    clock.schedule_interval(update, 1 / 60)
    app.run()
