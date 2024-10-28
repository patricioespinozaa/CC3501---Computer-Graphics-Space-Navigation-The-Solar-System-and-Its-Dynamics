import pyglet
import numpy as np
from pyglet.gl import *

# Tarea 1 - Sistema Solar
# Nombre: Patricio Espinoza A.
# Curso: CC3501


# Se utiliza adicionalmente la libreria random para la generación aleatoria de estrellas y asteroides
import random 

WIDTH = 1000
HEIGHT = 1000
DEFINITION = 36 

window = pyglet.window.Window(WIDTH, HEIGHT, "Tarea 1 - Sistema Solar")

def crear_planeta(x, y, r, g, b, radius):
    N = DEFINITION
    # Discretizamos un circulo en DEFINITION pasos
    # Cada punto tiene 3 coordenadas y 3 componentes de color
    # Consideramos tambien el centro del circulo
    positions = np.zeros((DEFINITION + 1)*3, dtype=np.float32) 
    colors = np.zeros((DEFINITION + 1) * 3, dtype=np.float32)
    dtheta = 2*np.pi / DEFINITION

    for i in range(DEFINITION):                                                                 # Para cada punto
        theta = i*dtheta                                                                        # Angulo del punto                                                          
        positions[i*3:(i+1)*3] = [x + np.cos(theta)*radius, y + np.sin(theta)*radius, 0.0]      # Coordenadas del punto
        colors[i*3:(i+1)*3] = [r, g, b]                                                         # Color del punto

    # Se agrega el centro
    positions[3*DEFINITION:] = [x, y, 0.0]    # Coordenadas del centro
    colors[3*DEFINITION:] = [r, g, b]         # Color del centro

    # Retorna las posiciones y colores
    return positions, colors


def create_planeta_indices():
    # Calculo de los indices
    indices = np.zeros(3*( DEFINITION + 1 ), dtype=np.int32)                                    # Cada triangulo tiene 3 indices 
    
    # Para cada punto
    for i in range(DEFINITION):
        # Cada triangulo se forma por el centro, el punto actual y el siguiente
        indices[3*i: 3*(i+1)] = [DEFINITION, i, i+1]
   
    # Completamos el circulo
    indices[3*DEFINITION:] = [DEFINITION, DEFINITION - 1, 0]

    # Retorna los indices
    return indices


# Clase para dibujar cuerpos celestes
class CuerpoCeleste:
    # Recibe las coordenadas x, y, los colores r, g, b y el radio del planeta
    def __init__(self, x, y, r, g, b, radius, pipeline):
        self.position = [x, y]
        self.color = [r, g, b]
        self.radius = radius
        self._buffer = pipeline.vertex_list_indexed(DEFINITION+1, GL_TRIANGLES, create_planeta_indices())

    # Dibuja el planeta
    def draw(self):
        cdata, ccolors = crear_planeta(*self.position, *self.color, self.radius)
        self._buffer.position[:] = cdata
        self._buffer.color[:] = ccolors
        self._buffer.draw(GL_TRIANGLES)


def crear_orbita(x, y, r, g, b, radius):
    # Definir la cantidad de puntos en la órbita
    positions = np.zeros((DEFINITION) * 3, dtype=np.float32)       # Cada punto tiene 3 coordenadas
    colors = np.zeros((DEFINITION) * 3, dtype=np.float32)          # Cada punto tiene 3 componentes de color
    dtheta = 2 * np.pi / DEFINITION                                # Ángulo entre cada segmento

    # Calculamos las posiciones y colores de cada punto
    for i in range(DEFINITION):
        theta = i * dtheta
        positions[i * 3:(i + 1) * 3] = [x + np.cos(theta) * radius, y + np.sin(theta) * radius, 0.0]
        colors[i * 3:(i + 1) * 3] = [r, g, b]

    return positions, colors


# Clase para dibujar órbitas
class Orbita:
    # Recibe las coordenadas x, y, los colores r, g, b y el radio de la órbita
    def __init__(self, x, y, r, g, b, radius, pipeline):
        self.position = [x, y]
        self.color = [r, g, b]
        self.radius = radius
        self._buffer = pipeline.vertex_list(DEFINITION, GL_LINE_LOOP)

    # Dibuja la órbita
    def draw(self):
        orbit_data, orbit_colors = crear_orbita(*self.position, *self.color, self.radius)
        self._buffer.position[:] = orbit_data
        self._buffer.color[:] = orbit_colors
        self._buffer.draw(GL_LINE_LOOP)



#### BONUS ####

# Planeta tierra con diseño
def crear_planeta_tierra(x, y, radius):
    N = DEFINITION                                                              # Cantidad de puntos
    positions = np.zeros((N + 1) * 3, dtype=np.float32)                         # Cada punto tiene 3 coordenadas
    colors = np.zeros((N + 1) * 3, dtype=np.float32)                            # Cada punto tiene 3 componentes de color
    dtheta = 2 * np.pi / N                                                      # Ángulo entre cada segmento

    # Calculamos las posiciones y colores de cada punto
    for i in range(N):
        theta = i * dtheta
        positions[i * 3:(i + 1) * 3] = [x + np.cos(theta) * radius, y + np.sin(theta) * radius, 0.0]
        
        # Simular zonas de tierra (verdes) y de agua (azules)
        if 0 <= theta < np.pi / 3 or 2 * np.pi / 3 <= theta < np.pi or 4 * np.pi / 3 <= theta < 5 * np.pi / 3:
            # Azul para océanos
            colors[i * 3:(i + 1) * 3] = [0, 0, 1]
        else:
            # Verde para continentes
            colors[i * 3:(i + 1) * 3] = [0, 1, 0]

    # Se agrega el centro
    positions[3 * N:] = [x, y, 0.0]       
    colors[3 * N:] = [0, 1, 0] 

    return positions, colors


# Clase para dibujar la Tierra con diseño
class Tierra(CuerpoCeleste):
    def draw(self):
        cdata, ccolors = crear_planeta_tierra(*self.position, self.radius)
        self._buffer.position[:] = cdata
        self._buffer.color[:] = ccolors
        self._buffer.draw(GL_TRIANGLES)

# Clase para dibujar las estrellas de fondo
class Estrella:
    # Recibe las coordenadas x, y, los colores r, g, b, el tamaño y el pipeline
    def __init__(self, x, y, r, g, b, size, pipeline):
        self.position = [x, y, 0.0]
        self.color = [r, g, b]
        self.size = size
        self._buffer = pipeline.vertex_list(1, GL_POINTS,                              # Dibujar puntos
                                            position=('f', self.position),             # Posición del punto
                                            color=('f', self.color))                   # Color del punto

    def draw(self):
        self._buffer.draw(GL_POINTS)                                                   # Dibujar el punto

# Permite generar estrellas aleatorias en la escena 
def generar_estrellas(cantidad, pipeline):
    estrellas = []
    for _ in range(cantidad):
        x = random.uniform(-1, 1)                                                      # Posición X aleatoria en el rango de la ventana
        y = random.uniform(-1, 1)                                                      # Posición Y aleatoria en el rango de la ventana
        r = random.uniform(0.7, 1.0)                                                   # Colores para simular brillo de estrellas
        g = random.uniform(0.7, 1.0)
        b = random.uniform(0.7, 1.0)
        size = random.uniform(1, 3)                                                    # Tamaño pequeño para las estrellas
        estrellas.append(Estrella(x, y, r, g, b, size, pipeline))                      # Se agrega la estrella a la lista de estrellas 
    return estrellas

# Extra: Cinturon de asteroides
def generar_cinturon_asteroides(cantidad, radio_interno, radio_externo, pipeline):
    asteroides = []                                                           # Lista de asteroides
    size = 0.005                                                              # Tamaño estándar para todos los asteroides
    
    for _ in range(cantidad):
        # Generar un ángulo aleatorio alrededor del Sol
        theta = random.uniform(0, 2 * np.pi)
        
        # Generar una distancia aleatoria dentro del rango del cinturón
        radio = random.uniform(radio_interno, radio_externo)
        
        # Convertir el ángulo y la distancia a coordenadas x, y
        x = radio * np.cos(theta)
        y = radio * np.sin(theta)
        
        # Color café oscuro para cada asteroide
        r, g, b = 0.36, 0.25, 0.20
        
        # Se agrega el asteroide a la lista luego de crearlo con la clase CuerpoCeleste
        asteroides.append(CuerpoCeleste(x, y, r, g, b, size, pipeline))

    return asteroides


# Crear el programa de shaders (sin modificaciones)
if __name__ == "__main__":
    
    vertex_source = """
#version 330

in vec3 position;
in vec3 color;

out vec3 fragColor;

void main() {
    fragColor = color;
    gl_Position = vec4(position, 1.0f);
}
    """

    fragment_source = """
#version 330

in vec3 fragColor;
out vec4 outColor;

void main()
{
    outColor = vec4(fragColor, 1.0f);
}
    """

    vert_program = pyglet.graphics.shader.Shader(vertex_source, "vertex")
    frag_program = pyglet.graphics.shader.Shader(fragment_source, "fragment")
    pipeline = pyglet.graphics.shader.ShaderProgram(vert_program, frag_program)

    # Crear los cuerpos celestes y órbitas

    # Sol en el centro
    sol = CuerpoCeleste(0, 0, 255, 255, 0, 0.2, pipeline)  # El Sol, color amarillo y el más grande, en el centro

    # Mercurio
    mercurio = CuerpoCeleste(0.21, 0.21, 0.627, 0.322, 0.176, 0.02, pipeline)       # Color marrón y el más pequeño
    orbita_mercurio = Orbita(0, 0, 1, 1, 1, 0.3, pipeline)                          # Órbita más cercana al Sol

    # Venus
    venus = CuerpoCeleste(0.22, 0.3, 1, 0.647, 0, 0.03, pipeline)                   # Color naranjo y segundo más cercano
    orbita_venus = Orbita(0, 0, 1, 1, 1, 0.37, pipeline)                            # Órbita de Venus

    # Tierra
    tierra = Tierra(0.32, 0.34, 0, 1, 0, 0.035, pipeline)                           # Color verde y tercer más cercano
    orbita_tierra = Orbita(0, 0, 1, 1, 1, 0.46, pipeline)                           # Órbita de la Tierra

    # Marte
    marte = CuerpoCeleste(0.45, 0.31, 1, 0, 0, 0.04, pipeline)                      # Color rojo y cuarto más cercano
    orbita_marte = Orbita(0, 0, 1, 1, 1, 0.55, pipeline)                            # Órbita de Marte

    # Júpiter
    jupiter = CuerpoCeleste(0.3, 0.65, 0.82, 0.71, 0.55, 0.1, pipeline)             # El más grande y quinto más cercano
    orbita_jupiter = Orbita(0, 0, 1, 1, 1, 0.73, pipeline)                          # Órbita de Júpiter

    # Saturno
    orbita_saturno = Orbita(0, 0, 1, 1, 1, 0.85, pipeline)                          # Órbita de Saturno
    saturno = CuerpoCeleste(0, 0.85, 0.65, 0.45, 0.2, 0.09, pipeline)               # Sexto más cercano y con anillo
    anillo_saturno = CuerpoCeleste(0, 0.85, 0.95, 0.84, 0.4, 0.12, pipeline)          # Anillo de Saturno
    
    # Urano
    urano = CuerpoCeleste(0.88, 0.43, 0.68, 0.85, 0.90, 0.07, pipeline)             # Color celeste y séptimo más cercano
    orbita_urano = Orbita(0, 0, 1, 1, 1, 0.98, pipeline)                            # Órbita de Urano

    # Neptuno
    neptuno = CuerpoCeleste(0.76, 0.8, 0, 0, 1, 0.082, pipeline)                    # Color azul y el más lejano
    orbita_neptuno = Orbita(0, 0, 1, 1, 1, 1.1, pipeline)                           # Órbita de Neptuno


    #### BONUS ####

    # Luna
    luna = CuerpoCeleste(0.3, 0.4, 0.5, 0.5, 0.5, 0.015, pipeline)                  # Color gris y órbita alrededor de la Tierra

    # Phobos
    phobos = CuerpoCeleste(0.51, 0.27, 0.72, 0.45, 0.20, 0.01, pipeline)           # Color cobrizo y órbita alrededor de Marte

    # Deimos
    deimos = CuerpoCeleste(0.45, 0.24, 0.50, 0.45, 0.40, 0.008, pipeline)          # Color cobrizo-gris y órbita alrededor de Marte
    
    #  Estrellas de fondo
    estrellas = generar_estrellas(1000, pipeline)                                   # Genera 100 estrellas aleatorias

    # Crear el cinturón de asteroides
    cinturon_asteroides = generar_cinturon_asteroides(200, 0.6, 0.65, pipeline)    # Genera 200 asteroides en el cinturon de asteroides

    @window.event
    def on_draw():
        glClearColor(0.1, 0.1, 0.1, 0.0)

        # Esta linea limpia la pantalla entre frames
        window.clear()

        with pipeline:
            # BONUS: Dibuja las estrellas de fondo
            for estrella in estrellas:
                estrella.draw()  

            # EXTRA: Cinturon de asteroides
            for asteroide in cinturon_asteroides:
                asteroide.draw()  

            # TAREA BASE
            sol.draw()                      # Dibuja el Sol
            
            orbita_mercurio.draw()
            mercurio.draw()                 # Dibuja Mercurio

            orbita_venus.draw()
            venus.draw()                    # Dibuja Venus

            orbita_tierra.draw()
            tierra.draw()                   # Dibuja la Tierra

            orbita_marte.draw()
            marte.draw()                    # Dibuja Marte

            orbita_jupiter.draw()
            jupiter.draw()                  # Dibuja Júpiter

            orbita_saturno.draw()
            anillo_saturno.draw()
            saturno.draw()                  # Dibuja Saturno
            

            orbita_urano.draw()
            urano.draw()                    # Dibuja Urano

            orbita_neptuno.draw()
            neptuno.draw()                  # Dibuja Neptuno
            
            #### BONUS ####

            luna.draw()                     # Dibuja la Luna
            phobos.draw()                   # Dibuja Phobos
            deimos.draw()                   # Dibuja Deimos
            

    @window.event
    def update(dt):
        pass

    pyglet.app.run()

    
