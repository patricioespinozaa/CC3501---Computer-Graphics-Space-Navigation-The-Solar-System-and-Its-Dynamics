#version 330

//Escriba aquí su flat shader
//Este debe considerar el efecto de Directional lights y Pointlights
//También debe evitar la interpolación de las normales (revise el enunciado)

//funcion que procesa directional light
vec3 computeDirectionalLight(vec3 normal, DirectionalLight light) {
    ???
}

//funcion que procesa pointlight
vec3 computePointLight(vec3 normal, PointLight light) {
    ???
}


void main()
{
    outColor = ???
}