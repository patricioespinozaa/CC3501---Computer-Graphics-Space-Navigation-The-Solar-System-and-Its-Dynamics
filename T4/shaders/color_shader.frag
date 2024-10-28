#version 330 core

uniform vec3 u_color;  // Color uniforme

out vec4 outColor;

void main()
{
    outColor = vec4(u_color, 1.0f);  // Color fijo sin interacción con luces
}
