#version 330 core

in vec3 position;

uniform mat4 u_model = mat4(1.0);
uniform mat4 u_view = mat4(1.0);
uniform mat4 u_projection = mat4(1.0);

void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0f);
}
