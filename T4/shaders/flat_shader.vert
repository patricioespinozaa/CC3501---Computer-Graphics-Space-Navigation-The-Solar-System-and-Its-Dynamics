#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

flat out vec3 fragNormal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    fragNormal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
