#version 330

in vec3 fragPos;
in vec2 fragTexCoord;
in vec3 fragNormal;

out vec4 outColor;

uniform sampler2D u_texture;
uniform vec3 u_color;

void main() {
    outColor =   vec4(u_color, 1.0);
}
