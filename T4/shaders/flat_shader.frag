#version 330 core
flat in vec3 fragNormal;
out vec4 fragColor;
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 objectColor;
void main()
{
    float diff = max(dot(normalize(fragNormal), normalize(lightDir)), 0.0);
    vec3 diffuse = diff * lightColor;
    fragColor = vec4(diffuse * objectColor, 1.0);
}