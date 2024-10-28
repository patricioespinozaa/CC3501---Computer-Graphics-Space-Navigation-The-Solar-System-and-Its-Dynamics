#version 330 core
in vec3 fragNormal;
out vec4 fragColor;
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 objectColor;
void main()
{
    float diff = max(dot(normalize(fragNormal), normalize(lightDir)), 0.0);
    float levels = 4.0; // Niveles de intensidad
    float toonShade = ceil(diff * levels) / levels;
    vec3 diffuse = toonShade * lightColor;
    fragColor = vec4(diffuse * objectColor, 1.0);
}