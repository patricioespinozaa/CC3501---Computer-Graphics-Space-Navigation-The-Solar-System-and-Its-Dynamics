#version 330

in vec3 fragPos;
in vec3 fragNormal;
in vec2 fragTexCoord;

out vec4 outColor;

// Material
struct Material {
    vec3 diffuse;
    vec3 ambient;
    vec3 specular;
    float shininess;
};

uniform int N=6;                  // N para discretizacion
uniform Material u_material;

// Lighting
uniform vec3 u_viewPos;

// Directional
struct DirectionalLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform DirectionalLight u_dirLight;

// Pointlight
const int MAX_POINT_LIGHTS = 16;
uniform int u_numPointLights;

struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

uniform PointLight u_pointLights[MAX_POINT_LIGHTS];

// Discretizar
float discretizeLight(float intensity) {
    float step = 1.0 / float(N - 1);            // Determina el tamaño de cada paso
    return step * floor(intensity / step);      // Discretiza el valor
}

vec3 computeDirectionalLight(vec3 normal, DirectionalLight light) {
    //ambient
    vec3 ambient = light.ambient * u_material.ambient;

    // diffuse
    float diff = max(dot(normal, light.direction), 0.0f);
    diff = discretizeLight(diff);                                       // discretizar intensidad
    vec3 diffuse = light.diffuse * (diff * u_material.diffuse);

    
    return (ambient + diffuse);
}

vec3 computePointLight(vec3 normal, PointLight light) {
    // attenuation
    vec3 lightVec = light.position - fragPos;
    float distance = length(lightVec);
    float attenuation = 1.0f / ( light.linear * distance + light.quadratic * distance * distance + light.constant );

    // ambient
    vec3 ambient = light.ambient * u_material.ambient;

    // diffuse
    vec3 lightDir = normalize(lightVec);
    float diff = max(dot(normal, lightDir), 0.0f);
    diff = discretizeLight(diff);                                                                       // discretizar intensidad
    vec3 diffuse = light.diffuse * (diff * u_material.diffuse);

    // specular blinn phong
    vec3 halfwayDir = normalize(lightDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0f), u_material.shininess);
    vec3 specular = light.specular * (spec * u_material.specular);

    return (ambient + diffuse) * attenuation;
}

void main()
{
    vec3 normal = normalize(fragNormal);

    vec3 result = vec3(0.0);

    result += computeDirectionalLight(normal, u_dirLight);

    if (u_numPointLights > 0 && u_numPointLights <= MAX_POINT_LIGHTS) {
        for (int i = 0; i < u_numPointLights; i++)
            result += computePointLight(normal, u_pointLights[i]);
    }

    outColor = vec4(result, 1.0f);
}