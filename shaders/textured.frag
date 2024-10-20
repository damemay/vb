#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D texImage;

void main() {
    // outColor = vec4(fragColor, 1.0);
    outColor = texture(texImage, fragUV);
}
