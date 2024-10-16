#version 450
#extension GL_EXT_buffer_reference: require

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;

struct Vertex {
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer{
    Vertex vertices[];
};

layout(push_constant) uniform constants {
    mat4 render_matrix;
    VertexBuffer vertex_buffer;
} PushConstants;

void main() {
    Vertex vert = PushConstants.vertex_buffer.vertices[gl_VertexIndex];
    gl_Position = PushConstants.render_matrix * vec4(vert.position, 1.0f);
    fragColor = vec3(vert.color);
    fragUV = vec2(vert.uv_x, vert.uv_y);
}
