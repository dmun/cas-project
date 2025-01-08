#version 410

in vec3 vertexPosition;

out vec4 fragColor;
uniform vec2 positions[1000];

void main()
{
    vec2 position = positions[gl_VertexID];
    vec2 normalizedPos = vec2(
        (position.x / 800.0) * 2.0 - 1.0,
        -((position.y / 600.0) * 2.0 - 1.0)
    );
    gl_Position = vec4(normalizedPos, 0.0, 1.0);
    gl_PointSize = 10.0;
}
