#!/usr/bin/env python3
"""Visualize SfM pointcloud + cameras in the browser.

Usage:
    python3 view_pointcloud.py sfm_result.json
"""

import http.server
import json
import os
import sys
import threading
import webbrowser
import numpy as np

PORT = 8770


def generate_html(sfm_path):
    with open(sfm_path) as f:
        data = json.load(f)

    pts = np.array(data["points3d"])
    cols = np.array(data["point_colors"])
    cams = [np.array(c) for c in data["cameras"]]

    # Center the point cloud
    center = pts.mean(axis=0)
    pts_centered = pts - center
    scale = np.abs(pts_centered).max()

    # Normalize to [-1, 1]
    pts_norm = pts_centered / max(scale, 1e-6)

    # Camera positions (world coordinates)
    cam_positions = []
    cam_forwards = []
    for vm in cams:
        R = np.array(vm[:3])[:, :3]
        t = np.array(vm[:3])[:, 3]
        pos = -R.T @ t
        pos_norm = (pos - center) / max(scale, 1e-6)
        fwd = R[2, :]  # Z axis = forward
        cam_positions.append(pos_norm.tolist())
        cam_forwards.append(fwd.tolist())

    # Build HTML
    pts_js = json.dumps(pts_norm.tolist())
    cols_js = json.dumps((cols / 255.0).tolist())
    cam_pos_js = json.dumps(cam_positions)
    cam_fwd_js = json.dumps(cam_forwards)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>Point Cloud Viewer</title>
<style>
    body {{ margin: 0; overflow: hidden; background: #111; font-family: system-ui; color: white; }}
    canvas {{ display: block; width: 100vw; height: 100vh; }}
    #info {{ position: fixed; top: 10px; left: 10px; font-size: 13px; background: rgba(0,0,0,0.7); padding: 8px 12px; border-radius: 6px; }}
    #controls {{ position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 11px; color: rgba(255,255,255,0.4); }}
</style>
</head>
<body>
<div id="info">{len(pts)} points, {len(cams)} cameras</div>
<div id="controls">Drag: orbit | Scroll: zoom</div>
<canvas id="c"></canvas>
<script>
const pts = {pts_js};
const cols = {cols_js};
const camPos = {cam_pos_js};
const camFwd = {cam_fwd_js};

const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2');

let yaw = 0.5, pitch = -0.3, dist = 3;
let dragging = false, lastX = 0, lastY = 0;

canvas.onmousedown = e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }};
onmouseup = () => dragging = false;
onmousemove = e => {{
    if (!dragging) return;
    yaw += (e.clientX - lastX) * 0.005;
    pitch += (e.clientY - lastY) * 0.005;
    pitch = Math.max(-Math.PI/2+0.01, Math.min(Math.PI/2-0.01, pitch));
    lastX = e.clientX; lastY = e.clientY;
}};
canvas.onwheel = e => {{ dist *= 1 + e.deltaY * 0.001; dist = Math.max(0.1, Math.min(20, dist)); e.preventDefault(); }};
canvas.oncontextmenu = e => e.preventDefault();

function resize() {{
    canvas.width = innerWidth * devicePixelRatio;
    canvas.height = innerHeight * devicePixelRatio;
    gl.viewport(0, 0, canvas.width, canvas.height);
}}
onresize = resize; resize();

// Shaders
const vs = `#version 300 es
uniform mat4 mvp;
uniform float pointSize;
in vec3 pos;
in vec3 col;
out vec3 vCol;
void main() {{
    gl_Position = mvp * vec4(pos, 1.0);
    gl_PointSize = pointSize / gl_Position.w;
    vCol = col;
}}`;
const fs = `#version 300 es
precision highp float;
in vec3 vCol;
out vec4 frag;
void main() {{
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    if (dot(c,c) > 1.0) discard;
    frag = vec4(vCol, 1.0);
}}`;

function mkShader(type, src) {{
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
    return s;
}}
const prog = gl.createProgram();
gl.attachShader(prog, mkShader(gl.VERTEX_SHADER, vs));
gl.attachShader(prog, mkShader(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(prog);

const uMvp = gl.getUniformLocation(prog, 'mvp');
const uPSize = gl.getUniformLocation(prog, 'pointSize');

// Point data
const posData = new Float32Array(pts.flat());
const colData = new Float32Array(cols.flat());
const posBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
gl.bufferData(gl.ARRAY_BUFFER, posData, gl.STATIC_DRAW);
const colBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
gl.bufferData(gl.ARRAY_BUFFER, colData, gl.STATIC_DRAW);

// Camera frustum lines
const camLines = [];
for (let i = 0; i < camPos.length; i++) {{
    const p = camPos[i];
    const f = camFwd[i];
    const len = 0.05;
    camLines.push(p[0], p[1], p[2]);
    camLines.push(p[0]+f[0]*len, p[1]+f[1]*len, p[2]+f[2]*len);
}}
const camBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, camBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(camLines), gl.STATIC_DRAW);

// Line shader (no point size)
const lvs = `#version 300 es
uniform mat4 mvp;
in vec3 pos;
void main() {{ gl_Position = mvp * vec4(pos, 1.0); }}`;
const lfs = `#version 300 es
precision highp float;
uniform vec3 lineColor;
out vec4 frag;
void main() {{ frag = vec4(lineColor, 1.0); }}`;
const lprog = gl.createProgram();
gl.attachShader(lprog, mkShader(gl.VERTEX_SHADER, lvs));
gl.attachShader(lprog, mkShader(gl.FRAGMENT_SHADER, lfs));
gl.linkProgram(lprog);
const luMvp = gl.getUniformLocation(lprog, 'mvp');
const luColor = gl.getUniformLocation(lprog, 'lineColor');

// Camera dot data (larger points for camera positions)
const camPosBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, camPosBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(camPos.flat()), gl.STATIC_DRAW);
const camColBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, camColBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(Array(camPos.length * 3).fill(0).map((_, i) => i % 3 === 1 ? 1 : 0.3)), gl.STATIC_DRAW);

function mat4Perspective(fov, aspect, near, far) {{
    const f = 1/Math.tan(fov/2);
    return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0];
}}
function mat4Mul(a, b) {{
    const r = new Array(16).fill(0);
    for (let i=0;i<4;i++) for (let j=0;j<4;j++) for (let k=0;k<4;k++) r[i*4+j]+=a[i*4+k]*b[k*4+j];
    return r;
}}

function render() {{
    requestAnimationFrame(render);
    gl.clearColor(0.07, 0.07, 0.07, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.PROGRAM_POINT_SIZE);

    const aspect = canvas.width / canvas.height;
    const proj = mat4Perspective(1.0, aspect, 0.01, 100);

    const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
    const view = [
        cy,sp*sy,-cp*sy,0,
        0,cp,sp,0,
        sy,-sp*cy,cp*cy,0,
        0,0,-dist,1
    ];
    const mvp = mat4Mul(view, proj);

    // Draw points
    gl.useProgram(prog);
    gl.uniformMatrix4fv(uMvp, false, mvp);
    gl.uniform1f(uPSize, 30);
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, pts.length);

    // Draw camera positions (larger green dots)
    gl.uniform1f(uPSize, 60);
    gl.bindBuffer(gl.ARRAY_BUFFER, camPosBuf);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, camColBuf);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, camPos.length);

    // Draw camera direction lines
    gl.useProgram(lprog);
    gl.uniformMatrix4fv(luMvp, false, mvp);
    gl.uniform3f(luColor, 1, 1, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, camBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, camPos.length * 2);
}}
requestAnimationFrame(render);
</script>
</body>
</html>"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 view_pointcloud.py sfm_result.json")
        sys.exit(1)

    sfm_path = sys.argv[1]
    html = generate_html(sfm_path)

    tmpdir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(tmpdir, "_pointcloud.html")
    with open(html_path, "w") as f:
        f.write(html)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=tmpdir, **kw)
        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"Viewer at http://localhost:{PORT}/_pointcloud.html")
    webbrowser.open(f"http://localhost:{PORT}/_pointcloud.html")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
