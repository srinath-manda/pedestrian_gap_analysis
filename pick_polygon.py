"""
pick_polygon.py — Extract first frame and open an interactive polygon picker in browser.

Usage:
    python pick_polygon.py --video "C:\path\to\video.mp4"

This will:
1. Save the first frame as first_frame.jpg
2. Open a browser window where you click to define the polygon
3. Print the --polygon argument to use with main.py
"""

import argparse
import base64
import json
import os
import sys
import webbrowser
import http.server
import threading
import cv2


def extract_first_frame(video_path: str, out_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("ERROR: Could not read first frame.")
    cv2.imwrite(out_path, frame)
    h, w = frame.shape[:2]
    print(f"[PickPolygon] First frame saved → {out_path}  ({w}x{h})")
    return w, h


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Polygon Picker — Pedestrian Crossing Zone</title>
<style>
  body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: Arial, sans-serif; }}
  h2 {{ text-align: center; padding: 10px; color: #00d4ff; }}
  p  {{ text-align: center; color: #aaa; margin: 4px; }}
  #wrap {{ text-align: center; position: relative; display: inline-block; }}
  canvas {{ cursor: crosshair; border: 2px solid #00d4ff; display: block; }}
  #controls {{ text-align: center; margin: 12px; }}
  button {{ margin: 6px; padding: 10px 24px; font-size: 15px; border: none;
            border-radius: 6px; cursor: pointer; }}
  #btnUndo  {{ background: #e67e22; color: white; }}
  #btnReset {{ background: #c0392b; color: white; }}
  #btnDone  {{ background: #27ae60; color: white; font-weight: bold; }}
  #result   {{ background: #0d0d1a; border: 1px solid #00d4ff; border-radius: 8px;
               padding: 16px; margin: 12px auto; max-width: 800px;
               font-family: monospace; font-size: 14px; display: none; }}
  #copyBtn  {{ background: #2980b9; color: white; margin-top: 8px; }}
</style>
</head>
<body>
<h2>🚶 Pedestrian Crossing Zone — Polygon Picker</h2>
<p>Left-click on the image to place polygon vertices around the crossing zone.</p>
<p>Click near the first point to close the polygon, or press <b>Done</b>.</p>
<div id="controls">
  <button id="btnUndo"  onclick="undo()">↩ Undo Last</button>
  <button id="btnReset" onclick="reset()">🗑 Reset</button>
  <button id="btnDone"  onclick="done()">✅ Done — Copy Command</button>
</div>
<div style="text-align:center">
  <div id="wrap">
    <canvas id="canvas"></canvas>
  </div>
</div>
<div id="result">
  <b>Run this command in your terminal:</b><br><br>
  <span id="cmd"></span><br><br>
  <button id="copyBtn" onclick="copyCmd()">📋 Copy Command</button>
</div>

<script>
const IMG_SRC = "data:image/jpeg;base64,{B64}";
const VIDEO_PATH = {VIDEO_PATH_JSON};
const OUTPUT_DIR = {OUTPUT_DIR_JSON};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const img = new Image();
let points = [];
let scale = 1.0;

img.onload = function() {{
  const maxW = Math.min(window.innerWidth - 40, 1280);
  scale = maxW / img.width;
  canvas.width  = img.width  * scale;
  canvas.height = img.height * scale;
  draw();
}};
img.src = IMG_SRC;

canvas.addEventListener("click", function(e) {{
  const rect = canvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left);
  const cy = (e.clientY - rect.top);
  // Snap-close if near first point
  if (points.length >= 3) {{
    const dx = cx - points[0][0] * scale;
    const dy = cy - points[0][1] * scale;
    if (Math.sqrt(dx*dx + dy*dy) < 12) {{ done(); return; }}
  }}
  points.push([Math.round(cx / scale), Math.round(cy / scale)]);
  draw();
}});

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  if (points.length === 0) return;

  // Draw filled polygon (semi-transparent)
  ctx.beginPath();
  ctx.moveTo(points[0][0]*scale, points[0][1]*scale);
  for (let i = 1; i < points.length; i++)
    ctx.lineTo(points[i][0]*scale, points[i][1]*scale);
  ctx.closePath();
  ctx.fillStyle = "rgba(0, 212, 255, 0.15)";
  ctx.fill();
  ctx.strokeStyle = "#00d4ff";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Draw points
  points.forEach((p, i) => {{
    ctx.beginPath();
    ctx.arc(p[0]*scale, p[1]*scale, i===0 ? 8 : 5, 0, 2*Math.PI);
    ctx.fillStyle = i===0 ? "#ff6b6b" : "#00d4ff";
    ctx.fill();
    ctx.fillStyle = "white";
    ctx.font = "bold 12px Arial";
    ctx.fillText(i+1, p[0]*scale + 8, p[1]*scale - 6);
  }});
}}

function undo()  {{ points.pop(); draw(); }}
function reset() {{ points = []; draw(); document.getElementById("result").style.display="none"; }}

function done() {{
  if (points.length < 3) {{ alert("Need at least 3 points!"); return; }}
  const flat = points.map(p => p[0]+","+p[1]).join(" ");
  const cmd = `python -m pedestrian_gap_analysis.main --video ${{VIDEO_PATH}} --output ${{OUTPUT_DIR}} --polygon "${{flat}}"`;
  document.getElementById("cmd").textContent = cmd;
  document.getElementById("result").style.display = "block";
  document.getElementById("result").scrollIntoView({{behavior:"smooth"}});
  console.log("POLYGON_RESULT:" + JSON.stringify(points));
}}

function copyCmd() {{
  const text = document.getElementById("cmd").textContent;
  navigator.clipboard.writeText(text).then(() => alert("Command copied to clipboard!"));
}}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Interactive polygon picker for crossing zone")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="output", help="Output directory for main pipeline")
    parser.add_argument("--port", type=int, default=8765, help="Local server port")
    args = parser.parse_args()

    # Extract first frame
    frame_path = os.path.join(os.path.dirname(args.video), "first_frame.jpg")
    extract_first_frame(args.video, frame_path)

    # Encode frame as base64
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # Build HTML
    html = HTML_TEMPLATE.format(
        B64=b64,
        VIDEO_PATH_JSON=json.dumps(args.video),
        OUTPUT_DIR_JSON=json.dumps(args.output),
    )

    html_path = os.path.join(os.path.dirname(args.video), "polygon_picker.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[PickPolygon] Opening polygon picker in your browser...")
    print(f"[PickPolygon] HTML saved → {html_path}")
    print(f"\n  1. Click on the image to place polygon vertices around the crossing zone")
    print(f"  2. Click 'Done' to generate the run command")
    print(f"  3. Copy and run the command in your terminal\n")

    webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
