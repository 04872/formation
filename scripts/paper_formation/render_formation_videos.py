#!/usr/bin/env python3
"""Render recorded UAV trajectories; no dynamics or planning is recomputed."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2] / 'results/paper_formation/uav_dynamics_20261116'
FFMPEG = '/home/eai/projects/env-rebuilt/bin/ffmpeg'
COLORS = ['#e63946', '#0077b6', '#2a9d50', '#ae4bd0', '#ed9a16', '#00a5a5']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def open_run(count, mode):
    path = ROOT / f'random_pillars_{mode}_triangle6_spacing3.5_seed0_uav_mapseed1_n{count}_width0.3'
    data = np.load(path / 'trajectory.npz')
    return {'data': data, 'metrics': json.loads((path / 'metrics.json').read_text()),
            'count': count, 'mode': mode}


def map_point(point, rect, bounds):
    x, y, w, h = rect
    xmin, xmax, ymin, ymax = bounds
    scale = min(w / (xmax - xmin), h / (ymax - ymin))
    ox = x + (w - scale * (xmax - xmin)) / 2
    oy = y + (h - scale * (ymax - ymin)) / 2
    return (ox + (point[0] - xmin) * scale, oy + (ymax - point[1]) * scale), scale


def view(image, run, tick, rect, bounds, detail=False):
    dest_x, dest_y, width, height = rect
    layer = Image.new("RGB", (width, height), "#f6f8fb")
    draw = ImageDraw.Draw(layer)
    rect = (0, 0, width, height)
    data = run['data']
    states = data['states']; idx = min(tick, len(states) - 1)
    points = states[idx, :, :2]
    ref = data['reference']; assignment = data['assignment_history'][idx]
    q = data['q_history'][idx]; targets = q[:2] + (ref @ q[2:].reshape(2, 2).T)[assignment]
    goal = data['goal_center'] + data['centered_reference']
    x, y, w, h = rect
    draw.rectangle((x, y, x+w, y+h), fill='#f6f8fb', outline='#aab4c4', width=1)
    def xy(p):
        return map_point(p, rect, bounds)[0]
    for x0, y0, x1, y1 in data['rectangles']:
        if x1 < bounds[0] or x0 > bounds[1] or y1 < bounds[2] or y0 > bounds[3]:
            continue
        a, b = xy((x0, y1)), xy((x1, y0))
        draw.rectangle((a[0], a[1], b[0], b[1]), fill='#374151')
    def polygon(points, color, width):
        indices = ConvexHull(points).vertices
        line = [xy(points[i]) for i in indices] + [xy(points[indices[0]])]
        draw.line(line, fill=color, width=width, joint='curve')
    polygon(goal, '#8e7ad1', 2)
    polygon(targets, '#ed9d22', 2)
    polygon(points, '#263e64', 2)
    start = max(0, idx - 90)
    for robot, color in enumerate(COLORS):
        path = [xy(p) for p in states[start:idx+1:2, robot, :2]]
        if len(path) > 1:
            draw.line(path, fill=color, width=2)
        p, scale = map_point(points[robot], rect, bounds)
        radius = float(data['robot_radius']) * scale
        draw.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius),outline=color,width=1)
        r = 4 if detail else 3
        draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r), fill=color, outline='white')
        target = xy(targets[robot]); s = 4
        draw.line((target[0]-s,target[1],target[0]+s,target[1]),fill='#da8500',width=2)
        draw.line((target[0],target[1]-s,target[0],target[1]+s),fill='#da8500',width=2)
        if detail:
            draw.line((p[0],p[1],target[0],target[1]),fill='#c2a878',width=1)
    image.paste(layer, (dest_x, dest_y))


def draw_case(run, tick, width, height, comparison=False):
    d = run['data']; m = run['metrics']; idx = min(tick, len(d['states'])-1)
    img = Image.new('RGB', (width,height), '#ffffff'); draw = ImageDraw.Draw(img)
    status = m['termination'].upper() if tick >= len(d['states'])-1 else 'RUNNING'
    time = min(tick, len(d['states'])-1) * float(d['dt'])
    target = d['goal'] + d['reference'][d['assignment_history'][idx]]
    error = np.linalg.norm(d['states'][idx,:,:2]-target,axis=1).max()
    speed = np.linalg.norm(d['uav_states'][idx,:,3:6],axis=1).max()
    title = f"{run['count']} pillars  |  {run['mode'].upper()}  |  {status}"
    if comparison:
        draw.text((12,7), title, font=font(17,True), fill='#172234')
        draw.text((12,31),f'global {tick*float(d["dt"]):.1f}s  |  case {time:.1f}s  |  vmax {speed:.2f}m/s  |  goal error {error:.2f}m',font=font(13),fill='#36455a')
        view(img, run, tick, (10,55,width-20,height-65), (0,50,0,40))
    else:
        draw.text((25,15),title,font=font(25,True),fill='#172234')
        draw.text((25,50),f't = {time:.1f} s    max speed = {speed:.2f} m/s    max goal error = {error:.2f} m',font=font(18),fill='#36455a')
        draw.text((25,83),'Full map',font=font(16,True),fill='#36455a')
        draw.text((805,83),'Formation detail',font=font(16,True),fill='#36455a')
        view(img, run, tick, (25,110,750,570), (0,50,0,40))
        center = d['states'][idx,:,:2].mean(axis=0); half = 8
        view(img, run, tick, (805,110,450,450), (center[0]-half,center[0]+half,center[1]-half,center[1]+half), True)
        draw.text((805,580),'Dark: actual   Orange: reference',font=font(16),fill='#36455a')
        draw.text((805,607),'Purple: goal   +: reference targets',font=font(16),fill='#36455a')
        draw.text((805,648),'Recorded states  |  4x playback',font=font(15),fill='#6b7280')
    return img


def encode(path, size, fps):
    cmd = [FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
           '-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-','-an','-c:v','libx264',
           '-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps',type=int,default=20)
    parser.add_argument('--playback-speed',type=float,default=4.0)
    args = parser.parse_args()
    runs = [open_run(count,mode) for count in (50,100,150) for mode in ('uniform','affine')]
    output=ROOT/'videos';output.mkdir(exist_ok=True)
    dt=float(runs[0]['data']['dt']); sim_step=args.playback_speed / args.fps
    def index(frame): return int(round(frame * sim_step / dt))
    frames=[math.floor((len(r['data']['states'])-1)*dt/sim_step)+1+2*args.fps for r in runs]
    (output/'manifest.json').write_text(json.dumps({'fps':args.fps,'playback_speed':args.playback_speed,'final_frame_hold_seconds':2,'sampling':'Recorded states only, no invented controls; simulation time synchronized.'},indent=2))
    sheets=[]
    for run,count in zip(runs,frames):
        path=output/f"{run['count']}_{run['mode']}.mp4";proc=encode(path,(1280,720),args.fps)
        for frame in range(count):
            img=draw_case(run,index(frame),1280,720);proc.stdin.write(img.tobytes())
            if frame in (count//2,count-1):
                img.thumbnail((640,360));sheets.append(img.copy())
        proc.stdin.close()
        if proc.wait()!=0: raise RuntimeError(f'ffmpeg failed: {path}')
        print(f'completed {path} ({count} frames)',flush=True)
    proc=encode(output/'comparison.mp4',(1280,1120),args.fps)
    for frame in range(max(frames)):
        img=Image.new('RGB',(1280,1120),'white')
        global_tick = min(index(frame), max(len(r['data']['states'])-1 for r in runs))
        for i,run in enumerate(runs):img.paste(draw_case(run,global_tick,640,360,True),((i%2)*640,(i//2)*360))
        footer = ImageDraw.Draw(img)
        footer.text((18,1091),'Dark: actual formation   |   Orange: reference + targets   |   Purple: goal formation   |   4x playback',font=font(16),fill='#36455a')
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed: comparison.mp4')
    sheet=Image.new('RGB',(1280,2160),'white')
    for i,img in enumerate(sheets):sheet.paste(img,((i%2)*640,(i//2)*360))
    sheet.save(output/'contact_sheet.png')
    print(f'completed {output}/comparison.mp4 and contact_sheet.png',flush=True)


if __name__ == '__main__':
    main()
