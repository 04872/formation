#!/usr/bin/env python3
"""Render matched-rate runs on one simulation clock, without normalizing duration."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[2]/'results/three_routes/matched_rates'
FFMPEG='/home/eai/projects/env-rebuilt/bin/ffmpeg'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
COLORS=['#e63946','#0077b6','#2a9d50','#ae4bd0','#ed9a16','#00a5a5']
ROUTES=[('paper_uniform','Paper uniform + MPPI'),('paper_affine','Paper affine + MPPI'),('rrt_affine','Local RRT* + Bezier')]


def font(size,bold=False):
    return ImageFont.truetype(BOLD if bold else FONT,size)


def read_case(root,route,seed):
    path=root/route/f'seed_{seed}'
    with np.load(path/'trajectory.npz') as archive:
        data={key:archive[key] for key in archive.files}
    metrics=json.loads((path/'metrics.json').read_text())
    reference=data['centered_reference'] if 'centered_reference' in data else data['reference']
    return {'data':data,'metrics':metrics,'reference':reference,'route':route}


def point(p,rect,extent):
    x,y,w,h=rect; xmin,xmax,ymin,ymax=extent
    scale=min(w/(xmax-xmin),h/(ymax-ymin));ox=x+(w-scale*(xmax-xmin))/2;oy=y+(h-scale*(ymax-ymin))/2
    return (ox+(p[0]-xmin)*scale,oy+(ymax-p[1])*scale),scale


def panel(case,label,seed,time_s,width=640,height=590,full_trail=False):
    data=case['data'];metrics=case['metrics'];states=data['states'];dt=float(data['dt']);last=len(states)-1
    tick=min(int(np.floor(time_s/dt+1e-8)),last);actual=states[tick];endtime=last*dt
    status=str(metrics['termination']).upper() if time_s>=endtime-1e-10 else 'RUNNING'
    image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image)
    draw.text((12,8),f'{label}  |  seed {seed}',font=font(19,True),fill='#172234')
    draw.text((12,37),f'{status}  |  case time {min(time_s,endtime):.1f} s',font=font(15),fill='#34445a')
    resolution=float(data['resolution']);size=data['occupancy'].shape;origin=data['origin'];extent=(origin[0],origin[0]+size[0]*resolution,origin[1],origin[1]+size[1]*resolution)
    box=(8,65,width-16,height-74);draw.rectangle((box[0],box[1],box[0]+box[2],box[1]+box[3]),fill='#f6f8fb',outline='#aab4c4')
    def xy(p):return point(p,box,extent)[0]
    for x0,y0,x1,y1 in data['rectangles']:
        a,b=xy((x0,y1)),xy((x1,y0));draw.rectangle((a[0],a[1],b[0],b[1]),fill='#374151')
    lower=0 if full_trail else max(0,tick-int(round(20/dt)))
    stride=max(1,int(round(.10/dt)))
    for i in range(len(actual)):
        c=COLORS[i%len(COLORS)];trail=[xy(v) for v in states[lower:tick+1:stride,i,:2]]
        if len(trail)>1:draw.line(trail,fill=c,width=2)
        p,scale=point(actual[i,:2],box,extent);r=float(data['robot_radius'])*scale
        draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),outline=c,width=1)
        draw.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=c)
        theta=actual[i,2];draw.line((p[0],p[1],p[0]+9*np.cos(theta),p[1]-9*np.sin(theta)),fill=c,width=2)
    for points,color in ((actual[:,:2],'#253d62'),(data['goal_center']+case['reference'],'#8c72c3')):
        hull=ConvexHull(points).vertices
        draw.line([xy(points[i]) for i in hull]+[xy(points[hull[0]])],fill=color,width=2)
    return image


def encode(path,width,height,fps):
    command=[FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    return subprocess.Popen(command,stdin=subprocess.PIPE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--fps',type=int,default=20);parser.add_argument('--speed',type=float,default=4.)
    args=parser.parse_args();cases={(route,seed):read_case(args.root,route,seed) for route,_ in ROUTES for seed in (0,1)}
    config=json.loads((args.root/'rrt_affine/seed_0/configuration.json').read_text())
    if config.get('backend')=='mppi':ROUTES[-1]=('rrt_affine','Local RRT* + MPPI')
    for seed in (0,1):
        maximum=max((len(cases[route,seed]['data']['states'])-1)*float(cases[route,seed]['data']['dt']) for route,_ in ROUTES)
        frames=math.floor(maximum*args.fps/args.speed)+1+2*args.fps;out=args.root/f'seed{seed}_comparison.mp4';proc=encode(out,1920,640,args.fps)
        for frame in range(frames):
            clock=min(frame*args.speed/args.fps,maximum);image=Image.new('RGB',(1920,640),'white')
            for i,(route,label) in enumerate(ROUTES):image.paste(panel(cases[route,seed],label,seed,clock),(640*i,0))
            draw=ImageDraw.Draw(image);draw.text((18,604),f'Global simulation time {clock:.1f} s  |  {args.speed:g}x playback  |  20 s trail  |  dark: actual formation, purple: goal  |  recorded states',font=font(18),fill='#34445a')
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError(f'ffmpeg failed for {out}')
        print(f'completed {out} frames={frames}',flush=True)
    image=Image.new('RGB',(1280,1770),'white')
    for row,(route,label) in enumerate(ROUTES):
        for seed in (0,1):
            case=cases[route,seed];end=(len(case['data']['states'])-1)*float(case['data']['dt'])
            image.paste(panel(case,label,seed,end,full_trail=True),(640*seed,590*row))
    image.save(args.root/'comparison.png')
    print(f'completed {args.root}/comparison.png',flush=True)


if __name__=='__main__':main()
