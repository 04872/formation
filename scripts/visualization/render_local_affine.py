#!/usr/bin/env python3
"""Render recorded local affine execution, evaluating saved Bézier segments."""
import argparse
import json
import subprocess
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull
from formation.local_affine.simulation import evaluate
from formation.local_affine.geometry import robot_positions
from formation.local_affine.trajectory import matrix

COLORS=['#e63946','#0077b6','#2a9d50','#ae4bd0','#ed9a16','#00a5a5']
FFMPEG='/home/eai/projects/env-rebuilt/bin/ffmpeg'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path)
    parser.add_argument('--fps',type=int,default=40);parser.add_argument('--speed',type=float,default=4.)
    args=parser.parse_args();p=args.result;d=np.load(p/'trajectory.npz');records=json.loads((p/'plans.json').read_text());m=json.loads((p/'metrics.json').read_text())
    events=records['execution'];plans=records['plans'];states=d['states'];ref=d['reference'];dt=float(d['dt'])
    duration=len(events)*dt;frames=int(np.ceil(duration*args.fps/args.speed))+2*args.fps
    out=p/'trajectory.mp4';cmd=[FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1200x960','-r',str(args.fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);font=ImageFont.truetype(FONT,22);small=ImageFont.truetype(FONT,17)
    width,height=d['occupancy'].shape;res=float(d['resolution']);xmin,ymin=d['origin'];xmax=xmin+width*res;ymax=ymin+height*res
    def pixel(point):return (75+(point[0]-xmin)*21.,75+(ymax-point[1])*21.)
    for frame in range(frames):
        clock=min(frame*args.speed/args.fps,duration);idx=min(int(clock/dt),len(events)-1);fraction=np.clip((clock-idx*dt)/dt,0,1);event=events[idx]
        if event['kind']=='curve':
            seg=plans[event['plan']]['segments'][event['segment']];u=event['u0']+fraction*(event['u1']-event['u0']);q,qv,_=evaluate(seg,u);current=robot_positions(q,ref)
            velocity=np.einsum('nij,j->ni',matrix(ref),qv)
        else:
            current=states[idx,:,:2];velocity=np.zeros((len(ref),2))
        delta=states[idx+1,:,2]-states[idx,:,2]
        delta=np.arctan2(np.sin(delta),np.cos(delta))
        heading=states[idx,:,2]+fraction*delta
        moving=np.linalg.norm(velocity,axis=1)>1e-8
        heading[moving]=np.arctan2(velocity[moving,1],velocity[moving,0])
        img=Image.new('RGB',(1200,960),'white');draw=ImageDraw.Draw(img)
        status=m['termination'].upper() if clock>=duration else 'RUNNING'
        draw.text((40,12),f"Local affine RRT*  |  {m['mode'].upper()}  |  {status}  |  t = {clock:.1f} s",font=font,fill='#172234')
        draw.text((40,44),f'Analytic flat-output simulation  |  recorded trajectory  |  {args.speed:g}x playback',font=small,fill='#4b5563')
        draw.rectangle((75,75,1125,915),fill='#f6f8fb',outline='#aab4c4')
        for x0,y0,x1,y1 in d['rectangles']:
            a,b=pixel((x0,y1)),pixel((x1,y0));draw.rectangle((a[0],a[1],b[0],b[1]),fill='#374151')
        for i,point in enumerate(current):
            trail_steps=max(1,int(round(20./dt)))
            trail_stride=max(1,int(round(.4/dt)))
            path=[pixel(v) for v in states[max(0,idx-trail_steps):idx+1:trail_stride,i,:2]]
            if len(path)>1:draw.line(path,fill=COLORS[i],width=2)
            px,py=pixel(point);r=float(d['robot_radius'])*21.
            draw.ellipse((px-r,py-r,px+r,py+r),outline=COLORS[i],width=2);draw.ellipse((px-4,py-4,px+4,py+4),fill=COLORS[i])
            draw.line((px,py,px+12*np.cos(heading[i]),py-12*np.sin(heading[i])),fill=COLORS[i],width=2)
        for pts,color in ((current,'#243e64'),(d['goal_center']+ref,'#9374ca')):
            hull=ConvexHull(pts).vertices;draw.line([pixel(pts[i]) for i in hull]+[pixel(pts[hull[0]])],fill=color,width=2)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
    print(out)


if __name__=='__main__':main()
