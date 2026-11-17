#!/usr/bin/env python3
"""Compare two recorded local-planner runs on the same simulation clock."""
import argparse
import math
from pathlib import Path
from PIL import Image, ImageDraw
from render_three_routes import read_case, panel, encode, font


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fps',type=int,default=20)
    parser.add_argument('--speed',type=float,default=4.)
    parser.add_argument('--before-label',default='Before: stop at each local endpoint')
    parser.add_argument('--after-label',default='After: free local terminal velocity')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for seed in (0,1):
        cases=[read_case(root,'rrt_affine',seed) for root in (args.before,args.after)]
        end=max((len(c['data']['states'])-1)*float(c['data']['dt']) for c in cases)
        video=args.output/f'seed{seed}_before_after.mp4'
        proc=encode(video,1280,640,args.fps)
        frames=math.floor(end*args.fps/args.speed)+1+2*args.fps
        labels=[args.before_label,args.after_label]
        for frame in range(frames):
            clock=min(frame*args.speed/args.fps,end)
            canvas=Image.new('RGB',(1280,640),'white')
            for i,(case,label) in enumerate(zip(cases,labels)):
                canvas.paste(panel(case,label,seed,clock),(640*i,0))
            ImageDraw.Draw(canvas).text((18,604),f'Simulation time {clock:.1f} s  |  {args.speed:g}x playback  |  same map and rates  |  20 s trail',font=font(18),fill='#34445a')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError(f'ffmpeg failed for {video}')
        print(video,flush=True)


if __name__=='__main__':main()
