#!/usr/bin/env python3
"""Audit executed split motion around the left end of the upper U wall."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2] / 'results/three_routes/local_backup_closed_loop_500s'
GROUPS = {1: {'inner': [0, 1, 3], 'outer': [2, 4, 5]}, 2: {'inner': [0, 1, 2], 'outer': [3, 4, 5]}}
# Explicit independent windows chosen from fixed-group peaks and member peaks.
WINDOWS = {1: [(201,331,445,'inner'), (200,435,458,'outer'), (458,473,491,'outer')],
           2: [(290,310,494,'inner'), (201,400,425,'member_3')]}

def stats(x):
    delta = np.diff(x, axis=0)
    return {'net_m': (x[-1]-x[0]).tolist(), 'positive_m': np.maximum(delta,0).sum(axis=0).tolist(),
            'negative_m': (-np.minimum(delta,0)).sum(axis=0).tolist()}

def main():
    report={'definition': {'sample_period_s': 1, 'tip_xy_m': [14,25.75],
      'progress': 'sign(y-25.75)*hypot(max(x-14,0),y-25.75): inner approaching tip and outer moving right are positive; continuous around x<=14',
      'limitations': 'Signed radial coordinate is a geometric proxy, not traveled path length; motion normal to wall also changes it. Report separate dx/dy. 1 s sampling can miss faster reversals.',
      'group_definition': 'Fixed identities based on actual positions after splitting; classification at 200 s is retrospective, not side membership at every time.',
      'event_definition': 'Specific candidate peaks selected by >=0.5 m group/member mean advance and >=0.5 m subsequent drawdown. Windows identify before-peak and peak-to-trough; each member must be checked individually. Distinct outer events may reuse long prior advance only for the first event.',
      'phases': '50 s bins; positive and negative are sums of 1 s signed progress increments.'}, 'runs':{}}
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
    for row,d in enumerate((1,2)):
        a=np.load(ROOT/f'depth{d}/rrt_affine/seed_0/trajectory.npz')
        stride=round(1/float(a['dt'])); p=a['states'][::stride,:,:2]; t=np.arange(len(p)); y=p[:,:,1]-25.75
        s=np.sign(y)*np.hypot(np.maximum(p[:,:,0]-14,0),y)
        run={'groups':GROUPS[d], 'phases':[], 'events':[], 'member_peak_candidates':{}}
        for start in range(200,500,50):
            end=min(start+50,len(t)-1); z=p[start:end+1]
            run['phases'].append({'start_s':start,'end_s':end,**stats(s[start:end+1]), 'dx_m':(z[-1,:,0]-z[0,:,0]).tolist(),'dy_m':(z[-1,:,1]-z[0,:,1]).tolist()})
        for start,peak,end,group in WINDOWS[d]:
            indices=GROUPS[d].get(group,[3]); f=s[peak]-s[start]; b=s[end]-s[peak]
            run['events'].append({'advance_window_s':[start,peak], 'drawdown_window_s':[peak,end], 'group':group, 'members':indices,
                'advance_each_member_m':f.tolist(),'drawdown_each_member_m':b.tolist(),
                'group_mean_advance_m':float(f[indices].mean()),'group_mean_drawdown_m':float(b[indices].mean()),
                'each_member_dx_drawdown_m':(p[end,:,0]-p[peak,:,0]).tolist(),'each_member_dy_drawdown_m':(p[end,:,1]-p[peak,:,1]).tolist()})
        for m in range(6):
            z=s[200:,m]; peaks, props=find_peaks(z,prominence=.5,distance=8)
            run['member_peak_candidates'][str(m)]=[{'peak_s':int(k+200),'left_base_s':int(props['left_bases'][i]+200),'right_base_s':int(props['right_bases'][i]+200),'prominence_m':float(props['prominences'][i])} for i,k in enumerate(peaks)]
            axes[row,0].plot(t,s[:,m],label=f'{m} ({"in" if m in GROUPS[d]["inner"] else "out"})')
        for name,ids in GROUPS[d].items(): axes[row,1].plot(t,s[:,ids].mean(axis=1),label=f'{name} {ids}')
        for ax in axes[row]:
            ax.axvline(200,color='gray',lw=.8); ax.set_xlim(190,500); ax.grid(alpha=.25); ax.legend(fontsize=8); ax.set_ylabel(f'depth {d}: signed progress (m)')
        report['runs'][f'depth{d}']=run
    axes[0,0].set_title('Fixed individual identities');axes[0,1].set_title('Fixed group means')
    for ax in axes[-1]:ax.set_xlabel('simulation time (s)')
    fig.tight_layout();fig.savefig(ROOT/'split_progress_events.png',dpi=160)
    (ROOT/'split_progress_events.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
