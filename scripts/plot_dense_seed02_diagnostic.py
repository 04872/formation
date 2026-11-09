"""Plot stored diagnostic witnesses; does not modify planner results."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/formation-matplotlib')
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_exit_baseline_paths import backdrop
from compare_coupling import scenario
folder=ROOT/'results/exit_baselines/diagnostics'
d=json.loads((folder/'dense_seed02.json').read_text())
m,_,_=scenario('dense')
fig,axs=plt.subplots(1,2,figsize=(14,5.2))
series=[('Online region tree',d['original']['tree_route'],'#258b48','-',20.706),('Same chain, joint entry optimum',d['original']['route'],'#225dbe','--',20.416),('BIT*',d['bit_snapshot']['tree_route'],'#df8619','-',18.697),('Green prefix + BIT* suffix (offline witness)',d['bit_suffix_splice']['route'],'#b42db8',':',18.909)]
for ax in axs:
 backdrop(ax,m)
 for label,route,color,style,cost in series:
  a=np.array(route);ax.plot(a[:,0],a[:,1],style,color=color,lw=2,label=f'{label}: J={cost:.3f}')
 gaps=np.array([q['pose'] for q in d['bit_suffix_coverage'] if not q['any_yaw_covered']])
 ax.scatter(gaps[:,0],gaps[:,1],s=55,c='red',marker='x',zorder=10,label='No existing cell at this XY, for any yaw')
axs[0].set_title('dense seed 2: matched snapshot (region tree at 2750 proposals)')
axs[1].set(xlim=(3.8,8),ylim=(.2,3.5),title='Tail detail: the fixed chain retains the upper detour')
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9)
fig.subplots_adjust(bottom=.23,wspace=.2)
fig.savefig(folder/'dense_seed02_diagnosis.png',dpi=160,bbox_inches='tight')
plt.close(fig)
