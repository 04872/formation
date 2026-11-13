"""Display physical obstacle geometry separately from the planning grid."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def draw_obstacles(ax, occupancy, resolution, origin, circles):
    width, height = np.array(occupancy.shape) * resolution
    x, y = origin
    if len(circles):
        for cx, cy, radius in circles:
            ax.add_patch(Circle((cx, cy), radius, facecolor='#c8cdd3',
                                edgecolor='#626d78', linewidth=0.8))
        wall = 2 * resolution
        for rx, ry, rw, rh in [(x,y,width,wall), (x,y+height-wall,width,wall),
                              (x,y,wall,height), (x+width-wall,y,wall,height)]:
            ax.add_patch(Rectangle((rx,ry),rw,rh,color='#626d78'))
    else:
        ax.imshow(occupancy.T, origin='lower', extent=(x,x+width,y,y+height),
                  cmap='Greys', alpha=0.5)
    ax.set(xlim=(x,x+width), ylim=(y,y+height), xlabel='x [m]', ylabel='y [m]')
    ax.set_aspect('equal')


def redraw(directory):
    """Replot recorded runs without rerunning or altering simulation data."""
    directory = Path(directory)
    with np.load(directory / 'trajectory.npz') as data:
        states = data['states']
        fig, ax = plt.subplots(figsize=(12, 8))
        draw_obstacles(ax, data['occupancy'], float(data['resolution']),
                       data['origin'], data['circles'] if 'circles' in data else [])
        for i in range(states.shape[1]):
            line, = ax.plot(states[:,i,0],states[:,i,1],lw=1.6,label=f'robot {i}')
            ax.scatter(*states[0,i,:2],color=line.get_color(),s=25)
            ax.scatter(*states[-1,i,:2],color=line.get_color(),marker='x',s=35)
        goal_center = data['goal'] + data['reference'].mean(axis=0)
        ax.scatter(*goal_center,marker='*',color='#c23b32',s=110,label='goal center')
    metrics = json.loads((directory / 'metrics.json').read_text())
    ax.set_title(f"{directory.name} | {metrics['termination']}")
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(directory/'trajectory.png', dpi=180, bbox_inches='tight')
    fig.savefig(directory/'trajectory.svg', bbox_inches='tight')
    xy = states[:,:,:2].reshape(-1,2)
    ax.set_xlim(xy[:,0].min()-2,xy[:,0].max()+2)
    ax.set_ylim(xy[:,1].min()-2,xy[:,1].max()+2)
    fig.savefig(directory/'trajectory_detail.png',dpi=180, bbox_inches='tight')
    plt.close(fig)
