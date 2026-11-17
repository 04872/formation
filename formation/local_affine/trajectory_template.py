"""Parameterized continuous-time Bézier tracking problems."""
from math import comb
import cvxpy as cp
import numpy as np
from formation.paper_formation.config import RADIUS


def gram(n):
    """Gram matrix of degree-n Bernstein basis on the unit interval."""
    return np.array([[comb(n,i)*comb(n,j)/((2*n+1)*comb(2*n,i+j))
                      for j in range(n+1)] for i in range(n+1)])


class TrajectoryTemplate:
    def __init__(self, segments, corridor_rows, E, mode, vmax, amax, terminal_goal,
                 terminal_stop=False, reserve=False):
        self.x=x=cp.Variable((segments*6,6))
        self.inv_t=cp.Parameter(segments,nonneg=True); self.inv_t2=cp.Parameter(segments,nonneg=True)
        self.duration=cp.Parameter(segments,nonneg=True); self.inv_t3=cp.Parameter(segments,nonneg=True)
        self.initial_velocity=cp.Parameter(6);self.initial_acceleration=cp.Parameter(6)
        self.route=cp.Parameter((segments+1,6))
        self.tracking_linear=[cp.Parameter((6,2*len(E))) for _ in range(segments)]
        self.velocity_reference=[cp.Parameter(2*len(E)) for _ in range(segments)]
        self.rotations=[cp.Parameter((2,2)) for _ in range(segments)]
        self.corridors=[cp.Parameter((rows,6)) for rows in corridor_rows]
        self.bounds=[cp.Parameter(rows) for rows in corridor_rows]
        self.pairs=[cp.Parameter((len(E)*(len(E)-1)//2,6)) for _ in range(segments)]
        constraints=[];objective=0
        robot_map=np.concatenate(E,axis=0)
        L5=np.linalg.cholesky(gram(5)).T;L4=np.linalg.cholesky(gram(4)).T;L3=np.linalg.cholesky(gram(3)).T
        for k in range(segments):
            P=x[k*6:(k+1)*6,:]
            if not reserve:
                workspace=P@robot_map.T
                objective += self.duration[k]*cp.sum_squares(L5@workspace)-2*cp.sum(cp.multiply(self.tracking_linear[k],workspace))
            V=5*(P[1:]-P[:-1]);A2=20*(P[2:]-2*P[1:-1]+P[:-2])
            if not reserve:
                objective += self.inv_t[k]*cp.sum_squares(L4@V@robot_map.T)
                objective -= 2*cp.sum(cp.multiply(self.velocity_reference[k],
                                                 (P[-1]-P[0])@robot_map.T))
            objective += 2*self.inv_t3[k]*cp.sum_squares(L3@A2@robot_map.T)
            for j in range(6):
                A=cp.reshape(P[j,2:],(2,2),order='C');local=self.rotations[k].T@A
                constraints += [(local+local.T)/2 >> .42*np.eye(2),cp.norm(A,2)<=1.35,
                                cp.abs(A[0,1])+cp.abs(A[1,0])<=1.5]
                if mode=='uniform':constraints += [A[0,1]==0,A[1,0]==0,A[0,0]==A[1,1]]
            constraints += [P@self.corridors[k].T<=cp.vstack([self.bounds[k]]*6),
                            P@self.pairs[k].T>=2*RADIUS+.08]
            for e in E:
                constraints += [cp.norm((V*self.inv_t[k])@e.T,axis=1)<=vmax,
                                cp.norm((A2*self.inv_t2[k])@e.T,axis=1)<=amax]
            if k:
                prev=x[(k-1)*6:k*6,:]
                constraints += [prev[5]==P[0],(prev[5]-prev[4])*self.inv_t[k-1]==(P[1]-P[0])*self.inv_t[k],
                                (prev[5]-2*prev[4]+prev[3])*self.inv_t2[k-1]==(P[2]-2*P[1]+P[0])*self.inv_t2[k]]
        constraints += [x[0]==self.route[0],5*(x[1]-x[0])*self.inv_t[0]==self.initial_velocity,
                        20*(x[2]-2*x[1]+x[0])*self.inv_t2[0]==self.initial_acceleration]
        if terminal_goal or terminal_stop:constraints += [x[-1]==x[-2],x[-1]-2*x[-2]+x[-3]==0]
        if terminal_goal:constraints += [x[-1]==self.route[-1]]
        self.problem=cp.Problem(cp.Minimize(objective),constraints)
