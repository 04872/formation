"""Algorithm 2 with differential-drive acceleration and yaw-rate control."""
import numpy as np
from .config import DT, CONFIG
from .paper_costs import guidance_cost, dynamic_cost, smoothness_cost, distance_penalty


def integrate(state, controls):
    """Integrate constant linear and angular velocity exactly for one step."""
    x=state.copy()
    v,w=controls[...,0],controls[...,1]
    mid=x[...,2]+w*DT/2
    factor=np.sinc(w*DT/(2*np.pi))*DT
    x[...,0]+=v*factor*np.cos(mid)
    x[...,1]+=v*factor*np.sin(mid)
    x[...,2]+=w*DT
    return x


def integrate_acceleration(state, velocity, control):
    """Advance pose with midpoint speed for constant acceleration, and return final speed."""
    acceleration,omega=control[...,0],control[...,1]
    new_velocity=velocity+acceleration*DT
    pose=integrate(state,np.stack(((velocity+new_velocity)/2,omega),axis=-1))
    return pose,new_velocity


def mppi(states, velocities, guidance, nominal, previous_predictions, world, rng,
         samples=CONFIG.samples, horizon=CONFIG.horizon, temperature=CONFIG.mppi_temperature, amax=1., noise_std=CONFIG.mppi_noise_std):
    """Use one nominal U, additive Gaussian noise, costs (13)-(17), and weighted noise."""
    n=len(states)
    result=np.zeros_like(nominal)
    predictions=np.zeros((n,horizon,3))
    diagnostics=np.zeros((n,6))
    for i in range(n):
        noise=rng.normal(0.,noise_std,(samples,horizon,2))
        controls=nominal[i][None,:,:]+noise
        x=np.repeat(states[i][None,:],samples,axis=0)
        v=np.full(samples,velocities[i]);cost=np.zeros(samples)
        closest=np.full(samples,np.inf)
        for t in range(horizon):
            command=controls[:,t]
            x,v=integrate_acceleration(x,v,command)
            acceleration=np.sqrt(command[:,0]**2+(v*command[:,1])**2)
            cost+=guidance_cost(x[:,:2],guidance[t,i])
            cost+=dynamic_cost(v,acceleration,CONFIG.max_speed,amax)
            cost+=smoothness_cost(command)
            cost+=distance_penalty(world.clearance(x[:,:2]),.4,.8,2600.,1.8)
            for j in range(n):
                if j!=i:
                    distance=np.linalg.norm(x[:,:2]-previous_predictions[j,t,:2],axis=1)
                    closest=np.minimum(closest,distance)
                    cost+=distance_penalty(distance,.5,1.2,1600.,1.5)
        weights=np.exp(-DT*(cost-cost.min())/temperature)
        weights/=weights.sum()
        result[i]=nominal[i]+np.einsum('k,ktd->td',weights,noise)
        # Hard actuator bounds apply after the weighted-noise update.
        result[i,:,0]=np.clip(result[i,:,0],-amax,amax)
        result[i,:,1]=np.clip(result[i,:,1],-CONFIG.max_angular_speed,CONFIG.max_angular_speed)
        x=states[i].copy(); v=velocities[i]
        weighted_closest=np.inf
        for t in range(horizon):
            control=result[i,t].copy()
            control[0]=np.clip(control[0],(-CONFIG.max_speed-v)/DT,(CONFIG.max_speed-v)/DT)
            x,v=integrate_acceleration(x,v,control)
            predictions[i,t]=x
            for j in range(n):
                if j!=i:
                    weighted_closest=min(weighted_closest,np.linalg.norm(x[:2]-previous_predictions[j,t,:2]))
        diagnostics[i]=[1./np.sum(weights**2),np.mean(closest>=.32),np.mean(closest>=.5),
                        closest[np.argmin(cost)],weighted_closest,np.min(cost)*DT]
    return result,predictions,diagnostics


class DistributedMPPI:
    """Retain nominal control and peer predictions across execution cycles."""
    def __init__(self,rng,states,samples,horizon=CONFIG.horizon):
        self.rng,self.samples,self.horizon=rng,samples,horizon
        self.nominal=np.zeros((len(states),horizon,2))
        self.previous_predictions=np.repeat(states[:,None,:],horizon,axis=1)
        self.velocities=np.zeros(len(states))
        self.u_init=np.zeros((len(states),2))

    def step(self,states,guidance,world):
        opt,pred,self.last_diagnostics=mppi(states,self.velocities,guidance,self.nominal,self.previous_predictions,
                      world,self.rng,self.samples,self.horizon)
        command=opt[:,0].copy()
        command[:,0]=np.clip(command[:,0],(-CONFIG.max_speed-self.velocities)/DT,
                             (CONFIG.max_speed-self.velocities)/DT)
        new_velocity=self.velocities+command[:,0]*DT
        execution=np.column_stack(((self.velocities+new_velocity)/2,command[:,1]))
        self.velocities=new_velocity
        self.nominal=np.concatenate((opt[:,1:],self.u_init[:,None,:]),axis=1)
        self.previous_predictions=np.concatenate((pred[:,1:],pred[:,-1:]),axis=1)
        return execution,command
