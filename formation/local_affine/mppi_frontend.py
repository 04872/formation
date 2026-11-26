"""Local affine RRT* guidance adapter for the distributed MPPI backend."""
from itertools import combinations
import numpy as np
from .geometry import certify_edge, robot_positions, shape_valid
from .rrt import plan
from formation.paper_formation.config import RADIUS, DT
from formation.paper_formation.formation import fit_q


def actual_join_safe(actual, target, regions):
    """Check continuous straight-line joins in single convex regions and pair clearance."""
    for start,end in zip(actual,target):
        if not any(np.all(H@start<=b+1e-9) and np.all(H@end<=b+1e-9) for H,b in regions):
            return False
    for i,j in combinations(range(len(actual)),2):
        d=actual[i]-actual[j];change=(target[i]-target[j])-d
        denominator=change@change
        u=np.clip(-(d@change)/denominator,0,1) if denominator else 0.
        if np.linalg.norm(d+u*change)<2*RADIUS+.08:return False
    return True


class RRTMPPIFrontend:
    """Plan in centered coordinates; return leader-anchored configurations."""
    def __init__(self, mode, rng, geometry, iterations=250, reference_speed=.55, history_decision=False, history_joint_path=False, history_link_epsilon=.4, history_current_motion_only=False):
        self.mode=mode;self.rng=rng;self.geometry=geometry
        self.iterations=iterations;self.reference_speed=reference_speed
        self.previous_shape=np.eye(2)
        self.history_decision = None
        self.use_joint_history = bool(history_joint_path)
        if history_decision:
            from .history_decision import HistoryDecision
            self.history_decision = HistoryDecision()
        if history_joint_path:
            if history_decision:
                raise ValueError('history decision modes are mutually exclusive')
            from .history_joint_path import HistoryJointPath
            self.history_decision = HistoryJointPath(history_link_epsilon, current_motion_only=history_current_motion_only)

    def plan(self, states, assignment, goal_center, world, regions):
        geom=self.geometry;ref=geom.centered_reference
        leader_q=fit_q(states,assignment,geom);fit_leader=leader_q.copy();A=leader_q[2:].reshape(2,2)
        root=leader_q.copy();root[:2]-=A@ref[geom.leader_slot]
        goal=np.r_[goal_center,np.eye(2).ravel()]
        source='fitted'
        if not shape_valid(root,self.mode) or not actual_join_safe(states[:,:2],robot_positions(root,ref)[assignment],regions):
            source='previous_reference'
            A=self.previous_shape
            root=np.r_[states[0,:2]-A@ref[assignment[0]],A.ravel()]
            valid_shape=shape_valid(root,self.mode)
            safe_join=valid_shape and actual_join_safe(states[:,:2],robot_positions(root,ref)[assignment],regions)
            if valid_shape and not safe_join:
                # For a fixed shape, this translation minimizes total squared join distance.
                source='previous_reference_centroid'
                root[:2]=(states[:,:2]-ref[assignment]@A.T).mean(axis=0)
                safe_join=actual_join_safe(states[:,:2],robot_positions(root,ref)[assignment],regions)
            leader_q=root.copy();leader_q[:2]+=A@ref[geom.leader_slot]
            if not safe_join:
                return np.empty((0,6)),{'status':'no_safe_nominal_root','root_source':source,'tree':[],'root_leader':leader_q.tolist(),'actual_fit_root':fit_leader.tolist()}
        if certify_edge(root,goal,regions,ref) is not None:
            nodes=[];route=[];stats={'direct_goal':True}
            if self.history_decision is not None and self.use_joint_history:
                stats['history_decision'] = self.history_decision.record_direct(root, goal, world, regions)
            vertices=[root.copy(),goal.copy()]
        else:
            nodes, route, stats = plan(
                root, goal, ref, regions, world, self.rng, self.mode, self.iterations,
                collect_history_candidates=self.history_decision is not None,
                collect_certified_edges=self.history_decision is not None and self.use_joint_history)
            if self.history_decision is not None:
                if self.use_joint_history:
                    selected, evidence = self.history_decision.select(nodes, stats, ref, goal, world, regions)
                    stats.pop('certified_edges', None)
                else:
                    selected, evidence = self.history_decision.select(nodes, stats, ref, goal, world)
                # The tree and candidate evidence already contain endpoint
                # IDs; avoid exporting duplicate complete branch lists.
                stats.pop('candidate_full_routes')
                stats.pop('baseline_route')
                stats['history_decision'] = evidence
                route = [int(index) for index in selected]
            vertices=[nodes[i].q.copy() for i in route]
            if len(vertices)>1 and certify_edge(vertices[-1],goal,regions,ref) is not None:
                vertices.append(goal.copy())
        if len(vertices)<2:
            return np.empty((0,6)),{'status':'no_rrt_route','root_source':source,'tree':[],'statistics':stats,'root_leader':leader_q.tolist(),'actual_fit_root':fit_leader.tolist()}
        samples=[vertices[0]]
        for a,b in zip(vertices[:-1],vertices[1:]):
            displacement=np.linalg.norm(robot_positions(b,ref)-robot_positions(a,ref),axis=1).max()
            count=max(1,int(np.ceil(displacement/(self.reference_speed*DT))))
            samples.extend(a+(b-a)*(j/count) for j in range(1,count+1))
        centered=np.asarray(samples);leader=centered.copy()
        leader[:,:2]+=np.einsum('nij,j->ni',centered[:,2:].reshape(-1,2,2),ref[geom.leader_slot])
        # Include vertices even when their timing falls between nominal steps.
        self.previous_shape=root[2:].reshape(2,2).copy()
        return leader,{'status':'ok','root_source':source,'statistics':stats,'route':route,'root_leader':leader_q.tolist(),'actual_fit_root':fit_leader.tolist(),
                       'vertices_centered':[q.tolist() for q in vertices],
                       'tree':[{'q':n.q.tolist(),'parent':n.parent,'cost':n.cost,'edge_cost':n.edge_cost} for n in nodes],
                       'reference_speed_m_s':self.reference_speed,'assignment_policy':'fixed_current'}
