import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formation.paper_formation.simulation import simulate
from formation.paper_formation.outputs import write_outputs
from formation.paper_formation.config import CONFIG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['open', 'narrow', 'random_circles'], default='open')
    parser.add_argument('--mode', choices=['uniform', 'affine'], default='affine')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--map-seed', type=int, default=0)
    parser.add_argument('--obstacle-count', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=350)
    parser.add_argument('--sequences', type=int, default=CONFIG.sequences)
    parser.add_argument('--steps', type=int, default=CONFIG.steps)
    parser.add_argument('--candidates', type=int, default=CONFIG.candidates)
    parser.add_argument('--samples', type=int, default=CONFIG.samples)
    parser.add_argument('--horizon', type=int, default=CONFIG.horizon)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--output', default='results/paper_formation')
    args = parser.parse_args()
    start_time = time.perf_counter()
    result = simulate(args.scene, args.mode, args.seed, args.max_steps, args.sequences, args.steps, args.candidates, args.samples, args.obstacle_count, args.map_seed, args.horizon, args.progress)
    args.wall_seconds = time.perf_counter() - start_time
    write_outputs(args, result)
if __name__ == '__main__':
    main()
