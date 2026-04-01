"""
cli.py — Command-line argument definitions for the training script.

Exports:
    build_parser()          -> argparse.ArgumentParser
    post_process_args(args) -> args  (auto-detection, defaults)
"""

import os
import argparse
from info.module_logger import get_module_logger

logger = get_module_logger('train.cli')


def build_parser() -> argparse.ArgumentParser:
    """Build and return the full argument parser with all training options."""

    parser = argparse.ArgumentParser(
        description='Train an RL agent for Monster Hunter Tri using PPO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
    Usage examples:

      # Basic new training run (single instance)
      python train.py --timesteps 100000

      # Custom experiment name + learning rate
      python train.py --name phase_A --lr 3e-4 --timesteps 500000

      # With grayscale vision (more economical in GPU ressources)
      python train.py --grayscale --timesteps 50000

      # Resume single-agent from checkpoint
      python train.py --resume ./models/my_exp/final_model.zip

      # Resume multi-agent from models directory (auto-discovers agent_N/ folders)
      python train.py --resume ./models/my_exp/ --num-agents 6 --num-instances 6

      # Quick debug run
      python train.py --debug-steps 1000 --small-rollout --no-gui

      # Multi-agent independent mode
      python train.py --num-agents 6 --num-instances 6 --multi-agent-mode independent --timesteps 500000

      # Multi-agent genetic evolution
      python train.py --num-agents 8 --num-instances 6 --multi-agent-mode genetic --genetic-generations 15 --timesteps 2000000

      # Above 10 agents/instances: requires --nolimit
      python train.py --num-agents 16 --num-instances 12 --nolimit --multi-agent-mode genetic

    More info in the readme or in : https://github.com/Dmsday/Monster-Hunter-Tri-IA
        '''
    )

    # -- Training ----------------------------------------------------------
    g_train = parser.add_argument_group('Training')
    g_train.add_argument('--timesteps', type=int, default=100000, metavar='N',
                         help='Total training timesteps (default: 100000)')
    g_train.add_argument('--name', type=str, default=None, metavar='NAME',
                         help='Experiment name (auto-generated if omitted)')
    g_train.add_argument('--resume', type=str, default=None, metavar='PATH',
                         help='Resume from a checkpoint (.zip file or directory for multi-agent)')
    g_train.add_argument('--force-new-vecnormalize', action='store_true',
                         help='Force new VecNormalize (ignore existing .pkl)')
    g_train.add_argument('--save-state', type=int, default=5,
                         choices=range(1, 9), metavar='N',
                         help='Save state slot to load (1-8, default: 5)')
    g_train.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                         help='PPO learning rate (default: 0.0003 --> better for large action spaces)')
    g_train.add_argument('--cpu', action='store_true',
                         help='Force CPU only (ignore CUDA)')

    # -- Environment -------------------------------------------------------
    g_env = parser.add_argument_group('Environment')
    g_env.add_argument('--env', type=str, default='hybrid',
                       choices=['visual', 'memory', 'hybrid'],
                       help='Environment mode (default: hybrid = vision + memory)')

    # -- Vision ------------------------------------------------------------
    g_vis = parser.add_argument_group('Vision & Capture')
    g_vis.add_argument('--grayscale', action='store_true',
                       help='Grayscale vision (reduces input dimensions)')
    g_vis.add_argument('--rtvision', action='store_true',
                       help='Show real-time AI vision window (OpenCV)')
    g_vis.add_argument('--rtminimap', action='store_true',
                       help='Show real-time exploration minimap (requires --rtvision)')

    # -- Dolphin -----------------------------------------------------------
    g_dolphin = parser.add_argument_group('Dolphin')
    g_dolphin.add_argument('--dolphin-path', type=str, default=None, metavar='PATH',
                           help='Path to Dolphin.exe or its parent folder')
    g_dolphin.add_argument('--dolphin-timeout', type=int, default=60, metavar='SECONDS',
                           help='Timeout for Dolphin window detection (default: 60s)')

    # -- Multi-Agent / Multi-Instance --------------------------------------
    g_multi = parser.add_argument_group('Multi-Agent & Multi-Instance')
    g_multi.add_argument('--num-agents', type=int, default=None, metavar='N',
                         help='Number of PPO agents (default: same as --num-instances)')
    g_multi.add_argument('--num-instances', type=int, default=1, metavar='N',
                         help='Number of Dolphin instances (default: 1)')
    g_multi.add_argument('--allocation-mode', type=str, default='auto',
                         choices=['auto', 'manual', 'weighted'],
                         help='Instance-to-agent allocation mode (default: auto)')
    g_multi.add_argument('--allocation-map', type=str, default=None, metavar='MAP',
                         help='Manual agent-to-instance mapping. '
                              'Format: "agent:inst1,inst2;agent2:inst3".')
    g_multi.add_argument('--multi-agent-mode', type=str, default='independent',
                         choices=['independent', 'round_robin', 'majority_vote', 'genetic', 'weighted'],
                         help='Multi-agent scheduling mode (default: independent)')
    g_multi.add_argument('--steps-per-agent', type=int, default=4096, metavar='N',
                         help='Steps collected per agent before update (default: 4096)')
    g_multi.add_argument('--weighted-eval-freq', type=int, default=100, metavar='N',
                         help='Re-evaluation frequency (weighted mode)')
    g_multi.add_argument('--nolimit', action='store_true',
                         help='Bypass the safety cap of 10 agents / 10 instances')

    # Genetics arguments
    g_multi.add_argument('--genetic-generations', type=int, default=10,
                         help='Number of generations (genetic mode)')
    g_multi.add_argument('--genetic-elite-ratio', type=float, default=0.25,
                         help='Elite retention ratio (genetic mode)')
    g_multi.add_argument('--genetic-mutation-rate', type=float, default=0.3,
                         help='Mutation rate (genetic mode)')
    g_multi.add_argument('--block-size', type=int, default=100,
                         help='Block size (round_robin mode)')

    # -- Interface ---------------------------------------------------------
    g_ui = parser.add_argument_group('Interface & Visualization')
    g_ui.add_argument('--no-gui', action='store_true',
                      help='Disable graphical interface')

    # -- Action heads configuration -------------------------------------------------------------
    g_heads = parser.add_argument_group('Action heads (all enabled by default except menu)')
    g_heads.add_argument(
        '--disabled-heads',
        nargs='*',
        default=['menu'],
        dest='disabled_heads',
        metavar='HEAD',
        choices=['movement', 'camera', 'combat', 'use_item', 'select_item', 'menu', 'sprint'],
        help=(
            'Action heads to disable (default: menu). '
            'Pass names separated by spaces, e.g. --disabled-heads menu use_item. '
            'Pass --disabled-heads with no argument to enable all heads.'
        ),
    )

    # -- Debug -------------------------------------------------------------
    g_dbg = parser.add_argument_group('Debug & Tests')
    g_dbg.add_argument('--debug-steps', type=int, default=None, metavar='N',
                       help='Quick test: override --timesteps with a small value')
    g_dbg.add_argument('--small-rollout', action='store_true',
                       help='Use n_steps=512 (short rollouts for debugging)')
    g_dbg.add_argument('--log-level', type=str, default='WARNING',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Global log level (default: WARNING)')

    return parser


def post_process_args(args) -> argparse.Namespace:
    """
    Apply post-parse defaults and auto-detection:
      - Detect experiment name from --resume path
      - Default num_agents to num_instances when unset
      - Enforce safety cap (10 agents / 10 instances) unless --nolimit
      - Ensure config/user/ directory exists
    """
    # Auto-detect experiment name from resume path
    if args.resume and args.name is None:
        resume_abs = os.path.abspath(args.resume)
        model_dir = os.path.dirname(resume_abs)
        exp_name = os.path.basename(model_dir)
        if exp_name and exp_name != 'models':
            args.name = exp_name
            logger.debug(f"Experiment name detected from --resume: {args.name}")
        else:
            logger.warning("Could not detect experiment name from --resume path")

    # Default num_agents to match num_instances (ONE_TO_ONE scenario)
    # Use --num-agents 1 --num-instances 5 explicitly for 1-agent-many-instances
    if args.num_agents is None:
        args.num_agents = args.num_instances

    # Safety cap: prevent accidental launch of too many instances
    SAFETY_CAP = 10
    nolimit = getattr(args, 'nolimit', False)

    if not nolimit and (args.num_agents > SAFETY_CAP or args.num_instances > SAFETY_CAP):
        print()
        print("=" * 60)
        print("  SAFETY LIMIT REACHED")
        print("=" * 60)
        print(f"  Requested: {args.num_agents} agent(s), {args.num_instances} instance(s)")
        print(f"  Safety cap: {SAFETY_CAP} max for each")
        print()
        print("  Launching too many instances can freeze your PC,")
        print("  consume all RAM, and require a hard reboot.")
        print()
        print("  Options:")
        print(f"    [Y] Proceed anyway (at your own risk)")
        print(f"    [N] Cap at {SAFETY_CAP} and continue (default)")
        print(f"    Or re-run with --nolimit to skip this prompt")
        print("=" * 60)

        try:
            answer = input("  Proceed with high values? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = 'n'

        if answer != 'y':
            old_agents = args.num_agents
            old_instances = args.num_instances
            args.num_agents = min(args.num_agents, SAFETY_CAP)
            args.num_instances = min(args.num_instances, SAFETY_CAP)
            print(f"  Capped: {old_agents} -> {args.num_agents} agents, "
                  f"{old_instances} -> {args.num_instances} instances")
        else:
            print("  Proceeding with requested values...")
        print()

    # Ensure config/user/ directory exists with .gitignore
    config_user_dir = os.path.join(".", "config", "user")
    os.makedirs(config_user_dir, exist_ok=True)
    gitignore = os.path.join(config_user_dir, ".gitignore")
    if not os.path.exists(gitignore):
        try:
            with open(gitignore, 'w') as f:
                f.write("# Ignore all user-specific config files\n*\n\n!.gitignore\n")
        except Exception:
            pass

    return args