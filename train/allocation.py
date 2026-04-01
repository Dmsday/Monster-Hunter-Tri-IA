"""
allocation.py — Agent-to-instance allocation logic and validation.

Exports:
    detect_scenario(num_agents, num_instances) -> str
    calculate_agent_allocation(...)            -> dict
    parse_allocation_map(...)                  -> dict
    validate_multi_agent_args(args)            -> str  (scenario)
    validate_genetic_params(args)
    validate_round_robin_params(args)
    validate_weighted_params(args)
    generate_example_allocation_map(...)       -> str
"""

from collections import Counter
from info.module_logger import get_module_logger

logger = get_module_logger('train.allocation')


# ======================================================================
#  SCENARIO DETECTION
# ======================================================================

def detect_scenario(num_agents: int, num_instances: int) -> str:
    """Determine the allocation scenario from agent/instance counts."""
    if num_agents == num_instances:
        return "ONE_TO_ONE"
    elif num_agents < num_instances:
        return "AGENT_MULTIPLE_INSTANCES"
    else:
        return "INSTANCE_SHARING"


# ======================================================================
#  ALLOCATION COMPUTATION
# ======================================================================

def calculate_agent_allocation(
    num_agents: int,
    num_instances: int,
    allocation_mode: str = 'auto',
    allocation_map: str = None,
    multi_agent_mode: str = 'independent',
) -> dict:
    """
    Compute the agent -> [instance_ids] mapping.

    Returns:
        {
          'scenario':      str,
          'allocation':    {agent_id: [instance_ids]},
          'num_agents':    int,
          'num_instances': int,
        }
    """
    scenario = detect_scenario(num_agents, num_instances)

    if scenario == "ONE_TO_ONE":
        allocation = {i: [i] for i in range(num_agents)}

    elif scenario == "AGENT_MULTIPLE_INSTANCES":
        if allocation_mode == 'manual' and allocation_map:
            allocation = parse_allocation_map(allocation_map, num_agents, num_instances)
        else:
            allocation = _distribute_instances_to_agents(num_agents, num_instances)

    else:  # INSTANCE_SHARING
        if allocation_mode == 'manual' and allocation_map:
            allocation = parse_allocation_map(allocation_map, num_agents, num_instances)
        else:
            allocation = _distribute_agents_to_instances(num_agents, num_instances)

        # At least one instance must be shared by multiple agents
        usage = {}
        for aid, insts in allocation.items():
            for i in insts:
                usage.setdefault(i, []).append(aid)
        if not any(len(a) > 1 for a in usage.values()):
            raise ValueError("INSTANCE_SHARING allocation has no shared instances")

    _log_allocation(scenario, allocation, num_agents, num_instances)

    return {
        'scenario': scenario,
        'allocation': allocation,
        'num_agents': num_agents,
        'num_instances': num_instances,
    }


def _distribute_instances_to_agents(num_agents: int, num_instances: int) -> dict:
    """Evenly spread instances across agents (scenario 2: fewer agents than instances)."""
    allocation = {}
    per_agent = num_instances // num_agents
    remainder = num_instances % num_agents
    idx = 0
    for aid in range(num_agents):
        count = per_agent + (1 if aid < remainder else 0)
        allocation[aid] = list(range(idx, idx + count))
        idx += count
    return allocation


def _distribute_agents_to_instances(num_agents: int, num_instances: int) -> dict:
    """Evenly spread agents across instances (scenario 3: more agents than instances)."""
    allocation = {}
    per_inst = num_agents // num_instances
    remainder = num_agents % num_instances
    current_agent = 0
    for iid in range(num_instances):
        count = per_inst + (1 if iid < remainder else 0)
        for _ in range(count):
            if current_agent < num_agents:
                allocation[current_agent] = [iid]
                current_agent += 1
    return allocation


def _log_allocation(scenario, allocation, num_agents, num_instances):
    """Print the computed allocation to the debug log."""
    logger.debug(f"Scenario: {scenario}")
    for aid in sorted(allocation):
        logger.debug(f"  Agent {aid:2d} -> Instances {allocation[aid]}")
    total = sum(len(v) for v in allocation.values())
    logger.debug(f"  Total connections: {total}, avg/agent: {total / max(num_agents, 1):.1f}")


# ======================================================================
#  ALLOCATION MAP PARSING
# ======================================================================

def parse_allocation_map(allocation_map: str, num_agents: int, num_instances: int) -> dict:
    """
    Parse a manual allocation string.

    Format: "0:1,2;1:3,4"  ->  {0: [1, 2], 1: [3, 4]}
    """
    allocation = {}
    instances_used = []

    try:
        for pair in allocation_map.split(';'):
            if ':' not in pair:
                raise ValueError(f"Bad format (expected 'agent:instances'): {pair}")
            agent_str, inst_str = pair.split(':', 1)
            aid = int(agent_str.strip())
            insts = [int(x.strip()) for x in inst_str.split(',')]
            allocation.setdefault(aid, []).extend(insts)
            instances_used.extend(insts)

        # All expected agents must be present
        expected = set(range(num_agents))
        actual = set(allocation.keys())
        if expected != actual:
            parts = []
            if expected - actual:
                parts.append(f"Missing agents: {sorted(expected - actual)}")
            if actual - expected:
                parts.append(f"Invalid agents: {sorted(actual - expected)}")
            raise ValueError('; '.join(parts))

        # Instance IDs must be in range
        for aid, insts in allocation.items():
            for i in insts:
                if not (0 <= i < num_instances):
                    raise ValueError(
                        f"Agent {aid}: instance {i} out of range 0..{num_instances - 1}")

        # Scenario-specific checks
        scenario = detect_scenario(num_agents, num_instances)

        if scenario == "AGENT_MULTIPLE_INSTANCES":
            counts = Counter(instances_used)
            dups = [i for i, c in counts.items() if c > 1]
            if dups:
                raise ValueError(f"Duplicate instances (not allowed in scenario 2): {dups}")
            if set(instances_used) != set(range(num_instances)):
                missing = set(range(num_instances)) - set(instances_used)
                raise ValueError(f"Unassigned instances: {sorted(missing)}")

        elif scenario == "INSTANCE_SHARING":
            usage = {}
            for aid, insts in allocation.items():
                for i in insts:
                    usage.setdefault(i, []).append(aid)
            if not any(len(a) > 1 for a in usage.values()):
                raise ValueError("No shared instances found (required in scenario 3)")

        return allocation

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Error parsing allocation_map: {exc}")


# ======================================================================
#  VALIDATION
# ======================================================================

def validate_multi_agent_args(args) -> str:
    """
    Validate multi-agent / multi-instance arguments.

    Returns the detected scenario string.
    """
    if not (1 <= args.num_agents <= 32):
        raise ValueError(f"num_agents must be 1..32 (got {args.num_agents})")
    if not (1 <= args.num_instances <= 16):
        raise ValueError(f"num_instances must be 1..16 (got {args.num_instances})")

    if hasattr(args, 'steps_per_agent') and args.steps_per_agent < 256:
        logger.warning(f"steps_per_agent is very low ({args.steps_per_agent}); "
                       "recommended >= 2048 for PPO stability")

    scenario = detect_scenario(args.num_agents, args.num_instances)

    if scenario == "INSTANCE_SHARING":
        supported = ['independent', 'round_robin', 'majority_vote']
        if args.multi_agent_mode == 'genetic':
            raise NotImplementedError("Genetic mode is not implemented for instance sharing")
        if args.multi_agent_mode not in supported:
            raise ValueError(f"Unknown multi-agent mode: {args.multi_agent_mode}")

    if args.allocation_mode == 'manual' and args.allocation_map is None:
        raise ValueError("--allocation-map is required when --allocation-mode=manual")

    if args.multi_agent_mode == 'genetic':
        validate_genetic_params(args)
    if args.multi_agent_mode == 'round_robin':
        validate_round_robin_params(args)
    if args.multi_agent_mode == 'weighted':
        validate_weighted_params(args)

    return scenario


def validate_genetic_params(args):
    """Raise ValueError if genetic-mode parameters are invalid."""
    errors = []
    if args.genetic_generations < 1:
        errors.append("genetic_generations must be >= 1")
    if not (0.0 < args.genetic_elite_ratio < 1.0):
        errors.append("genetic_elite_ratio must be in (0, 1)")
    if not (0.0 <= args.genetic_mutation_rate <= 1.0):
        errors.append("genetic_mutation_rate must be in [0, 1]")
    if errors:
        raise ValueError("\n".join(errors))


def validate_round_robin_params(args):
    """Raise ValueError if round-robin parameters are invalid."""
    if args.block_size < 1:
        raise ValueError("block_size must be >= 1")


def validate_weighted_params(args):
    """Raise ValueError if weighted-mode parameters are invalid."""
    if hasattr(args, 'weighted_eval_freq') and args.weighted_eval_freq < 1:
        raise ValueError("weighted_eval_freq must be >= 1")


def generate_example_allocation_map(num_agents: int, num_instances: int) -> str:
    """Generate an example allocation-map string for display purposes."""
    scenario = detect_scenario(num_agents, num_instances)
    if scenario == "AGENT_MULTIPLE_INSTANCES":
        alloc = _distribute_instances_to_agents(num_agents, num_instances)
    elif scenario == "INSTANCE_SHARING":
        alloc = _distribute_agents_to_instances(num_agents, num_instances)
    else:
        alloc = {i: [i] for i in range(num_agents)}
    return ';'.join(
        f"{aid}:{','.join(map(str, insts))}"
        for aid, insts in sorted(alloc.items())
    )
