"""
reward — Reward calculation subsystem for Monster Hunter Tri RL agent.

Contains:
    - MonsterHunterRewardCalculator  (orchestrator)
    - ExplorationTracker             (octree-based exploration mapping)
    - CubeMarkerSystem               (zone/monster/water markers)
    - CampTracker                    (camp stay penalties & exit bonuses)
    - MonsterZoneTracker             (combat detection & zone-leave penalties)
    - OxygenTracker                  (drowning penalties & recovery bonuses)
"""