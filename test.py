from maze_solver import AStarMazeEnvironment, AStarAgent, TurnResult, Action, AStarAgentPlanningState
import sys

env = AStarMazeEnvironment('manual_hazards')
agent = AStarAgent()

start = env.reset()
agent.reset_episode()
result = TurnResult()
result.current_position = start
result.actions_executed = 5

for ep in range(5):
    for i in range(100000):
        prev_state = agent.plan_state
        prev_exp = agent.expansion_state
        actions = agent.plan_turn(result)
        result = env.step(actions)
        if result.is_dead:
            sys.stderr.write('DEATH at turn %d: prev_plan=%s prev_exp=%s pos=%s fire_map_size=%d\n' % (i, prev_state, prev_exp, result.current_position, len(agent.fire_map)))
            sys.stderr.write('  _last_turn_planned=%s actions_executed=%d\n' % (agent._last_turn_planned_cells, result.actions_executed))
            sys.stderr.write('  fire_groups_active: %s\n' % list(env._active_fire_cells()))
        if result.is_goal_reached:
            break

    stats = env.get_episode_stats()
    sys.stderr.write(f'ep {ep}: turns=%d cells=%d deaths=%d fire_map=%d\n' % (stats['turns_taken'], stats['cells_explored'], stats['deaths'], len(agent.fire_map)))

