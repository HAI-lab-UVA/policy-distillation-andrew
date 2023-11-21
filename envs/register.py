from gymnasium.envs.registration import register

register(
    id='NewGoal-Pusher-v4',
    entry_point='envs.pusher.new_goal:NewGoalPusherEnv',
    max_episode_steps=100,
)