def reward_func(env, state):
    x, x_dot, theta, theta_dot = state
    r1 = ((env.x_threshold - abs(x))/env.x_threshold - 0.5)
    r2 = ((env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5) * 1.5
    return r1 + r2