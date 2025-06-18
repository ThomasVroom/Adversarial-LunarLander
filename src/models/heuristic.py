import numpy as np

# heuristic for lunar lander
def heuristic(obs):
    angle_targ = obs[:, 0] * 0.5 + obs[:, 2] * 1.0 # angle should point towards center
    angle_targ = angle_targ.clip(-0.4, 0.4) # more than 0.4 radians (22 degrees) is bad
    hover_targ = 0.55 * np.abs(obs[:, 0]) # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - obs[:, 4]) * 0.5 - (obs[:, 5]) * 1.0
    hover_todo = (hover_targ - obs[:, 1]) * 0.5 - (obs[:, 3]) * 0.5

    angle_todo[(obs[:, 6] or obs[:, 7]).astype(bool)] = 0
    hover_todo[(obs[:, 6] or obs[:, 7]).astype(bool)] = -obs[:, 3] * 0.5 # override to reduce fall speed

    a = np.zeros(1, dtype=int)
    a[hover_todo > np.abs(angle_todo) and hover_todo > 0.05] = 2
    a[a == 0 and angle_todo < -0.05] = 3
    a[a == 0 and angle_todo > +0.05] = 1
    return a
