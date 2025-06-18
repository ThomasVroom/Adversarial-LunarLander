from src.env.CustomLunarLander import CustomLunarLander
from src.env.CustomLunarLander import (
    SCALE,
    MAIN_ENGINE_Y_LOCATION,
    MAIN_ENGINE_POWER,
    SIDE_ENGINE_AWAY,
    SIDE_ENGINE_HEIGHT,
    SIDE_ENGINE_POWER,
    FPS,
    VIEWPORT_H,
    VIEWPORT_W,
    LEG_DOWN
)
from gymnasium import spaces
from typing import Optional
import math
import numpy as np

class AdversarialLunarLander(CustomLunarLander):

    def __init__(
        self,
        render_mode: Optional[str] = None,
        gravity: float = -10.0,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        super().__init__(
            render_mode,
            gravity,
            True,
            wind_power,
            turbulence_power
        )

        # multi-agent environment
        self.action_space = spaces.MultiDiscrete([4, 4])

    # action_protagonist = [0, 1, 2, 3]
    # action_adversary = [0, 1, 2, 3]
    def step(self, action):
        assert self.lander is not None

        if hasattr(action, "__getitem__"):
            action_protagonist = action[0]
            action_adversary = action[1]
        else:
            action_protagonist = action
            action_adversary = self.action_space[1].sample()

        # update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)), which is proven to never be periodic (k = 0.01)
            # wind_mag = math.tanh(math.sin(0.02 * self.wind_idx) + math.sin(math.pi * 0.01 * self.wind_idx)) * self.wind_power
            # self.wind_idx += 1
            wind_mag = (-1 if action_adversary % 2 == 0 else 1) * self.wind_power
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)), which is proven to never be periodic (k = 0.01)
            # torque_mag = math.tanh(math.sin(0.02 * self.torque_idx) + math.sin(math.pi * 0.01 * self.torque_idx))* self.turbulence_power
            # self.torque_idx += 1
            torque_mag = (-1 if action_adversary < 2 else 1) * self.turbulence_power
            self.lander.ApplyTorque(torque_mag, True)

        # apply engine impulses

        # tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if action_protagonist == 2:
            # main engine
            m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)

        s_power = 0.0
        if action_protagonist in [1, 3]:
            # action = 1 is left, action = 3 is right
            direction = action_protagonist - 2
            s_power = 1.0

            # the components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation dependent torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # add ten points for legs contact, the idea is if you lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30 # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}
