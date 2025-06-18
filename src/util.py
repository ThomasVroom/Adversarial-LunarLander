import time
import torch
from tqdm import tqdm
import gymnasium as gym

# helper method for creating environments
def make_env(env_id, gravity, enable_wind, wind_power, turbulence_power, max_episode_steps=None):
    def thunk():
        env = gym.make(
            env_id,
            max_episode_steps=max_episode_steps,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk
def make_adv_env(env_id, gravity, wind_power, turbulence_power, max_episode_steps=None):
    def thunk():
        env = gym.make(
            env_id,
            max_episode_steps=max_episode_steps,
            gravity=gravity,
            wind_power=wind_power,
            turbulence_power=turbulence_power
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

# helper method that visualizes or records a random episode
def visualize_episode(
        env_id,
        gravity,
        enable_wind,
        wind_power,
        turbulence_power,
        agent,
        seed=None,
        device="cpu",
        max_time=30,
        video_name=None
    ):
    env = gym.make(
        env_id,
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        render_mode="rgb_array" if video_name else "human"
    )
    if video_name: 
        env = gym.wrappers.RecordVideo(env, f"videos/{video_name}")

    # reset environment
    obs, _ = env.reset(seed=seed)
    done = False
    target_time = time.time() + max_time
    reward_sum = 0
    while not (done or time.time() > target_time):
        # convert observation to tensor
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        # sample action from agent
        with torch.no_grad():
            action = agent.get_action(obs)

        # execute action
        obs, reward, terminated, truncated, __ = env.step(action.detach().cpu().numpy())
        done = terminated or truncated
        reward_sum += reward
        env.render()

    # close ui
    env.close()
    print(f"Collected a total reward of: {reward_sum}")

# helper method for evaluating an agent on given seeds
def evaluate(agent, seeds, config, device, adversary=None):
    # create environment
    if adversary:
        env = gym.make(
            "AdversarialLunarLander-v0",
            gravity=config["gravity"],
            wind_power=config["wind_power"],
            turbulence_power=config["turbulence_power"],
            max_episode_steps=config["max_episode_steps"]
        )
    else:
        env = gym.make(
            "CustomLunarLander-v0",
            gravity=config["gravity"],
            enable_wind=config["enable_wind"],
            wind_power=config["wind_power"],
            turbulence_power=config["turbulence_power"],
            max_episode_steps=config["max_episode_steps"]
        )

    total_rewards = []

    # evaluate on given seeds
    for seed in tqdm(seeds):
        # reset environment
        obs, _ = env.reset(seed=seed)
        reward_sum = 0
        done = False

        # run out environment
        while not done:
            # convert obs to tensor
            obs = torch.tensor(obs, dtype=torch.float32, device=device)

            # get action from agent
            with torch.no_grad():
                action = agent.get_action(obs).item()
                if adversary:
                    adversary_action = adversary.get_action(obs).item()
                    action = [action, adversary_action]

            # perform action
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            done = terminated or truncated

        # save reward_sum
        total_rewards.append(reward_sum)
    return total_rewards
