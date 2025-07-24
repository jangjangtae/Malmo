from envs.malmo_env import MalmoEnv
from models.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer
import os

def run_episode(env, agent, train=False):
    state = env.reset()
    total_reward = 0
    max_steps = 50
    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if train:
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward

def main():
    mission_path = os.path.join("missions", "bug_mission.xml")
    env = MalmoEnv(mission_path)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space)
    
    for episode in range(30):
        reward = run_episode(env, agent, train=True)
        print(f"[Episode {episode+1}] Reward: {reward:.2f}")

if __name__ == "__main__":
    main()
