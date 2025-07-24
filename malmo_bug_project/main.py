from envs.malmo_env import MalmoEnv
from models.dqn_agent import DQNAgent

env = MalmoEnv("missions/bugged_item_mission.xml")
agent = DQNAgent(input_dim=4, output_dim=4)

for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember((state, action, reward, next_state, done))
        agent.train()
        state = next_state
        total_reward += reward
    print(f"Episode {episode}, Reward: {total_reward}")
