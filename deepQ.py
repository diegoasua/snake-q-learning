import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from qlearning import SnakeGame
from datetime import datetime
import os

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.width = int(np.sqrt(input_size))  # 20 for 20x20 grid
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        # Adjust hyperparameters
        self.batch_size = 128  # Increased from 64
        self.gamma = 0.99      # Increased from 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Increased from 0.01
        self.epsilon_decay = 0.998  # Slower decay
        self.learning_rate = 0.0005  # Decreased from 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural Networks (one for selecting actions, one for evaluating actions)
        self.policy_net = DQN(state_size, 256, action_size).to(self.device)
        self.target_net = DQN(state_size, 256, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Convert lists to numpy arrays first, then to tensors
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Rest of the method remains the same
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def train_model(episodes=10000):
    env = SnakeGame()
    state_size = env.width * env.height  # 400 for 20x20 grid
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    best_score = 0
    target_update_frequency = 10  # Update target network every 10 episodes
    
    # Create directory for checkpoints
    checkpoint_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            
            if done:
                scores.append(env.score)
                if env.score > best_score:
                    best_score = env.score
                    agent.save(f"{checkpoint_dir}/best_model.pth")
                
                if episode % 100 == 0:
                    avg_score = np.mean(scores[-100:])
                    print(f"Episode: {episode}, Score: {env.score}, "
                          f"Average Score: {avg_score:.2f}, "
                          f"Best Score: {best_score}, "
                          f"Epsilon: {agent.epsilon:.2f}")
                
                if episode % target_update_frequency == 0:
                    agent.update_target_network()
                
                break
    
    return agent, scores

if __name__ == "__main__":
    trained_agent, training_scores = train_model()
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(training_scores)
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Add moving average
    window_size = 100
    moving_avg = np.convolve(training_scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(training_scores)), moving_avg, 'r', 
             label=f'{window_size}-episode moving average')
    
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.show()
    
    # Save final model
    trained_agent.save("final_model.pth")