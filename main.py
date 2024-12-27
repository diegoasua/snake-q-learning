import numpy as np
import random
from collections import deque
import pickle
import pygame
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class SnakeGame:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.reset()
    
    def reset(self):
        self.snake = deque([(self.height//2, self.width//2)])
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.place_food()
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        return self._get_state()
    
    def place_food(self):
        while True:
            self.food = (random.randint(0, self.height-1), 
                        random.randint(0, self.width-1))
            if self.food not in self.snake:
                break
    
    def step(self, action):
        self.steps += 1
        self.steps_without_food += 1
        
        # Actions: 0 (straight), 1 (right), 2 (left)
        if action == 1:
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 2:
            self.direction = (self.direction[1], -self.direction[0])
        
        new_head = (self.snake[0][0] + self.direction[0],
                   self.snake[0][1] + self.direction[1])
        
        # Check if game is over
        game_over = (
            new_head[0] < 0 or new_head[0] >= self.height or
            new_head[1] < 0 or new_head[1] >= self.width or
            new_head in self.snake or
            self.steps_without_food > 100  # Prevent infinite loops
        )
        
        if game_over:
            reward = -1 - (len(self.snake) / 4)  # Bigger penalty for dying with longer snake
            return self._get_state(), reward, True
        
        self.snake.appendleft(new_head)
        
        # Calculate distance-based reward
        old_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        distance_reward = (old_distance - new_distance) * 0.1
        
        # Check if food is eaten
        if new_head == self.food:
            self.score += 1
            self.steps_without_food = 0
            reward = 1 + (len(self.snake) * 0.05)  # Bigger reward for eating with longer snake
            self.place_food()
        else:
            self.snake.pop()
            reward = distance_reward - 0.005  # Small penalty for each step
        
        return self._get_state(), reward, False
    
    def _is_danger(self, direction):
        head = self.snake[0]
        next_pos = (head[0] + direction[0], head[1] + direction[1])
        
        # Check if next position is out of bounds
        if (next_pos[0] < 0 or next_pos[0] >= self.height or
            next_pos[1] < 0 or next_pos[1] >= self.width):
            return True
        
        # Check if next position contains snake body
        if next_pos in self.snake:
            return True
        
        return False

    def _get_state(self):
        head = self.snake[0]
        
        # Calculate relative positions of body parts
        body_positions = set()
        for i in range(1, min(4, len(self.snake))):  # Consider up to 3 body segments
            rel_pos = (self.snake[i][0] - head[0], 
                      self.snake[i][1] - head[1])
            body_positions.add(rel_pos)
        
        state = [
            # Danger in 8 directions (N, NE, E, SE, S, SW, W, NW)
            *[self._is_danger((dx, dy)) for dx, dy in [
                (-1, 0), (-1, 1), (0, 1), (1, 1),
                (1, 0), (1, -1), (0, -1), (-1, -1)]],
            
            # Food direction (8 directions)
            self.food[0] < head[0],  # N
            self.food[0] < head[0] and self.food[1] > head[1],  # NE
            self.food[1] > head[1],  # E
            self.food[0] > head[0] and self.food[1] > head[1],  # SE
            self.food[0] > head[0],  # S
            self.food[0] > head[0] and self.food[1] < head[1],  # SW
            self.food[1] < head[1],  # W
            self.food[0] < head[0] and self.food[1] < head[1],  # NW
            
            # Current direction
            self.direction == (-1, 0),  # N
            self.direction == (0, 1),   # E
            self.direction == (1, 0),   # S
            self.direction == (0, -1),  # W
            
            # Length of snake (normalized)
            len(self.snake) / (self.width * self.height)
        ]
        return np.array(state, dtype=float)
    
    def render(self, screen):
        """New method to render the game state"""
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw food
        pygame.draw.rect(screen, (255, 0, 0), 
                        (self.food[1]*20, self.food[0]*20, 20, 20))
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(screen, (0, 255, 0),
                           (segment[1]*20, segment[0]*20, 20, 20))
            
        pygame.display.flip()

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_min = 0.001593721112625894
        self.epsilon_decay = 0.9962050178281874
        self.learning_rate = 0.4525171577269225
        self.learning_rate_decay = 0.9996387403565626
        self.gamma = 0.9634379410341871
        self.memory = deque(maxlen=20000)
        self.batch_size = 188
    
    def get_state_key(self, state):
        # Discretize continuous values
        return tuple(np.round(state, 3))
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def train(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            for state, action, reward, next_state, done in batch:
                self._update(state, action, reward, next_state, done)
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.learning_rate *= self.learning_rate_decay
    
    def _update(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.gamma * next_max * (1 - done))
        
        self.q_table[state_key][action] = new_value

    def save(self, filename):
        """Save the Q-table and agent parameters"""
        save_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filename):
        """Load a saved Q-table and agent parameters"""
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)
        self.q_table = save_data['q_table']
        self.epsilon = save_data['epsilon']
        self.learning_rate = save_data['learning_rate']

def visualize_agent(agent, env, speed=5):  # Reduced default speed from 100 to 5
    """Visualize the agent playing"""
    pygame.init()
    screen = pygame.display.set_mode((env.width * 20, env.height * 20))
    pygame.display.set_caption('Snake AI')
    
    state = env.reset()
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        action = agent.act(state)
        state, _, done = env.step(action)
        env.render(screen)
        time.sleep(1/speed)  # Now 0.2 seconds per move at default speed
    
    pygame.quit()
    

def train_model(episodes=10000):
    env = SnakeGame()
    state_size = 29  # Updated state size
    action_size = 3
    agent = QLearningAgent(state_size, action_size)
    
    scores = []
    best_score = 0
    checkpoint_interval = 1000  # Save every 1000 episodes

    # Create directory for checkpoints
    checkpoint_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                scores.append(env.score)
                if env.score > best_score:
                    best_score = env.score
                    # Save best model
                    agent.save(f"{checkpoint_dir}/best_model.pkl")
                if episode % 100 == 0:
                    avg_score = np.mean(scores[-100:])
                    print(f"Episode: {episode}, Average Score: {avg_score:.2f}, "
                          f"Best Score: {best_score}, Epsilon: {agent.epsilon:.2f}, "
                          f"Learning Rate: {agent.learning_rate:.3f}")
                    
                # Save checkpoint
                if episode % checkpoint_interval == 0:
                    agent.save(f"{checkpoint_dir}/checkpoint_{episode}.pkl")
                break
    
    return agent, scores

if __name__ == "__main__":
    # Train the model
    trained_agent, training_scores = train_model()
    
    # Plot learning curve
    figure(figsize=(10, 6))
    plt.plot(training_scores)
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Add moving average
    window_size = 100
    moving_avg = np.convolve(training_scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(training_scores)), moving_avg, 'r', label=f'{window_size}-episode moving average')
    
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.show()
    
    # Save the final model
    trained_agent.save("final_model.pkl")
    
    # Find and visualize different checkpoints
    import glob
    checkpoint_dirs = glob.glob("checkpoints_*")
    if checkpoint_dirs:
        latest_dir = max(checkpoint_dirs)
        checkpoints = sorted(glob.glob(f"{latest_dir}/checkpoint_*.pkl"))
        
        for checkpoint in checkpoints:
            episode_num = checkpoint.split('_')[-1].split('.')[0]  # Extract episode number
            pygame.display.set_caption(f'Snake AI - Checkpoint Episode {episode_num}')
            print(f"\nVisualizing Checkpoint from Episode {episode_num}")
            env = SnakeGame()
            agent = QLearningAgent(29, 3)
            agent.load(checkpoint)
            visualize_agent(agent, env)
        
        # Visualize the best model
        pygame.display.set_caption('Snake AI - Best Model')
        print("\nVisualizing Best Model")
        env = SnakeGame()
        agent = QLearningAgent(29, 3)
        agent.load(f"{latest_dir}/best_model.pkl")
        visualize_agent(agent, env)