import numpy as np
import random
from collections import deque

class SnakeGame:
    def __init__(self, width=10, height=10):
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
            reward = -1 - (len(self.snake) / 2)  # Bigger penalty for dying with longer snake
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
            reward = 1 + (len(self.snake) * 0.1)  # Bigger reward for eating with longer snake
            self.place_food()
        else:
            self.snake.pop()
            reward = distance_reward - 0.01  # Small penalty for each step
        
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

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.2
        self.learning_rate_decay = 0.9999
        self.gamma = 0.99
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
    
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

def train_model(episodes=10000):
    env = SnakeGame()
    state_size = 29  # Updated state size
    action_size = 3
    agent = QLearningAgent(state_size, action_size)
    
    scores = []
    best_score = 0
    
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
                if episode % 100 == 0:
                    avg_score = np.mean(scores[-100:])
                    print(f"Episode: {episode}, Average Score: {avg_score:.2f}, "
                          f"Best Score: {best_score}, Epsilon: {agent.epsilon:.2f}, "
                          f"Learning Rate: {agent.learning_rate:.3f}")
                break
    
    return agent, scores

if __name__ == "__main__":
    trained_agent, training_scores = train_model()