import optuna
import numpy as np
from qlearning import SnakeGame, QLearningAgent
import matplotlib.pyplot as plt
from datetime import datetime
import os

def objective(trial):
    # Define hyperparameter search space
    params = {
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.995, 0.9999),
        'epsilon_min': trial.suggest_float('epsilon_min', 0.001, 0.1),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'learning_rate_decay': trial.suggest_float('learning_rate_decay', 0.9995, 0.9999),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'batch_size': trial.suggest_int('batch_size', 32, 256)
    }
    
    # Training configuration
    episodes = 2000  # Reduced for faster optimization
    env = SnakeGame()
    agent = QLearningAgent(state_size=29, action_size=3)
    
    # Update agent with trial parameters
    agent.epsilon_decay = params['epsilon_decay']
    agent.epsilon_min = params['epsilon_min']
    agent.learning_rate = params['learning_rate']
    agent.learning_rate_decay = params['learning_rate_decay']
    agent.gamma = params['gamma']
    agent.batch_size = params['batch_size']
    
    scores = []
    for episode in range(episodes):
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                scores.append(env.score)
                break
        
        # Report intermediate values
        if episode % 200 == 0:
            trial.report(np.mean(scores[-100:] if len(scores) > 100 else scores), episode)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return np.mean(scores[-100:])  # Return average of last 100 episodes

def run_optimization(n_trials=100):
    study_name = f"snake_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=200),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(f'{study_name}_history.png')
    
    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(f'{study_name}_importance.png')
    
    return study

if __name__ == "__main__":
    study = run_optimization(n_trials=100)