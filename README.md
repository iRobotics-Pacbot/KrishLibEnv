# Pacbot stable-baselines

This repository uses Krishlib and trains stable-baselines

## Summary of files
env.py: Contains training environment, reward mechanics, action space, and Pygame rendering

evaluate.py: Evaluates a trained model, prints a stats summary, and saves to a .csv (bin/results/)

game.py: Represents an instance of a PacMan game and its actions

gameState.py: Represents game states, ghost behavior, pellets, and power ups

terminalColors.py: data for printing colors to terminal

testEnv.py: test environment run for setup and rendering

train.py: Creates final model (bin/*.zip), saves intermediate models (bin/best_model/), saves periodic evaluation results (bin/eval_logs/), and saves training metrics (bin/tensorboard/)

walls.py: bitmap of the PacMan game

### Training sequence
1. Train a baseline model using train.py

2. Review learning curves in TensorBoard and results in eval_logs/; Use command tensorboard --logdir bin/tensorboard and then open local URL

3. Compare best checkpoint model against final checkpoint and view metrics using evaluate.py

4. Start with modifying environment rewards and then if necessary model hyperparameters
