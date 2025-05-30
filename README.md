# Pong Battle AI

Pong Battle AI is a reinforcement learning project that trains an AI agent to play a custom version of Pong. The left paddle is controlled by a trained PPO agent, while the right paddle is a simple scripted opponent that follows the ball vertically.

## How It Works

- **Environment**: Built with Gymnasium and PyGame (`pong_env.py`), featuring real time rendering.
- **Agents**:
  - Left Paddle: Controlled by a reinforcement learning agent (PPO).
  - Right Paddle: Scripted to track the ball's vertical position.
- **Game Mechanics**:
  - Ball speed increases after each paddle hit.
  - +1 reward for scoring, -1 for conceding.
  - +0.1 reward for hitting the ball.
  - Distance penalty for poor paddle alignment.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

- Train the AI:
  ```bash
  python train.py
  ```
- Evaluate the AI:
  ```bash
  python evaluate.py
  ```

