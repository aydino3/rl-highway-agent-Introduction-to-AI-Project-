RL Highway Agent (PPO) — Speed vs Safety in Dense Traffic
Header & Visual Proof
Embedded evolution media (place files in the repo under assets/ and link/embed them in GitHub README):
• assets/evolution.gif (recommended for automatic display on GitHub)
• assets/evolution.mp4 (optional)
Stages shown: Untrained Agent → Half-Trained Agent → Fully Trained Agent
Objective
Train a reinforcement learning agent to drive as fast as possible in dense traffic without crashing, using Gymnasium + highway-env and Stable-Baselines3 (PPO).
Environment
Gymnasium Env: highway-v0
Observation (State): kinematics/features of the ego vehicle and nearby vehicles (numeric, not pixels).
Action Space: discrete high-level actions (lane changes + speed control) provided by highway-env.
Methodology
Custom Reward Function (Math)
A custom reward wrapper shapes behavior toward safe high-speed driving. Let v be ego speed, 1_collision indicate collision, r_lane encourage efficient lane choice, and r_lc penalize excessive lane changes.
R(s,a) =
w_speed * clip((v - v_min) / (v_max - v_min), 0, 1)
+ w_right * r_lane
- w_collision * 1_collision
- w_lc * r_lc

Weights and thresholds are defined in src/config.py and applied in src/envs/reward_wrapper.py.
The Model: PPO
PPO (Proximal Policy Optimization) was chosen for its stability and strong performance on continuous control-style objectives, while remaining CPU-friendly for a laptop-scale experiment.
Key Hyperparameters
Hyperparameters are stored in src/config.py (e.g., total_timesteps, learning_rate, gamma, gae_lambda, clip_range, n_steps, batch_size, n_envs).
Policy Network
Policy: MlpPolicy
Architecture: a 2-layer MLP with separate policy/value heads (see src/agents training script).
Training Analysis
Training curve image (place in assets/ and link in README): assets/reward_vs_episodes.png
Graph Commentary
Early training is noisy because the agent explores and crashes frequently. With reward shaping and PPO’s clipped updates, the agent learns safer behaviors while maintaining higher speed, which increases the average episode reward and episode length.
Challenges & Failures
1) Video Recording / Rendering Pitfalls
While generating the evolution media, recording could fail if the environment was not initialized with an RGB render mode or if video recording was triggered at the wrong time.
Fix: use render_mode="rgb_array" and a robust frame capture pipeline; export MP4/GIF into assets/.
2) Progress Bar Dependencies
Stable-Baselines3 progress bars require extra packages (rich + tqdm).
Fix: install rich (tqdm is already a dependency in this repo).
How to Run
1) Install
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install rich

2) Train
PYTHONPATH=. python -m src.train

3) Generate Evolution Video (MP4 + GIF)
PYTHONPATH=. python -m src.video.make_evolution_video \
  --env-id highway-v0 \
  --mid-model outputs/models/ppo_mid.zip \
  --final-model outputs/models/ppo_final.zip \
  --steps 1200 \
  --make-gif \
  --out-dir outputs/videos

# Then copy the outputs into assets/
mkdir -p assets
cp -f outputs/videos/evolution.mp4 assets/evolution.mp4
cp -f outputs/videos/evolution.gif assets/evolution.gif

4) Plot Reward Curve
PYTHONPATH=. python -m src.plots.plot_reward_curve \
  --csv outputs/logs/monitor.csv \
  --out outputs/plots/reward_vs_episodes.png \
  --title "Reward vs Episodes"

cp -f outputs/plots/reward_vs_episodes.png assets/reward_vs_episodes.png

Repository Structure
src/
  agents/        # training scripts (PPO)
  envs/          # custom reward wrapper
  plots/         # plotting utilities
  video/         # evolution video generator
assets/
  evolution.gif
  evolution.mp4
  reward_vs_episodes.png

