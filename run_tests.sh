num_seeds=1
for ((i=0; i<num_seeds; i++)); do
  # ball-1D
  # python3 -m safe_explorer.main --main_trainer_task ballnd --env_ballnd_n 1 --main_trainer_seed `expr $i`
  # python3 -m safe_explorer.main --main_trainer_task ballnd --env_ballnd_n 1 --env_ballnd_enable_reward_shaping --main_trainer_seed `expr $i`
  # python3 -m safe_explorer.main --main_trainer_task ballnd --env_ballnd_n 1 --main_trainer_use_safety_layer --main_trainer_seed `expr $i`
  # ball-3D
  # python3 -m safe_explorer.main --main_trainer_task ballnd --env_ballnd_n 3 --main_trainer_seed `expr $i`
  # python3 -m safe_explorer.main --main_trainer_task ballnd --env_ballnd_n 3 --env_ballnd_enable_reward_shaping --main_trainer_seed `expr $i`
  python3 -m safe_explorer.main --main_trainer_task ballnd --env_ballnd_n 3 --main_trainer_use_safety_layer --main_trainer_seed `expr $i`
  # spacehip corridor
  # python3 -m safe_explorer.main --main_trainer_task spaceship --main_trainer_seed `expr $i`
  # python3 -m safe_explorer.main --main_trainer_task spaceship --env_spaceship_enable_reward_shaping --main_trainer_seed `expr $i`
  # python3 -m safe_explorer.main --main_trainer_task spaceship --main_trainer_use_safety_layer --main_trainer_seed `expr $i`
done