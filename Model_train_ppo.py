import gym
import sys
sys.path.append('/home/bargavan/VIT_submission/environment.py')
sys.path.append('/home/bargavan/VIT_submission/gym-pybullet-drones')
#from environment import DroneNavigationAviary
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import pickle
from gym_pybullet_drones.envs import HoverAviary
env = HoverAviary(gui=True,record=False)
env = Monitor(env,'monitor_name')
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=75000,
                             render=False)
env.reset()

# Customising Neural Network architecture
policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[512,512,256,128])

model = PPO('MlpPolicy',env,policy_kwargs=policy_kwargs,verbose=1,device='cuda')


model.learn(300000,callback=eval_callback)

t = env.get_episode_rewards()
model.save("drone_model")



file_name = "rewards_val.pkl"
op_file = open(file_name,'wb')
pickle.dump(t, op_file)
op_file.close()

fi,a = plt.subplots()
a.plot(np.arange(len(t)),t)
plt.show()
