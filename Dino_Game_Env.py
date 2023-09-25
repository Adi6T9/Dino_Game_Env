#!/usr/bin/env python
# coding: utf-8

# # Importing Dependencies

# In[1]:


#mss for screen capture
from PIL import Image,ImageGrab
#sending commands
import pydirectinput
#Image trasnformation to data
import numpy as np
#OCR
import pytesseract
#For visualization
import matplotlib.pyplot as plt
import cv2
#time for pauses
import time
#Environment components:
from gym import Env
from gym.spaces import Box,Discrete


# # Environment Setup
# 

# In[10]:


class Dino(Env):
    #Setup the environment action and observation boxes
    def __init__(self):
        #subclass
        super().__init__()
        self.observation_space=Box(low=0,high=255,shape=(1,83,100),dtype=np.uint8)
        self.action_space=Discrete(3)
        #Extraction Parameters
        self.cap=ImageGrab
        self.game_location=(49,400,600,800)
        self.done_location=(650,420,1319,500)
    #Actions in the game
    def step(self,action):
        #Actions:0=Jump,1=Duck,2=No action
        action_map={0:"up",1:"down",2:"no_op"}
        if action!=2:
            pydirectinput.press(action_map[action])
        done,done_cap=self.game_over()
        new_observation=self.get_observation()
        reward=1
        if done==True:
            reward-=2
        info={}
        return new_observation,reward,done,info
    #Visualize the game
    def render(self):
        pass
    #Close the observation
    def close(self):
        pass
    #Restart the game
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150,y=150)
        pydirectinput.press("space")
        return self.get_observation()
    #Observations
    def get_observation(self):
        raw=np.array(self.cap.grab(bbox=self.game_location).convert("L"))
        return raw
    def game_over(self):
        done_cap=np.array(self.cap.grab(bbox=self.done_location))
        done=False
        res= pytesseract.image_to_string(done_cap)[:4]
        if res == "GAME":
            done=True
        return done, done_cap
    
    


# In[11]:


env = Dino()


# In[12]:


env.action_space.sample()


# In[5]:


plt.imshow(env.get_observation())


# In[6]:


env.get_observation().shape


# In[7]:


env.close()


# In[13]:


env=Dino()
for episode in range(10):
    obs=env.reset()
    done=False
    total_reward=0
    
    while not done:
        obs,reward,done,info= env.step(env.action_space.sample())
        total_reward+=reward
    print(f'Total Reward for episode{episode}is{total_reward}')


# In[ ]:




