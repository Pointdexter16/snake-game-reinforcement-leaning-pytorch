import torch
import numpy as np
import random
from game import SnakeGame,Point,Direction
from collections import deque
from model import Linear_Qnet,QTrainer
from helper import plot

MAX_MEM=100_000
BATCH_SIZE=1000
LR=0.001

class Agent():

    def __init__(self):
        self.no_games=0
        self.epsilon=0
        self.memory=deque(maxlen=MAX_MEM)
        self.gamma=0.9 
        self.model=Linear_Qnet(11,256,3)
        self.trainer=QTrainer(self.model,LR,self.gamma)

    def get_state(self,game):
        head=game.snake[0]
        point_l=Point(head.x-20,head.y)
        point_r=Point(head.x+20,head.y)
        point_u=Point(head.x,head.y-20)
        point_d=Point(head.x,head.y+20)
    
        dir_r=Direction.RIGHT==game.direction
        dir_l=Direction.LEFT==game.direction
        dir_u=Direction.UP==game.direction
        dir_d=Direction.DOWN==game.direction

        state=[
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x<game.head.x,
            game.food.x>game.head.x,
            game.food.y<game.head.y,
            game.food.y>game.head.y,

        ]
        return np.array(state,dtype=int)
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    def train_long_mem(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample=random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample=self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)
    def train_short_mem(self,state,action,reward,new_state,done):
        self.trainer.train_step(state,action,reward,new_state,done)
    def get_action(self,state):
        self.epsilon=80-self.no_games
        final_move=[0,0,0]
        if random.randint(0,200)<self.epsilon:
            move=random.randint(0,2)
            final_move[move]=1
        else:
            state0=torch.tensor(state,dtype=torch.float)
            prediction=self.model(state0)
            move=torch.argmax(prediction).item()
            final_move[move]=1
        
        return final_move

def train():
    plot_score=[]
    plot_mean_score=[]
    total_score=0
    reward=0
    record=0
    agent=Agent()
    game=SnakeGame()
    while True:
        state_old=agent.get_state(game)
        final_mov=agent.get_action(state_old)
        reward, done , score=game.play_step(final_mov)

        state_new=agent.get_state(game)

        agent.train_short_mem(state_old,final_mov,reward,state_new,done)
        agent.remember(state_old,final_mov,reward,state_new,done)

        if done:
            game.reset()
            agent.no_games+=1
            agent.train_long_mem()
            if score>record:
                record=score
                agent.model.save()
            print(f"Game:{agent.no_games} Score:{score} Record:{record} ")
            plot_score.append(score)
            total_score+=score
            mean_score=total_score/agent.no_games
            plot_mean_score.append(mean_score)
            plot(plot_score,plot_mean_score)

if __name__=='__main__':
    train()