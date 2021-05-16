"""
This is the main scource code for training and testing. This code is modified from the previous work of our group, names are omitted.
"""

from __future__ import print_function
import datetime
import numpy as np
import torch 
import copy
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions import Categorical 
import math
import time
import sys
import os

import json
from datetime import datetime

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import matplotlib.backends
from IPython.display import HTML
from matplotlib import rc, colors
import matplotlib.cm as cm
import matplotlib.patches as patches


import argparse
import pickle

#fcns for print and get_time
import fcn as fcn
# from skimage.util import view_as_windows


################################################################
###################### DEFINE THE MODEL ########################
################################################################

class CModel(nn.Module):
    def __init__(self,HD=32,out=10, _msg_dim=3,frame_size=3,n_agent=3,device=None,random_action_flag=False,no_channels=1,edge_pr=0.5,kern_size=1, model = None):
        super(CModel, self).__init__()
        
        self.n_agent = n_agent
        self.HD = HD
        self.device=device
        self.MessageSize = _msg_dim
        self.ConvSize=69
        self.accuracy=0.1
        self.no_channels=no_channels
        self.edge_pr=edge_pr
        self.random_action_flag=random_action_flag
        self.probs=None
        self.preds=None
        self.lstmcom_flag = False
        self.downOp = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.imageDims=(28,28)
        self.maxDim=self.imageDims[0]
        
        # actually frame size
        self.patchSize=4
        
        # input vgg model here
        self.model = model
        self.in_f = self.model.classifier[6].out_features
        
        self.in_neighbors=[[] for i in range(n_agent)]#,[],[]]
        for i in range(n_agent):
            self.in_neighbors[i]=[(i+1)%n_agent]#[(i+j)%n_agent for j in range(1,n_agent)]
            
        self.frame_size=frame_size
        
        # LSTM 
        self.lstm = nn.LSTMCell(8, 64)
        # LSTMCOM
        self.lstmf1 = nn.LSTMCell(8, 64)
#         self.lstmf2 = nn.LSTMCell(4*self.in_f, self.in_f)
        
        self.pred = nn.Linear(64, out)
        
        #Add goal and action here, try without device first
        self.gn1 = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.gn2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gp=nn.Conv2d(64, 3, kernel_size=3,padding=1)
        
        self.inf_f = 0
        

        
    def forward(self, loc, images, states=None, m=None,n_agent=2):
        
        '''
        loc: [[bs x2], next agents] # NEEDS REVISING :D
        x: [ [bs x 1 x FS x FS] , next agents ...]
        states: [ ([bs x HD], [bs x HD]), next agents ... ]
        m: [ ([bs x 2]), next agents]
        
        '''
        n_agent = self.n_agent
        batch_size=images.shape[0]/n_agent
        
        if states is None: 
            statesH = 0*torch.randn(n_agent*batch_size,64,device=self.device)
            statesC = 0*torch.randn(n_agent*batch_size,64,device=self.device)
            states=(statesH,statesC)

        else:
            statesH=states[0]
            statesC=states[1]
        
        b_img = images #input image
        b_imgg = b_img.clone()
        b_img = self.model(b_img) #output from vgg, need one linear layer to be class
        total_input=b_img.to(self.device) #[batch * # of agents, feature size]
        
        states_input=(states[0],states[1]) 

        hx, cx=self.lstm(total_input,states_input)
        
        max_inf_f = 0
        for i in range(batch_size):
            f = torch.mm(self.lstm.weight_ih[64:128],b_img.to(self.device)[i].view(8,1)) + torch.mm(self.lstm.weight_hh[64:128],states_input[0][i].view(64,1))+self.lstm.bias_hh[64:128].view(64,1)+self.lstm.bias_ih[64:128].view(64,1)
            sig = nn.Sigmoid()
            f = sig(f)
            inf_f = torch.norm(f.view(1,64),p=float('inf')).data.cpu().numpy()
            
            if inf_f > self.inf_f:
                self.inf_f = inf_f
        
        del total_input,states_input
        
        pred = self.pred(hx)
       
        states_new=(hx,cx)
            
        hxg = torch.cat((hx,loc.float()),1)
        x_goal = hxg.unsqueeze(1)
        x_goal = x_goal.unsqueeze(1)
        x_goal = self.relu(self.gn1(x_goal))
        x_goal = self.relu(self.gn2(x_goal))  
        x_goal=torch.mean(self.gp(x_goal),[2,3])
        mu_goal=self.maxDim/2.0+self.maxDim*torch.tanh(x_goal[:,0:2])/2.0
        sigma_goal=torch.exp(x_goal[:,2])*10
        x= torch.stack(mu_goal.shape[0]*[torch.range(0,self.maxDim)],0).to(self.device)
        y=torch.stack(mu_goal.shape[0]*[torch.range(0,self.maxDim)],0).to(self.device)

        mu_x_vector=mu_goal[:,0].view(-1,1)+x*0 
        mu_y_vector=mu_goal[:,1].view(-1,1)+x*0 
        gx = torch.exp(-(x-mu_x_vector)*(x-mu_x_vector)/((2*sigma_goal*sigma_goal).view(-1,1)))
        gy = torch.exp(-(y-mu_y_vector)*(y-mu_y_vector)/((2*sigma_goal*sigma_goal).view(-1,1)))

        x_goal=torch.bmm(gx.unsqueeze(2),gy.unsqueeze(1))
        x_goal=x_goal+0.000001
        x_goal=x_goal/(torch.sum(x_goal,[1,2]).view(-1,1,1))
        self.plot = x_goal[0]
        goalDist=Categorical(x_goal.view(x_goal.shape[0],-1))
        setGoals=goalDist.sample()
        setGoals=torch.stack([setGoals/(self.maxDim+1),setGoals%(self.maxDim+1)],1).unsqueeze(0)
        setGoals=torch.clamp(setGoals,min=0,max=self.maxDim -self.patchSize).int()   
        
        del hx,cx,x_goal,mu_goal,sigma_goal,x,y,mu_x_vector,mu_y_vector,gx,gy,goalDist

        pred_average=pred[0*batch_size:(0+1)*batch_size]*0
        
        # added for consensus of predictions 
        for i in range(n_agent):
            pred_average= pred_average+(pred[i*batch_size:(i+1)*batch_size])/(n_agent*1.0)

        pred=pred_average
        self.preds=pred_average
        
        return pred, states_new, setGoals


    
################################################################
###################### DEFINE THE ACTOR ########################
################################################################

class Actor():
    # this class defines a policy network with two layer NN
    def __init__(self, model=None, env=None,time_horizon=10,device=None):
        
        self.device=device
        self.model = model # model of the actor
        self.env=env # environment of the actor
        self.final_preds=None
        
        self.step_preds=None
        self.time_horizon=time_horizon
        self.n_agent=env.n_agent
        self.target_n_r=None
        self.random_action_flag = False
        
    # simply transfering here
    def select_action(self, loc, image, states):
       
        preds, states, setGoals = self.model(loc, image, states, self.n_agent)
        return preds, states, setGoals
    
    def rollout(self):#,render=False,pause=.0):
        states=None
        iter_counter =0
        rewards_sum=0
        log_probs_sum=0
        goal_counter = 0
        
        log_probs_summed=torch.zeros(self.env.batch_size,device=self.device)

        # Loop for time horizon
        for iter_counter in xrange(self.time_horizon):

            loc=self.env.get_agent_locs().to(self.device)
            image=self.env.get_agent_images().to(self.device)
            
            preds, states, setGoals = self.select_action(loc,image, states)
            
            if iter_counter%4 == 0:
                setG_f = setGoals
        
            loc_f = loc.float()
            setG_f = setG_f.float().squeeze(0)
            a_m = setG_f - loc_f
            a_m2 = loc_f - setG_f
            arr0 = a_m[:,0].to(self.device)
            arr2 = a_m[:,1].to(self.device)
            arr1 = a_m2[:,0].to(self.device)
            arr3 = a_m2[:,1].to(self.device)
            
            del a_m, a_m2
            
            a_up = torch.tensor(arr1.shape).to(self.device)
            a_final = torch.cat((arr0.unsqueeze(1),arr1.unsqueeze(1),arr2.unsqueeze(1),arr3.unsqueeze(1)),dim = 1)
            am = a_final.argmax(1)
            action = (am % a_up).view(-1, 1)
            action = action.squeeze(1)
            
            if self.random_action_flag:
                action = torch.LongTensor(arr1.shape).random_(0, 4)
                
            self.env.step(action,setG_f)
            
    
        mmm = nn.Softmax(dim=1)
        self.final_preds=mmm(preds)
        self.final_reward=torch.norm((mmm(preds)-self.env.true_labels_single),p=2,dim=1)
        self.final_reward=-torch.pow(self.final_reward,2)

        rewards_sum=self.final_reward   #reward
#         print(penalty[1])
#         print(rewards_sum[1])
        
        return  rewards_sum #log_probs_sum #s,

    def multi_rollout(self,img,target,n_rollout=50):
        
#         log_prob_list=torch.zeros(n_rollout,self.time_horizon)
        n_batches=img.shape[0]
        rewards_sum_list_means=torch.zeros(n_batches,device=self.device) 

        img_n_r=torch.cat([img for i in range(n_rollout)],0)
        target_n_r=torch.cat([target for i in range(n_rollout)],0)
        self.target_n_r=target_n_r
        self.env.reset(img_n_r,target_n_r)
        rewards_sum_n_r = self.rollout() 
        
        # sum all the rollouts
        for i in range(n_rollout):
            rewards_sum_list_means=rewards_sum_list_means+rewards_sum_n_r[i*n_batches:(i+1)*n_batches]/(1.0*n_rollout)
            
        baseline=torch.cat([rewards_sum_list_means for i in range(n_rollout)])
        obj_function=rewards_sum_n_r#-baseline

        return torch.mean(obj_function)

    
################################################################
####################### DEFINE THE ENV #########################
################################################################
    
class EnvMNIST(object):
    def __init__(self,full_images,targets,filter_size=4,n_agent=2,seed=50,device=None,max_dim=28):
        '''
        Input:
            full_image: the image that the agents walk on
            filter_size: the number of pixel width and heights 
            n
        '''
        self.max_dim=max_dim
        self.device=device
        self.batch_size=full_images.shape[0]
        self.filter_size=filter_size # same with frame size
        self.n_agent = n_agent
        self.seed=seed
        self.rnd = np.random.RandomState(seed=self.seed)
        self.targets=targets

        self.agents = [MnistAgent(full_images, loc_init=torch.IntTensor([self.rnd.randint(0,self.max_dim-self.filter_size,size=2)]),filter_size=self.filter_size,device=self.device,max_dim=max_dim) for i in range(n_agent)]
        self.full_images=full_images
        self.reset(full_images,targets)
        

    def reset(self, full_images=None,targets=0):
        
        # reset all agents
        s_lst = []
        
        if full_images is not None:
            self.full_images = full_images
            self.batch_size=full_images.shape[0] #[167,1,28,28]

        self.locs=[]  
        true_labels_single=torch.zeros(self.batch_size,10,device=self.device)  

        for i in range(self.n_agent):
            s_lst.append(self.agents[i].reset(self.full_images))            
            #print 's',self.agents[i].filter_size
            self.locs.append(self.agents[i].locs)
#         print "environemnt reset Part 0.5",time.time()-st2
            
            
        true_labels_single[ range(self.batch_size), targets] = 1
        self.true_labels_single=true_labels_single
        self.targets=targets
             
        self.true_labels=torch.cat([true_labels_single for i in range(self.n_agent)],0)

        #print 
        self.s_lst=s_lst    
        self.s=torch.cat(s_lst,0)
        self.locs=torch.cat(self.locs,0)

    def step(self, actions,setG_f):
        for i in range(self.n_agent):
            self.agents[i].step(actions[i*self.batch_size:((i+1)*self.batch_size)],setG_f[i*self.batch_size:((i+1)*self.batch_size)])
        
    def get_agent_images(self):
        
        ag_image_lst=[]
        
        for i in range (self.n_agent):
            ag_image_lst.append(self.agents[i].s)
        
        return torch.cat(ag_image_lst,0)  
    
    def get_agent_locs(self):
        
        ag_loc_lst=[]
        
        for i in range (self.n_agent):
            ag_loc_lst.append(self.agents[i].locs)
            
        return torch.cat(ag_loc_lst,0)
    
    def show_domain(self,num=0):

        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
#         ax.imshow(np.squeeze(self.full_images[num]))
        myimshow(ax,self.full_images[num])
   
        rect = patches.Rectangle((self.locs[num,0],self.locs[num,1]),
                                 self.filter_size,self.filter_size,linewidth=2,
                                 edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.show()
        
    def show_history(self,num=0):
        
        colors = ['w', 'red', 'yellow' ,'green','blue','purple','orange','c']
        frames = len(self.agents[0].loc_history)
        print("Rendering %d frames..." % frames)
        # Create figure and axes
        fig,ax = plt.subplots(1)

        def render_frame(i):
            # Display the image
#             ax.imshow(np.squeeze(self.full_images[num]))
            myimshow(ax,self.full_images[num])
            if i > 0:
                [p.remove() for p in reversed(ax.patches)]
                for j in range(len(self.agents)):

                    rect = patches.Rectangle((self.agents[j].loc_history[i-1][num,0],self.agents[j].loc_history[i-1][num,1]),
                                         self.filter_size,self.filter_size,linewidth=2,
                                         edgecolor=colors[j],facecolor='none')
                    rect1 = patches.Ellipse((self.agents[j].goal_history[i-1][num,0],self.agents[j].goal_history[i-1][num,1]),
                                         self.filter_size/2,self.filter_size/2,
                                         edgecolor=colors[j],facecolor=colors[j])
                    
                    ax.add_patch(rect)
                    ax.add_patch(rect1)
                
            for j in range(len(self.agents)):
                    comc = 0
                    for k in range(len(self.agents)):
                        if k > j:
                            comc = comc +1
                            arrow = patches.Arrow(self.agents[j].loc_history[i-1][num,0]+self.filter_size/2.0, self.agents[j].loc_history[i-1][num,1]+self.filter_size/2.0, (self.agents[k].loc_history[i-1][num,0]-self.agents[j].loc_history[i-1][num,0]).cpu().numpy(), (self.agents[k].loc_history[i-1][num,1]-self.agents[j].loc_history[i-1][num,1]).cpu().numpy(), color = 'palegreen', linestyle = '-.', linewidth = 1)
                            comx = (self.agents[k].loc_history[i-1][num,0]-self.agents[j].loc_history[i-1][num,0]).cpu().numpy()
                            comy = (self.agents[k].loc_history[i-1][num,1]-self.agents[j].loc_history[i-1][num,1]).cpu().numpy()
                            comr = math.sqrt(comx**2 + comy**2)
                            if comr < 15 and comc<4:
                                ax.add_patch(arrow)

        anim = matplotlib.animation.FuncAnimation(
            fig, render_frame, frames=frames+1, interval=600)
        plt.close()
        display(HTML(anim.to_html5_video()))
   

class MnistAgent(object):
    def __init__(self,full_images,loc_init=[0,0],filter_size=4,device=None,max_dim=28):
        self.device=device
        self.decMap = torch.IntTensor([[1,0],[-1,0],[0,1],[0,-1]]).to(self.device)

        self.max_dim=max_dim
        self.batch_size=full_images.shape[0]
        self.prev_batch_size=0
        
        self.loc_init=loc_init.to(self.device)

        self.filter_size = filter_size

        self.reset(full_images)
        
        
    def reset(self, full_images):
        #self.h = []   # hidden state
        self.batch_size=full_images.shape[0]
        if self.batch_size== self.prev_batch_size:
            pass
        else:
            self.raw_grid=self.generate_grid(self.filter_size,self.filter_size,self.max_dim)
        
        self.prev_batch_size=self.batch_size
        
#         print (self.loc_init)
        self.locs = torch.cat([self.loc_init for i in range(self.batch_size)],0)

        self.full_images = full_images
        self.s = self.get_observation()
        self.loc_history=[]
        self.goal_history=[]
        self.image_history=[]
        self.loc_history.append(self.locs)
        self.goal_history.append(self.locs)
        self.image_history.append(self.s)
        
        return self.s

    def reset_FAST(self, full_images):
        #self.h = []   # hidden state
        self.batch_size=full_images.shape[0]
        if self.batch_size== self.prev_batch_size:
            pass
        else:
            self.raw_grid=self.generate_grid(self.filter_size,self.filter_size,self.max_dim)
        
        self.prev_batch_size=self.batch_size
            
        self.locs = torch.cat([self.loc_init for i in range(self.batch_size)],0)

        self.full_images = full_images
#         self.s = self.get_observation()
        self.loc_history=[]
#         self.image_history=[]
        self.loc_history.append(self.locs)
#         self.image_history.append(self.s)
        
        return self.s    
    
    def generate_grid(self,h, w,filter_size):
        
        x = (torch.range(0, h-1)-(filter_size-1)/2.0)/((filter_size-1)/2.0)
        y = (torch.range(0, w-1)-(filter_size-1)/2.0)/((filter_size-1)/2.0)
        grid = torch.stack([x.repeat(w), y.repeat(h,1).t().contiguous().view(-1)],1).to(self.device)
        grid=torch.stack([grid for i in range(self.batch_size)],0)

        return grid

    def translate_grid(self,locs,grid,filter_size):
        
        #print grid.shape,locs.unsqueeze(1).shape
        translated_grid=grid+(locs.unsqueeze(1).float().to(self.device))/((filter_size-1)/2.0)

        return translated_grid
    

    
    def get_observation(self):
        
        cropping_grid=\
        self.translate_grid(self.locs,self.raw_grid,self.max_dim).view(self.batch_size,self.filter_size,self.filter_size,2).to(self.device)
        
        s=torch.nn.functional.grid_sample(self.full_images,cropping_grid)

        return s  # shape of [channels, filter_size, filter_size]
    
    def step(self,actions,setG_f):
        # 0:down 1:up 2:right 3: left
        decision_vector =  self.decMap[actions]
        self.goal = setG_f
#         self.locs = self.goal
        del setG_f
        self.locs=torch.clamp(self.filter_size*decision_vector+self.locs,min=0,max=self.max_dim -self.filter_size)
        self.s = self.get_observation()
        self.image_history.append(self.s)
        self.loc_history.append(self.locs)
        self.goal_history.append(self.goal)

        return self.s#, r, done 
    
    def step_FAST(self,actions):
        # 0:down 1:up 2:right 3: left
        decision_vector =  self.decMap[actions]
        
        
        self.locs=torch.clamp(self.filter_size*decision_vector+self.locs,min=0,max=self.max_dim -self.filter_size)

        self.loc_history.append(self.locs)

        return None
        
#         return self.s#, r, done 
    
def myimshow(ax,img):
    img = torch.squeeze(img / 2 + 0.5 )    # unnormalize
#     img = img[0]
    img = img.permute(1,2,0)
    npimg = img.cpu().data.numpy()
    ax.imshow(npimg) 
    plt.show()
    
