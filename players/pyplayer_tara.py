'''
### Key Implementation Notes ###

Don't touch the function definitions (only the contents) for 'player1PlaceShip' and 'player1Action' - it will crash the program
The data passed to these functions is essentially the same as for the C implementation, except using lists rather than arrays
 - board is a 1D list
 - remaining_ships is a 1D list
 - history is a nested list
    - last element is the most recent
    - elements are [x,y,hit]

The C macro/enum constants are provided automatically by the main program, such as ship and square ids, and the board size


### Important details ###

For an unknown reason, modification of global variables (variables defined outside the scope of a function) from inside 
 functions will crash the program. If you want to preserve variables between function calls, which also being able to modify
 them, use function attributes 
 (a helper decorator has been provided for you - example is in+above the default `player1Action` function)

'''



'''
Game constants - set automatically when the script is integrated into the C environment
- They exist here only so they are valid variables according to python syntax highlighters
'''

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np

import pandas as pd
import csv

# ship ids - used as an id, and index for remaining_ships
SHIP_PATROL_BOAT = SHIP_SUBMARINE = SHIP_DESTROYER = SHIP_BATTLESHIP = SHIP_CARRIER = 0

# square types - used in board
SQUARE_EMPTY = SQUARE_MISS = SQUARE_HIT = SQUARE_SHIP_PATROL_BOAT = SQUARE_SHIP_SUBMARINE = SQUARE_SHIP_DESTROYER = SQUARE_SHIP_BATTLESHIP = SQUARE_SHIP_CARRIER = 0

# hit types - used in history
SHOT_MISS = SHOT_HIT = SHOT_SUNK = 0

BOARD_SIZE = 10

#ship details 2, 3, 4, 5  L2x3   in 10x10 board NWSE

# class OurNetwork(nn.Module):
#   def __init__(self):
#       super(OurNetwork, self).__init__()   # initialises parent class, 
#       # makes passed variables dynamic rather than static
#       self.conv1 = nn.Conv2d(1,10, padding=0, kernel_size=3)  # output [10,8,8]
#       self.conv2 = nn.Conv2d(10,20, padding=0, kernel_size=3)  # output [20,6,6]
#       self.pooling = nn.MaxPool2d(2,2)    # output shape [20,3,3]
#       # MAKE SURE ALL SIZES MATCH
#       self.dense1 = nn.Linear(20*3*3, 256)  
#       self.dense2 = nn.Linear(256,100)
      
#   def forward(self,x):
#       # x = x.view(x.size(0),10,10)
#       # x = x.reshape(1,10,10)
#       x = F.relu(self.conv1(x))
#       x = F.relu(self.conv2(x))
#       # print(x.shape, "conv")
#       x = self.pooling(x)
#       # print(x.shape, "pool")
#       x = x.view(x.size(0),-1)
#       x = F.relu(self.dense1(x))
#       x = F.relu(self.dense2(x))
      
#       return x

class OurNetwork(nn.Module):
  def __init__(self):
      super(OurNetwork, self).__init__()   # initialises parent class, 
      # makes passed variables dynamic rather than static
      self.conv1 = nn.Conv2d(1,20, padding=2, kernel_size=5)  # output [20,8,8]
      self.conv2 = nn.Conv2d(20,40, padding=2, kernel_size=5)  # output [40,6,6]
      self.pooling = nn.MaxPool2d(2,2)    # output shape [40,5,5]
      # MAKE SURE ALL SIZES MATCH
      self.dense1 = nn.Linear(40*5*5, 300)  
      self.dense2 = nn.Linear(300,100)
      
  def forward(self,x):
      # x = x.view(x.size(0),10,10)
      # x = x.reshape(1,10,10)
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      # print(x.shape, "conv")
      x = self.pooling(x)
      # print(x.shape, "pool")
      x = x.view(x.size(0),-1)
      x = F.relu(self.dense1(x))
      x = F.relu(self.dense2(x))
      
      return x
      

''' 
Helper function for querying the board list at given coordinates rather than directly indexing
'''
def getBoardSquare(board: list, x : int, y : int):
  return board[y + x*BOARD_SIZE]

'''
Helper decorator used for "static" variables
'''
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def player1PlaceShip(board : list, remaining_ships : list, ship : int):
  # [x,y,dir]
  if ship == SHIP_PATROL_BOAT:  #2
    return [8,1,1]
  if ship == SHIP_SUBMARINE: #3
    return [7, 7, 0]
  if ship == SHIP_DESTROYER: #L
    return [1, 1, 0]
  if ship == SHIP_BATTLESHIP: #4
    return [3, 4, 0]
  if ship == SHIP_CARRIER: #5
    return [3, 7, 1]
  return [-1, -1, -1]

@static_vars(x = -1)
@static_vars(y = 0)
def player1Action(board: list, remaining_ships : list, history : list):
  for spot in board:
    if spot == 1:
      spot = -1
    elif spot == 2:
      spot = 1
      
  # print("AAAAAAA before loading")
  model = OurNetwork()
  # print("before path")
  model.load_state_dict(torch.load("players/model_tara.pth"))
  # print("found path")
  model.eval()
  # print("AAAAAAA after eval")
  board = torch.tensor(board)
  board = board.view(10,10)
  # print("tensor worked")
  board = board.unsqueeze(0).unsqueeze(0).float()
  # print("before output")
  output = model(board).view(10,10)
  
  # print("AAAAAAA before shoot here,", output.shape)
  maxx = 0
  maxy = 0
  biggest = 0
  # print("reset biggest")
  coord = [0,0]
  # output = np.array(output)
  # print("as numpy array ", output)
  
  # print("before finding max")
  # biggest = torch.max(output)
  # print(biggest)
  
  # print(output)

  # torch.unbind(output)
  # print("unbind ", output)
  for i in range(10):
    for j in range(10):
      if board[0][0][j][i] == 0:  # hasn't been shot at
        if output[i][j] > biggest:
          coord = [j,i]
          biggest = output[i][j]
          
  # print("AAAAAAA found spot to shoot")
  # print(coord)
  
  return coord

