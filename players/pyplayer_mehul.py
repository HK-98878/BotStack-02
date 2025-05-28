import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

'''
### Key Implementation Notes ###

Don't touch the function definitions (only the contents) for 'player1PlaceShip' and 'player1Action' - it will crash the program
The data passed to these functions is essentially the same as for the C implementation, except using lists rather than arrays
 - board is a 1D list
 - remaining_ships is a 1D list (5 elements, true or false for hit or sunk)
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

# ship ids - used as an id, and index for remaining_ships
SHIP_PATROL_BOAT = SHIP_SUBMARINE = SHIP_DESTROYER = SHIP_BATTLESHIP = SHIP_CARRIER = 0

# square types - used in board
SQUARE_EMPTY = SQUARE_MISS = SQUARE_HIT = SQUARE_SHIP_PATROL_BOAT = SQUARE_SHIP_SUBMARINE = SQUARE_SHIP_DESTROYER = SQUARE_SHIP_BATTLESHIP = SQUARE_SHIP_CARRIER = 0

# hit types - used in history
SHOT_MISS = SHOT_HIT = SHOT_SUNK = 0

BOARD_SIZE = 0

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
  if ship == SHIP_PATROL_BOAT:
    return [1,1,0]
  if ship == SHIP_SUBMARINE:
    return [3, 1, 0]
  if ship == SHIP_DESTROYER:
    return [7, 2, 0]
  if ship == SHIP_BATTLESHIP:
    return [1, 5, 0]
  if ship == SHIP_CARRIER:
    return [5, 5, 0]
  return [-1, -1, -1]

class CNN(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_filters,    kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_filters*2, base_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base_filters, 1,              kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)

def player1Action(board: list, remaining_ships : list, history : list):
  # In board variable:
  # 0 is no shot
  # 1 is miss
  # 2 is hit

  # Create a mask of valid targets (no shot yet)
  valid_mask = np.array(board) == 0

  # Copy and transform board for model input:
  proc_board = board.copy()
  for i in range(len(proc_board)):
      if proc_board[i] == 1:
          proc_board[i] = -1.0
      elif proc_board[i] == 2:
          proc_board[i] = 1.0

  # Load and run the model
  model = CNN()
  model.load_state_dict(torch.load('players/model_mehul.pt', map_location=torch.device('cpu')))
  model.eval()
  
  # Prepare input tensor
  input_tensor = torch.tensor(np.array(proc_board).reshape(1, 10, 10)).float()
  with torch.no_grad():
      output = model(input_tensor).view(-1)  # flatten to shape (100,)
  
  mask_tensor = torch.tensor(valid_mask.astype(np.float32)).view(-1)  # 1 for valid, 0 for invalid
  
  # Set invalid positions' scores to a very low value
  neg_inf = torch.tensor(-1e9)
  masked_output = torch.where(mask_tensor.bool(), output, neg_inf)

  # Choose the best valid move
  best_idx = torch.argmax(masked_output).item()
  
  # Convert flat index to row, col
  row = best_idx // 10
  col = best_idx % 10

  return [row, col]