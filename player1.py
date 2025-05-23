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

@static_vars(x = -1)
@static_vars(y = 0)
def player1Action(board: list, remaining_ships : list, history : list):
  if player1Action.x == BOARD_SIZE:
    player1Action.x = -1
    player1Action.y += 1
  if player1Action.y == BOARD_SIZE:
    return [-1, -1]
  player1Action.x += 1

  return [player1Action.x, player1Action.y]