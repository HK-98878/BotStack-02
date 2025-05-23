import pandas as pd
import numpy as np
df = pd.read_csv(r"/Users/xavierparker/Desktop/Bot Stack/battleships/battleship_dataset.csv")
data = df.values.astype('float32')

boards = np.array(data[:, :100])    # First 100 columns = board (hits)
ships = np.array(data[:, 100:])     # Last 100 columns = ships

print(boards.shape)
print(ships.shape)
