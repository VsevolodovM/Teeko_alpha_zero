import numpy as np
from MCTS import *
import torch
from model import *

class Move:
    def __init__(self, from_x, from_y, to_x, to_y):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
    def give_move(self):
        return [self.from_x, self.from_y, self.to_x, self.to_y]




class TeeKo:
    def __init__(self):
        self.row_count = 5
        self.column_count = 5
        self.action_size = 25
        self.move_num = 0
        self.max_pieces_per_player = 4
        
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, move, player):
        if move.from_x == -1 and move.from_y == -1:
            # New piece placement
            if state[move.to_y][move.to_x] == 0:
                state[move.to_y][move.to_x] = player
        else:
            # Moving an existing piece
            if state[move.from_y][move.from_x] == player and state[move.to_y][move.to_x] == 0:
                state[move.from_y][move.from_x] = 0
                state[move.to_y][move.to_x] = player
        self.move_num += 1
        return state

    
    def get_valid_moves(self, state, player = 1):
        valid_moves = []

        # Count how many pieces the player has on board
        player_pieces_count = np.sum(state == player)

        # If the player hasn't placed all their pieces yet
        if player_pieces_count < self.max_pieces_per_player:
            for y in range(self.row_count):
                for x in range(self.column_count):
                    if state[y, x] == 0:
                        valid_moves.append(Move(-1, -1, x, y))  # Add a move to place a new piece

        # If all pieces are placed, check for moving existing pieces
        else:
            for y in range(self.row_count):
                for x in range(self.column_count):
                    if state[y, x] == player:
                        # Check all adjacent cells (up, down, left, right, and diagonals)
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue  # Skip the current cell
                                ny, nx = y + dy, x + dx
                                # Check if the new position is within the board and is empty
                                if 0 <= ny < self.row_count and 0 <= nx < self.column_count and state[ny, nx] == 0:
                                    valid_moves.append(Move(x, y, nx, ny))

        return np.array(valid_moves, dtype=object) 
    
    def check_win(self, state, action):

        if action == None:
            return False

        # Rows
        for row in range(5):
            if (state[row, 0] == state[row, 1] == state[row, 2] == state[row, 3] != 0) or \
               (state[row, 1] == state[row, 2] == state[row, 3] == state[row, 4] != 0):
                return state[row, 0]  # or state[row, 1], since they are the same

        # Cols
        for col in range(5):
            if (state[0, col] == state[1, col] == state[2, col] == state[3, col] != 0) or \
               (state[1, col] == state[2, col] == state[3, col] == state[4, col] != 0):
                return state[0, col]  # or state[1, col], since they are the same

        # Check squares (2x2)
        for i in range(4):
            for j in range(4):
                if state[i, j] == state[i, j+1] == state[i+1, j] == state[i+1, j+1] != 0:
                    return state[i, j]

        # Check larger squares
        for i in range(3):
            for j in range(3):
                if state[i, j] == state[i, j+2] == state[i+2, j] == state[i+2, j+2] != 0:
                    return state[i, j]

        #9times
        for row in range(4):  # up to the 4th row
            for col in range(4):  # up to the 4th column
                if (state[row, col] == state[row, col + 1] == 
                    state[row + 1, col] == state[row + 1, col + 1]):
                    return state[row, col]

        # 8 times (normal and mirrored)
        for row in range(1, 4):  # from the 2nd to the 4th row
            for col in range(1, 4):  # from the 2nd to the 4th column
                if (state[row, col] == 
                    state[row - 1, col] == 
                    state[row, col - 1] == 
                    state[row, col + 1] ==
                    state[row + 1, col]):
                    return state[row, col]
                
        for row in range(1, 4):  # from the 2nd to the 4th row
            for col in range(1, 4):  # from the 2nd to the 4th column
                if (state[row, col] ==  
                    state[row - 1, col - 1] ==  
                    state[row - 1, col + 1] ==  
                    state[row + 1, col - 1] ==  
                    state[row + 1, col + 1]):
                    return state[row, col]

        # 2 times
        if (state[0, 1] == state[1, 4] == state[3, 0] == state[4, 3] != 0) or \
           (state[0, 3] == state[1, 0] == state[3, 4] == state[4, 1] != 0):
            return state[0, 1]  # or state[0, 3]

        # 1 time
        if (state[0, 2] == state[2, 0] == state[2, 4] == state[4, 2] != 0):
            return state[0, 2]

        return 0  # No winner
        
    def get_value_and_terminated(self, state, action):
        if self.move_num > 199:
            self.move_num = 0
            return 0, True
        if self.check_win(state, action) == 1 or self.check_win(state, action) == -1:
            self.move_num = 0
            return 1, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        return encoded_state

    

def index_to_move(index):
    board_size = 5  # Assuming a 5x5 board
    num_positions = board_size * board_size  # 25 possible positions

    if index < num_positions:
        # The index is for placing a new piece
        to_y = index // board_size
        to_x = index % board_size
        return Move(-1, -1, to_x, to_y)
    else:
        # The index is for moving a piece
        index -= num_positions  # Adjust index to account for the placement actions
        from_index = index // num_positions
        to_index = index % num_positions

        from_y = from_index // board_size
        from_x = from_index % board_size
        to_y = to_index // board_size
        to_x = to_index % board_size

        return Move(from_x, from_y, to_x, to_y)
    
