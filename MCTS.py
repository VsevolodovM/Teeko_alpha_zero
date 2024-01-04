import numpy as np
import math
import torch
from TeeKo_Game import *


def create_valid_moves_array(valid_moves, row_count, column_count):
    # 25 for placement + 625 for moves
    action_size = 25 + row_count * column_count * row_count * column_count
    valid_moves_array = np.zeros(action_size)

    # Mark placements and moves
    for move in valid_moves:
        if move.from_x == -1 and move.from_y == -1:  # Placement
            action_index = move.to_y * column_count + move.to_x
        else:  # Movement
            action_index_from = move.from_y * column_count + move.from_x
            action_index_to = move.to_y * column_count + move.to_x
            action_index = 25 + action_index_from * (row_count * column_count) + action_index_to
        valid_moves_array[action_index] = 1

    return valid_moves_array

def index_to_move(index):
    board_size = 5  # Assuming a 5x5 board
    num_positions = board_size * board_size  # 25 possible positions

    if index < num_positions:
        # The index is for placing a new piece
        to_y = int(index // board_size)
        to_x = int(index % board_size)
        return Move(-1, -1, to_x, to_y)
    else:
        # The index is for moving a piece
        index -= num_positions  # Adjust index to account for the placement actions
        from_index = index // num_positions
        to_index = index % num_positions

        from_y = int(from_index // board_size)
        from_x = int(from_index % board_size)
        to_y = int(to_index // board_size)
        to_x = int(to_index % board_size)

        return Move(from_x, from_y, to_x, to_y)

class Node:
    def __init__(self, game, args, state, parent = None, action_taken = None, prior = 0, visit_count = 0):
        self.game = game
        self.args = args
        self.state = state
        self.action_taken = action_taken

        self.parent = parent
        self.children = []
        # self.expandable_moves = game.get_valid_moves(state)
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        return best_child
    
    def get_ucb(self, child):

        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt((self.visit_count) / (child.visit_count + 1)) * child.prior
    

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:


        # index = np.random.choice(range(len(self.expandable_moves)))
        # action = self.expandable_moves[index]
        # self.expandable_moves = np.delete(self.expandable_moves, index)

                child_state = self.state.copy()
                action = index_to_move(action)
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player = -1)


                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
    
    # def simulate(self):
    #     value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
    #     value = self.game.get_opponent_value(value)

    #     if is_terminal:
    #         return value
        
    #     rollout_state = self.state.copy()
    #     rollout_player = 1

    #     while True:
    #         valid_moves = self.game.get_valid_moves(rollout_state)
    #         index = np.random.choice(range(len(self.expandable_moves)))
    #         action = self.expandable_moves[index]
    #         rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
    #         value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
    #         if is_terminal:
    #             if rollout_player == -1:
    #                 value = self.game.get_opponent_value(value)
    #             return value
            
    #         rollout_player = self.game.get_opponent(rollout_player)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self,game,args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count = 1)
        for search in range(self.args['num_searches']):
            node = root

            while len(node.children) > 0:
                node = node.select()


            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            


            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                valid_moves = create_valid_moves_array(valid_moves, self.game.row_count, self.game.column_count)
                policy *= valid_moves
                policy /= np.sum(policy)


                value = value.item()


                node.expand(policy)


            node.backpropagate(value)

                # selection
                # expansion
                # simulation
                # backpropagation
        
        action_probs = np.zeros(25 + 25 * 25)  # 25 for placement, 625 for moves
        for child in root.children:
            move = child.action_taken.give_move()
            if move[0] == -1 and move[1] == -1:  # Check if it's a placement
                action_index = move[3] * 5 + move[2]  # Placement index
            else:  # It's a move
                action_index_from = move[1] * 5 + move[0]  # 'from' move index
                action_index_to = move[3] * 5 + move[2]    # 'to' move index
                # The movement index is offset by 25 to account for placements
                action_index = 25 + action_index_from * 25 + action_index_to
            action_probs[action_index] = child.visit_count
        
        action_probs /= np.sum(action_probs)  # Normalize to get probabilities
        return action_probs
            

