import numpy as np
import math

class Node:
    def __init__(self, game, args, state, parent = None, action_taken = None):
        self.game = game
        self.args = args
        self.state = state
        self.action_taken = action_taken

        self.parent = parent
        self.children = []
        self.expandable_moves = game.get_valid_moves(state)
        self.visit_count = 0
        self.value_sum = 0

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
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    

    def expand(self):
        index = np.random.choice(range(len(self.expandable_moves)))
        action = self.expandable_moves[index]
        self.expandable_moves = np.delete(self.expandable_moves, index)

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player = -1)


        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child 
    
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)

        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1

        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            index = np.random.choice(range(len(self.expandable_moves)))
            action = self.expandable_moves[index]
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value
            
            rollout_player = self.game.get_opponent(rollout_player)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self,game,args):
        self.game = game
        self.args = args


    def search(self, state):
        root = Node(self.game, self.args, state)
        for search in range(self.args['num_searches']):
            node = root

            while self.game.move_num < 200 and len(node.children) > 0:
                node = node.select()


            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)


            if not is_terminal:

                node = node.expand()
                value = node.simulate()


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
            

