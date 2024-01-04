import numpy as np
from MCTS import *
import torch
from model import *
from TeeKo_Game import *
from tqdm import tqdm
from alpha_zero import *

# TeeKo_1 = TeeKo()
# player  = 1


# args = {
#     'C' : 2,
#     'num_searches': 400
# }

# model = ResNet(TeeKo_1, 4 , 64)
# model.eval()

# mcts = MCTS(TeeKo_1, args, model)

# state = TeeKo_1.get_initial_state()


# while True:
#     print(state)

#     if player == 1:
#         valid_moves = TeeKo_1.get_valid_moves(state, player)
#         print(len(valid_moves))
    
#         print("valide_moves", [valid_moves[i].give_move() for i in range(len(valid_moves))])

#         action = Move(int(input(f"{player} From x:")), int(input(f"{player} From y:")), int(input(f"{player} To x:")), int(input(f"{player} To y:")))
#         # # action =  Move(-1, -1 , 0,  0)


#         # neuteral_state = TeeKo_1.change_perspective(state, player)
#         # valid_moves = TeeKo_1.get_valid_moves(state, player)
#         # print(f"valide_moves for {player}", [valid_moves[i].give_move() for i in range(len(valid_moves))])
#         # mcts_ptobs = mcts.search(neuteral_state)
#         # action = np.argmax(mcts_ptobs)
#         # action = index_to_move(action)
#         # print("fromX:",action.from_x, "\nFromY:", action.from_y,"\nToX:", action.to_x,"\nToY:", action.to_y)
        
#     else:
#         neuteral_state = TeeKo_1.change_perspective(state, player)
#         valid_moves = TeeKo_1.get_valid_moves(state, player)
#         print(f"valide_moves for {player}", [valid_moves[i].give_move() for i in range(len(valid_moves))])
#         mcts_ptobs = mcts.search(neuteral_state)
#         action = np.argmax(mcts_ptobs)
#         action = index_to_move(action)
#         print("fromX:",action.from_x, "\nFromY:", action.from_y,"\nToX:", action.to_x,"\nToY:", action.to_y)


#     state = TeeKo_1.get_next_state(state, action, player)

#     value, is_terminal = TeeKo_1.get_value_and_terminated(state, action)

#     if is_terminal:
#         print(state)
#         if value == 1:
#             print(player, "won")
#         else:
#             print("draw")
#         break
#     player = TeeKo_1.get_opponent(player)


# # ------------------------------------------------ #
# # TeeKo_1 = TeeKo()
# # state = TeeKo_1.get_initial_state()
# # state = TeeKo_1.get_next_state(state, Move(-1, -1 , 0,  0), 1)
# # state = TeeKo_1.get_next_state(state, Move(-1, -1 , 1,  3), -1)
# # print(state)

# # encoded_state = TeeKo_1.get_encode_state(state)
# # print(encoded_state)

# # tensor_state = torch.tensor(encoded_state).unsqueeze(0)

# # model = ResNet(TeeKo_1, 4, 128)
# # model.eval()
# # policy, value = model(tensor_state)
# # value = value.item()
# # policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()


# # print("value:",value,"\nPolicy:", policy)



# ============================


TeeKo_1 = TeeKo()
player  = 1


args = {
    'C' : 2,
    'num_searches': 60,
    'num_iteration' : 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size':64
}



model = ResNet(TeeKo_1, 4 , 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

alphaZero = AlphaZero(model, optimizer, TeeKo_1, args)

alphaZero.learn()

