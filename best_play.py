import threading

import numpy as np

from dino_env import DinoEnv
from tile_coding import get_tile, sum_weights, create_tilings

feature_ranges = [[5, 15], [0, 10], [0, 2], [0, 2]]
number_tilings = 8
step = [(feature[1] - feature[0]) / number_tilings for feature in feature_ranges]
print(step)
bins = [[20, 60, 10, 10] for i in range(number_tilings)]

# 4 Tilings
# offsets = [
#     [0.0, 0.0, 0.0, 0.0],
#     [2.5, 2.5, 0.5, 0.5],
#     [-2.5, -2.5, -0.5, -0.5],
#     [-2.5, -2.5, 0.5, 0.5]
# ]

# 8 tilings
offsets = [
    [0.0, 0.0, 0.0, 0.0],
    [1.25, 1.25, 0.25, 0.25],
    [-1.25, -1.25, -0.25, -0.25],
    [-1.25, -1.25, 0.25, 0.25],
    [1.25, 1.25, -0.25, -0.25],
    [2.5, 2.5, 0.5, 0.5],
    [-2.5, -2.5, -0.5, -0.5],
    [-2.5, -2.5, 0.5, 0.5],
]

tilings = create_tilings(feature_ranges, number_tilings, bins, offsets)

with open("ws.txt", "rb") as f:
    ws = np.load(f)

def best_play(ws, best_scores, episodes, tilings,number_tilings, thread):
    env = DinoEnv()
    for i in range(episodes):
        state = env.reset()
        current_state = get_tile(state, tilings)
        done = False
        steps, score = 0, 0
        while not done:
            pred_0 = sum_weights(ws, current_state, 0, number_tilings)
            pred_1 = sum_weights(ws, current_state, 1, number_tilings)
            pred_2 = sum_weights(ws, current_state, 2, number_tilings)
            preds = [pred_0, pred_1, pred_2]
            action = preds.index(max(preds))

            obs, reward, done = env.step(action)

            score += reward

            next_state = get_tile(obs, tilings)

            current_state = next_state
            steps += 1
        best_scores.append(env.get_score())
        print(f'Thread: {thread} -> Episódio {i} : {env.get_score()}')

threads = []
num_threads = 4
best_scores = []
for i in range(num_threads):
    threads.append(threading.Thread(target=best_play,
                                    args=(ws, best_scores, 200, tilings, number_tilings, i)))
for i in range(num_threads):
    threads[i].start()

for i in range(num_threads):
    threads[i].join()

with open("best_scores.txt", "wb") as f:
    np.save(f, best_scores)
