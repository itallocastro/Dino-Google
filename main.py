import numpy as np
from dino_env import DinoEnv
from tile_coding import create_tilings, get_tile, create_weight, sum_weights
import threading

feature_ranges = [[5, 15], [0, 10], [0, 2], [0, 2], [0, 150]]
number_tilings = 8
step = [(feature[1] - feature[0]) / number_tilings for feature in feature_ranges]
print(step)
# Velocity; Distance; Height; Width; Coord Y
bins = [[30, 50, 10, 10, 10] for i in range(number_tilings)]

# 8 tilings
offsets = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.25, 1.25, 0.25, 0.25, 0.0],
    [-1.25, -1.25, -0.25, -0.25, 0.0],
    [-1.25, -1.25, 0.25, 0.25, 0.0],
    [1.25, 1.25, -0.25, -0.25, 0.0],
    [2.5, 2.5, 0.5, 0.5, 0.0],
    [-2.5, -2.5, -0.5, -0.5, 0.0],
    [-2.5, -2.5, 0.5, 0.5, 0.0],
]

tilings = create_tilings(feature_ranges, number_tilings, bins, offsets)
# print(tilings.shape)
# print(
#     get_tile([6.5, 4.2, 1.3, 0.9], tilings)
# )

# ws = np.array(create_weight(number_tilings, bins, 3))
# scores = []
# scores_per_100 = []
with open("ws.txt", "rb") as f:
    ws = np.load(f)
with open("scores.txt", "rb") as f:
    scores = list(np.load(f))
    print(scores)
with open("score_per_100.txt", "rb") as f:
    scores_per_100 = list(np.load(f))

def Q_Learning(ws, tilings, number_tilings, scores=[], scores_per_100=[], episodes=20000, thread=1, alpha=0.1,
               gamma=0.99, epsilon=0.001):
    env = DinoEnv()
    print(episodes)
    # alpha = alpha / number_tilings
    for i in range(episodes):
        state = env.reset()
        # current_state = get_tile([state[0], state[2]], tilings)
        current_state = get_tile(state, tilings)
        done = False
        steps, score = 0, 0
        while not done:
            if (i % 100 == 0):
                pred_0 = sum_weights(ws, current_state, 0, number_tilings)
                pred_1 = sum_weights(ws, current_state, 1, number_tilings)
                pred_2 = sum_weights(ws, current_state, 2, number_tilings)
                preds = [pred_0, pred_1, pred_2]
                action = preds.index(max(preds))
            else:
                if (np.random.uniform(0, 1) < epsilon):
                    action = env.action_space.sample()
                else:
                    pred_0 = sum_weights(ws, current_state, 0, number_tilings)
                    pred_1 = sum_weights(ws, current_state, 1, number_tilings)
                    pred_2 = sum_weights(ws, current_state, 2, number_tilings)
                    preds = [pred_0, pred_1, pred_2]
                    action = preds.index(max(preds))

            obs, reward, done = env.step(action)

            score += reward

            old_value = sum_weights(ws, current_state, action, number_tilings)

            # next_state = get_tile([obs[0], obs[2]], tilings)
            next_state = get_tile(obs, tilings)
            sum_next_0 = sum_weights(ws, next_state, 0, number_tilings)
            sum_next_1 = sum_weights(ws, next_state, 1, number_tilings)
            sum_next_2 = sum_weights(ws, next_state, 2, number_tilings)
            sums_next = [sum_next_0, sum_next_1, sum_next_2]
            max_next = max(sums_next)

            new_value = old_value + alpha * (reward + (gamma * max_next) - old_value)

            # print(f'i = {i} max value of next is {max_next}')
            k = 0
            coef = 0
            for j in range(number_tilings):
                coef += 1 / (j + 1)
            k = new_value / coef
            for j in range(number_tilings):
                ws[j][tuple(np.concatenate((current_state[j], [action])))] = k * (1 / (j + 1))

            current_state = next_state
            steps += 1
        if (i % 100 == 0):
            print(score)
            scores_per_100.append(env.get_score())
        if score > 1000:
            print(f'Score {score} com {i} episódios')
        print(f'Thread: {thread} -> Episódio {i} : {env.get_score()}')
        scores.append(env.get_score())

    return ws, scores, scores_per_100



threads = []
num_threads = 8
for i in range(num_threads):
    threads.append(threading.Thread(target=Q_Learning,
                                    args=(ws, tilings, number_tilings, scores, scores_per_100, int(1000/num_threads), i)))
for i in range(num_threads):
    threads[i].start()

for i in range(num_threads):
    threads[i].join()

with open("ws.txt", "wb") as f:
    np.save(f, ws)
with open("scores.txt", "wb") as f:
    np.save(f, scores)
with open("score_per_100.txt", "wb") as f:
    np.save(f, scores_per_100)

with open("ws.txt", "rb") as f:
    a = np.load(f)
    # print(a)
with open("scores.txt", "rb") as f:
    print(np.load(f))
with open("score_per_100.txt", "rb") as f:
    print(np.load(f))
