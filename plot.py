import numpy as np
import matplotlib.pyplot as plt

with open("scores.txt", "rb") as f:
    scores = np.load(f)
with open("score_per_100.txt", "rb") as f:
    scores_per_100 = np.load(f)

with open("best_scores.txt", "rb") as f:
    bests = np.load(f)

figure, axis = plt.subplots(2)

axis[0].plot(range(0, len(scores)), scores)

axis[1].plot(range(0, len(scores_per_100)), scores_per_100)

# axis[2].plot(range(0, len(bests)), bests)

plt.show()