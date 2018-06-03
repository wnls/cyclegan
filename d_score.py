import matplotlib.pyplot as plt
import json

with open("checkpoints/d_scores.json") as f:
    j = json.load(f)
    plt.plot(j['D_A'], '.')
    plt.title('D_A Scores')
    plt.show()

    plt.plot(j['D_B'], '.')
    plt.title('D_B Scores')
    plt.show()

    plt.plot(j['A_gt'], '.')
    plt.title('A_gt Scores')
    plt.show()