# import matplotlib.pyplot as plt
# import json
#
# with open("checkpoints/d_scores.json") as f:
#     j = json.load(f)
#     plt.plot(j['D_A'], '.')
#     plt.title('D_A Scores')
#     plt.show()
#
#     plt.plot(j['D_B'], '.')
#     plt.title('D_B Scores')
#     plt.show()
#
#     plt.plot(j['A_gt'], '.')
#     plt.title('A_gt Scores')
#     plt.show()

import matplotlib.pyplot as plt
import json

with open("checkpoints/d_scores_test.json") as f:
    j = json.load(f)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))

    x = j['D_A']
    # x = sorted(j['D_A'], reverse=True)
    ax1.plot(x, '.')
    ax1.set_title('D_A Scores')
    ax1.axhline(y=0.5, color='g')

    x = j['D_B']
    # x = sorted(j['D_B'], reverse=True)
    ax2.plot(x, '.')
    ax2.set_title('D_B Scores')
    ax2.axhline(y=0.5, color='g')

    x = j['A_gt']
    # x = sorted(j['D_B'], reverse=True)
    ax2.plot(x, '.')
    ax2.set_title('gt Scores')
    ax2.axhline(y=0.5, color='g')

    fig.tight_layout()
    # plt.show()

    plt.savefig("checkpoints/d_scores_test.png")