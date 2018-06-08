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

with open("checkpoints/d_scores_train.json") as f:
    j_train = json.load(f)
with open("checkpoints/d_scores_test.json") as f:
    j_test = json.load(f)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,5))

x = j_train['D_A']
# x = sorted(j['D_A'], reverse=True)
ax[0].plot(x, '.')
ax[0].set_title('D_A Scores Train')
ax[0].axhline(y=0.5, color='g')

x = j_train['D_B']
# x = sorted(j['D_B'], reverse=True)
ax[1].plot(x, '.')
ax[1].set_title('D_B Scores Train')
ax[1].axhline(y=0.5, color='g')

x = j_train['A_gt']
# x = sorted(j['D_B'], reverse=True)
ax[2].plot(x, '.')
ax[2].set_title('Groundtruth Scores Train')
ax[2].axhline(y=0.5, color='g')

x = j_test['D_A']
# x = sorted(j['D_A'], reverse=True)
ax[0].plot(x, '.')
ax[0].set_title('D_A Scores Test')
ax[0].axhline(y=0.5, color='g')

x = j_test['D_B']
# x = sorted(j['D_B'], reverse=True)
ax[1].plot(x, '.')
ax[1].set_title('D_B Scores Test')
ax[1].axhline(y=0.5, color='g')

x = j_test['A_gt']
# x = sorted(j['D_B'], reverse=True)
ax[2].plot(x, '.')
ax[2].set_title('Groundtruth Scores Test')
ax[2].axhline(y=0.5, color='g')

fig.tight_layout()
# plt.show()

plt.savefig("checkpoints/d_scores.png")