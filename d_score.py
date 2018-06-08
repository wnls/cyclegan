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

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,3))

# x = j_train['D_A']
x = sorted(j_train['D_A'], reverse=True)
ax1.plot(x, '.', label="D_X")
# x = j_train['D_B']
x = sorted(j_train['D_B'], reverse=True)
ax1.plot(x, '.', label="D_Y")
# x = j_train['A_gt']
x = sorted(j_train['A_gt'], reverse=True)
ax1.plot(x, '.', label='Groundtruth')

ax1.set_title('D Scores Train')
ax1.axhline(y=0.5, color='brown')
ax1.legend(fontsize=9)
ax1.set_ylim([-0.7, 1.25])


# x = j_test['D_A']
x = sorted(j_test['D_A'], reverse=True)
ax2.plot(x, '.', label='D_X')
# x = j_test['D_B']
x = sorted(j_test['D_B'], reverse=True)
ax2.plot(x, '.', label="D_Y")

# x = j_test['A_gt']
x = sorted(j_test['A_gt'], reverse=True)
ax2.plot(x, '.', label='Groundtruth')

ax2.set_title('D Scores Test')
ax2.axhline(y=0.5, color='brown')
ax2.legend(fontsize=9)
ax2.set_ylim([-0.7, 1.3])

# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(13,6))
#
# # x = j_train['D_A']
# x = sorted(j_train['D_A'], reverse=True)
# ax[0][0].plot(x, '.')
# ax[0][0].set_title('D_A Scores Train')
# ax[0][0].axhline(y=0.5, color='g')
# # ax[0][0].set_xlabel("Examples")
# # ax[0][0].set_ylabel("")
#
# # x = j_train['D_B']
# x = sorted(j_train['D_B'], reverse=True)
# ax[0][1].plot(x, '.')
# ax[0][1].set_title('D_B Scores Train')
# ax[0][1].axhline(y=0.5, color='g')
#
# # x = j_train['A_gt']
# x = sorted(j_train['D_B'], reverse=True)
# ax[0][2].plot(x, '.')
# ax[0][2].set_title('Groundtruth Scores Train')
# ax[0][2].axhline(y=0.5, color='g')
#
# # x = j_test['D_A']
# x = sorted(j_test['D_A'], reverse=True)
# ax[1][0].plot(x, '.')
# ax[1][0].set_title('D_A Scores Test')
# ax[1][0].axhline(y=0.5, color='g')
#
# # x = j_test['D_B']
# x = sorted(j_test['D_B'], reverse=True)
# ax[1][1].plot(x, '.')
# ax[1][1].set_title('D_B Scores Test')
# ax[1][1].axhline(y=0.5, color='g')
#
# # x = j_test['A_gt']
# x = sorted(j_test['D_B'], reverse=True)
# ax[1][2].plot(x, '.')
# ax[1][2].set_title('Groundtruth Scores Test')
# ax[1][2].axhline(y=0.5, color='g')

fig.tight_layout()
# plt.show()

plt.savefig("checkpoints/d_scores.png")