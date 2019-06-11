import matplotlib.pyplot as plt


## History
history_list = []
for _ in range(16):
    history_list.append([])
with open('./history', 'r') as f:
    data = f.readlines()
    for line in data:
        line = line[1:-2].split(',')
        max_idx = len(line)
        acc = 0
        for idx in range(max_idx)[::-1]:
            acc += float(line[idx])
            history_list[idx].append(acc)
for i in range(16):
    if not history_list[i]:
        break
    plt.plot(history_list[i], label=f'{2**i}')
plt.xlabel('episode(k)')
plt.ylabel('ratio')
plt.legend()
plt.savefig('./history.png')
plt.close()

## Reward
with open('./reward','r') as f:
    reward = f.readline().split(',')[:-1]
    for idx, i in enumerate(reward):
        reward[idx] = int(i)
plt.plot(reward)
plt.xlabel('episode')
plt.ylabel('score')
plt.savefig('./reward.png')
