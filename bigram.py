import torch
import matplotlib.pyplot as plt


words = open('names.txt','r').read().splitlines()

"""
using a dictionary to keep track of frequency of bigram (what character likely follows another character)
"""
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        # print(ch1, ch2)

sorted(b.items(), key = lambda kv: -kv[1])


"""
using pytorch instead to keep track of frequency of bigram (what character likely follows another character)
"""
N = torch.zeros((27, 27), dtype = torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:(i+1) for i,s in enumerate(chars)}
stoi['.'] = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


# visualizing data above in a nicer way using matplotlib
itos = {i:s for s, i in stoi.items()}

plt.figure(figsize = (16, 16))
plt.imshow(N, cmap = 'Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha = "center", va = "bottom", color = 'gray')
        plt.text(j, i, N[i, j].item(), ha = "center", va = "top", color = 'gray')
plt.axis('off')
# plt.show()


"""
sampling from the model
"""
# p = N[0].float()
# p = p / p.sum()
# g = torch.Generator().manual_seed(2147483647)
# p = torch.rand(3, generator = g)
# p = p / p.sum()
# ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g)

g = torch.Generator().manual_seed(2147483647)
P = N.float()
P = P / P.sum(1, keepdim = True)

for i in range(20):
    out = []
    ix = 0
    while True:
        # p = N[ix].float()
        # p = p / p.sum()
        # p = torch.ones(27) / 27.0
        p = P[ix]
        ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))


