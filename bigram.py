"""
simple implementation of bigram, output probability predictions of names created by using probabilities of a character following another
"""

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
P = (N+1).float()
P /= P.sum(1, keepdim = True)

for i in range(5):
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

"""
loss function
finding how good bigram model is at creating valid names based on bigram probabilities, want to maximize likelihood
the model parameters are the frequencies giving by plot S
"""

# GOAL: maxmize likelihood of the data model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log loss: log(a * b * c) = log(a) + log(b) + log(c)

# EXAMPLE (according to this trained model on this data set):
# 'andrej': negLogProb=3.039  unlikely name
# 'angela': negLogProb=2.130  semi likely name
# 'emily':  negLogProb=2.3866 semi likely name

neg_log_likelihood = 0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        neg_log_likelihood += (-logprob)

        n += 1
        print(f'{ch1}{ch2} prob: {prob:.4f}   negative logprob: {neg_log_likelihood:.4f}')

# print(f'{log_likelihood=}')

# print(f'{nll=}')
print(f'average negative log likelihood: {(neg_log_likelihood/n):.4f}')