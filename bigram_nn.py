"""
neural network implementation of bigram, output a probability predictions of each character following input character
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


words = open('names.txt','r').read().splitlines()

"""
create training set of all the bigrams (x, y)
"""

xs, ys = [], []
chars = sorted(list(set(''.join(words))))
stoi = {s:(i+1) for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
W = torch.randn((27, 27), requires_grad=True)       # 27 inputs into a layer of 27 neurons


"""
gradient descent
"""
iterations = 100
for k in range(iterations):

    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()    # input to the network
    logits = xenc @ W                               # log(counts) done with matrix multiplication
    counts = logits.exp()                           # log(counts) exponentiated -> equivalent to the N matrix (counts for the next character)
    probs = counts / counts.sum(1, keepdims=True)   # probabilities for next character
    # log(counts) exponentiated is 'softmax' takes outputs and produces probabilities (positive values that sum to 1)

    # loss function
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    print(f'loss: {loss.item()}')

    # backward pass
    W.grad = None
    loss.backward()

    # update weights
    W.data += -50 * W.grad


"""
sampling from model (testing from model, getting names outputted from trained model)
getting 5 new names from the model
"""

for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

