import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""
read in all words from name.txt
"""
words = open('other/names.txt', 'r').read().splitlines()


"""
build the vocabulary of characters and mappings to and from integers
"""
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)


"""
build the dataset
training: data used for training model, training parameters (weights and biases)
validation / dev: data used for tuning hyperparameters
testing: data used for testing model
"""
block_size = 3        # context length: how many characters do we take to predict the next one?
def build_dataset(words):
    X, Y = [], []

    # note: encoding letters within dataset with numbers corresponding to letters as defined in stoi
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]    # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])       # training data is 80% of total data
Xdev, Ydev = build_dataset(words[n1:n2])   # dev data is 10% of total data
Xte, Yte = build_dataset(words[n2:])       # testing data is 10% of total data



n_dimensions = 27
n_hidden_layers = 200
C = torch.randn((27, n_dimensions))
W1 = torch.randn((block_size * n_dimensions, n_hidden_layers )) * 0.2
b1 = torch.randn(n_hidden_layers ) * 0.01
W2 = torch.randn((n_hidden_layers , 27)) * 0.01
b2 = torch.randn(27) * 0
bngain = torch.ones((1, n_hidden_layers))
bnbias = torch.zeros((1, n_hidden_layers))
parameters = [C, W1, b1, W2, b2, bngain, bnbias]

for p in parameters:
    p.requires_grad = True

learning_rate_exp = torch.linspace(-3, 0, 1000)
learning_rate_steps = 10 ** learning_rate_exp

steps = 200000
batch_size = 32
learning_rate_i = []
lossi = []
stepi = []

for i in range(steps):
    """
    minibatch construct
    """
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    """
    forward pass
    """
    emb = C[Xtr[ix] ]                                                                                       # embedding all integers within X with lookup table C (2 dimensions) into vectors
    embcat = emb.view(emb.shape[0], -1)                                                                     # concatenate the vectors from above
    hpreact = embcat @ W1 + b1                                                                              # hidden layer pre-activation
    hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias    # batch normalization
    h = torch.tanh(hpreact)                                       # hidden layer
    logits = h @ W2 + b2                                                                                    # output layer
    loss = F.cross_entropy(logits, Ytr[ix])                                                                 # implementing loss function (nll)
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims = True)
    # loss = -prob[torch.arange(32), Y].log().mean()

    """
    backward pass
    """
    for p in parameters:
        p.grad = None
    loss.backward()

    """
    update
    """
    # learning_rate = learning_rate_steps[i]
    learning_rate = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -learning_rate * p.grad

    """
    track stats to find optimal learning rate
    """
    if i % 10000 == 0:
        print(f'{i:7d}/{steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    # stepi.append(i)
    # lossi.append(loss.log10().item())

# plt.plot(stepi, lossi)
# plt.show()

"""
plotting graph to visualize character embedding in 2 dimensions
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
plt.grid('minor')
plt.show()
"""



"""
finding final loss with trained model by comparing predicted outputs with actual outputs
"""
@torch.no_grad()
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x] 
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')


"""
sampling from the model
"""
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        # forward pass the neural net
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
            break

    #  decode and print the generated word
    print(''.join(itos[i] for i in out))    



# torch.cat(torch.unbind(emb, 1))    # embeddings of the input first three characters (block_size)

# a = torch.arange(18)
# a.view(2, 9)










