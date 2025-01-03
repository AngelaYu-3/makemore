# makemore

makemore takes one text file as the input, where each line is assumed to be one piece of training data, and generates more text that's like the given text file (ex: if the given text file is a list of names, makemore generates more names). It is an autoregressive character-level language model, with a wide choice of models from bigrams to transformers (like as seen in GPT).

This project is for educational purposes and follows the course [Neural Networks: Zero to Hero by Andrej Karpathy](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2).

***Source code is commented to demonstrate independent understanding of core concepts as well as for ease of future read-throughs. Additional coding files used during independent learning process are also included. Annotated research papers that implemented models are based on are also included.***
___

## Key Papers Current Implementation Follows
- bigram (one character predicts the next character with a lookup table of counts from name.txt data)
- bigram_nn (one character predicts the next character with neural network trained on name.txt data)
- [A Neural Probablistic Language Model](https://github.com/AngelaYu-3/makemore/blob/main/annotated_papers/MLP_paper.pdf)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://github.com/AngelaYu-3/makemore/blob/main/annotated_papers/batchNorm_paper.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

___

## Included Dataset

The included example **names.txt** dataset has the most common 32K names taken from [ssa.gov](https://www.ssa.gov/) for the year 2018.

___

## Questions

**Q: For bigram.py, why do we want to use nll loss function? Why does using log loss function make working with probabilities in MLPs better?**

A: log loss function is monotonic (when a function is always increasing or always decreasing) **(why does log being monotonic important for finding loss of probabilities in this mlp model?)**

**Q: What is the point of regularization?**

A: Encourages model to be more generalized instead of memorizing the data. Reduces overfitting of the model (when the model learns training data too closely, including noise and irrelevant details, leading to poor performance on new training data). 

___

## License
MIT
