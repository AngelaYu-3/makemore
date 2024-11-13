# makemore

makemore takes one text file as the input, where each line is assumed to be one piece of training data, and generates more text that's like the given text file (ex: if the given text file is a list of names, makemore generates more names). It is an autoregressive character-level language model, with a wide choice of models from bigrams to transformers (like as seen in GPT).

This project is for educational purposes and follows the course [Neural Networks: Zero to Hero by Andrej Karpathy](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2).

***Source code is commented to demonstrate deep, independent understanding of core concepts as well as for ease of future read-throughs. Additional coding files used during independent learning process also included. Annotated research papers models are implemented on are also included.***
___

## Key Papers Current Implementation Follows
- bigram (one character predicts the next character with a lookup table of counts from name.txt)
- bigram_nn (one character predicts the next character with neural network trained on name.txt data)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

___

## Included Dataset

The included example **names.txt** dataset has the most common 32K names taken from [ssa.gov](https://www.ssa.gov/) for the year 2018.

___

## License
MIT
