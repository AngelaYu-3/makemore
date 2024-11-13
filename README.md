# makemore

makemore takes one text file as the input, where each line is assumed to be one piece of training data, and generates more text that's like the given text file (ex: if the given text file is a list of names, makemore generates more names). It is an autoregressive character-level language model, with a wide choice of models from bigrams to transformers (like as seen in GPT).

This project is for educational purposes and follows the course [Neural Networks: Zero to Hero by Andrej Karpathy](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2).

***Source code is commented to demonstrate deep, independent understanding of core concepts as well as for ease of future read-throughs. Additional files are included as part of learning process.***

___

## Included Dataset

The included names.txt dataset, as an example, has the most common 32K names taken from [ssa.gov](https://www.ssa.gov/) for the year 2018.

___

## License
MIT
