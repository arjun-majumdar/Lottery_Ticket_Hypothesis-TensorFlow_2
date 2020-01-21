# The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks - Using TensorFlow 2.0

A GitHub repository implementing __The Lottery Ticket Hypothesis__ paper by _Jonathan Frankle, Michael Carbin_

"lottery ticket hypothesis:" dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations. The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective. 

The paper can be downloaded from:
[The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)




### Prerequisites for the code to run:
- Python 3.X
- TensorFlow 2.0
- _tensorflow_model_optimization_ package
