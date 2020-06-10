# Feed-Forward Neural Network (FFNN) for Regression Problems

## Reference

- Mathematical background: ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/index.html).

- Datasets: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).

## Characteristics

- The code has been written and tested in Python 3.7.7.
- Usage: *python test.py example*.

## Parameters

`example` Name of the example to run (wine, stock, wifi, pulsar.)

`problem` Defines the type of problem. Equal to C specifies logistic regression, anything else specifies linear regression. The default value is `None`.

## Examples

There are four examples in *test.py*: wine, stock, wifi, pulsar. Since GDO is used, `use_grad` is set to `True`. For all examples `init_weights` is also set to `True`.

### Single-label linear regression examples: wine

```python
data_file = 'wine_dataset.csv'
n_features = 11
hidden_layers = [20]
split_factor = 0.7
L2 = 0.0
epochs = 50000
alpha = 0.2
d_alpha = 1.0
tolX = 1.e-7
tolF = 1.e-7
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/Wine+Quality>.

The dataset has 11 features, 1 label, 4898 samples, 261 variables, and a layout of [11, 20, 1].

Correlation predicted/actual values: 0.708 (training), 0.601 (test).

Exit on `epochs` with `tolX` = 2.0e-4 and `tolF` = 1.1e-7.
