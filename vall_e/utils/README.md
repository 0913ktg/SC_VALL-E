# PyTorch Training Utilities (WIP)

This is a collection of PyTorch training utilities that I am currently using. Please note that the interfaces may change frequently. If you have checked out the correct commit, you should not encounter any issues.

To run the training utility, use one of the following commands:

```
deepspeed <your-entry>.py
```

or

```
python <your-entry>.py
```

## Warning

- Be cautious when using the deepspeed branch to train GANs. The gradients of different engines must be properly managed, as the zero_grad function is not called before feedforward to support gradient accumulation.
