## WF-Bench
This is the source code for "WF-Bench: A Benchmark for Neural-Network WaveFunction Expressivity and Scaling Laws". The pfaffian.py are adopted from arXiv:2602.05255, with github link https://github.com/ChenAo-Phys/lrux.

## Using the Code

The codebase is designed to support, without loss of generality, any Flax-based neural-network wavefunction.

Both the **network** and **target** wavefunctions are expected to take an unbatched input of the form

```python
(position, spin)
```

and return

```python
(sign, logabspsi)
```

where `sign` represents the complex phase or sign structure of the wavefunction, and `logabspsi` is the logarithm of the absolute value of the wavefunction amplitude.

The loss functions provided in this codebase are batched. They expect inputs in the form of a nested dictionary, where each key is the name of a sampler and each value contains the corresponding batched `(position, spin)` data.

For example, the mixed-probability loss expects samples from both the `"network"` sampler and the `"target"` sampler:

```python
{
    "network": (position_network, spin_network),
    "target": (position_target, spin_target),
}
```

Here, `position_network` and `spin_network` are batched samples drawn from the network distribution, while `position_target` and `spin_target` are batched samples drawn from the target distribution.

In general, a loss function may use one or more samplers, depending on the estimator being optimized. The required sampler names and input structure should match the corresponding loss implementation.

## Citing WF-Bench
Please cite our work as the following
```bibtex
@misc{aaa,
      title={WF-Bench: A Benchmark for Neural-Network WaveFunction \\ Expressivity and Scaling Laws},
      author={Lixing Zhang and Guijing Duan and Di Luo},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={},
      url={},
}
```

## Contact information
Lixing Zhang : zlx@ucla.edu / zhanglixing1108@gmail.com


