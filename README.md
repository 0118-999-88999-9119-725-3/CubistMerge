# CubistMerge (CuMe)

This repository is forked from the [ToMe (Token Merging) repository](https://github.com/facebookresearch/ToMe) and extends it with our CubistMerge method along with additional testing capabilities for DeiT models.

### CuMe

To run with CubistMerge:
```bash
python3 test_deit.py -m cume -rh 1 -rw 1 --data_path /path/to/imagenet
```

- **Parameters**:
  - `rh`: num of tokens to reduce each row
  - `rw`: num of tokens to reduce each column
  - given H × W input tokens, this will reduce to (H - rₕ) × (W - rw)

### Baseline (Original Model)

To run the original baseline model without any token merging:
```bash
python3 test_deit.py --data_path /path/to/imagenet
```

### ToMe (Token Merging)

To run with ToMe token merging:

```bash
python3 test_deit.py -m tome -r 6 --data_path /path/to/imagenet
```

**Note**: For ToMe, we have kept their recommended merging schedule, which differs from our CubistMerge approach. Therefore, the `r` parameter in ToMe has a different meaning compared to our `rh` and `rw` parameters.

- **Parameters**:
    - `r`: num of tokens to be reduced at every layer.

## Comparison Example

To compare ToMe and CubistMerge at similar runtime, you can use:

```bash
python3 test_deit.py -m cume -rh 1 -rw 1 --data_path /path/to/imagenet
python3 test_deit.py -m tome -r 6 --data_path /path/to/imagenet
```

Or at a higher token reduction rate
```bash
python3 test_deit.py -m cume -rh 2 -rw 2 --data_path /path/to/imagenet
python3 test_deit.py -m tome -r 9 --data_path /path/to/imagenet
```