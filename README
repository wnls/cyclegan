Clone https://github.com/facebookresearch/visdom.git and run
```
pip install -e .
```
in that directory first before running.

## How to Enable Visualization

1. run `python -m visdom.server [-port xxxx]`

2. run `python main.py --vis [--port xxxx]`

## Train model from scratch

```bash
python main.py --mode train --n_epoch 200
```

## Use pretrained model
```bash
python main.py --mode [train|test] --pretrain_path ./checkpoints/xxx/xxx.pt
```

## Plot stats from train.json
```bash
python plot.py --dir ./checkpoints/xxx
```
It will look for train.json in the directory and output plots as result.png.
