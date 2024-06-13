
### Installation

## Pytorch Geometric

For this code to work you will need to install PyG via [Anaconda](https://anaconda.org/pyg/pyg) for all major OS/PyTorch/CUDA combinations 🤗
If you have not yet installed PyTorch, install it via `conda` as described in the [official PyTorch documentation](https://pytorch.org/get-started/locally/).
Given that you have PyTorch installed (`>=1.8.0`), simply run

```
conda install pyg -c pyg
```

## Additional dependencies

```
pip install requests fake_useragent optuna
```

## Running experiments

```
python ./graphgym/main_optuna.py --cfg configs/pyg/ecoli_static.yaml --study grn-static-convergence-large-set

python ./graphgym/main_optuna.py --cfg configs/pyg/ecoli_temporal.yaml --study grn-temporal-convergence-large-set
```

### Notebooks

- [Data Training Tutorial Notebook](./graphgym/datatraining_tutorial.ipynb)

Check the ./graphgym/datatraining_tutorial.ipynb for an example of loading the datasets and training models.g