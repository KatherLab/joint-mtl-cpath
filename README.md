# joint-mtl-cpath
Joint multi-task learning improves weakly-supervised biomarker prediction in computational pathology

## Data prerequisites
Preprocess your WSI data to obtain feature matrices, slide table and clinical table according to the [STAMP protocol](https://github.com/KatherLab/STAMP). The paper focuses on MSI and HRD combined with signatures from the TME in colorectal cancer and lung adenocarcinoma, respectively. However, this framework is applicable for essentially any combination of categorical and continuous weak labels.

## Install the environment
```bash
conda create -n joint-mtl-cpath python=3.10 -y
conda activate joint-mtl-cpath
pip install -r requirements.txt
```

## Training the model
1. Edit TrainConfig in `config.py` with the desired targets, data paths and modeling configurations;
2. Run `python trainer_mtl.py`.

## Externally evaluate the model
1. Edit TestConfig in `config.py` with the respective targets, data paths and modeling configurations;
2. Run `python test.py`.

## Multi-task balancing methods:
The following balancing methods can be selected in the config file:

### Weighting-based:
- **Naive** - `weight = "equal"`
- **[Uncertainty](https://arxiv.org/abs/1705.07115)** - `weight = "uncert"`
- **[Dynamic Weight Average](https://arxiv.org/abs/1803.10704)** - `weight = "dwa"`
- **[Auto-Lambda](https://arxiv.org/abs/2202.03091)** - `weight = "autol"`

### Gradient-based:
- **Naive** - `grad_method = "none"`
- **[GradDrop](https://arxiv.org/abs/2010.06808)** - `grad_method = "graddrop"`
- **[PCGrad](https://arxiv.org/abs/2001.06782)** - `grad_method = "pcgrad"`
- **[CAGrad](https://arxiv.org/abs/2110.14048)** - `grad_method = "cagrad"`
