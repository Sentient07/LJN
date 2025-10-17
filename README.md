# Deformation Recovery: Localized Learning for Detail-Preserving Deformations

This repository contains the official implementation for the paper:

**[Deformation Recovery: Localized Learning for Detail-Preserving Deformations](https://arxiv.org/abs/2410.08225)**
[Ramana Sundararaman](https://scholar.google.com/citations?user=3G5I0GIAAAAJ&hl=en), [Nicolas Donati](https://scholar.google.fr/citations?user=g2S2g8kAAAAJ&hl=fr), [Simone Melzi](https://scholar.google.com/citations?user=lPfq1eIAAAAJ&hl=en), [Etienne Corman](https://scholar.google.com/citations?user=YJ7oB-cAAAAJ&hl=en), [Maks Ovsjanikov](https://scholar.google.com/citations?user=45k3T2wAAAAJ&hl=en)



## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/geometry-processing/deformation-recovery.git
    cd deformation-recovery
    ```
    *(Note: You might want to replace the URL with your actual repository URL)*

2.  **Install dependencies:**
    It is recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    pip install -r requirements.txt
    ```

## Datasets

This project supports several datasets. To use them, you'll need to configure your data paths.

1.  **Download the datasets:**
    *   **FAUST** ([website](http://faust.is.tue.mpg.de/))
    *   **SURREAL** ([website](https://www.di.ens.fr/willow/research/surreal/data/))
    *   **SCAPE** ([website](http://robotics.cs.stanford.edu/~drago/Projects/scape/scape.html))
    *   **SHREC'19** ([website](https://www.dais.unive.it/shrec19))
    *   **DeformingThings4D** ([website](https://gvv.mpi-inf.mpg.de/deformingthings4d/))

2.  **Configure data paths:**
    Copy the template file `DATA_PATHS.py.template` to `DATA_PATHS.py`:
    ```bash
    cp DATA_PATHS.py.template DATA_PATHS.py
    ```
    Then, edit `DATA_PATHS.py` to point to the locations of the datasets on your local machine.

## Usage

### Training

To train the model, run the `train.py` script. Here's an example:

```bash
python train.py --exp_name faust_experiment --dataset faust_o --lr 1e-3 --n_epoch 120
```

**Arguments:**

*   `--exp_name`: (required) Name for the experiment. Checkpoints will be saved in `./Logs/<exp_name>/`.
*   `--dataset`: The dataset to use for training. Choices: `faust_r`, `faust_o`, `surreal`, `scape_r`, `scape_o`, `shrec19_r`. (default: `faust_o`)
*   `--lr`: Learning rate. (default: `1e-3`)
*   `--n_epoch`: Number of epochs. (default: `120`)
*   `--chkpt`: Path to a checkpoint to resume training from.
*   `--spec_inp`: Use spectral input features.
*   `--pos_loss_weight`: Weight for the position loss. (default: `10.0`)
*   `--jac_loss_2_weight`: Weight for the second Jacobian loss. (default: `0.5`)

### Evaluation

The `--only_eval` flag is available but not yet implemented. Evaluation functionality will be added soon.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{sundararaman2024deformation,
      title={Deformation Recovery: Localized Learning for Detail-Preserving Deformations},
      author={Ramana Sundararaman and Nicolas Donati and Simone Melzi and Etienne Corman and Maks Ovsjanikov},
      year={2024},
      eprint={2410.08225},
      archivePrefix={arXiv},
      primaryClass={cs.GR}
}
```

## License

This project is licensed under the MIT License.
