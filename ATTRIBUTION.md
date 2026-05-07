# Attribution

This project builds upon the following work:

## Hamiltonian Neural Networks (Greydanus et al., 2019)

**Repository:** https://github.com/gregregregor/hamiltonian-nn  
**Paper:** Greydanus, S., Dzamba, M., and Yosinski, J. *Hamiltonian Neural Networks.* arXiv:1906.01563, 2019.  
**License:** MIT

The following files in this repository are derived from or directly adapted from the above:

| This repo | Derived from (upstream) | Changes made |
|-----------|------------------------|--------------|
| `src/models/hnn.py` | `hnn.py` | Extended `HNN` class with Poisson structure via `get_solenoidal()`; replaced symplectic `M` matrix with Poisson bracket `Π(x)` |
| `src/models/mlp.py` | `nn_models.py` | Minor changes to layer depth and initialization |
| `src/utils/utils.py` | `utils.py` | Kept `rk4`, `L2_loss`, `choose_nonlinearity`, `integrate_model` largely unchanged |
| `src/data/data.py` | `data.py` | Replaced harmonic oscillator Hamiltonian with Lotka-Volterra/rigid-body systems; replaced symplectic `dynamics_fn` with Poisson version |

All modifications are the work of the authors of this repository and are
released under the MIT License. The original copyright of the upstream code
belongs to its respective authors.

## Citation

If you use this code, please cite both this work and the upstream HNN paper:

```bibtex
@misc{greydanus2019hamiltonian,
  title     = {Hamiltonian Neural Networks},
  author    = {Greydanus, Samuel and Dzamba, Misko and Yosinski, Jason},
  year      = {2019},
  eprint    = {1906.01563},
  archivePrefix = {arXiv},
}

@inproceedings{araujo2025phnn,
  title     = {Poisson Hamiltonian Neural Networks: Structure-Preserving Learning of Dynamical Systems},
  author    = {Araújo, Adérito and Oliveira, Gonçalo Inocêncio and Mestre, João Nuno},
  year      = {2025},
  institution = {CMUC, University of Coimbra}
}
```
