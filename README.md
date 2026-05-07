# Poisson Hamiltonian Neural Networks (PHNNs)

> **Structure-Preserving Learning of Dynamical Systems**  
> Adérito Araújo, Gonçalo Inocêncio Oliveira, João Nuno Mestre  
> CMUC, Department of Mathematics, University of Coimbra

---

## Overview

This repository contains the implementation of **Poisson Hamiltonian Neural Networks (PHNNs)**, an extension of Hamiltonian Neural Networks (HNNs) designed to learn the dynamics of Poisson-Hamiltonian systems while preserving their underlying geometric structure.

Rather than approximating the vector field directly, PHNNs learn an approximation $H_\theta(x)$ of the Hamiltonian function, ensuring that learned dynamics respect conservation laws inherent to the system. The key contribution is the integration of **Poisson-Hamiltonian Integrators (PHI)** into both the training and testing pipelines, enabling better long-term stability and structure preservation compared to classical explicit methods.

---

## Background

A **Poisson-Hamiltonian system** is defined by:

$$\dot{x} = \Pi(x) \nabla H(x)$$

where $\Pi(x)$ is a skew-symmetric matrix encoding the Poisson structure and $H$ is the Hamiltonian (energy) of the system. This framework generalizes classical symplectic Hamiltonian mechanics — symplectic systems are a special case — and can be defined in any dimension, including odd dimensions.

PHNNs extend HNNs (Greydanus et al., 2019) to this broader class of systems by incorporating the Poisson bracket structure and, optionally, structure-preserving numerical integrators during training.

---

## Benchmark Systems

The method is validated on four dynamical systems:

| System | Poisson Structure | Dimension |
|--------|------------------|-----------|
| Ideal Pendulum | Symplectic (canonical) | 2D |
| Lotka-Volterra | Quadratic | 2D |
| Lotka-Volterra | Quadratic | 3D |
| Rigid Body Rotation | Linear (so(3)) | 3D |

---

## Method

### Training Strategies

Two first-order integrators are compared during training and testing, leading to four experimental combinations:

| Combination | Training | Testing |
|-------------|----------|---------|
| EE-EE | Explicit Euler | Explicit Euler |
| EE-PHI | Explicit Euler | Poisson-Hamiltonian Integrator |
| PHI-EE | Poisson-Hamiltonian Integrator | Explicit Euler |
| PHI-PHI | Poisson-Hamiltonian Integrator | Poisson-Hamiltonian Integrator |

### PHI Method (Poisson-Hamiltonian Integrator)

The PHI method, based on the theory of symplectic groupoids and bi-realizations, proceeds as:

1. Choose the Lagrangian bisection $L_{\Delta t}$ for the given time step;
2. For current state $x_k$, find $y_k \in L_{\Delta t}$ such that $\alpha(y_k) = x_k$;
3. Compute the next state as $x_{k+1} = \beta(y_k)$.

This guarantees preservation of the Poisson structure and controlled conservation of the Hamiltonian.

### Loss Functions

**When using direct vector field data:**

$$\mathcal{L}(x, \dot{x}) = \frac{1}{N} \sum_{k=1}^{N} \left\| \dot{x}_k - \Pi(x_k)\nabla H_\theta(x_k) \right\|^2$$

**When using an integrator (single-step prediction):**

$$\mathcal{L}(x, x^*) = \frac{1}{N} \sum_{k=1}^{N} \left\| x_k - x^*_k \right\|^2$$

---

## Neural Network Architecture

- **Layers:** 4 (2 hidden layers)
- **Neurons per hidden layer:** 200 (vector field training) / 256 (integrator training)
- **Activation function:** $\tanh$
- **Optimizer:** Adam, learning rate $10^{-3}$
- **Epochs:** 2000 (3000 for Rigid Body)
- **Training noise:** Gaussian, $\sigma^2 = 0.1$

---

## Key Results

- **EE training** yields better short-term accuracy due to explicit gradient updates.
- **PHI testing** ensures long-term stability and geometric structure preservation.
- **EE-PHI (hybrid)** achieves the best balance between accuracy and stability across all tested systems.
- **PHI-PHI** achieves the best energy conservation (Hamiltonian preservation) over long trajectories.
- The rigid-body rotation case requires special treatment via the $so(3)$ isomorphism; the PHI integrator encounters convergence issues in its implicit step for this system.

---

## Repository Structure

```
phnn/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── models/
│   │   └── phnn.py              # PHNN model definition
│   ├── integrators/
│   │   ├── euler.py             # Explicit Euler integrator
│   │   └── phi.py               # Poisson-Hamiltonian Integrator
│   ├── systems/
│   │   ├── pendulum.py          # Ideal pendulum
│   │   ├── lotka_volterra_2d.py # Lotka-Volterra 2D
│   │   ├── lotka_volterra_3d.py # Lotka-Volterra 3D
│   │   └── rigid_body.py        # Rigid body rotation
│   ├── training/
│   │   └── train.py             # Training loop
│   └── utils/
│       ├── data.py              # Trajectory generation, noise
│       └── plotting.py          # Visualization utilities
│
├── experiments/
│   ├── run_vector_field.py      # Section 4.1 experiments
│   └── run_integrators.py       # Section 4.2 experiments (EE-EE, EE-PHI, PHI-EE, PHI-PHI)
│
├── notebooks/
│   └── results_visualization.ipynb
│
└── paper/
    └── phnn_paper.pdf
```

---

## Installation

```bash
git clone https://github.com/<your-username>/phnn.git
cd phnn
pip install -r requirements.txt
```

**Requirements** (suggested):
```
torch
numpy
scipy
matplotlib
```

---

## Usage

**Train on Lotka-Volterra 2D with EE-PHI strategy:**
```bash
python experiments/run_integrators.py --system lv2d --train euler --test phi
```

**Reproduce vector field training results (Section 4.1):**
```bash
python experiments/run_vector_field.py --system pendulum
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{araujo2025phnn,
  title     = {Poisson Hamiltonian Neural Networks: Structure-Preserving Learning of Dynamical Systems},
  author    = {Araújo, Adérito and Oliveira, Gonçalo Inocêncio and Mestre, João Nuno},
  year      = {2025},
  institution = {CMUC, University of Coimbra}
}
```

---

## References

- Greydanus, S., Dzamba, M., and Yosinski, J. *Hamiltonian Neural Networks.* arXiv:1906.01563, 2019.
- Cosserat, O., Laurent-Gengoux, C., and Salnikov, V. *Numerical Methods in Poisson Geometry and their Application to Mechanics.* Mathematics and Mechanics of Solids, 2024.
- Chen, Z., Zhang, J., Arjovsky, M., and Bottou, L. *Symplectic Recurrent Neural Networks.* ICLR, 2020.
- Eldred, C. et al. *Lie-Poisson Neural Networks (LPNets).* Neural Networks, 2024.

---

## Acknowledgements

This work was financially supported by the **Fundação para a Ciência e a Tecnologia (FCT)** under project UID/00324 — Center for Mathematics of the University of Coimbra. Gonçalo Oliveira acknowledges FCT support under Ph.D. Scholarship UIDP/00324/2020.
