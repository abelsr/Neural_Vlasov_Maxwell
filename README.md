# Neural Operator for Vlasov-Maxwell Equations

This repository contains an implementation of the neural operator for the Vlasov-Maxwell equations. 

## Installation

First, clone the repository and navigate to the directory:

```bash
git clone https://github.com/abelsr/Neural_Vlasov_Maxwell.git
cd Neural_Vlasov_Maxwell
```

Then, create a new conda environment and install the required packages:

```bash
conda env create -f environment.yml
conda activate neural_vlasov_maxwell
```

or, if you prefer to use pip:

```bash
pip install -r requirements.txt
```

## AFNO

This repository contains code based on the paper [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895) and [FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators](https://arxiv.org/pdf/2202.11214.pdf).
