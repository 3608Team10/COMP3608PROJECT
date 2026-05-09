# Fake News Detection

## How to Run

### Colab Notebooks

Bagging

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3608Team10/COMP3608PROJECT/blob/main/Bagging.ipynb)

Boosting

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3608Team10/COMP3608PROJECT/blob/main/Boosting.ipynb)

Stacking

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3608Team10/COMP3608PROJECT/blob/main/Stacking.ipynb)

### Colab Runtime

The notebooks leverage NVIDIA RAPIDS for GPU-acceleration. Some dependencies require a GPU runtime to be selected. \
Select the best GPU runtime available to you.

<img src="images/colab-runtime.png" alt="Colab Runtime Type" height="350" width="350" />

### Colab Files

The notebooks are configured to download the ingest_data script from github via the following command

```py
!wget https://raw.githubusercontent.com/3608Team10/COMP3608PROJECT/refs/heads/main/ingest_data.py
```

Your colab session files should look like this:

<img src="images/colab-files.png" alt="Colab Files Script Upload" height="350" width="375" />

## Optimization Problem

MIN

$$
\min_{\theta_{1}, \dots, \theta_{M}} \quad Z^{(e)} = - \frac{1}{N} \sum_{i=1}^{N} \alpha_{c_i} [w_1 y_i \log(F^{(e)}(x)) + w_0 (1 - y_i) \log(1 - F^{(e)}(x))] + \sum_{m=1}^{M} \lambda_{m} \Omega(\theta_{m})
$$

Where

- $e \in \\\{bag, boost, stack\\\}$
- $F(x) \in \\\{0, 1\\\}$

SUBJECT TO

$C_{1}^{bag}$: Ensemble Prediction Function

$$F(x) = \frac{1}{M} \displaystyle\sum_{m=1}^{M} f_m(x)$$

Where

- $f_m$ is trained on a bootstrap sample $B_m \subset \mathcal{D}$ with replacement

$C_{1}^{boost}$: Ensemble Prediction Function

$$F(x) = \displaystyle\sum_{m=1}^{M} \eta \beta_{m} f_m(x), \qquad \beta_{m} > 0$$

Where

- $\eta$ is the learning rate shrinkage factor

$C_{1}^{stack}$: Ensemble Prediction Function

$$C_{1}^{train}: \hat{p}_m^{(i)} = f_{m}^{(-k(i))}(x_i) \qquad \text{(OOF meta-features)}$$

$$C_{1}^{test}: F(x) = g(f_1(x), \ldots, f_M(x))\qquad \text{(full retrain inference)}$$

Where

- train: base model $f_m$ predicts on fold $k(i)$ using version trained on all other folds
- test: base models $f_1, \dots, f_M$ are fully retrained on complete training set $\mathcal{D}$

$C_2$: Feature Mapping

$$x_i = \phi (title_i, text_i, category_i, dataset_i)$$

$C_3$: Label Constraint

$$y_i \in \\\{0, 1\\\}, \qquad 0 = fake, \quad 1 = real$$

$C_4$: Category Weight

$$\alpha_{c_i} = \frac{N}{K \cdot N_{c_i}}$$

Where

- $K$ is the number of distinct categories
- $N_{c_i}$ is the number of samples in category $c_i$
- $N$ is the total number of samples

$C_5$: Class Weight

$$w_1 = \frac{N}{2N_1}, \qquad w_0 = \frac{N}{2N_0}$$

Where

- $N_1$ is the number of real samples
- $N_0$ is the number of fake samples
- $N = N_0 + N_1$

$C_{6}^{stack}$: OOF Meta-Feature Matrix

$$\mathbf{M} = \left[\hat{p}_1, \hat{p}_2, \dots, \hat{p}_M\right] \in \mathbb{R}^{N_{\text{train}} \times M}$$

$C_{7}^{stack}$: Meta-Learner Sub-Objective

$$
\min_{\theta_{g}} \quad Z_g = - \frac{1}{N}\sum_{i=1}^{N} [w_1 y_i \log (g(m_i)) + w_0 (1 - y_i) \log (1 - g(m_i))] + \lambda_{g} \Omega(\theta_{g})
$$

Where

- $m_i = [\hat{p}_1^{(i)}, \dots, \hat{p}_M^{(i)}]$ is the $i$-th row of $\mathbf{M}$
- category weights are not applied during meta-learner optimization
