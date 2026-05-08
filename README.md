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

$$\text{where } e \in \\\{bag, boost, stack\\\}, \quad F(x) \in \\\{0, 1\\\}$$

SUBJECT TO

$C_{1}^{bag}$: Ensemble Prediction Function

$$F(x) = \frac{1}{M} \displaystyle\sum_{m=1}^{M} f_m(x)$$

$$\text{where } f_m \text{ is trained on a bootstrap sample } B_m \subset \mathcal{D} \text{ with replacement}$$

$C_{1}^{boost}$: Ensemble Prediction Function

$$F(x) = \displaystyle\sum_{m=1}^{M} \eta \beta_{m} f_m(x), \qquad \beta_{m} > 0$$

$$\text{where } \eta \text{ is the learning rate shrinkage factor}$$

$C_{1}^{stack}$: Ensemble Prediction Function

$$C_{1}^{train}: \hat{p}_m^{(i)} = f_{m}^{(-k(i))}(x_i) \qquad \text{(OOF meta-features)}$$

$$\text{where base model } f_m \text{ predicts on fold } k(i) \text{ using version trained on all other folds}$$

$$C_{1}^{test}: F(x) = g(f_1(x), \ldots, f_M(x))\qquad \text{(full retrain inference)}$$

$$\text{where base models } f_1, \dots, f_M \text{ are fully retrained on complete training set } \mathcal{D}$$

$C_2$: Feature Mapping

$$x_i = \phi (title_i, text_i, category_i, dataset_i)$$

$C_3$: Label Constraint

$$y_i \in \\\{0, 1\\\}, \qquad 0 = fake, \quad 1 = real$$

$C_4$: Category Weight

$$\alpha_{c_i} = \frac{N}{K \cdot N_{c_i}}$$

$$\text{where } K \text{ is the number of distinct categories, } N_{c_i} \text{ is the number of samples in category } c_i \text{ and N is the total number of samples}$$

$C_5$: Class Weight

$$w_1 = \frac{N}{2N_1}, \qquad w_0 = \frac{N}{2N_0}$$

$$\text{where } N_1 \text{ is the number of real samples, } N_0 \text{ is the number of fake samples and } N = N_0 + N_1$$

$C_{6}^{stack}$: OOF Meta-Feature Matrix

$$\mathbf{M} = \left[\hat{p}_1, \hat{p}_2, \dots, \hat{p}_M\right] \in \mathbb{R}^{N_{\text{train}} \times M}$$

$C_{7}^{stack}$: Meta-Learner Sub-Objective

$$
\min_{\theta_{g}} \quad Z_g = - \frac{1}{N}\sum_{i=1}^{N} [w_1 y_i \log (g(m_i)) + w_0 (1 - y_i) \log (1 - g(m_i))] + \lambda_{g} \Omega(\theta_{g})
$$

$$\text{where } m_i = [\hat{p}_1^{(i)}, \dots, \hat{p}_M^{(i)}] \text{ is the } i \text{-th row of } \mathbf{M} \text{,}$$

$$\text{category weights are not applied during meta-learner optimization}$$
