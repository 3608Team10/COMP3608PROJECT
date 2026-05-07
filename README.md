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
\min_{\theta_{1}, \dots, \theta_{M}, \theta_{g}} \quad
Z = - \frac{1}{N} \displaystyle\sum_{i=1}^{N}
\alpha_{c_i} [w_1 y_i log(F(x_i)) + w_0 (1 - y_i)
log(1 - F(x_i))] + \displaystyle\sum_{m=1}^{M} \lambda_{m} \Omega (\theta_{m})
+ \lambda_{g} \Omega (\theta_{g})
$$

SUBJECT TO

$C_{1}^{bag}$: Ensemble Prediction Function

$$
F(x) = \frac{1}{M} \displaystyle\sum_{m=1}^{M} f_m(x)
$$

$$
\text{where } f_m \text{ is trained on a bootstrap sample }
B_m \subset \mathcal{D} \text{ with replacement}
$$

$C_{1}^{boost}$: Ensemble Prediction Function

$$
F(x) = \displaystyle\sum_{m=1}^{M} \beta_{m} f_m(x),
\qquad
\beta_{m} > 0,
\quad
\displaystyle\sum_{m=1}^{M} \beta_{m} \eta \leq 1
$$

$$
\text{where } \eta \text{ is the learning rate shrinkage factor}
$$

$C_{1}^{stack}$: Ensemble Prediction Function

$$C_{1}^{train}: \hat{p}_m^{(i)} = f_{m}^{(-k(i))}(x_i) \qquad \text{(OOF meta-features)}$$

$$
\text{where base model } f_m \text{ predicts on fold } k(i)
\text{ using version trained on all other folds}
$$

$$
C_{1}^{test}: F(x) = g(f_1(x), \ldots, f_M(x))
\qquad \text{(full retrain inference)}
$$

$$
\text{where base models } f_1, \dots, f_M
\text{ are fully retrained on complete training set }
\mathcal{D}
$$

$C_2$: Feature Mapping

$$x_i = \phi (title_i, text_i, category_i, dataset_i)$$

$C_3$: Label Constraint

$$
y_i \in \\\{0, 1\\\},
\qquad
0 = fake, \quad 1 = real
$$

$C_4$: Category Weight

$$\alpha_{c_i} = \frac{N}{K \cdot freq(c_i)}$$

$$
\text{where } K \text{ is the number of distinct categories and }
freq(c_i) = \frac{N_{c_i}}{N}
\text{ is the proportion of samples in category } c_i
$$

$C_5$: Class Weight

$$w_1 = \frac{N}{2N_1}, \qquad w_0 = \frac{N}{2N_0}$$

$$
\text{where } N_1 \text{ and } N_0 \text{ are the counts of real and fake samples respectively}
$$

$C_{6}^{stack}$: OOF Meta-Feature Matrix

$$
\mathbf{M} = \left[\hat{p}_1, \hat{p}_2, \dots, \hat{p}_M\right] \in
\mathbb{R}^{N_{\text{train}} \times M}
$$

$C_{7}^{stack}$: Meta-Learner Sub-Objective

$$
\min_{\theta_g} \quad Z_g = - \frac{1}{N}\sum_{i=1}^{N}
[w_1\, y_i \log g(m_i) + w_0 (1 - y_i) \log (1 - g(m_i))]
+ \lambda_g \Omega(\theta_g)
$$

$$
\text{where } m_i = [\hat{p}_1^{(i)}, \dots, \hat{p}_M^{(i)}]
\text{ is the } i \text{-th row of } \mathbf{M}
$$

## Colab Workflow (Developers)

### Google Drive Structure

![Drive Structure](images/folder-structure.png)

1. In your Google Drive create a folder, 'project', to store your repository
2. In your project folder create a Google colaboratory, e.g. Commands.ipynb, to store your git commands.
3. Open your commands notebook and follow the sections below to manage your git commands and clone your repository.

### Mount Google Drive

Check your current directory

```py
!pwd
```

Mount Google Drive to access your drive storage directly as a local repository

```py
from google.colab import drive
drive.mount('/content/drive/')
```

<!-- To unmount your drive if necessary you can use the following command. It ensures all pending writes are flushed and saved to drive before disconnecting.

```py
drive.flush_and_unmount()
``` -->

Change directory to your project folder

```py
%cd /content/drive/MyDrive/project
```

### Clone Repository

Clone the repository (for first time setup or when necessary)

```py
!git clone https://github.com/3608Team10/COMP3608PROJECT.git
```

### Github Token

Before we have the ability to push to github you need to create a token

1. Nagivate to github > Click on your profile icon in the top right > Settings
2. Developer settings > Personal Access Tokens > Fine-grained tokens
3. Generate new token
    - Under Resource Owner changes this to '3608Team10'
    - Under Repository Access change this to 'All repositories'
    - Add permissions
        - Tick Contents
        - Tick Workflows
    - Change Contents Access to 'Read and write'
    - Change Workflows Access to 'Read and write'
4. Generate token and COPY THE TOKEN IMMEDIATELY

### Colab Secrets

Now use colab secrets (key icon) and add the following (place the actual values in the value column):

<img src="images/colab-secrets.png" alt="Colab Secrets - Github" height="350" width="350" />

### Configuring your github credentials to local environment

Now change your directory to the local repository folder

```py
%cd /content/drive/MyDrive/project/COMP3608PROJECT
```

Configure your credentials

```py
from google.colab import userdata

USER = userdata.get('USER')
TOKEN = userdata.get('TOKEN')

!git remote set-url origin https://{USER}:{TOKEN}@github.com/3608Team10/COMP3608PROJECT.git
```

```py
!git config --global user.email "Your GitHub email"
!git config --global user.name "Your GitHub Username"
```

### Git Commands

Create a new branch from an origin branch and switch your working directory to that branch

```py
!git switch -c <new-branch-name> origin/<remote-branch>
# !git switch -c <new-branch-name> origin/main
```

Switch to an existing branch

```py
!git switch <branch-name>
```

From this point you can open and edit other colab notebooks in the project then come back to the commands notebook to push/pull changes. Follow the Colab Secrets guide in the How to Run section to upload your Kaggle API Token.

Source Control Command

- The following command adds all files under the github repository directory with the . operator
- Adds a commit message

```py
!git add .
!git commit -m 'message'
```

Git push command

```py
!git push origin <branch-name>
```

Git pull command

```py
!git pull origin <branch-name> 
```
