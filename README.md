# Fake News Detection

## How to Run

### Prerequisites

- Google Drive Account
- Kaggle Account
- ingest_data.py (Download python script and upload to your Google Colab environment)

### Colab Notebooks

Bagging 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3608Team10/COMP3608PROJECT/blob/main/Bagging.ipynb)

Boosting 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3608Team10/COMP3608PROJECT/blob/main/Boosting.ipynb)

Stacking 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3608Team10/COMP3608PROJECT/blob/main/Stacking.ipynb)

### Colab Secrets

Before you can start working in the notebooks above you need to get your Kaggle API Key. <br>
The data ingestion script supports Kaggle API Tokens and Kaggle Legacy API Credentials. <br>
Place the actual values in the value field. <br>

API Tokens (Recommended):

<img src="images/colab-secrets-kaggle-token.png" alt="Colab Secrets - Kaggle (Token)" height="350" width="500"/> <br>

Legacy API Credentials:

<img src="images/colab-secrets-kaggle-legacy.png" alt="Colab Secrets - Kaggle (Legacy)" height="350" width="500"/> <br>

### Colab Files

Download the ingest_data.py script from the github repository. <br>
Drag and drop the script or right-click in the file window and click upload to add the script to your session. <br>

<img src="images/colab-files-script-upload.png" alt="Colab Files - Script Upload" height="350" width="350"/> <br>

## Optimization Problem

MIN

$$
Z = - \frac{1}{N} \displaystyle\sum_{i=1}^{N} \alpha_{c_i} [w_1 y_i log(F(x_i)) + w_0 (1 - y_i)
log(1 - F(x_i))] + \lambda \Omega (\Theta)
$$

SUBJECT TO

$C_1$: Ensemble Prediction Function

1. Bagging (Bootstrap Aggregating):

    Model:

    $$F(x) = \frac{1}{M} \displaystyle\sum_{m=1}^{M} f_m(x)$$

    Where:
    - Each $f_m$ is trained on a bootstrap sample

2. Boosting (Gradient Boosting):

    Model:
    
    $$F(x) = \displaystyle\sum_{m=1}^{M} \beta_{m} f_m(x)$$

    Where:
    - $\beta_{m} > 0$
    - $\beta_{m}$ represents weight of weak learner

3. Stacking (Meta-Learning):

    Model:

    $$F(x) = g(f_1(x), f_2(x), \cdots , f_M(x))$$

    Where:
    - $f_1 \cdots f_M$ : represents base models
    - $g$ : meta-learner

Ensemble Prediction Function $F(x)$

$$
F(x) = 
\begin{cases} 
\frac{1}{M} \sum f_m(x) & \qquad \text{Bagging} \\
\sum \beta_{m} f_m(x) &  \qquad \text{Boosting} \\
g(f_1(x), \cdots , f_M(x)) & \qquad \text{Stacking} \\
\end{cases}
$$

$C_2$: Feature Mapping

$$x_i = \phi (title_i, text_i, category_i, dataset_i)$$

$C_3$: Label Constraint

$$y_i \in \{ 0, 1 \}$$

$C_4$: Category Weight

$$\alpha_{c_i} = \frac{1}{freq(c_i)}$$

$C_5$: Class Weight

$$w_1 = \frac{N}{2N_1}, \qquad w_0 = \frac{N}{2N_0}$$

## Colab Workflow (Developers)

### Google Drive Structure

![Drive Structure](images/folder-structure.png)

1. In your Google Drive create a folder, 'project', to store your repository
2. In your project folder create a Google colaboratory, e.g. Commands.ipynb, to store your git commands.
3. Follow the commands in the section below to clone your repository and manage your git commands

### Clone Repository

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

Clone the repository

```py
!git clone https://github.com/3608Team10/COMP3608PROJECT.git
```

Now change your directory to the local repository folder

```py
%cd /content/drive/MyDrive/project/COMP3608PROJECT
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

<img src="images/colab-secrets-github.png" alt="Colab Secrets - Github" height="350" width="350"/> <br>

Configuring your github credentials to local environment

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

From this point you can open and edit other colab notebooks in the project then come back to the commands notebook to push/pull changes. 

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

```
!git pull origin <branch-name> 
```

