# Fake News Detection

## How to Run

### Prerequisites

- Google Drive Account
- Kaggle Account
- data_ingestion.py (Download data_ingestion.py and upload to your Google Colab environment)

### Colab Notebooks

Bagging 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3608Team10/COMP3608PROJECT/blob/main/Bagging.ipynb)

Boosting 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3608Team10/COMP3608PROJECT/blob/main/Boosting.ipynb)

Stacking 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3608Team10/COMP3608PROJECT/blob/main/Stacking.ipynb)

### Colab Secrets

Before you can start working in the notebooks above you need to get your kaggle api key. <br>
The data ingestion script supports Kaggle API Tokens and Kaggle Legacy Keys. <br>
Place the actual values in the value field. <br>

For the new api token:

<img src="images/colab-secrets-kaggle-token.png" alt="Colab Secrets - Kaggle (Token)" height="350" width="500"/> <br>

For the legacy api key:

<img src="images/colab-secrets-kaggle-legacy.png" alt="Colab Secrets - Kaggle (Key)" height="350" width="500"/> <br>

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
!git clone https://github.com/3608Team10/COMP3608PROJECT
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

```py
from google.colab import userdata

USER = userdata.get('USER')
TOKEN = userdata.get('TOKEN')

!git remote set-url origin https://{USER}:{TOKEN}@github.com/3608Team10/COMP3608PROJECT.git
```

### Git Commands

Create a new branch and switch your working directory to that branch

```py
!git switch -c <branch-name>
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

