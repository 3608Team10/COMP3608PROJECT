# Fake News Detection

## How to Run

Bagging 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/3608Team10/COMP3608PROJECT/blob/main/Bagging.ipynb)

Boosting 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/3608Team10/COMP3608PROJECT/blob/main/Boosting.ipynb)

Stacking 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/3608Team10/COMP3608PROJECT/blob/main/Stacking.ipynb)

## Colab Workflow

### Google Drive Structure

![Drive Structure](images/folder-structure.png)

1. In your Google Drive create a folder to store your repository (e.g. project)
2. Create a Google colaboratory to store your git commands (e.g. Commands.ipynb).
3. Follow the commands in the section below to clone your repository and manage your git commands

### Git Commands

Check your current directory

```py
!pwd
```

Mount Google Drive to access your drive storage directly as a local repository

```py
from google.colab import drive
drive.mount('/content/drive/')
```

To unmount your drive if necessary you can use the following command. It ensures all pending writes are flushed and saved to drive before disconnecting.

```py
drive.flush_and_unmount()
```

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

Before we have the ability to push to github you need to create a token

1. Nagivate to github > click on your profile icon in the top right > settings
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


Now use colab secrets (key icon) and add the following (place the actual values in place of the placeholder):

- Name: USER    VALUE: placeholder
- Name: TOKEN   VALUE: placeholder


```py
from google.colab import userdata

USER = userdata.get('USER')
ORG = userdata.get('ORG')
TOKEN = userdata.get('TOKEN')
REPO = userdata.get('REPO')

!git remote set-url origin https://{USER}:{TOKEN}@github.com/3608Team10/COMP3608PROJECT.git
```

Create a new branch and switch your working directory to that branch

```py
!git switch -c <branch-name>
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

