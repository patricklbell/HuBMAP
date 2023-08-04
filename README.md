# Setup
This project requires Python 3.11.4. 
## Download
Click download zip from Github and unzip the folder. Navigate to the root of the project.
## Install Dependencies
```
pip install -r requirements.txt
```

# Usage
This repository contains two notebooks, 'colab-notebook.ipynb' is designed to run in Google Colab,
'local-notebook.ipynb' is designed to run locally using Jupyter. The repository also contains commandline
programs for training and testing out a U-Net model, to understand these commands run
```
python train.py --help
```
or
```
python evaluate.py --help
```

Additionally, 'view.py' is a PyQt applet for exploring the dataset. Simply run
```
python view.py
```
to run the applet.
