This Repo has been forked from the original SORT repo in order to modify the SORT algorithm and implement our EECS 6111 project.

To set up this project, pull the repository and create a conda environment with the required packages as follows (Make sure Conda is installed on your machine):
```bash
conda create -n sort python=3.7.16
conda activate sort
pip3 install -r requirements.txt
```
Once the conda environment is set up we need to set up the data. For the purposes of reproduction we simply provide a download link to the detection results used as input for the methods. A zip file of the data can be downloaded [here](https://yuoffice-my.sharepoint.com/:u:/r/personal/kj323_yorku_ca/Documents/dataset.zip?csf=1&web=1&e=wAltP3) (If unable to access please reach out to me at my email kj323@yorku.ca)

The dataset contains the detection results along with ground truths as well as the edge probability files. Extract the zip to the same directory as the repository. Once they have been extracted, run the file main.py with the following commands (making sure you also have params1d.npy)

```bash
python3 main.py --Dataset=./dataset/ --ParamsPth=params1d.npy
```

This tests the performance of the methods, the results can be found in ./dataset/Results/Evaluate/evaluate[0,1,2].txt. Here evaluate 0 corresponds to base sort, 1 corresponds to GNN advice 2 corresponds to min cost flow advice.

To test speed:
```bash
python3 test_speed.py --Dataset=./dataset/ --ParamsPth=params1d.npy 
```
This tests the speed of the methods, the results can be found in ./dataset/Results/Evaluate/evaluate[0,1,2]_runtime.txt. Here evaluate 0 corresponds to base sort, 1 corresponds to GNN advice 2 corresponds to min cost flow advice.

For this project, we had to edit the sort source code to ensure it can encorporate various types of advice. Along with this, we also edited the sort repository to include evaluation code so that we could evaluate the results. Along with this, we also provide two easy to use scripts in main.py and test_speeds.py to ensure that the project is reproducable. 

For the offline methods, our implementations for the GNN method can be found [Here](https://github.com/Kumarj95/mot_neural_solver). For this method, we had to train a model using the training data from the MOT17 and MOT20 datasets. To do this, we rewrote the repository with a modern version of torch. Along with this, we also edited the source code to be able to extract the edge probabilities required for this project. 