# MGGCF
Multi-Graph Group Collaborative Filtering
---
This is our Tensorflow implementation for the paper:Multi-Graph Group Collaborative Filtering.
We also provide two baselines code.
## Introduction
---
Multi-Graph Group Collaborative Filtering is a new group recommendation method based on graph neural network.
## Environment Requirement
---
The  MGGCF code has been tested running under Python 3.6. The required packages are as follows:
+ tensorflow == 1.8.0
+ numpy == 1.14.3
+ scipy == 1.1.0
+ sklearn == 0.19.1
The AGREE and NeuMF baselines code are based on pytorch framework.
+ pytorch version: '0.3.0'
+ python version: '3.5'
## Example to Run the Codes
Copy data file folder to the corresponding method file folder like MGGCF or AGREE.
Then
run AGREE or NeuMF:
```
python main.py
```
run MGGCF:
```
python MGGCF.py
```
## Parameter Tuning
we put all the parameters in the config.py for AGREE and NeuMF.
and we put all the parameters in the parser.py for MGGCF.

## Dataset
We provide two processed datasets: CAMRa2011 and Movielens-Simi.
group(user)RatingTrain.:
+ Train file.
+ Each Line is a training instance: groupID(userID)\t itemID\t rating\t timestamp (if have)

group(user)RatingTest:
+ group(user) Test file (positive instances).
+ Each Line is a testing instance: groupID(userID)\t itemID\t rating\t timestamp (if have)

group(user)RatingNegative
+ group(user) Test file (negative instances).
+ Each line corresponds to the line of test.rating, containing 100 negative samples.
+ Each line is in the format: (groupID(userID),itemID)\t negativeItemID1\t negativeItemID2 ...

groupMember
+ group member file
+ Each line is a group information: groupID\t userID1 \t userID2 ...


