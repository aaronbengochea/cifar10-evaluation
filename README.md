## Repo Inspiration 
This repository takes inspiration from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). We explore model architectures inspired by ResNet18.

## Study Details
The study focused on 3 types of ResNets: Shallow, Medium-Depth, and Deep ResNets. We investigate how different training strategies—including data regularization, optimizer selection, and learning rate scheduling—affect each model's training time, performance, and generalizability. 

## Recreate Findings - General Setup on Kaggle
1. Git clone this repository
2. Open Kaggle.com and sign in for GPU access 
3. Navigate to Code -> New Notebook
4. Once inside the kaggle notebook enviorment, select File -> Upload notebook
5. Upload kaggle.ipynb which can be found in the cloned repository
6. Follow further instructions which can be found on kaggle.ipynb
7. Initialize the model, train the model, and finally perform inference on the hidden test competition set using all model checkpoints

## Maximum Benchmark Model - "Most Accurate Model"
### Architecture
- **Block Type:** `BasicBlock`
- **Blocks per Layer:** `[5, 7, 4, 3]`
- **Channels per Layer:** `[32, 64, 128, 256]`
- **Convolutional Filter Sizes:** `3x3`
- **Identity Filter Sizes:** `1x1`
- **Parameter Count:** `4.99M`

### Training Parameters
- **Epochs:** `310`
- **Regulerization:** `RandomHorizontalFlip, `
- **Optimizer:** `SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)`
- **Scheduler:** `CosineAnnealingLR(Tmax=Epochs)`

### Training & Performance
- **Train Set Avg Loss** `0.3973`
- **Test Set Avg Loss** `0.1440`
- **Train Set Accuracy:** `86.22%`
- **Test Set Accuracy:** `96.01%`
- **Hidden Test Set Accuracy:** `85.095%`
- **Optimal model found at:** `305/310 epoch`

## Directory Details
- `data` - stores the CIFAR10 training and test data locally
- `testset` - stores the hidden kaggle CIFAR10 testset used for scoring in the competition
- `experiments` - an assortment of different experiments and their details, conducted on various ResNets with varying depth/channel/optimizer/schedular choices, all with trainable parameter counts below 5M


## File Details
- `kaggle.ipynb` - notebook to be imported and used in kaggle enviornment for recreation of model initialization, training, and inference
- `resnet.py` - contains the implementation of the ResNet, and BasicBlock classes
- `training.py` - contains the implementation of the training loop
- `inference.py` - contains the implementation of the inference loop using the hidden competition dataset and all checkpoint models
- `research.pdf` - A comprehensive research report that captures our methodology, experiments, and key learnings from training various ResNet architectures on CIFAR-10


