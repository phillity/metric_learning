# metric_learning
Metric Learning TF 2.0+Keras Algorithm Implementations for Facial Recognition

## Requirements:
1. Python 3.6+
2. pip install -r requirements.txt
3. If GPU is available, pip install tensorflow-gpu==2.0.0 (highly recommended)

## Datset setup:
These steps will download the VGGFace2 dataset (caution: ~40GB of disk space is required)
1. Create an account at http://zeus.robots.ox.ac.uk/vgg_face2/login/
2. Run the dataset download scripts (metric_learning/utils/vggface2_train_download.py and metric_learning/utils/vggface2_test_download.py)
3. Unzip the resutling "vggface2_train.tar.gz" and "vggface2_test.tar.gz" files into the metric_learning/data directory
4. Run python utils/create_dataset.py in order to create "vggface2_train_dataset" and "vggface2_test_dataset" directories of preprocessed images

## Pretrain softmax model on vggface2_train_dataset (preprocessed VGG2Face train set):
These are the steps to pretrain a softmax loss-based model on the preprocessed "train_dataset"
1. cd metric_learning
2. python pretrain_softmax_model.py

## Finetune softmax model using metric learning on vggface2_test_dataset (preprocessed VGG2Face test set):
These are the steps to finetune a pretrained model using a metric learning and the preprocessed "test_dataset"
1. cd metric_learning
2. python finetune_softmax_model.py -m ["softmax", "contrastive", "triplet", "lmcl", "aaml"]
