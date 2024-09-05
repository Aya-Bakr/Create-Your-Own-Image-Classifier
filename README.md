# Create Your Own Image Classifier
 This was the project of course 7 in AI Programming with Python From Udacity

This repository contains the implementation of an image classifier using a pre-trained deep learning model. The project is divided into two main parts: a development notebook and a command-line application. Below is a breakdown of the criteria and specifications covered in each part.

## Part 1: Development Notebook

This section outlines the development process for building and training the image classifier.

### Criteria and Specifications

- **Package Imports**: 
  - All necessary packages and modules are imported in the first cell of the notebook.

- **Training Data Augmentation**: 
  - Utilizes `torchvision.transforms` to augment training data with techniques such as random scaling, rotations, mirroring, and/or cropping.

- **Data Normalization**: 
  - The training, validation, and testing datasets are appropriately cropped and normalized.

- **Data Loading**: 
  - The data for each set (train, validation, test) is loaded using `torchvision.datasets.ImageFolder`.

- **Data Batching**: 
  - The data for each set is loaded into batches using `torch.utils.data.DataLoader`.

- **Pretrained Network**: 
  - A pre-trained network (e.g., VGG16) is loaded from `torchvision.models`, with its parameters frozen.

- **Feedforward Classifier**: 
  - A new feedforward network is defined as a classifier using the features from the pre-trained model as input.

- **Training the Network**: 
  - The parameters of the feedforward classifier are trained while keeping the parameters of the feature network static.

- **Validation Loss and Accuracy**: 
  - During training, the validation loss and accuracy are displayed to monitor performance.

- **Testing Accuracy**: 
  - After training, the network's accuracy is evaluated on the test dataset.

- **Saving the Model**: 
  - The trained model is saved as a checkpoint, including associated hyperparameters and the `class_to_idx` dictionary.

- **Loading Checkpoints**: 
  - A function is provided to load a checkpoint and rebuild the model.

- **Image Processing**: 
  - The `process_image` function successfully converts a PIL image into a format suitable for input into the trained model.

- **Class Prediction**: 
  - The `predict` function takes the path to an image and a checkpoint, returning the top K most probable classes for the image.

- **Sanity Check with Matplotlib**: 
  - A Matplotlib figure is created that displays an image alongside its top 5 predicted classes, with actual flower names for verification.

## Part 2: Command-Line Application

This section describes the functionality of the command-line tools for training and predicting image classes.

### Criteria and Specifications

- **Training a Network**: 
  - The `train.py` script successfully trains a new network on a dataset of images.

- **Training Validation Log**: 
  - The script prints out the training loss, validation loss, and validation accuracy during the training process.

- **Model Architecture**: 
  - The training script allows users to choose from at least two different architectures available from `torchvision.models`.

- **Model Hyperparameters**: 
  - The script allows users to set hyperparameters such as learning rate, number of hidden units, and training epochs.

- **Training with GPU**: 
  - Users can opt to train the model on a GPU, improving training speed.

- **Predicting Classes**: 
  - The `predict.py` script reads an image and a checkpoint, then prints the most likely image class along with its associated probability.

- **Top K Classes**: 
  - The script can output the top K classes along with their associated probabilities.

- **Displaying Class Names**: 
  - Users can load a JSON file that maps class values to category names, enhancing interpretability.

- **Predicting with GPU**: 
  - The `predict.py` script supports GPU usage for calculating predictions.
