# transformer-classification-of-iris
transformer classification of  iris (PyTorch framework)

# The code will be described here.

## This command is meant to install a Python package called "datasets" using the Python package management tool pip. This package provides a number of datasets and data loaders to help users easily access and use various datasets in Python. These datasets can be used for research and development in areas such as machine learning, natural language processing, and computer vision. By installing this package, users can quickly access and use these datasets to accelerate their research and development efforts.👇

pip install datasets


## 1. Importing the necessary libraries👇

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from torchvision.transforms import ToTensor

from torch.nn.utils.rnn import pad_sequence

##  2.Defining the Transformer model

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 
        x = self.fc(x)
        return x
        
Description：This is a neural network model based on the Transformer model for processing input data and producing output results. Specifically, this model has the following characteristics:

- The dimension of the input data is input_dim and the dimension of the output data is output_dim.
- The dimension of the hidden layer in the middle of the model is hidden_dim, and the complexity of the model can be controlled by adjusting this parameter.
- The model uses num_layers TransformerEncoder layers, and each layer has num_heads attention heads.
- A linear layer (nn.Linear) and a fully connected layer (nn.Linear) are used in the model to process the input data and generate the output results.
- In the forward propagation process, the input data is first embedding through a linear layer, then processed through the TransformerEncoder layer, and finally the output results are generated through the fully connected layer.
- After processing the TransformerEncoder layer, the model aggregates the output at all times by calculating the mean to obtain a fixed dimensional vector, and then produces the final output through the fully-connected layer.

This model can be used to process sequential data, such as natural language text, image sequences, and so on. It has the advantage of being able to handle long sequential data and has better parallelism and generalization capability.

 👇 def forward(self, x)👇  
       
The forward propagation process of the model. In the forward propagation process, the input data x is first embedded through a linear layer and then processed through a TransformerEncoder layer to produce a sequence of outputs processed by an attention mechanism. Next, the model aggregates the outputs at all times by computing the mean to obtain a vector of fixed dimensions. Finally, the model maps this vector to the output dimension through a fully connected layer (fc) to produce the final output.

Specifically, the input parameter x of this function is a tensor representing the input data in the shape of (batch_size, sequence_length, input_dim), where batch_size denotes the batch size, sequence_length denotes the sequence length, and input_dim denotes the dimension of the input data . In the function, the input data is first embedded through the self.embedding linear layer to get a tensor of shape (batch_size, sequence_length, hidden_dim). Then, this tensor is processed by self.transformer to obtain a tensor of shape (batch_size, sequence_length, hidden_dim), which represents the output sequence processed by the attention mechanism. Then, this tensor is aggregated in the 1st dimension by computing the mean (mean) to obtain a tensor of the shape (batch_size, hidden_dim), representing the mean of the outputs at all moments. Finally, this tensor is mapped by self.fc fully connected layer to get a tensor of shape (batch_size, output_dim), which represents the final output result. The function returns this tensor as the output of the model by return statement.

Here x.mean(dim=1) represents the mean pooling (Mean Pooling) operation on the input data x in the 1st dimension, i.e., the vector at each moment (1st dimension) is averaged to obtain a vector with a fixed dimension. In this model, after the input data is processed by the TransformerEncoder layer, a vector is generated at each moment, and this operation can aggregate these vectors into a fixed dimensional vector for subsequent processing and output. In natural language processing, this operation is also called Sentence-level Representation (SLR).

## 3.# Loading the Iris dataset👇

iris_data = load_iris()

data = iris_data.data

targets = iris_data.target

This code is used to load the Iris dataset and divide it into two parts: data and targets.
First, the code calls the load_iris() function, which is a function in the sklearn library, to load the Iris dataset. Once loaded, the dataset is stored in the iris_data object.

Then, the code stores the data and targets in the dataset into the data and targets variables, respectively. Where data is a two-dimensional array of shape (n_samples, n_features), which represents all the sample data in the dataset, and each sample data has n_features. And targets is a one-dimensional array of shape (n_samples,), which represents the target values of all samples in the dataset.

It should be noted that the Iris dataset is a classical dataset for classification problems, which contains 150 sample data, each sample data has 4 features (calyx length, calyx width, petal length, petal width) and targets have 3 categories (Iris Setosa, Iris Versicolour, Iris Virginica).

## 4.Dividing the training set and test set👇

   train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=42)


This code uses the train_test_split function from the sklearn library to divide the dataset into two parts: the training set (train_data and train_targets) and the test set (test_data and test_targets).

Specifically, the code first passes in two parameters, the data (data) and the targets (targets) in the whole dataset. Then, the code sets the size of the test set to 20% of the entire dataset (test_size=0.2) and sets the random seed to 42 to ensure that the random results obtained from each run of the program are the same.

Finally, the train_test_split function returns a tuple containing four elements, namely the data of the training set, the data of the test set, the target of the training set, and the target of the test set. These variables can be assigned to the four variables train_data, test_data, train_targets, and test_targets, respectively.

It should be noted that the data set is divided to test the performance of the model during model training, and the test set should be the data set that is not used in building the model and is used to evaluate the model performance.

##  5.Defining custom data sets 👇
class IrisDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y
        
This code defines a custom dataset class, IrisDataset, for converting the raw data format to a data format acceptable to the PyTorch framework.

The definition of a dataset class typically includes the following functions:
- __init__ function: initializes the parameters of the dataset class.
- __len__ function: returns the total number of samples in the dataset.
- __getitem__ function: returns a sample in the dataset based on the given index idx.

Specifically, the __init__ function of IrisDataset takes two arguments: data and targets, which represent the data in the dataset and the corresponding target values, respectively.

The __len__ function returns the total number of samples in the dataset, that is, the number of data in the dataset.

The __getitem__ function takes the index idx as input parameter and returns the samples in the dataset whose index is idx. In the function, the code first obtains the corresponding data and target values in the dataset based on the index idx, then converts the data and target values to the tensor data type in the PyTorch framework, and returns them as a tuple (x, y) on return.

This custom dataset class can be used to define a data loader in PyTorch to provide data for model training with the following code:
```
train_dataset = IrisDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = IrisDataset(test_data, test_targets)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
```
where train_dataset and test_dataset are the dataset classes of the training set and test set, respectively, and train_loader and test_loader are the data loaders of the training set and test set, respectively, which can be used to provide the corresponding data during model training. batch_size indicates the batch size, shuffle indicates whether to disrupt the data order at the end of each epoch.

## 6. Set random seeds 👇
torch.manual_seed(42)

# Creating a data loader
train_dataset = IrisDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = IrisDataset(test_data, test_targets)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

In this code, torch.manual_seed(42) is used to set the seed of the random number generator to ensure that the results obtained are consistent each time the program is run with the same parameters.

Next, the code defines the data loaders train_loader and test_loader for the training and test sets. where train_loader has a batch size of 16 and a shuffle of True to disrupt the data order at the end of each epoch; test_loader has a batch size of 1 and a shuffle of True indicates that the data order is disrupted after each epoch.

The data loader is a tool in PyTorch that is used to split the dataset into smaller batches for easier model processing. The data loader can also perform data augmentation and shuffle the data to improve the training of the model.

It should be noted that the parameter shuffle is set differently for the training and test sets. For the training set, a shuffle of True disrupts the order of the data in each epoch to increase the robustness of the model; for the test set, a shuffle of False ensures that the order of the test set in each epoch is the same, which facilitates the model to compare the test results.

# 7.Define hyperparameters  👇
    input_dim = 4
    hidden_dim = 128
    output_dim = 3
    num_layers = 2
    num_heads = 4
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001

# Create models and optimizers
    model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers, num_heads)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
 
 In this code, torch.manual_seed(42) is used to set the seed of the random number generator to ensure that the results obtained are consistent each time the program is run with the same parameters.

Next, the code defines the data loaders train_loader and test_loader for the training and test sets. where train_loader has a batch size of 16 and a shuffle of True to disrupt the data order at the end of each epoch; test_loader has a batch size of 1 and a shuffle of True indicates that the data order is disrupted after each epoch.

The data loader is a tool in PyTorch that is used to split the dataset into smaller batches for easier model processing. The data loader can also perform data augmentation and shuffle the data to improve the training of the model.

It should be noted that the parameter shuffle is set differently for the training and test sets. For the training set, a shuffle of True disrupts the order of the data in each epoch to increase the robustness of the model; for the test set, a shuffle of False ensures that the order of the test set in each epoch is the same, which facilitates the model to compare the test results.

##  8.Training Model  👇
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Adjusting the dimensionality of input data
        inputs = inputs.unsqueeze(1)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)
        
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * total_correct / total_samples
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
 
This code implements the training process of the model. The code iterates through num_epochs of epochs through a for loop, each epoch contains multiple batches, and each batch contains batch_size samples. The code uses the train_loader to iterate over the data in the training set.

For each batch, the code performs the following operations:

- The gradient of the parameters in the model is set to 0 by optimizer.zero_grad() to avoid the impact on the previously calculated gradient when back-propagating the gradient calculation.
- inputs.squeeze(1) changes the dimension of the input data from (batch_size, feature_size) to (batch_size, 1, feature_size), because the Transformer model requires the input data to have the dimension of (batch_size, seq_len. feature_size), set seq_len to 1.
- model(inputs) propagates the input data forward to get the output of the model.
- criterion(outputs, targets) calculates the difference between the model output and the true label, i.e. the value of the loss function.
- loss.item() gets the value of the loss function and adds it to the total_loss for cumulative calculation.
- torch.max(outputs, 1) returns the position where each sample achieves the maximum value in the output tensor outputs, and returns the index and the maximum value of these positions, where the index is the category predicted by the model.
- (predicted == targets).sum().item() calculates the number of correctly predicted samples and adds them up cumulatively.
- targets.size(0) gets the number of samples in this batch and adds it to total_samples for cumulative calculation.
- loss.backward() calculates the gradient of the parameters in the model according to the loss function.
- optimizer.step() updates the parameters in the model according to the gradient calculated by the loss function.

For this epoch, the code cumulatively computes total_loss, total_correct and total_samples. using these variables the average loss avg_loss and the accuracy accuracy of the training set are computed and then they are printed out.

The code training process is executed cyclically, with each epoch containing multiple batches, and the parameters of the model are updated based on the data from each batch, allowing the model to gradually converge to optimal values.

## 9. Evaluating the model on a test set 👇
      correct = 0
      total = 0

      with torch.no_grad():
          for inputs, targets in test_loader:
              # Adjusting the dimensionality of input data
              inputs = inputs.unsqueeze(1)

              outputs = model(inputs)
              _, predicted = torch.max(outputs.data, 1)
              total += targets.size(0)
              correct += (predicted == targets).sum().item()

      accuracy = 100 * correct / total
      print(f"Test Accuracy: {accuracy:.2f}%")


This code tests the model with the test set data and calculates the test accuracy of the model. The code uses test_loader to iterate over the data in the test set.
For each test sample, the code performs the following actions:

- inputs.squeeze(1) changes the dimension of the test sample from (feature_size,) to (1, feature_size). This is because the test sample is a one-dimensional tensor, while the Transformer model requires a three-dimensional tensor of the input data, and setting seq_len to 1 is sufficient.
- model(inputs) propagates the input data forward to get the model output.
- torch.max(outputs.data, 1) returns the location of the maximum value of each sample in the output tensor outputs, and returns the index and maximum value of these locations, where the index is the category predicted by the model.
- targets.size(0) gets the number of samples in this batch and adds it to the total for cumulative calculation.
- (predicted == targets).sum().item() calculates the number of correctly predicted samples and adds them to total.

For this test set, the code cumulatively calculates correct and total, where correct is the number of samples correctly predicted by the model and total is the total number of samples in the test set. Finally, the code calculates the accuracyaccuracy of the model on the test set and prints it out.

The testing process is similar to the training process, but there is no need to compute gradients during testing and the parameters of the model are fixed and will not be updated.
