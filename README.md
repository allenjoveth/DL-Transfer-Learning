# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Transfer Learning is a technique where a pre-trained model (trained on a large dataset such as ImageNet) is used as a starting point for a different but related task. It leverages learned features from the original task to improve learning efficiency and performance on the new task.

VGG19 is a convolutional neural network with 19 layers. It consists of multiple convolutional layers for feature extraction, followed by fully connected layers for classification. In transfer learning, we typically freeze the convolutional layers and retrain the final fully connected layers to match our dataset.


## Neural Network Model


## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 
Visualize sample images from the dataset.


### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.




## PROGRAM

### Name: ALLEN JOVETH P

### Register Number: 212223240007

```python
print(f"Total number of test samples: {len(test_dataset)}")
first_image1,label=test_dataset[0]
print("Image shape:",first_image1.shape)

model=models.vgg19(weights=VGG19_Weights.DEFAULT)
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images = images.to(device)
            labels = labels.to(device) 
            optimizer.zero_grad()
            outputs=model(images)

            target_labels = labels.unsqueeze(1).float().to(device)
            loss=criterion(outputs,target_labels)

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))

        
        model.eval()
        val_loss=0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs=model(images)
            target_labels = labels.unsqueeze(1).float().to(device)
            loss=criterion(outputs,target_labels)
            val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Allen Joveth P")
    print("Register Number:  212223240007")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="616" height="487" alt="Screenshot 2026-05-18 113531" src="https://github.com/user-attachments/assets/eaa84475-dcb5-4c08-9585-28b68c13df76" />


## Confusion Matrix
<img width="549" height="673" alt="Screenshot 2026-05-18 113510" src="https://github.com/user-attachments/assets/407c4a06-00c2-48e5-aa01-d87f8f4f7ea7" />

## Classification Report
<img width="402" height="177" alt="image" src="https://github.com/user-attachments/assets/6ba4ad19-b144-4a11-99a6-528f905d5b66" />

### New Sample Data Prediction
<img width="299" height="324" alt="Screenshot 2026-05-18 113437" src="https://github.com/user-attachments/assets/d65a639e-b37d-4d0f-9af9-1b624122962c" />
<img width="402" height="177" alt="image" src="https://github.com/user-attachments/assets/2041017f-d6ad-4d59-a7e2-cd68849402ae" />

## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
