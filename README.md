# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Problem: Digital images are often corrupted by various forms of noise during acquisition, transmission, or storage. This noise degrades image quality, obscures important features, and can negatively impact subsequent image processing tasks such as object recognition, segmentation, or analysis.

Goal: Develop and implement a deep learning model, specifically a Denoising Autoencoder, using PyTorch to effectively remove noise from corrupted images. The model should learn to reconstruct clean images from their noisy counterparts, thereby improving image quality and clarity. The performance of the autoencoder will be evaluated based on its ability to accurately denoise images from the MNIST dataset.

## DESIGN STEPS
### STEP 1: 

Examine the current setup for loading the MNIST dataset and the add_noise function to understand how input data is prepared and corrupted. This includes checking the transform applied to images.

### STEP 2: 

Review the DenoisingAutoencoder class definition, paying attention to the encoder and decoder layers, and their corresponding activation functions. Also, confirm the criterion (loss function) and optimizer used for training.

### STEP 3: 

Examine the train function to understand how the model is trained, including the epoch loop, batch processing, noise addition during training, forward pass, loss calculation, backpropagation, and optimizer step.
### STEP 4: 

Review the visualize_denoising function to understand how the model's performance is evaluated and visualized. Pay attention to how original, noisy, and denoised images are displayed side-by-side for comparison.

### STEP 5: 

Based on the output of the executed visualize_denoising function, analyze the effectiveness of the current autoencoder in removing noise. Identify patterns in denoised images and consider if the model is underfitting or overfitting.

### STEP 6: 

Summarize the current understanding of the denoising autoencoder's implementation and performance, and suggest potential next steps for improvement or further analysis based on the assessment.


## PROGRAM

### Name:Rithika R

### Register Number:212224240136

```python
# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
      super(DenoisingAutoencoder, self).__init__()
      self.encoder=nn.Sequential(
          nn.Conv2d(1,16,3,stride=2,padding=1),
          nn.ReLU(),
          nn.Conv2d(16,32,3,stride=2,padding=1),
          nn.ReLU(),
      )
      self.decoder=nn.Sequential(
          nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(16,1,3,stride=2,padding=1,output_padding=1),
          nn.Sigmoid(),
      )



    def forward(self, x):
      x=self.encoder(x)
      x=self.decoder(x)
      return x


# Initialize model, loss function and optimizer
model =DenoisingAutoencoder().to(device)
criterion =nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)

# Training function
def train(model, loader, criterion, optimizer, epochs=5):
  print("Name: Rithika R                  ")
  print("Register Number:   212224240136               ")
  model.train()
  for epoch in range(epochs):
    running_loss=0.0
    for images, _ in loader:
      images=images.to(device)
      noisy_images=add_noise(images).to(device)

      outputs=model(noisy_images)
      loss=criterion(outputs,images)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss+=loss.item()

    print(f"Epoch[{epoch+1}/{epoch}],Loss:{running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Rithika R                  ")
    print("Register Number:212224240136                  ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


```

### OUTPUT

### Model Summary
```
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 14, 14]             160
              ReLU-2           [-1, 16, 14, 14]               0
            Conv2d-3             [-1, 32, 7, 7]           4,640
              ReLU-4             [-1, 32, 7, 7]               0
   ConvTranspose2d-5           [-1, 16, 14, 14]           4,624
              ReLU-6           [-1, 16, 14, 14]               0
   ConvTranspose2d-7            [-1, 1, 28, 28]             145
           Sigmoid-8            [-1, 1, 28, 28]               0
================================================================
Total params: 9,569
Trainable params: 9,569
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.13
Params size (MB): 0.04
Estimated Total Size (MB): 0.17
----------------------------------------------------------------
```
### Training loss
```
Name: Rithika R                  
Register Number:   212224240136               
Epoch[1/0],Loss:0.0158
Epoch[2/1],Loss:0.0149
Epoch[3/2],Loss:0.0146
Epoch[4/3],Loss:0.0143
Epoch[5/4],Loss:0.0141
```


## RESULT
The above code execute succesfully.
