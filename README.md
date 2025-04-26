# DCGAN Implementation

## Objective
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)**, inspired by the paper *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* by Radford et al. (2015). The goal is to train a generator that can create realistic images similar to those in the training dataset.

## Dataset Preprocessing
1. **Downloading the dataset**: The dataset is loaded from Kaggle.
2. **Transformations applied**:
   - Resize images to a fixed size (e.g., 64x64 pixels).
   - Convert images to tensors.
   - Normalize pixel values to the range [-1, 1] for stable GAN training.
3. **Creating DataLoader**:
   - DataLoader is used to efficiently load batches of images for training.

## Model Architecture
- **Generator**: A deep neural network with transposed convolutional layers to generate images from random noise.
- **Discriminator**: A convolutional neural network that classifies images as real or fake.
- **Loss Function**: Binary Cross-Entropy (BCE) loss is used for both generator and discriminator.
- **Optimizer**: Adam optimizer is used with tuned learning rates.

## Training the Model
1. **Set hyperparameters**: Batch size, learning rate, number of epochs, etc.
2. **Training loop**:
   - Train the discriminator to differentiate real and generated images.
   - Train the generator to produce realistic images that can fool the discriminator.
   - Loss values are monitored for both networks.
3. **Checkpointing**: Model weights are saved periodically.

## Testing the Model
- After training, the generator can be used to produce new images.
- Generate images by feeding random noise into the trained generator.
- Save or visualize the generated outputs.

## Expected Outputs
- The generator will produce images resembling the dataset.
- Training progression can be observed via loss curves.
- Sample outputs should improve in quality over epochs.

## Running the Code
1. Clone the repository and install dependencies.
2. Download the dataset (if not automatically fetched by the script).
3. Run the notebook or script to start training.
4. Use the trained generator to create new images.

## Dependencies
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- Kaggle API (if loading data from Kaggle)

