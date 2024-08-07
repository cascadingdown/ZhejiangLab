# Edge-Cloud Collaborative Satellite Image Analysis for Efficient Man-Made Structure Recognition（IEEE under review)

This project utilizes the UCMerced Land Use Dataset for training and evaluating various models to classify images into artificial and natural categories in an attempt to strike a balance between accuracy and latency. The code supports the research presented in the paper "Edge-Cloud Collaborative Satellite Image Analysis for Efficient Man-Made Structure Recognition".

## Dataset

The dataset can be downloaded from the following link:
[UCMerced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

## Project Structure

The project is organized into the following directories:

- `code`: Contains the Python scripts for data processing, model definition, training, and evaluation.
- `examples`: Contains results for the experiments.

## Code Structure

### Data processing

The data processing script (`data_processing.py`) handles the following tasks:
- Loading the UCMerced Land Use Dataset.
- Applying necessary transformations.
- Updating labels to classify images as artificial or natural.
- Splitting the dataset into training and testing sets.
- Creating data loaders for batch processing.

### Model Definitions

The model definitions script (`models.py`) includes:
- `SimpleCNN`: A simple convolutional neural network for baseline comparison.
- `ModifiedMobileNetV2`: A modified MobileNetV2 model for improved performance.
- `ModifiedShuffleNet`: A modified ShuffleNet model for efficiency.
- `MobileShuffleNet`: A custom hybrid model combining MobileNetV2 and ShuffleNet features.

### Training and Evaluation

The training and evaluation script (`train_and_evaluate.py`) performs:
- Training of the model using the training dataset.
- Evaluation of the model using the testing dataset.
- Calculation of accuracy, recall, and F1 score.

### Visualization

The visualization script (`visualization.py`) provides functions to:
- Display sample images from the dataset.
- Show incorrect predictions made by the model.
- Save images to local directories for further inspection.


### Data Transmission

The data transmission script (`transmit.py`) handles the following tasks:
- Classifying images and separating those predicted as "Artificial".
- Transferring images to a remote server using SFTP.
- Measuring and reporting the time taken for classification and data transfer.

## Examples

Examples of the experiment and results can be found in the `examples` folder. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

We acknowledge the creators of the UCMerced Land Use Dataset for providing the dataset used in this project.
