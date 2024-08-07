# Edge-Cloud Collaborative Satellite Image Analysis for Efficient Man-Made Structure Recognition（IEEE under review)

This project utilizes the UCMerced Land Use Dataset for training and evaluating various models to classify images into artificial and natural categories in an attempt to strike a balance between accuracy and latency. The code supports the research presented in the paper "Edge-Cloud Collaborative Satellite Image Analysis for Efficient Man-Made Structure Recognition".

## Dataset

The dataset can be downloaded from the following link:
[UCMerced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

## Project Structure

The project is organized into the following directories:

- `code`: Contains the Python scripts for data processing, model definition, training, and evaluation.
- `examples`: Contains Jupyter notebooks demonstrating the experiments and results.

## Code Structure

### Data Preparation

The data preparation script (`data_preparation.py`) handles the following tasks:
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

## Running the Project

To run the project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. **Download the dataset**:
    Download the UCMerced Land Use Dataset from [here](http://weegee.vision.ucmerced.edu/datasets/landuse.html) and extract it to the appropriate directory.

3. **Install dependencies**:
    Ensure you have Python and the necessary libraries installed:
    ```sh
    pip install torch torchvision matplotlib scikit-learn
    ```

4. **Run the data preparation script**:
    ```sh
    python code/data_preparation.py
    ```

5. **Train and evaluate the model**:
    ```sh
    python code/train_and_evaluate.py
    ```

6. **Visualize the results**:
    ```sh
    python code/visualization.py
    ```

## Examples

Examples of the experiment and results can be found in the `examples` folder. These Jupyter notebooks provide a detailed walkthrough of the experiments and visualize the results.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

We acknowledge the creators of the UCMerced Land Use Dataset for providing the dataset used in this project.
