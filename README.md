# ML_INTRODUCTION_WITH_KNN

## Classification Problems in Machine Learning

Classification is one of the main topics in Machine Learning. It involves assigning a category or label to a dataset based on its features. Examples include:

- **Medical Diagnosis:** Classify patients as "sick" or "not sick" based on data such as age, sex, weight, height, and symptoms.
- **Species Identification:** Determine the species of an animal based on physical characteristics.
- **Fraud Detection:** Classify bank transactions as "fraudulent" or "not fraudulent."

### Project: Flower Species Classification with k-NN

This project uses a basic Machine Learning algorithm, k-Nearest Neighbors (k-NN), to classify flower species based on simple features such as petal length and width. The k-NN algorithm is a classification method that works as follows:

1. **Training Dataset:** We use a known dataset, where the class of each sample is already known. This dataset is used to train the model.

2. **Classifying New Examples:** To classify a new example, k-NN calculates the distance between the example to be classified and all examples in the training dataset. In this case, we use Euclidean distance.

3. **Selecting the Nearest Neighbors:** The algorithm selects the k closest (nearest) examples based on the calculated distance.

4. **Class Inference:** The class of the new example is determined based on the most common class among the k nearest neighbors. k-NN makes the class prediction based on this majority.

This method is simple but effective for many classification problems, providing an intuitive and easy-to-implement approach.

# Guide to Run this project

## C

## 1. Clone the Repository 
<br>

1. Open your terminal (or command prompt).

<br>

2. Clone the repository using Git:

    ```bash
    git clone https://github.com/gabriel26077/ML_INTRODUCTION_WITH_KNN
    ```
3. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

   This will create a directory named `venv` containing the virtual environment.

## 2. Activate the Virtual Environment

- **On Windows:**

    ```bash
    venv\Scripts\activate
    ```

- **On macOS and Linux:**

    ```bash
    source venv/bin/activate
    ```

   After activation, your terminal prompt should change to indicate that the virtual environment is active.

## 3. Install Dependencies

1. Ensure you have a `requirements.txt` file in your project directory with the necessary dependencies listed.

2. Install the dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## 4. Run the program

1. With the virtual environment active, you can now run your Python program. For example:

    ```bash
    python knn.py
    ```

    or

    ```bash
    python knn_scatter_plot.py
    ```
## 5. Deactivate the Virtual Environment

- To exit the virtual environment when you are done:

    ```bash
    deactivate
    ```
