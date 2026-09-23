# Handwritten Math Symbol Classification

This project compares classic machine learning approaches for recognising handwritten math symbols. I wanted to see how much the choice of features matters compared to the choice of classifier, so I trained two classifiers (SVM and a simple neural network) on three kinds of features (raw pixels, HoG and LBP) and compared all six combinations.

## The Dataset

The dataset has 5,000 grayscale images of handwritten symbols, split evenly across 10 classes:

`(` &nbsp; `1` &nbsp; `7` &nbsp; `=` &nbsp; `C` &nbsp; `alpha` &nbsp; `div` &nbsp; `rightarrow` &nbsp; `sum` &nbsp; `v`

Each image is resized to 45 × 45 pixels. The data is split 70/30, giving 3,500 images for training and 1,500 for testing (with `random_state=42` so the results can be reproduced).

## What I Did

### Feature extraction

* **Raw pixels:** each image is flattened into a 2,025 value vector and scaled to the 0 to 1 range.
* **HoG (Histogram of Oriented Gradients):** 9 orientations, 10 × 10 pixels per cell and 2 × 2 cells per block, which gives a 324 value feature vector. HoG captures the direction of strokes, which turns out to be exactly what matters for handwriting.
* **LBP (Local Binary Patterns):** 24 points with a radius of 8, using the uniform method. The result is a normalised 26 bin histogram per image.

### Classifiers

* **SVM:** RBF kernel with C = 100.
* **ANN:** a small fully connected network built in Keras with two hidden layers (256 and 128 neurons, ReLU) and a softmax output for the 10 classes. It uses the Adam optimiser and sparse categorical cross entropy loss, and trains for 30 epochs.

For every combination, the notebook reports test accuracy, plots a confusion matrix and shows a few examples of correct and incorrect predictions so you can see where each model struggles.

## Results

| Classifier | Feature | Test Accuracy |
|:--|:--|:--|
| SVM | Raw Pixels | 94.5% |
| SVM | HoG | **96.9%** |
| SVM | LBP | 67.1% |
| ANN | Raw Pixels | 89.1% |
| ANN | HoG | **96.6%** |
| ANN | LBP | 56.9% |

### What stood out

* **HoG was the clear winner** for both classifiers, sitting just under 97%. Once the features were good, the choice of classifier barely mattered.
* **Raw pixels did better than expected with the SVM** (94.5%), but the ANN dropped to 89% on the same input. Its training accuracy was around 95%, so it was starting to overfit on the 2,025 pixel inputs.
* **LBP struggled badly.** LBP is designed to describe texture, and handwritten symbols are really about shape and stroke direction rather than texture. Squashing each image into a 26 bin histogram also throws away all the spatial information, so symbols like `v`, `alpha` and `sum` got mixed up a lot.
* The most common mistake across the stronger models was confusing `(` and `1`, which makes sense because they can look almost identical when written quickly.

## Tech Stack

* Python 3.12
* TensorFlow / Keras 2.18
* scikit learn (SVM and metrics)
* scikit image (HoG and LBP)
* OpenCV (loading and resizing images)
* NumPy, Pandas, Matplotlib and Seaborn

## Running It Yourself

1. Clone the repo:

```bash
   git clone https://github.com/pray900/<repo-name>.git
   cd <repo-name>
```

2. Install the dependencies:

```bash
   pip install numpy pandas matplotlib seaborn opencv-python-headless scikit-image scikit-learn tensorflow pydot
```

3. Put the dataset zip in the project folder. The notebook unzips it and expects one subfolder per class, each full of `.jpg` images.

4. Open `Math symbols image classfication.ipynb` in Jupyter and run the cells from top to bottom.

## Possible Next Steps

* Train a CNN, which should beat everything here because it learns its own features.
* Tune the SVM and ANN hyperparameters with a grid search instead of using fixed values.
* Add dropout or early stopping to the ANN to cut down on overfitting.
* Try combining HoG and LBP features to see if they add anything together.
* Use data augmentation (small rotations and shifts) to make the models more robust.
