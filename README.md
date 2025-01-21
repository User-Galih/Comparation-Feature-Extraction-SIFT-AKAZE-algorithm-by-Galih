# Comparison of Feature Extraction Using SIFT and AKAZE on NIST Dataset
## author : Galih Putra Pratama
This repository compares feature extraction using the SIFT and AKAZE algorithms. It includes code, the NIST facial dataset, comparison graphs, and output files (Excel and Word) containing feature extraction results and analysis. The project evaluates the performance of both algorithms for facial feature extraction.

## Overview

### The project involves:
- SIFT (Scale-Invariant Feature Transform): An algorithm for detecting and describing local features in images.
- AKAZE (Accelerated-KAZE): A faster alternative to SIFT, designed to provide competitive performance with lower computational cost.

### The evaluation includes:
- Keypoint detection comparison between SIFT and AKAZE.
- Matching keypoints between two images and measuring the Euclidean distance.
- Timing evaluation to compare the speed of both methods.
- Confusion matrix analysis for evaluating precision, recall, accuracy, and F1-score.

## Requirements
To run the code and generate the results, you will need:

1. Python 3.x
   
Libraries:
1. OpenCV : For image processing and feature detection.
2. NumPy : For handling data and exporting results to Excel.
3. scikit-learn: For evaluating regression and classification metrics.
4. Pandas : For data manipulation and exporting results.
5. Matplotlib : For visualizing results.
6. python-docx: For generating Word documents containing analysis results.
   
Install the dependencies using:

```bash
pip install opencv-python numpy pandas matplotlib python-docx scikit-learn
```
## Code Workflow
1. Feature Detection: The code first detects features using both SIFT and AKAZE algorithms.
2. Keypoint Matching: Keypoints are matched between pairs of images, and the Euclidean distance between matched points is calculated.
3. Timing: The time taken to process each image pair using both SIFT and AKAZE is recorded.
4. Evaluation: A detailed comparison of the two methods is performed, including confusion matrix metrics, keypoint matching accuracy, and computational efficiency.

## How It Works
1. Loading Images: The code loads images from a specified directory (training and validation sets).
2. Feature Detection and Matching: SIFT and AKAZE are used for feature detection, and keypoint matches between the images are computed.
3. Evaluation:
    - Regression Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² are calculated to compare image descriptors.
    - Classification Metrics: Precision, Recall, Accuracy, and F1-Score are computed for keypoint matching.
    - Timing Evaluation: Execution time for both SIFT and AKAZE is measured.
    - Image Visualization: Visualizes the matches, inliers, and outliers, and saves the images as PNG files.
4. Excel Output: Results are stored in an Excel file in multiple sheets:
    - Hasil Perbandingan: Overall comparison results.
    - Keypoint Matches: Detailed keypoint match data.
    - Timing Evaluation: Execution time comparison.
    - Confusion Matrix: Classification performance metrics.
    - Analisis: Analysis of total keypoints and best method comparison.
  
## File Structure
The code expects the images to be structured in folders as follows:

- train_path: Directory containing training images.
- val_path: Directory containing validation images.
- output_file: Excel file where results will be saved.
- output_image_dir: Directory to store the match visualization images.

## Running the Code
1. Mount Google Drive (Optional): If using Google Colab, the following line mounts the drive:
```bash
from google.colab import drive
drive.mount('/content/drive')
```
2. Execute the Main Function: To compare the images and generate results, call the compare_images function with the paths to your image directories and output files:
```bash
compare_images('train_images', 'val_images', 'results.xlsx', 'output_images')
```
3. Result Analysis: The output Excel file will contain multiple sheets with detailed results. Each sheet will display the comparison, keypoint matches, timing, and evaluation metrics for both SIFT and AKAZE methods.

## License
This code is provided under the MIT License. You may use, modify, and distribute it freely, provided that appropriate credit is given.
