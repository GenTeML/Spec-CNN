# Spec-CNN

**The code can be compiled under any local or browser-based Python environment on Mac, Windows, or Linux. Source code and datasets are copyrighted under Creative Commons BY-NC. The latest script versions and corresponding DOIs can be found in [Zenodo](https://zenodo.org/badge/latestdoi/290046588
).**

## Quickstart:
Python files as well as Google Colaboratory Python files are available for execution. There are two basic programs to run: one using the raw Raman test and training spectra and one using the continuous wavelet transform (CWT) processed Raman test and training spectra. 
Training and testing will take ~10-45 minutes depending on the settings used. Results and metrics are displayed at the bottom of your Python or Google Colaboratory environment along with classification uncertainties. Run the program multiple times to get average results as there is natural variability. 

### The Machine Learning Raman Open Dataset (MLROD) used by this code is available at [The Open Repository](https://odr.io/MLROD) ([DOI](https://doi.org/10.48484/PWRB-R137)) and as part of [NASA’s AHED](https://ahed.nasa.gov/). For further information on the software, training, and test datasets, please refer to the Earth and Space Sciences publication: Berlanga, Genesis., Williams, Quentin., & Temiquel, Nathan. (2022). Convolutional Neural Networks as a Tool for Raman Spectral Mineral Classification Under Low Signal, Dusty Mars Conditions. Please forward code questions to the authors. 

There are two test sets within the raw and CWT that can be used: The clean 0% dust cover datasets (the default) and the dusty 50% dust cover datasets. To run one raw set or the other edit the “testin_path” on the corresponding Python script “Classify Test Spectra with Trained CNN” code block, to point to “Data/Raw Data/Labeled Test/” or “Data/Raw Data/Labeled Test/Dusty/”. To run one CWT set or the other edit the “testin_path” on the corresponding Python script “Classify Test Spectra with Trained CNN” code block, to point to “Data/Preprocessed/Continuous Wavelet Transformation/Test Set/Labeled/” or “Data/Preprocessed/Continuous Wavelet Transformation/Test Set/Labeled/Dusty/”.
The scripts can also be directed to the RRUFF test set files used to externally verify CNN performance. Point the directory to “RRUFF_test” instead of “Dusty.”

## Technology used:
Python 3.x as the general programming language
Pandas and Numpy used for data processing and manipulation
SciKit Learn used for traditional models, signal processing (continuous wavelet transform), PCA, Scaling, and splitting data into test and dev sets
Tensorflow for CNN model
Jupyter Notebooks on Google Colaboratory for the development environment

## How to use the programs:
Adding new data
1. Point training or testing data to the corresponding fin_path or testin_path data paths in the CNN Raw and CNN CWT core scripts. 
2. Preprocess the data using DataProcessingBatch.ipynb
This processes all raw data in the data folders.
This program is described below in “What Happens In Preprocessing” 
The DataProcessingBatch program was made to point to "/.../Peaks Only/" and "/.../Continuous Wavelet Transformation/" folders as fout folders. Whatever folder it points to needs to have a subfolder called "/Labeled/" or "/Unlabeled/"

Training and Testing Models

3. Run CNN Raw or CNN CWT for all model training and testing within your local Python environment or Google Colaboratory. These are self contained scripts that will run as is or the user can edit CNN parameters as desired.

4. Upon completion, the confusion matrix, precision, recall, accuracy, and F1 scores are reported at the bottom of each model. Google Colaboratory should take ~10 minutes to execute on the full dataset depending on the settings. Your local Python environment execution times vary from system to system but can take ~45 minutes. 

When in doubt, check all input and output file paths. 

## What Happens In Preprocessing:
1. Columns with non-numeric values are dropped
2. The data set is trimmed to wavenumbers in the range [150,1100]
3. Data shape is standardized - data is grouped in bands of 5 wave numbers and the minimum value is taken: 
eg. all data between 150 and 155 is grouped, the minimum value of those data is used for the new datapoint “150”
This ensures that the data entering downstream processing is standardized and cleans up the noise created by cosmic rays
4. Continuous wavelet transformation (CWT) is performed on the data to smooth, baseline correct, and highlight the peaks in the data
It should be noted that once this is transformed, it is no longer spectral data but a processed form of that data
This pushes some non-peak data into negative values - this allows the peak-finding algorithm to better identify the peaks in the data using signal-to-noise ratio and will improve the speed with which our CNN model converges
5. A peak-finding algorithm using signal to noise ratio (SNR) to identify the peaks in the data
An artifact of CWT performed on Raman spectra with Ricker wavelets is that peaks always appear at the extreme ends of the data - the first and last peaks are dropped before the largest peaks are extracted
6. Two datasets are output from preprocessing: 
A 10-feature dataset in the format [highest peak, second highest peak...fifth highest peak, highest peak relative intensity,...fifth highest peak intensity]
This data is used in all models
The CWT of each spectrum with 190 features (one for every 5 wave numbers between 150 and 1100)
This data is used only in the neural network model: due to the large number of features and significant noise, it is not appropriate for traditional models like SVC or logistic regression

## What is going on in the CNN:
The convolutional neural network can be viewed here and is made up of the following layers:

Layer 1
1-Dimensional Convolution (16 filters, filter size 13)
Batch Normalization
Leaky ReLU activation function
1-Dimensional Maxpooling (pool size 3)

Layer 2
1-Dimensional Convolution (32 filters, filter size 5)
Batch Normalization
Leaky ReLU activation function
1-Dimensional Maxpooling (pool size 2)

Layer 3
1-Dimensional Convolution (64 filters, filter size 3)
Batch Normalization
Leaky ReLU activation function
1-Dimensional Maxpooling (pool size 2)
Flattening - concatenate all channels into one to prepare for the fully connected layers

Layer 4
Dense - a “fully connected layer” or traditional network layer, 2048 nodes (W dot X + b: except with no b - the bias parameter was not included because the following batch normalization would negate it)
Batch normalization
Tanh activation function
55% dropout (to reduce overfitting)

Layer 5
Dense with 8 nodes (to account for the labels that fall in the range [0,7])
Batch Normalization
Softmax activation layer - outputs an array of probabilities (one for each possible label [0,7]) that total to 1
