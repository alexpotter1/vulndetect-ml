# vulndetect-ml
Simple ML project to detect and classify vulnerable Java code

## How to get started?
Ensure you have a working virtualenv/anaconda environment to install packages to (or you don't mind installing globally)

#### Install TensorFlow
* `conda install tensorflow` is preferred, will install NVIDIA CUDA/Intel MKL-DNN acceleration dependencies automagically

* `pip install tensorflow` if you don't have conda

#### Install TensorFlow Datasets module
* `pip install tensorflow_datasets`
  
#### Begin training with simple_train.py
* **Note:** The first time will likely result in an error. This is normal for now (until I fix imports with `importlib`). This is due to registering the custom NIST Juliet dataset with TensorFlow Datasets.

* The program will download the dataset from the internet, extract it and begin building the encoder. Don't worry if it looks like it has frozen after it finishes downloading. It has not. You will eventually see it processing the encoded samples once it has built the vocabulary from the corpus

* You may want to train on a GPU-accelerated machine as it can take a while - ensure you have CUDA installed for this (or `conda install`...)
  
#### Run the prediction engine with predict.py
* For now, put some random Java sample into a folder called `new/`. Name it something, and change the `input_path` variable (predict.py:11) to match your new filename.
* Run `predict.py`
* It should load the saved encoder, model and give you a prediction with a confidence percentage
* Not always accurate, but not bad

#### Data visualisation
* `simple_train.py` saves two files useful for embedding projection visualisation:
  * `vecs.tsv` - saved embedding vectors
  * `meta.tsv` - associate metadata for each embedding vector
* You can load this with [projector.tensorflow.org](http://projector.tensorflow.org) for PCA/t-SNE etc
