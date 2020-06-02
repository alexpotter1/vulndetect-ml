# vulndetect-ml
Simple ML project to detect and classify vulnerable Java code

Check out [detect.alexpotter.net](https://detect.alexpotter.net) for a proof-of-concept!

## How to get started?
Ensure you have a working virtualenv/anaconda environment to install packages to (or you don't mind installing globally)

#### Install TensorFlow
* `conda install tensorflow` is preferred, will install NVIDIA CUDA/Intel MKL-DNN acceleration dependencies automagically

* `pip install tensorflow` if you don't have conda

#### Install TensorFlow Datasets module
* `pip install tensorflow_datasets`
  
#### Begin training with simple_train.py
* The program will download the dataset from the internet, extract it and begin building the encoder. Don't worry if it looks like it has frozen after it finishes downloading. It has not. You will eventually see it processing the encoded samples once it has built the vocabulary from the corpus

* You may want to train on a GPU-accelerated machine as it can take a while - ensure you have CUDA installed for this (or `conda install`...)

#### Data visualisation
* `simple_train.py` saves two files useful for embedding projection visualisation:
  * `vecs.tsv` - saved embedding vectors
  * `meta.tsv` - associate metadata for each embedding vector
* You can load this with [projector.tensorflow.org](http://projector.tensorflow.org) for PCA/t-SNE etc

## Prediction
Two ways to do this:
* Run `predict.py` with a Java file in a folder called `new/`
* Use the new React frontend to submit a file and check it

### Prediction - React web app (recommended)
A hosted version is available for testing [here](https://detect.alexpotter.net)

To run this locally, you will need (in addition `tensorflow`, `numpy`, all of the other modules for training):
* `python3`
* `flask`, `flask_cors` : `python3 -m pip install flask flask_cors python_dotenv`
* `node` with `npm`: `brew install node`, `sudo apt install nodejs npm`, https://nodejs.org/en/download/current/

1. Open a Terminal and `cd` to `web/`
2. Run `npm install`
3. Run `npm start` to start the frontend
4. Open another terminal and `cd` to `web/api`
5. Run `npm run link-api` to symlink the required Python modules (only need to run this for the first time)
6. Run `npm run start-api` to start the Python backend server
7. The React web app should be accessible on `http://localhost:3000` with a web browser.
  * If it does not connect for any reason, check your firewall configuration and check that there are no errors in the Terminal
  
This will start the app in **development mode**
