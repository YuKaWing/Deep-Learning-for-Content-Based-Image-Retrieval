# Deep-Learning-for-Content-Based-Image-Retrieval
Scripts of my final year project, Deep Learning for Conten-based Image Retrieval of my academic degree.

Some scripts are made but not used in the end.

Build_[Dataset]_Dataset.py is for building the hdf5 dataset

Build_[Dataset]_[Model]_Features.py is for building the hdf5 feature vectors

[Dataset]_[Model].py is for training the model

[Dataset]_[Model]_test.py is for testing the model

[Dataset]_[Model]_search_engine.py is for user to query. "query commands.txt" listed example query commands

[Dataset]_[Model]_search_evaluate.py is for evaluating search engine

Config_[Dataset].py stores the config

ver "_2" is used at last for the Oxford Building Dataset. e.g. oxbuild_images_105K_ResNet50_search_evaluate_2.py

ver "_1" is used at last for the Caltech-256 Dataset. e.g. Caltech256_VGG16_1.py

environment.yml is the exported yml file of used environment
conda env create -f environment.yml
this command should create the environment based on the yml file