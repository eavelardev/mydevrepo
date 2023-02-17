# TensorFlow on Google Cloud
https://www.cloudskillsboost.google/course_templates/12

## 0. Introduction to the Course
* ([Video](https://www.youtube.com/watch?v=rDxQ849LGGI) - Mar 10, 2022) ([Slide](https://docs.google.com/presentation/d/1QHxATK_kGbpyXA_9dVH62Z7bwevNxA73)) Introduction

## 1. Introduction to the TensorFlow Ecosystem
* ([Video](https://www.youtube.com/watch?v=N6zm6IoMoH0) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1QKjCw6KmuwBWYpo-Nv58-DUBiLf904JQ)) Introduction to the TensorFlow ecosystem
* ([Video](https://www.youtube.com/watch?v=DyrEEJT47Gs) - Mar 10, 2022) ([Slide](https://docs.google.com/presentation/d/1QhmNgkkK3AswSIOuFpVY-hXzNTYFDSo_)) Introduction to Tensorflow
* ([Video](https://www.youtube.com/watch?v=CLkXK9aJrhs) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1Qldxf2bQ-NKvNzEQNWnuY6wC2Dak0yR_)) TensorFlow API hierarchy
* ([Video](https://www.youtube.com/watch?v=c_83Wxv4NX0) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1QvcLrJZ60WPpzhU-2eOYlhcPi5q5RAmD)) Components of Tensorflow: Tensors and variables

## 2. Design and Build an Input Data Pipeline
* ([Video](https://www.youtube.com/watch?v=ZjC2t06Zexk) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1R-pPjfK-cwM4fX2gCCuDjrKk2LANDsFU)) Introduction
* ([Video](https://www.youtube.com/watch?v=LfzZMRaKQS0) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1R586b5QVwHVZ_m7zhCZ50VZwwDjrHZHC)) An ML recap
* ([Video](https://www.youtube.com/watch?v=U4ISWGuX-3E) - Mar 10, 2022) ([Slide](https://docs.google.com/presentation/d/1RA4nL6rkGbGWK7akXuEv9eglZC9LSrsU)) Training on large datasets with tf.data API
* ([Video](https://www.youtube.com/watch?v=o8xo-IT04Gc) - Mar 10, 2022) ([Slide](https://docs.google.com/presentation/d/1RUxxpq-F9vUJIIvxsd1FHwfEq0u6yrr9)) Working in-memory and with files
* ([Video](https://www.youtube.com/watch?v=7LqLueX4LmQ) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1R_HBJK_V_wXfGNWkrvS2oIzaCuIzsk7_)) Getting the data ready for model training
    * [tf.feature_column](https://www.tensorflow.org/api_docs/python/tf/feature_column) is deprecated.
    * [Working with preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers)
    * [Classify structured data using Keras preprocessing layers](https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers)
    * [Migrate tf.feature_columns to Keras preprocessing layers](https://www.tensorflow.org/guide/migrate/migrating_feature_columns)
    * [An Introduction to Keras Preprocessing Layers](https://blog.tensorflow.org/2021/11/an-introduction-to-keras-preprocessing.html) (November 24, 2021)
* ([Video](https://www.youtube.com/watch?v=WbxkYCDGaYw) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1Rhq4HkGOeQzSBrXgP_9Y1W38d7lSJhyB)) Embeddings
    * [tf.feature_column.embedding_column](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column) is deprecated.
    * [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)
    * [Word embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
* ([Video](https://www.youtube.com/watch?v=ByS99Z_Gd6M) - Mar 3, 2022) Lab intro: TensorFlow Dataset API
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438560/labs/318947)) `TensorFlow Dataset API`
    * training-data-analyst/courses/machine_learning/deepdive2/introduction_to_tensorflow/
        * labs/[2_dataset_api.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/2_dataset_api.ipynb)
        * solutions/[2_dataset_api.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/solutions/2_dataset_api.ipynb)
* ([Video](https://www.youtube.com/watch?v=SLFeLWONXfw) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1SH2UO0FawwEbzQ3O60PyPn2aLxNo9PbZ)) Scaling data processing with tf.data and Keras preprocessing layers
* ([Video](https://www.youtube.com/watch?v=jZ-EbMj_MsU) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1ShKso8ImE2tELJSe_mWl-dckWINEWgJ3)) Lab intro: Classifying structured data using Keras preprocessing layers
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438560/labs/318950)) `Classifying Structured Data using Keras Preprocessing Layers`
    * training-data-analyst/courses/machine_learning/deepdive2/introduction_to_tensorflow/
        * labs/[preprocessing_layers.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/preprocessing_layers.ipynb)
        * solutions/[preprocessing_layers.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/solutions/preprocessing_layers.ipynb)

## 3. Building Neural Networks with the TensorFlow and Keras API
* ([Video](https://www.youtube.com/watch?v=2rQLLREROd0) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1Su9_0cEtmaWKJaxMGWI-p8vbQG_CAwFZ)) Introduction
* ([Video](https://www.youtube.com/watch?v=OHUh5EUdD74) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1T28_JnYacO0kj-R9A6xAEP00o2_TJUZ8)) Activation functions
* ([Video](https://www.youtube.com/watch?v=bFYED2RZdPY) - Mar 10, 2022) ([Slide](https://docs.google.com/presentation/d/1T5BBUOXsJ53vVYn26bnacOhh9WMQG3Yc)) Training neural networks with TensorFlow 2 and the Keras Sequential API
* ([Video](https://www.youtube.com/watch?v=q0REuGXftaA) - Mar 10, 2022) ([Slide](https://docs.google.com/presentation/d/1T9N7_3tdV8peeeieNC_T9IvFJVwCMzCI)) Serving models in the cloud (AI platform)
* ([Video](https://www.youtube.com/watch?v=B4VH0e3t0qA) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1TAXvV7bEhcKU57TmBFIygSLBBmCbBIUz)) Lab intro: Introducing the Keras Sequential API on Vertex AI Platform (tf.feature_column)
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438560/labs/318958)) `Introducing the Keras Sequential API on Vertex AI Platform`
    * training-data-analyst/courses/machine_learning/deepdive2/introduction_to_tensorflow/
        * labs/[3_keras_sequential_api.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/3_keras_sequential_api.ipynb) (tf.feature_column)
        * solutions/[3_keras_sequential_api.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/solutions/3_keras_sequential_api.ipynb) (tf.feature_column)
* ([Video](https://www.youtube.com/watch?v=KJk-ADypwO8) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1TBBgI1i9N32AFtIkGhXlESTAvGo5zlAc)) Training neural networks with TensorFlow 2 and the Keras Functional API
* ([Video](https://www.youtube.com/watch?v=mPNsnojiWvk) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1TTk7vji2AxHfOO59xKXI7xpZTGFaPS3g)) Lab intro: Build a DNN using the Keras Functional API on Vertex AI Platform (tf.feature_column)
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438560/labs/318961)) `Build a DNN using the Keras Functional API`
    * training-data-analyst/courses/machine_learning/deepdive2/art_and_science_of_ml/
        * labs/[neural_network.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/art_and_science_of_ml/labs/neural_network.ipynb) (tf.feature_column)
        * solutions/[neural_network.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/art_and_science_of_ml/solutions/neural_network.ipynb) (tf.feature_column)
    * training-data-analyst/courses/machine_learning/deepdive2/introduction_to_tensorflow/
        * labs/[4_keras_functional_api.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/4_keras_functional_api.ipynb)
        * solutions/[4_keras_functional_api.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/solutions/4_keras_functional_api.ipynb)
* ([Video](https://www.youtube.com/watch?v=E-a0zZYOLDo) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1TRZElClZ0BtIi1t89Vz2n0R5MiTcpaQk)) Model subclassing
* ([Video](https://www.youtube.com/watch?v=AeCYZkRpF-E) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1TWGuXo4o1FU2Jx6VMuC24vUqlUPoFNAb)) (Optional) Lab intro: Making new layers and models via subclassing
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438560/labs/318964)) `(Optional) Making New Layers and Models via Subclassing`
    * training-data-analyst/courses/machine_learning/deepdive2/introduction_to_tensorflow/
        * labs/[custom_layers_and_models.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/custom_layers_and_models.ipynb)
        * solutions/[custom_layers_and_models.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/solutions/custom_layers_and_models.ipynb)
* ([Video](https://www.youtube.com/watch?v=7_Y-68QliVo) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1TeLLh902DDSKW58Q2lregkz-vTipGryP)) Regularization basics
* ([Video](https://www.youtube.com/watch?v=aQ_LW3eWMMM) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1TiOfMw-1PQUU2x9bkFbNDXeK5hDWVVg9)) How can we meaure model complexity: L1 vs. L2 Regularization

## 4. Training at Scale with Vertex AI
* ([Video](https://www.youtube.com/watch?v=ta_3sUir94A) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1U-KMf3OAhLl2iaTNRN8kxX2Pp29Xn2-v)) Introduction
* ([Video](https://www.youtube.com/watch?v=gx-vJzACbqk) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1UBleTHQ25t9l6YVojlvWjz6cXhndzhti)) Training at scale with Vertex AI
* ([Video](https://www.youtube.com/watch?v=eZpVr1bjlbI) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1U5d6jCypA-BMySuA7F8gML4FPkJY1Dty)) Lab intro: Training at scale with the Vertex AI Training Service
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438560/labs/318972)) `Training at Scale with Vertex AI Training Service`
    * training-data-analyst/courses/machine_learning/deepdive2/introduction_to_tensorflow/
        * labs/[1_training_at_scale_vertex.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/1_training_at_scale_vertex.ipynb)
        * solutions/[1_training_at_scale_vertex.ipynb](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/solutions/1_training_at_scale_vertex.ipynb)

## 5. Summary
* ([Document](https://drive.google.com/open?id=1Tp2ZcodY0G0zHNOvIhAql4Q4SSWtACxi)) Summary
* ([Document](https://drive.google.com/open?id=1TprYxd03b5xJOXXBqIpB_SPEPOA8wGPZ)) Resource: All quiz questions
* ([Document](https://drive.google.com/open?id=1Tqt0fb2jGbfitc33mVy4aZZfScIrzLX3)) Resource: All readings
* ([Document](https://drive.google.com/open?id=1TuEeQsN3RwNYZmAHbUY3A7EyzwmifHrk)) Resource: All slides