# Professional Machine Learning Engineer Sample Questions (20 questions)
# https://docs.google.com/forms/d/e/1FAIpQLSeYmkCANE81qSBqLW0g2X7RoskBX9yGYQu-m1TtsjMvHabGqg/viewform

# Cloud OnAir: Machine Learning Certification (4 questions)
# https://www.youtube.com/watch?v=Dd-RZICTWho

# Certification Study Group - Professional Machine Learning Engineer (8 questions)
# https://drive.google.com/drive/folders/1TWAYh4cBrYYwBZgCO2_u33aPM3bF74Q5

# Udemy - Google Cloud Machine Learning Engineer Certification Prep (50 questions)
# https://www.udemy.com/course/google-cloud-machine-learning-engineer-certification-prep/

# Packt - Journey to Become a Google Cloud Machine Learning Engineer (2022) - Dr. Logan Song (30 questions)
# https://www.packtpub.com/product/journey-to-become-a-google-cloud-machine-learning-engineer/9781803233727

# whizlabs - Google Cloud Certified Professional Machine Learning Engineer - Practice Test 1 (50 questions)
# https://www.whizlabs.com/learn/course/professional-machine-learning-engineer/394/quiz/25152

# whizlabs - Google Cloud Certified Professional Machine Learning Engineer - Practice Test 2 (50 questions)
# https://www.whizlabs.com/learn/course/professional-machine-learning-engineer/394/quiz/25153

# Google GCP-PMLE Certification Exam Sample Questions (10 questions) (AI Platform)
# https://www.vmexam.com/google/google-gcp-pmle-certification-exam-sample-questions

# Google Professional Machine Learning Engineer Practice Exam (30 questions) (AI Platform)
# https://gcp-examquestions.com/course/google-professional-machine-learning-engineer-practice-exam/

# Google Professional Machine Learning Engineer Exam Actual Questions (AI Platform)
# https://www.examtopics.com/exams/google/professional-machine-learning-engineer/view/

question_format = [
    {
        'question':
        "",
        'tags': [0],
        'options': {
            'A': "",
            'B': "",
            'C': "",
            'D': ""
        },
        'answers': [],
        'explanation':
        "",
        'references': []
    },
]

questions = [
    # Professional Machine Learning Engineer Sample Questions (20 questions)
    {
        'question':
        "You are developing a proof of concept for a real-time fraud detection model. After undersampling the training set to achieve a 50% fraud rate, you train and tune a tree classifier using area under the curve (AUC) as the metric, and then calibrate the model. You need to share metrics that represent your model’s effectiveness with business stakeholders in a way that is easily interpreted. Which approach should you take?",
        'tags': [1, 'sample'],
        'options': {
            'A': "Calculate the AUC on the holdout dataset at a classification threshold of 0.5, and report true positive rate, false positive rate, and false negative rate.",
            'B': "Undersample the minority class to achieve a 50% fraud rate in the holdout set. Plot the confusion matrix at a classification threshold of 0.5, and report precision and recall.",
            'C': "Select all transactions in the holdout dataset. Plot the area under the receiver operating characteristic curve (AUC ROC), and report the F1 score for all available thresholds.",
            'D': "Select all transactions in the holdout dataset. Plot the precision-recall curve with associated average precision, and report the true positive rate, false positive rate, and false negative rate for all available thresholds."
        },
        'answers': ['D'],
        'explanation':

        "* The precision-recall curve is an appropriate metric for imbalanced classification when the output can be set using different thresholds. Presenting the precision-recall curve together with the mentioned rates provides business stakeholders with all the information necessary to evaluate model performance.\n"

        "* You need business directions about the cost of misclassification to define the optimal threshold for both balanced and imbalanced classification\n"

        "* The holdout dataset needs to represent real-world transactions to have a meaningful model evaluation, and you should never change its distribution.\n"

        "* Classes in the holdout dataset are not balanced, so the ROC curve is not appropriate; also, neither F1 score nor ROC curve is recommended for communicating to business stakeholders. The F1 score aggregates precision and recall, but it is important to look at each metric separately to evaluate the model’s performance when the cost of misclassification is highly unbalanced between labels.",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc',
            'https://colab.research.google.com/github/Fraud-Detection-Handbook/fraud-detection-handbook/blob/main/Chapter_4_PerformanceMetrics/ThresholdBased.ipynb',
            'https://colab.research.google.com/github/Fraud-Detection-Handbook/fraud-detection-handbook/blob/main/Chapter_4_PerformanceMetrics/ThresholdFree.ipynb'
        ]
    },
    {
        'question':
        "Your organization’s marketing team wants to send biweekly scheduled emails to customers that are expected to spend above a variable threshold. This is the first machine learning (ML) use case for the marketing team, and you have been tasked with the implementation. After setting up a new Google Cloud project, you use Vertex AI Workbench to develop model training and batch inference with an XGBoost model on the transactional data stored in Cloud Storage. You want to automate the end-to-end pipeline that will securely provide the predictions to the marketing team, while minimizing cost and code maintenance. What should you do?",
        'tags': [2, 'sample'],
        'options': {
            'A': "Create a scheduled pipeline on Vertex AI Pipelines that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending.",
            'B': "Create a scheduled pipeline on Cloud Composer that accesses the data from Cloud Storage, copies the data to BigQuery, uses BigQuery ML to perform training and batch prediction, and outputs a table in BigQuery with customer emails and expected spending.",
            'C': "Create a scheduled notebook on Vertex AI Workbench that accesses the data from Cloud Storage, performs training and batch prediction on the managed notebook instance, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending.",
            'D': "Create a scheduled pipeline on Cloud Composer that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and sends an email to the marketing team’s Gmail group email with an attachment that contains an encrypted list of all customer emails and expected spending."
        },
        'answers': ['A'],
        'explanation':
        "* Vertex AI Pipelines and Cloud Storage are cost-effective and secure solutions. The solution requires the least number of code interactions because the marketing team can update the pipeline and schedule parameters from the Google Cloud console.\n"
        
        "* Cloud Composer is not a cost-efficient solution for one pipeline because its environment is always active. In addition, using BigQuery is not the most cost-effective solution.\n"
        
        "* The marketing team would have to enter the Vertex AI Workbench instance to update a pipeline parameter, which does not minimize code interactions.\n"
        
        "* Cloud Composer is not a cost-efficient solution for one pipeline because its environment is always active. Also, using email to send personally identifiable information (PII) is not a recommended approach.",
        
        'references': [
            'https://cloud.google.com/storage/docs/encryption',
            'https://cloud.google.com/vertex-ai/docs/pipelines/run-pipeline',
            'https://cloud.google.com/vertex-ai/docs/workbench/managed/schedule-managed-notebooks-run-quickstart',
            'https://cloud.google.com/architecture/setting-up-mlops-with-composer-and-mlflow'
        ]
    },
    {
        'question':
        "You have developed a very large network in TensorFlow Keras that is expected to train for multiple days. The model uses only built-in TensorFlow operations to perform training with high-precision arithmetic. You want to update the code to run distributed training using tf.distribute.Strategy and configure a corresponding machine instance in Compute Engine to minimize training time. What should you do?",
        'tags': [3, 'sample'],
        'options': {
            'A': "Select an instance with an attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add MirroredStrategy to the code, and create the model in the strategy’s scope with batch size dependent on the number of replicas.",
            'B': "Create an instance group with one instance with attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add TF_CONFIG and MultiWorkerMirroredStrategy to the code, create the model in the strategy’s scope, and set up data autosharing.",
            'C': " Create a TPU virtual machine, and gradually scale up the machine type until the optimal execution time is reached. Add TPU initialization at the start of the program, define a distributed TPUStrategy, and create the model in the strategy’s scope with batch size and training steps dependent on the number of TPUs.",
            'D': "Create a TPU node, and gradually scale up the machine type until the optimal execution time is reached. Add TPU initialization at the start of the program, define a distributed TPUStrategy, and create the model in the strategy’s scope with batch size and training steps dependent on the number of TPUs."
        },
        'answers': ['B'],
        'explanation':

        "* GPUs are the correct hardware for deep learning training with high-precision training, and distributing training with multiple instances will allow maximum flexibility in fine-tuning the accelerator selection to minimize execution time. Note that one worker could still be the best setting if the overhead of synchronizing the gradients across machines is too high, in which case this approach will be equivalent to MirroredStrategy.\n"
        
        "* It is suboptimal in minimizing execution time for model training. MirroredStrategy only supports multiple GPUs on one instance, which may not be as performant as running on multiple instances.\n"
        
        "* TPUs are not recommended for workloads that require high-precision arithmetic, and are recommended for models that train for weeks or months.\n"
        
        "* TPUs are not recommended for workloads that require high-precision arithmetic, and are recommended for models that train for weeks or months. Also, TPU nodes are not recommended unless required by the application.",

        'references': [
            'https://cloud.google.com/tpu/docs/intro-to-tpu#when_to_use_tpus',
            'https://www.tensorflow.org/guide/distributed_training',
            'https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl'
        ]
    },
    {
        'question':
        "You developed a tree model based on an extensive feature set of user behavioral data. The model has been in production for 6 months. New regulations were just introduced that require anonymizing personally identifiable information (PII), which you have identified in your feature set using the Cloud Data Loss Prevention API. You want to update your model pipeline to adhere to the new regulations while minimizing a reduction in model performance. What should you do?",
        'tags': [4, 'sample'],
        'options': {
            'A': "Redact the features containing PII data, and train the model from scratch.",
            'B': "Mask the features containing PII data, and tune the model from the last checkpoint.",
            'C': "Use key-based hashes to tokenize the features containing PII data, and train the model from scratch.",
            'D': "Use deterministic encryption to tokenize the features containing PII data, and tune the model from the last checkpoint."
        },
        'answers': ['C'],
        'explanation':

        "* Hashing is an irreversible transformation that ensures anonymization and does not lead to an expected drop in model performance because you keep the same feature set while enforcing referential integrity.\n"
        
        "* Removing features from the model does not keep referential integrity by maintaining the original relationship between records, and is likely to cause a drop in performance.\n"
        
        "* Masking does not enforce referential integrity, and a drop in model performance may happen. Also, tuning the existing model is not recommended because the model training on the original dataset may have memorized sensitive information.\n"
        
        "* Deterministic encryption is reversible, and anonymization requires irreversibility. Also, tuning the existing model is not recommended because the model training on the original dataset may have memorized sensitive information.",

        'references': [
            'https://cloud.google.com/dlp/docs/transformations-reference#transformation_methods',
            'https://cloud.google.com/dlp/docs/deidentify-sensitive-data',
            'https://cloud.google.com/blog/products/identity-security/next-onair20-security-week-session-guide',
            'https://cloud.google.com/dlp/docs/creating-job-triggers'
        ]
    },
    {
        'question':
        "You set up a Vertex AI Workbench instance with a TensorFlow Enterprise environment to perform exploratory data analysis for a new use case. Your training and evaluation datasets are stored in multiple partitioned CSV files in Cloud Storage. You want to use TensorFlow Data Validation (TFDV) to explore problems in your data before model tuning. You want to fix these problems as quickly as possible. What should you do?",
        'tags': [5, 'sample'],
        'options': {
            'A': "1. Use TFDV to generate statistics, and use Pandas to infer the schema for the training dataset that has been loaded from Cloud Storage. 2. Visualize both statistics and schema, and manually fix anomalies in the dataset’s schema and values.",
            'B': "1. Use TFDV to generate statistics and infer the schema for the training and evaluation datasets that have been loaded from Cloud Storage by using URI. 2. Visualize statistics for both datasets simultaneously to fix the datasets’ values, and fix the training dataset’s schema after displaying it together with anomalies in the evaluation dataset.",
            'C': "1. Use TFDV to generate statistics, and use Pandas to infer the schema for the training dataset that has been loaded from Cloud Storage. 2. Use TFRecordWriter to convert the training dataset into a TFRecord. 3. Visualize both statistics and schema, and manually fix anomalies in the dataset’s schema and values.",
            'D': "1. Use TFDV to generate statistics and infer the schema for the training and evaluation datasets that have been loaded with Pandas. 2. Use TFRecordWriter to convert the training and evaluation datasets into TFRecords. 3. Visualize statistics for both datasets simultaneously to fix the datasets’ values, and fix the training dataset’s schema after displaying it together with anomalies in the evaluation dataset."
        },
        'answers': ['B'],
        'explanation':
        
        "* It takes the minimum number of steps to correctly fix problems in the data with TFDV before model tuning. This process involves installing tensorflow_data_validation, loading the training and evaluation datasets directly from Cloud Storage, and fixing schema and values for both. Note that the schema is only stored for the training set because it is expected to match at evaluation.\n"
        
        "* You also need to use the evaluation dataset for analysis. If the features do not belong to approximately the same range as the training dataset, the accuracy of the model will be affected.\n"
        
        "* Transforming into TFRecord is an unnecessary step. Also, you need to use the evaluation dataset for analysis. If the features do not belong to approximately the same range as the training dataset, the accuracy of the model will be affected.\n"
        
        "* Transforming into TFRecord is an unnecessary step.",

        'references': [
            'https://www.tensorflow.org/tfx/guide/tfdv',
            'https://cloud.google.com/tensorflow-enterprise/docs/overview',
            'https://cloud.google.com/architecture/ml-modeling-monitoring-analyzing-training-server-skew-in-ai-platform-prediction-with-tfdv'
        ]
    },
    {
        'question':
        "You have developed a simple feedforward network on a very wide dataset. You trained the model with mini-batch gradient descent and L1 regularization. During training, you noticed the loss steadily decreasing before moving back to the top at a very sharp angle and starting to oscillate. You want to fix this behavior with minimal changes to the model. What should you do?",
        'tags': [6, 'sample'],
        'options': {
            'A': "Shuffle the data before training, and iteratively adjust the batch size until the loss improves.",
            'B': "Explore the feature set to remove NaNs and clip any noisy outliers. Shuffle the data before retraining.",
            'C': "Switch from L1 to L2 regularization, and iteratively adjust the L2 penalty until the loss improves.",
            'D': "Adjust the learning rate to exponentially decay with a larger decrease at the step where the loss jumped, and iteratively adjust the initial learning rate until the loss improves."
        },
        'answers': ['B'],
        'explanation':
        
        "* A large increase in loss is typically caused by anomalous values in the input data that cause NaN traps or exploding gradients.\n"
        
        "* Divergence due to repetitive behavior in the data typically shows a loss that starts oscillating after some steps but does not jump back to the top.\n"
        
        "* L2 is not clearly a better solution than L1 regularization for wide models. L1 helps with sparsity, and L2 helps with collinearity.\n"
        
        "* A learning rate schedule that is not tuned typically shows a loss that starts oscillating after some steps but does not jump back to the top.",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/representation/cleaning-data',
            'https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic',
            'https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/l1-regularization'
        ]
    },
    {
        'question':
        "You trained a neural network on a small normalized wide dataset. The model performs well without overfitting, but you want to improve how the model pipeline processes the features because they are not all expected to be relevant for the prediction. You want to implement changes that minimize model complexity while maintaining or improving the model’s offline performance. What should you do?",
        'tags': [7, 'sample'],
        'options': {
            'A': "Keep the original feature set, and add L1 regularization to the loss function.",
            'B': "Use principal component analysis (PCA), and select the first n components that explain 99% of the variance.",
            'C': "Perform correlation analysis. Remove features that are highly correlated to one another and features that are not correlated to the target.",
            'D': "Ensure that categorical features are one-hot encoded and that continuous variables are binned, and create feature crosses for a subset of relevant features."
        },
        'answers': ['C'],
        'explanation':
        
        "* Removing irrelevant features reduces model complexity and is expected to boost performance by removing noise.\n"
        
        "* Keep the original feature set ..., although the approach lets you reduce RAM requirements by pushing the weights for meaningless features to 0, regularization tends to cause the training error to increase. Consequently, the model performance is expected to decrease.\n"
        
        "* PCA is an unsupervised approach, and it is a valid method of feature selection only if the most important variables are the ones that also have the most variation. This is usually not true, and disregarding the last few components is likely to decrease model performance.\n"
        
        "* Ensure that categorical features ..., can make the model converge faster but it increases model RAM requirements, and it is not expected to boost model performance because neural networks inherently learn feature crosses.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/feature-crosses/video-lecture',
            'https://cloud.google.com/blog/products/ai-machine-learning/building-ml-models-with-eda-feature-selection',
            'https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/l1-regularization'
        ]
    },
    {
        'question':
        "You trained a model in a Vertex AI Workbench notebook that has good validation RMSE. You defined 20 parameters with the associated search spaces that you plan to use for model tuning. You want to use a tuning approach that maximizes tuning job speed. You also want to optimize cost, reproducibility, model performance, and scalability where possible if they do not affect speed. What should you do?",
        'tags': [8, 'sample'],
        'options': {
            'A': "Set up a cell to run a hyperparameter tuning job using Vertex AI Vizier with val_rmse specified as the metric in the study configuration.",
            'B': " Using a dedicated Python library such as Hyperopt or Optuna, configure a cell to run a local hyperparameter tuning job with Bayesian optimization.",
            'C': "Refactor the notebook into a parametrized and dockerized Python script, and push it to Container Registry. Use the UI to set up a hyperparameter tuning job in Vertex AI. Use the created image and include Grid Search as an algorithm.",
            'D': "Refactor the notebook into a parametrized and dockerized Python script, and push it to Container Registry. Use the command line to set up a hyperparameter tuning job in Vertex AI. Use the created image and include Random Search as an algorithm where maximum trial count is equal to parallel trial count."
        },
        'answers': ['D'],
        'explanation':
        "* Random Search can limit the search iterations on time and parallelize all trials so that the execution time of the tuning job corresponds to the longest training produced by your hyperparameter combination. This approach also optimizes for the other mentioned metrics.\n"
        
        "* Vertex AI Vizier should be used for systems that do not have a known objective function or are too costly to evaluate using the objective function. Neither applies to the specified use case. Vizier requires sequential trials and does not optimize for cost or tuning time.\n"
        
        "* Bayesian optimization can converge in fewer iterations than the other algorithms but not necessarily in a faster time because trials are dependent and thus require sequentiality. Also, running tuning locally does not optimize for reproducibility and scalability.\n"
        
        "* Grid Search is a brute-force approach and it is not feasible to fully parallelize. Because you need to try all hyperparameter combinations, that is an exponential number of trials with respect to the number of hyperparameters, Grid Search is inefficient for high spaces in time, cost, and computing power.",

        'references': [
            'https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview',
            'https://cloud.google.com/vertex-ai/docs/vizier/overview',
            'https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-1.0.7/google_cloud_pipeline_components.v1.hyperparameter_tuning_job.html#google_cloud_pipeline_components.v1.hyperparameter_tuning_job.HyperparameterTuningJobRunOp'
        ]
    },
    {
        'question':
        "You trained a deep model for a regression task. The model predicts the expected sale price for a house based on features that are not guaranteed to be independent. You want to evaluate your model by defining a baseline approach and selecting an evaluation metric for comparison that detects high variance in the model. What should you do?",
        'tags': [9, 'sample'],
        'options': {
            'A': "Use a heuristic that predicts the mean value as the baseline, and compare the trained model’s mean absolute error against the baseline.",
            'B': "Use a linear model trained on the most predictive features as the baseline, and compare the trained model’s root mean squared error against the baseline.",
            'C': "Determine the maximum acceptable mean absolute percentage error (MAPE) as the baseline, and compare the model’s MAPE against the baseline.",
            'D': "Use a simple neural network with one fully connected hidden layer as the baseline, and compare the trained model’s mean squared error against the baseline."
        },
        'answers': ['D'],
        'explanation':
        
        "* A one-layer neural network can handle collinearity and is a good baseline. The mean square error is a good metric because it gives more weight to errors with larger absolute values than to errors with smaller absolute values.\n"
        
        "* Always predicting the mean value is not expected to be a strong baseline; house prices could assume a wide range of values. Also, mean absolute error is not the best metric to detect variance because it gives the same weight to all errors.\n"
        
        "* A linear model is not expected to perform well with multicollinearity. Also, root mean squared error does not penalize high variance as much as mean squared error because the root operation reduces the importance of higher values.\n"
        
        "* While defining a threshold for acceptable performance is a good practice for blessing models, a baseline should aim to test statistically a model’s ability to learn by comparing it to a less complex data-driven approach. Also, this approach does not detect high variance in the model.",

        'references': [
            'https://developers.google.com/machine-learning/testing-debugging/common/model-errors#establish-a-baseline',
            'https://cloud.google.com/automl-tables/docs/evaluate#evaluation_metrics_for_regression_models',
            'https://developers.google.com/machine-learning/glossary#baseline'
        ]
    },
    {
        'question':
        "You designed a 5-billion-parameter language model in TensorFlow Keras that used autotuned tf.data to load the data in memory. You created a distributed training job in Vertex AI with tf.distribute.MirroredStrategy, and set the large_model_v100 machine for the primary instance. The training job fails with the following error:\n\n" 
        
        "“The replica 0 ran out of memory with a non-zero status of 9.”\n\n" 
        
        "You want to fix this error without vertically increasing the memory of the replicas. What should you do?",
        'tags': [10, 'sample'],
        'options': {
            'A': "Keep MirroredStrategy. Increase the number of attached V100 accelerators until the memory error is resolved.",
            'B': "Switch to ParameterServerStrategy, and add a parameter server worker pool with large_model_v100 instance type.",
            'C': "Switch to tf.distribute.MultiWorkerMirroredStrategy with Reduction Server. Increase the number of workers until the memory error is resolved.",
            'D': "Switch to a custom distribution strategy that uses TF_CONFIG to equally split model layers between workers. Increase the number of workers until the memory error is resolved."
        },
        'answers': ['D'],
        'explanation':
        
        "* This is an example of a model-parallel approach that splits the model between workers. You can use TensorFlow Mesh to implement this. This approach is expected to fix the error because the memory issues in the primary replica are caused by the size of the model itself.\n"
        
        "* MirroredStrategy is a data-parallel approach. This approach is not expected to fix the error because the memory issues in the primary replica are caused by the size of the model itself.\n"
        
        "* The parameter server alleviates some workload from the primary replica by coordinating the shared model state between the workers, but it still requires the whole model to be shared with workers. This approach is not expected to fix the error because the memory issues in the primary replica are caused by the size of the model itself.\n"
        
        "* MultiWorkerMirroredStrategy is a data-parallel approach. This approach is not expected to fix the error because the memory issues in the primary replica are caused by the size of the model itself. Reduction Server increases throughput and reduces latency of communication, but it does not help with memory issues.",

        'references': [
            'https://cloud.google.com/ai-platform/training/docs/training-at-scale',
            'https://cloud.google.com/ai-platform/training/docs/machine-types#scale_tiers',
            'https://cloud.google.com/vertex-ai/docs/training/distributed-training',
            'https://cloud.google.com/ai-platform/training/docs/overview#distributed_training_structure',
            'https://github.com/tensorflow/mesh'
        ]
    },
    {
        'question':
        "You need to develop an online model prediction service that accesses pre-computed near-real-time features and returns a customer churn probability value. The features are saved in BigQuery and updated hourly using a scheduled query. You want this service to be low latency and scalable and require minimal maintenance. What should you do?",
        'tags': [11, 'sample'],
        'options': {
            'A': "1. Configure a Cloud Function that exports features from BigQuery to Memorystore. 2. Use Memorystore to perform feature lookup. Deploy the model as a custom prediction endpoint in Vertex AI, and enable automatic scaling.",
            'B': "1. Configure a Cloud Function that exports features from BigQuery to Memorystore. 2. Use a custom container on Google Kubernetes Engine to deploy a service that performs feature lookup from Memorystore and performs inference with an in-memory model.",
            'C': "1. Configure a Cloud Function that exports features from BigQuery to Vertex AI Feature Store. 2. Use the online service API from Vertex AI Feature Store to perform feature lookup. Deploy the model as a custom prediction endpoint in Vertex AI, and enable automatic scaling.",
            'D': "1. Configure a Cloud Function that exports features from BigQuery to Vertex AI Feature Store. 2. Use a custom container on Google Kubernetes Engine to deploy a service that performs feature lookup from Vertex AI Feature Store’s online serving API and performs inference with an in-memory model."
        },
        'answers': ['A'],
        'explanation':

        "* This approach creates a fully managed autoscalable service that minimizes maintenance while providing low latency with the use of Memorystore.\n"
        
        "* Feature lookup and model inference can be performed in Cloud Function, and using Google Kubernetes Engine increases maintenance.\n"
        
        "* Vertex AI Feature Store is not as low-latency as Memorystore.\n"
        
        "* Feature lookup and model inference can be performed in Cloud Function, and using Google Kubernetes Engine increases maintenance. Also, Vertex AI Feature Store is not as low-latency as Memorystore",

        'references': [
            'https://cloud.google.com/architecture/ml-on-gcp-best-practices#model-deployment-and-serving',
            'https://cloud.google.com/vertex-ai/docs/featurestore/overview#benefits',
            'https://cloud.google.com/memorystore/docs/redis/redis-overview'
        ]
    },
    {
        'question':
        "You are logged into the Vertex AI Pipeline UI and noticed that an automated production TensorFlow training pipeline finished three hours earlier than a typical run. You do not have access to production data for security reasons, but you have verified that no alert was logged in any of the ML system’s monitoring systems and that the pipeline code has not been updated recently. You want to debug the pipeline as quickly as possible so you can determine whether to deploy the trained model. What should you do?",
        'tags': [12, 'sample'],
        'options': {
            'A': "Navigate to Vertex AI Pipelines, and open Vertex AI TensorBoard. Check whether the training regime and metrics converge.",
            'B': "Access the Pipeline run analysis pane from Vertex AI Pipelines, and check whether the input configuration and pipeline steps have the expected values.",
            'C': "Determine the trained model’s location from the pipeline’s metadata in Vertex ML Metadata, and compare the trained model’s size to the previous model.",
            'D': "Request access to production systems. Get the training data’s location from the pipeline’s metadata in Vertex ML Metadata, and compare data volumes of the current run to the previous run."
        },
        'answers': ['A'],
        'explanation':
        
        "* TensorBoard provides a compact and complete overview of training metrics such as loss and accuracy over time. If the training converges with the model’s expected accuracy, the model can be deployed.\n"
        
        "* Checking input configuration is a good test, but it is not sufficient to ensure that model performance is acceptable. You can access logs and outputs for each pipeline step to review model performance, but it would involve more steps than using TensorBoard.\n"
        
        "* Model size is a good indicator of health but does not provide a complete overview to make sure that the model can be safely deployed. Note that the pipeline’s metadata can also be accessed directly from Vertex AI Pipelines.\n"
        
        "* Data is the most probable cause of this behavior, but it is not the only possible cause. Also, access requests could take a long time and are not the most secure option. Note that the pipeline’s metadata can also be accessed directly from Vertex AI Pipelines.",

        'references': [
            'https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview',
            'https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction',
            'https://cloud.google.com/vertex-ai/docs/pipelines/visualize-pipeline'
        ]
    },
    {
        'question':
        "You recently developed a custom ML model that was trained in Vertex AI on a post-processed training dataset stored in BigQuery. You used a Cloud Run container to deploy the prediction service. The service performs feature lookup and pre-processing and sends a prediction request to a model endpoint in Vertex AI. You want to configure a comprehensive monitoring solution for training-serving skew that requires minimal maintenance. What should you do?",
        'tags': [13, 'sample'],
        'options': {
            'A': "Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses email to send alerts. When an alert is received, use the console to diagnose the issue.",
            'B': "Update the model hosted in Vertex AI to enable request-response logging. Create a Data Studio dashboard that compares training data and logged data for potential training-serving skew and uses email to send a daily scheduled report.",
            'C': "Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses Cloud Logging to send alerts. Set up a Cloud Function to initiate model retraining that is triggered when an alert is logged.",
            'D': "Update the model hosted in Vertex AI to enable request-response logging. Schedule a daily DataFlow Flex job that uses Tensorflow Data Validation to detect training-serving skew and uses Cloud Logging to send alerts. Set up a Cloud Function to initiate model retraining that is triggered when an alert is logged."
        },
        'answers': ['A'],
        'explanation':
        
        "* Vertex AI Model Monitoring is a fully managed solution for monitoring training-serving skew that, by definition, requires minimal maintenance. Using the console for diagnostics is recommended for a comprehensive monitoring solution because there could be multiple causes for the skew that require manual review.\n"
        
        "* This solution does not minimize maintenance. It involves multiple custom components that require additional updates for any schema change.\n"
        
        "* A model retrain does not necessarily fix skew. For example, differences in pre-processing logic between training and prediction can also cause skew.\n"
        
        "* Update the model hosted in Vertex AI to enable request-response logging. Schedule a daily DataFlow ..., does not minimize maintenance. It involves multiple components that require additional updates for any schema change. Also, a model retrain does not necessarily fix skew. For example, differences in pre-processing logic between training and prediction can also cause skew.",

        'references': [
            'https://cloud.google.com/architecture/ml-modeling-monitoring-automating-server-data-skew-detection-in-ai-platform-prediction',
            'https://cloud.google.com/vertex-ai/docs/model-monitoring/overview'
        ]
    },
    {
        'question':
        "You have a historical data set of the sale price of 10,000 houses and the 10 most important features resulting from principal component analysis (PCA). You need to develop a model that predicts whether a house will sell at one of the following equally distributed price ranges: 200-300k, 300-400k, 400-500k, 500-600k, or 600-700k. You want to use the simplest algorithmic and evaluative approach. What should you do?",
        'tags': [14, 'sample'],
        'options': {
            'A': "Define a one-vs-one classification task where each price range is a categorical label. Use F1 score as the metric.",
            'B': "Define a multi-class classification task where each price range is a categorical label. Use accuracy as the metric.",
            'C': "Define a regression task where the label is the sale price represented as an integer. Use mean absolute error as the metric.",
            'D': "Define a regression task where the label is the average of the price range that corresponds to the house sale price represented as an integer. Use root mean squared error as the metric."
        },
        'answers': ['B'],
        'explanation':

        "* The use case is an ordinal classification task which is most simply solved using multi-class classification. Accuracy as a metric is the best match for a use case with discrete and balanced labels.\n"
        
        "* Define a one-vs-one classification task ..., is more complex than the classification approach suggested in the correct option. F1 score is not useful with equally distributed labels, and one-vs-one classification is used for multi-label classification, but the use case would require only one label to be correct.\n"
        
        "* Regression is not the recommended approach when solving an ordinal classification task with a small number of discrete values. This specific regression approach adds complexity because it uses the exact sale price to predict a range. Finally, the mean absolute error would not be the recommended metric because it gives the same penalty for errors of any magnitude.\n"
        
        "* Regression is not the recommended approach when solving an ordinal classification task with a small number of discrete values. This specific regression approach would be recommended because it uses a less complex label and a recommended metric to minimize variance and bias.",

        'references': [
            'https://cloud.google.com/automl-tables/docs/problem-types',
            'https://cloud.google.com/blog/products/gcp/predicting-community-engagement-on-reddit-using-tensorflow-gdelt-and-cloud-dataflow-part-2',
            'https://www.tensorflow.org/tutorials/keras/regression',
            'https://www.tensorflow.org/tutorials/keras/classification'
        ]
    },
    {
        'question':
        "You downloaded a TensorFlow language model pre-trained on a proprietary dataset by another company, and you tuned the model with Vertex AI Training by replacing the last layer with a custom dense layer. The model achieves the expected offline accuracy; however, it exceeds the required online prediction latency by 20ms. You want to optimize the model to reduce latency while minimizing the offline performance drop before deploying the model to production. What should you do?",
        'tags': [15, 'sample'],
        'options': {
            'A': "Apply post-training quantization on the tuned model, and serve the quantized model.",
            'B': "Use quantization-aware training to tune the pre-trained model on your dataset, and serve the quantized model.",
            'C': "Use pruning to tune the pre-trained model on your dataset, and serve the pruned model after stripping it of training variables.",
            'D': "Use clustering to tune the pre-trained model on your dataset, and serve the clustered model after stripping it of training variables."
        },
        'answers': ['A'],
        'explanation':

        "* Post-training quantization is the recommended option for reducing model latency when re-training is not possible. Post-training quantization can minimally decrease model performance.\n"
        
        "* Tuning the whole model on the custom dataset only will cause a drop in offline performance.\n"
        
        "* Tuning the whole model on the custom dataset only will cause a drop in offline performance. Also, pruning helps in compressing model size, but it is expected to provide less latency improvements than quantization.\n"
        
        "* Tuning the whole model on the custom dataset only will cause a drop in offline performance. Also, clustering helps in compressing model size, but it does not reduce latency.",

        'references': [
            'https://cloud.google.com/architecture/best-practices-for-ml-performance-cost',
            'https://www.tensorflow.org/lite/performance/model_optimization',
            'https://www.tensorflow.org/tutorials/images/transfer_learning'
        ]
    },
    {
        'question':
        "You developed a model for a classification task where the minority class appears in 10% of the data set. You ran the training on the original imbalanced data set and have checked the resulting model performance. The confusion matrix indicates that the model did not learn the minority class. You want to improve the model performance while minimizing run time and keeping the predictions calibrated. What should you do?",
        'tags': [16, 'sample'],
        'options': {
            'A': "Update the weights of the classification function to penalize misclassifications of the minority class.",
            'B': "Tune the classification threshold, and calibrate the model with isotonic regression on the validation set.",
            'C': "Upsample the minority class in the training set, and update the weight of the upsampled class by the same sampling factor.",
            'D': "Downsample the majority class in the training set, and update the weight of the downsampled class by the same sampling factor."
        },
        'answers': ['D'],
        'explanation':

        "* Downsampling with upweighting improves performance on the minority class while speeding up convergence and keeping the predictions calibrated.\n"
        
        "* Update the weights ... does not guarantee calibrated predictions and does not improve training run time.\n"
        
        "* Tune the classification threshold ... increases run time by adding threshold tuning and calibration on top of model training.\n"
        
        "* Upsampling increases training run time by providing more data samples during training.",

        'references': [
            'https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data',
            'https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/imbalanced_data.ipynb',
            'https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/calibration/calibration-node-classification.ipynb',
            'https://developers.google.com/machine-learning/glossary#calibration-layer'
        ]
    },
    {
        'question':
        "You have a dataset that is split into training, validation, and test sets. All the sets have similar distributions. You have sub-selected the most relevant features and trained a neural network in TensorFlow. TensorBoard plots show the training loss oscillating around 0.9, with the validation loss higher than the training loss by 0.3. You want to update the training regime to maximize the convergence of both losses and reduce overfitting. What should you do?",
        'tags': [17, 'sample'],
        'options': {
            'A': "Decrease the learning rate to fix the validation loss, and increase the number of training epochs to improve the convergence of both losses.",
            'B': "Decrease the learning rate to fix the validation loss, and increase the number and dimension of the layers in the network to improve the convergence of both losses.",
            'C': "Introduce L1 regularization to fix the validation loss, and increase the learning rate and the number of training epochs to improve the convergence of both losses.",
            'D': "Introduce L2 regularization to fix the validation loss, and increase the number and dimension of the layers in the network to improve the convergence of both losses."
        },
        'answers': ['D'],
        'explanation':

        "* L2 regularization prevents overfitting. Increasing the model’s complexity boosts the predictive ability of the model, which is expected to optimize loss convergence when underfitting.\n"
        
        "* Changing the learning rate does not reduce overfitting. Increasing the number of training epochs is not expected to improve the losses significantly.\n"
        
        "* Changing the learning rate does not reduce overfitting.\n"
        
        "* Increasing the number of training epochs is not expected to improve the losses significantly, and increasing the learning rate could also make the model training unstable. L1 regularization could be used to stabilize the learning, but it is not expected to be particularly helpful because only the most relevant features have been used for training.",

        'references': [
            'https://developers.google.com/machine-learning/testing-debugging/common/overview',
            'https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization',
            'https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/l1-regularization',
            'https://cloud.google.com/bigquery-ml/docs/preventing-overfitting',
            'https://www.tensorflow.org/tutorials/keras/overfit_and_underfit',
            'https://www.tensorflow.org/tensorboard/get_started',
            'https://cloud.google.com/architecture/guidelines-for-developing-high-quality-ml-solutions#guidelines_for_model_quality'
        ]
    },
    {
        'question':
        "You recently used Vertex AI Prediction to deploy a custom-trained model in production. The automated re-training pipeline made available a new model version that passed all unit and infrastructure tests. You want to define a rollout strategy for the new model version that guarantees an optimal user experience with zero downtime. What should you do?",
        'tags': [18, 'sample'],
        'options': {
            'A': "Release the new model version in the same Vertex AI endpoint. Use traffic splitting in Vertex AI Prediction to route a small random subset of requests to the new version and, if the new version is successful, gradually route the remaining traffic to it.",
            'B': "Release the new model version in a new Vertex AI endpoint. Update the application to send all requests to both Vertex AI endpoints, and log the predictions from the new endpoint. If the new version is successful, route all traffic to the new application.",
            'C': "Deploy the current model version with an Istio resource in Google Kubernetes Engine, and route production traffic to it. Deploy the new model version, and use Istio to route a small random subset of traffic to it. If the new version is successful, gradually route the remaining traffic to it.",
            'D': "Install Seldon Core and deploy an Istio resource in Google Kubernetes Engine. Deploy the current model version and the new model version using the multi-armed bandit algorithm in Seldon to dynamically route requests between the two versions before eventually routing all traffic over to the best-performing version."
        },
        'answers': ['B'],
        'explanation':

        "* Shadow deployments minimize the risk of affecting user experience while ensuring zero downtime.\n"
        
        "* Canary deployments may affect user experience, even if on a small subset of users.\n"
        
        "* The multi-armed bandit approach may affect user experience, even if on a small subset of users. This approach could cause downtime when moving between services.",

        'references': [
            'https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#data_and_model_validation',
            'https://cloud.google.com/architecture/implementing-deployment-and-testing-strategies-on-gke',
            'https://cloud.google.com/architecture/application-deployment-and-testing-strategies#choosing_the_right_strategy',
            'https://cloud.google.com/vertex-ai/docs/general/deployment',
            'https://docs.seldon.io/projects/seldon-core/en/latest/analytics/routers.html'
        ]
    },
    {
        'question':
        "You trained a model for sentiment analysis in TensorFlow Keras, saved it in SavedModel format, and deployed it with Vertex AI Predictions as a custom container. You selected a random sentence from the test set, and used a REST API call to send a prediction request. The service returned the error:\n"
        
        "“Could not find matching concrete function to call loaded from the SavedModel. Got: Tensor('inputs:0\", shape=(None,), dtype=string). Expected: TensorSpec(shape=(None, None), dtype=tf.int64, name='inputs')”." 
        
        "You want to update the model’s code and fix the error while following Google-recommended best practices. What should you do?",
        'tags': [19, 'sample'],
        'options': {
            'A': "Combine all preprocessing steps in a function, and call the function on the string input before requesting the model’s prediction on the processed input.",
            'B': "Combine all preprocessing steps in a function, and update the default serving signature to accept a string input wrapped into the preprocessing function call.",
            'C': "Create a custom layer that performs all preprocessing steps, and update the Keras model to accept a string input followed by the custom preprocessing layer.",
            'D': "Combine all preprocessing steps in a function, and update the Keras model to accept a string input followed by a Lambda layer wrapping the preprocessing function."
        },
        'answers': ['B'],
        'explanation':

        "* Combine all preprocessing steps in a function, and update the default ..., efficiently updates the model while ensuring no training-serving skew.\n"
        
        "* Duplicating the preprocessing adds unnecessary dependencies between the training and serving code and could cause training-serving skew.\n"
        
        "* Create a custom layer ..., adds unnecessary complexity. Because you update the model directly, you will need to re-train the model.\n"
        
        "* Combine all preprocessing steps in a function, and update the Keras model ..., adds unnecessary complexity. Because you update the model directly, you will need to re-train the model. Note that using Lambda layers over custom layers is recommended for simple operations or quick experimentation only.",

        'references': [
            'https://cloud.google.com/blog/topics/developers-practitioners/add-preprocessing-functions-tensorflow-models-and-deploy-vertex-ai',
            'https://www.tensorflow.org/tutorials/customization/custom_layers',
            'https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda',
            'https://developers.google.com/machine-learning/guides/rules-of-ml#rule_32_re-use_code_between_your_training_pipeline_and_your_serving_pipeline_whenever_possible'
        ]
    },
    {
        'question':
        "You used Vertex AI Workbench user-managed notebooks to develop a TensorFlow model. The model pipeline accesses data from Cloud Storage, performs feature engineering and training locally, and outputs the trained model in Vertex AI Model Registry. The end-to-end pipeline takes 10 hours on the attached optimized instance type. You want to introduce model and data lineage for automated re-training runs for this pipeline only while minimizing the cost to run the pipeline. What should you do?",
        'tags': [20, 'sample'],
        'options': {
            'A': 
                "1. Use the Vertex AI SDK to create an experiment for the pipeline runs, and save metadata throughout the pipeline.\n"
                "2. Configure a scheduled recurring execution for the notebook.\n"
                "3. Access data and model metadata in Vertex ML Metadata.",
            'B': 
                "1. Use the Vertex AI SDK to create an experiment, launch a custom training job in Vertex training service with the same instance type configuration as the notebook, and save metadata throughout the pipeline.\n"
                "2. Configure a scheduled recurring execution for the notebook.\n"
                "3. Access data and model metadata in Vertex ML Metadata.",
            'C': 
                "1. Create a Cloud Storage bucket to store metadata.\n"
                "2. Write a function that saves data and model metadata by using TensorFlow ML Metadata in one time-stamped subfolder per pipeline run.\n"
                "3. Configure a scheduled recurring execution for the notebook. 4. Access data and model metadata in Cloud Storage.",
            'D': 
                "1. Refactor the pipeline code into a TensorFlow Extended (TFX) pipeline.\n"
                "2. Load the TFX pipeline in Vertex AI Pipelines, and configure the pipeline to use the same instance type configuration as the notebook.\n"
                "3. Use Cloud Scheduler to configure a recurring execution for the pipeline. 4. Access data and model metadata in Vertex AI Pipelines."
        },
        'answers': ['C'],
        'explanation':

        "* Create a Cloud Storage bucket ..., minimizes running costs by being self-managed. This approach is recommended to minimize running costs only for simple use cases such as deploying one pipeline only. When optimizing for maintenance and development costs or scaling to more than one pipeline or performing experimentation, using Vertex ML Metadata and Vertex AI Pipelines are recommended.\n"
        
        "* A managed solution does not minimize running costs, and Vertex AI ML Metadata is more managed than Cloud Storage.\n"
        
        "* A managed solution does not minimize running costs, and this approach introduces Vertex training service with Vertex ML Metadata, which are both managed services.\n"
        
        "* A managed solution does not minimize running costs, and this approach introduces Vertex AI Pipelines, which is a fully managed service.",

        'references': [
            'https://cloud.google.com/vertex-ai/docs/pipelines/lineage',
            'https://cloud.google.com/vertex-ai/docs/ml-metadata/tracking',
            'https://cloud.google.com/vertex-ai/pricing',
            'https://cloud.google.com/architecture/ml-on-gcp-best-practices#operationalized-training',
            'https://cloud.google.com/architecture/ml-on-gcp-best-practices#organize-your-ml-model-artifacts'
        ]
    },
    # Cloud OnAir: Machine Learning Certification (4 questions)
    {
        'question':
        "You need to write a generic test to verify wheter Deep Neural Network (DNN) models automatically released by your team have a sufficient number of parameters to learn the task for which they were built. What should you do?",
        'tags': [1],
        'options': {
            'A': "Train the model for a few iterations, and check for NaN values.",
            'B': "Train the model for a few iterations, and verify that the loss is constant.",
            'C': "Train a simple linear model, and determine if the DNN model outperforms it.",
            'D': "Train the model with no regularization, and verify that the loss function is close to zero."
        },
        'answers': ['D'],
        'explanation':
        "",
        'references': []
    },
    {
        'question':
        "You work for a large retailer. You want to use ML to forecast future sales leveraging 10 years of historical sales data. The historical data is stored in Cloud Storage in Avro format. You want to rapidly experiment with all the available data. How should you build and train your model for the sales forecast?",
        'tags': [2],
        'options': {
            'A': "Load data into BigQuery and use the ARIMA model type on BigQuery ML.",
            'B': "Convert the data into CSV format and create a regression model on AutoML Tables.",
            'C': "Convert the data into TFRecords and create an RNN model on TensorFlow on Vertex AI Workbench.",
            'D': "Convert and refactor the data into CSV format and use the built-in XGBoost algorithm on Vertex AI custom training."
        },
        'answers': ['A'],
        'explanation':
        "BigQuery ML is designed for fast and rapid experimentation and it is possible to use federated queries to read data directly from Cloud Storage. Moreover, ARIMA is considered one of the best in class for time series forecasting.\n"
        "* AutoML Tables is not ideal for fast iteration and rapid experimentation. Even if it does not require data cleanup and hyperparameter tuning, it takes at least one hour to create a model.\n"
        "* In order to build a custom TensorFlow model, you would still need to do data cleanup and hyperparameter tuning.\n"
        "* Using Vertex AI custom training requires preprocessing your data in a particular CSV structure and it is not ideal for fast iteration, as training times can take a long time because it cannot be distributed on multiple machines.",
        'references': []
    },
    {
        'question':
        "You are an ML engineer at a media company. You need to build an ML model to analyze video content frame by frame, identify objects, and alert users if there is inappropriate content. Which Google Cloud products should you use to build this project?",
        'tags': [3, ''],
        'options': {
            'A': "Pub/Sub, Cloud Functions, and Cloud Vision API",
            'B': "Pub/Sub, Cloud IoT, Dataflow, Cloud Vision API, and Cloud Logging",
            'C': "Pub/Sub, Cloud Functions, Video Intelligence API, and Cloud Logging",
            'D': "Pub/Sub, Cloud Functions, AutoML Video Intelligence, and Cloud Logging"
        },
        'answers': ['C'],
        'explanation':
        "Video Intelligence API can find inappropriate components and other components satisfy the requirements of real-time processing and notification. AutoML Video intelligence should be only used in case of customization",
        'references': []
    },
    {
        'question':
        "You need to build an object detection model for a small startup company to identify if and where the company’s logo appears in an image. You were given a large repository of images, some with logos and some without. These images are not yet labelled. You need to label these pictures, and then train and deploy the model. What should you do?",
        'tags': [4],
        'options': {
            'A': "Use Google Cloud’s Data Labelling Service to label your data. Use AutoML Object Detection to train and deploy the model.",
            'B': "Use Vision API to detect and identify logos in pictures and use it as a label. Use Vertex AI to build and train a convolutional neural network.",
            'C': "Create two folders: one where the logo appears and one where it doesn’t. Manually place images in each folder. Use Vertex AI to build and train a convolutional neural network.",
            'D': "Create two folders: one where the logo appears and one where it doesn’t. Manually place images in each folder. Use Vertex AI to build and train a real time object detection model."
        },
        'answers': ['A'],
        'explanation':
        "",
        'references': []
    },
    # Certification Study Group - Professional Machine Learning Engineer (8 questions)
    {
        'question':
        "You work for a manufacturing company that owns a high-value machine which has several machine settings and multiple sensors. A history of the machine’s hourly sensor readings and known failure event data are stored in BigQuery. You need to predict if the machine will fail within the next 3 days in order to schedule maintenance before the machine fails. Which data preparation and model training steps should you take?",
        'tags': [1, 'study_group'],
        'options': {
            'A': "Data preparation: Daily max value feature engineering with DataPrep; Model training: AutoML classification with BQML",
            'B': "Data preparation: Daily min value feature engineering with DataPrep; Model training: Logistic regression with BQML and AUTO_CLASS_WEIGHTS set to True",
            'C': "Data preparation: Rolling average feature engineering with DataPrep; Model training: Logistic regression with BQML and AUTO_CLASS_WEIGHTS set to False",
            'D': "Data preparation: Rolling average feature engineering with DataPrep; Model training: Logistic regression with BQML and AUTO_CLASS_WEIGHTS set to True"
        },
        'answers': ['D'],
        'explanation':
        "Considering the noise and fluctuations of the data, the moving average is more appropriate than min/max to show the trend.\n"
        "* BQML allows you to create and run machine learning models using standard SQL queries in BigQuery.\n"
        "* The AUTO_CLASS_WEIGHTS=TRUE option balances class labels in the training data. By default, the training data is not weighted. If the training data labels are out of balance, the model can train to predict by weighting the most popular label classes more.\n"
        "* It uses a moving average of the sensor data and balances the weights using the parameters of BQML, AUTO_CLASS_WEIGHTS.\n"
        "* Model training does not balance class labels for unbalanced data sets",
        'references': []
    },
    {
        'question':
        "You work for a large financial institution that is planning to use Dialogflow to create a chatbot for the company’s mobile app. You have reviewed old chat logs and tagged each conversation for intent based on each customer’s stated intention for contacting customer service. About 70% of customer inquiries are simple requests that are solved within 10 intents. The remaining 30% of inquiries require much longer and more complicated requests. Which intents should you automate first?",
        'tags': [2, 'study_group'],
        'options': {
            'A': "Automate a blend of the shortest and longest intents to be representative of all intents.",
            'B': "Automate the more complicated requests first because those require more of the agents’ time.",
            'C': "Automate the 10 intents that cover 70% of the requests so that live agents can handle the more complicated requests.",
            'D': "Automate intents in places where common words such as “payment” only appear once to avoid confusing the software."
        },
        'answers': ['C'],
        'explanation':
        "It enables a machine to handle the most simple requests and gives the live agents more opportunity to handle higher value requests.",
        'references': []
    },
    {
        'question':
        "You work for a maintenance company and have built and trained a deep learning model that identifies defects based on thermal images of underground electric cables. Your dataset contains 10,000 images, 100 of which contain visible defects. How should you evaluate the performance of the model on a test dataset?",
        'tags': [3, 'study_group'],
        'options': {
            'A': "Calculate the Area Under the Curve (AUC) value.",
            'B': "Calculate the number of true positive results predicted by the model.",
            'C': "Calculate the fraction of images predicted by the model to have a visible defect.",
            'D': "Calculate the Cosine Similarity to compare the model’s performance on the test dataset to the model’s performance on the training dataset."
        },
        'answers': ['A'],
        'explanation':
        "AUC measures how well predictions are ranked, rather than their absolute values. AUC is also classification-threshold invariant. It measures the quality of the model’s predictions irrespective of what classification threshold is chosen.\n"
        "* Calculating the number of true positives without considering false positives can lead to misleading results. For instance, the model could classify nearly every image as a defect. This would result in many true positives, but the model would in fact be a very poor discriminator.\n"
        "* Calculating the fraction of images that contain defects doesn’t indicate whether your model is accurate or not.\n"
        "* Cosine Similarity is more commonly used in distance-based models (e.g., K Nearest Neighbors). This isn’t an appropriate metric for checking the performance of an image classification model.",
        'references': []
    },
    {
        'question':
        "Different cities in California have markedly different housing prices. Suppose you must create a model to predict the housing prices. Which of the following sets of features, or features crosses could learn city-specific relationships between roomsPerPerson and housing price?",
        'tags': [4, 'study_group'],
        'options': {
            'A': "Three separated binned features: [binned latitude], [binned longitude], [roomsPerPerson]",
            'B': "Two feature crosses: [binned latitude x roomsPerPerson] and [binned longitude x roomsPerPerson]",
            'C': "One feature cross [latitude x longitude x roomsPerPerson]",
            'D': "One feature cross [binned latitude x binned longitude x binned roomsPerPerson]"
        },
        'answers': ['D'],
        'explanation':
        "",
        'references': []
    },
    {
        'question':
        "You work for a textile manufacturer and have been asked to build a model to detect and classify fabric defects. You trained a machine learning model with high recall based on high resolution images taken at the end of the production line. You want quality control inspectors to gain trust in your model. Which technique should you use to understand the rationale of your classifier?",
        'tags': [5, 'study_group'],
        'options': {
            'A': "Use K-fold cross validation to understand how the model performs on different test datasets.",
            'B': "Use the Integrated Gradients method to efficiently compute feature attributions for each predicted image.",
            'C': "Use PCA (Principal Component Analysis) to reduce the original feature set to a smaller set of easily understood features.",
            'D': "Use k-means clustering to group similar images together, and calculate the Davies-Bouldin index to evaluate the separation between clusters."
        },
        'answers': ['B'],
        'explanation':
        "It identifies the pixel of the input image that leads to the classification of the image itself",
        'references': []
    },
    {
        'question':
        "You work for a gaming company that develops and manages a popular massively multiplayer online (MMO) game. The game’s environment is open-ended, and a large number of positions and moves can be taken by a player. Your team has developed an ML model with TensorFlow that predicts the next move of each player. Edge deployment is not possible, but low-latency serving is required. How should you configure the deployment?",
        'tags': [6, 'study_group'],
        'options': {
            'A': "Use a Cloud TPU to optimize model training speed.",
            'B': "Use a Deep Learning Virtual Machine on Compute Engine with an NVIDIA GPU.",
            'C': "Use Vertex AI Prediction with a high-CPU machine type to get a batch prediction for the players.",
            'D': "Use Vertex AI Prediction with a high-memory machine type to get a batch prediction for the players."
        },
        'answers': ['B'],
        'explanation':
        "TPUs are not available for prediction on Vertex AI Endpoint (only GPUs).",
        'references': []
    },
    {
        'question':
        "Your team is using a TensorFlow Inception-v3 CNN model pretrained on ImageNet for an image classification prediction challenge on 10,000 images. You will use Vertex AI to perform the model training. What TensorFlow distribution strategy and Vertex AI custom training job configuration should you use to train the model and optimize for wall-clock time?",
        'tags': [7, 'study_group'],
        'options': {
            'A': "Default Strategy; Custom tier with a single master node and four v100 GPUs.",
            'B': "One Device Strategy; Custom tier with a single master node and four v100 GPUs.",
            'C': "One Device Strategy; Custom tier with a single master node and eight v100 GPUs.",
            'D': "MirroredStrategy; Custom tier with a single master node and four v100 GPUs."
        },
        'answers': ['D'],
        'explanation':
        "MirroredStrategy is the only strategy that can perform distributed training; albeit there is only a single copy of the variables on the CPU host.",
        'references': []
    },
    {
        'question':
        "You work on a team where the process for deploying a model into production starts with data scientists training different versions of models in a Kubeflow pipeline. The workflow then stores the new model artifact into the corresponding Cloud Storage bucket. You need to build the next steps of the pipeline after the submitted model is ready to be tested and deployed in production on Vertex AI. How should you configure the architecture before deploying the model to production?",
        'tags': [8, 'study_group'],
        'options': {
            'A': "Deploy model in test environment -> Evaluate and test model -> Create a new Vertex AI model version",
            'B': "Validate model -> Deploy model in test environment -> Create a new Vertex AI model version",
            'C': "Create a new Vertex AI model version -> Evaluate and test model -> Deploy model in test environment",
            'D': "Create a new Vertex AI model version - > Deploy model in test environment -> Validate model"
        },
        'answers': ['A'],
        'explanation':
        "",
        'references': []
    },   
    # Udemy - Google Cloud Machine Learning Engineer Certification Prep (50 questions)
    {
        'question':
        "You are supporting a group of data analysts who want to build ML models using a managed service. They also want the ability to customize their models and tune hyperparameters. What managed service in Google Cloud would you recommend?",
        'tags': [1, 'udemy'],
        'options': {
            'A': "Vertex AI custom training",
            'B': "Vertex AI AutoML",
            'C': "Cloud TPUs",
            'D': "Cloud GPUs"
        },
        'answers': ['A'],
        'explanation':
        "Vertex AI custom training allows for tuning hyperparameters. Vertex AI AutoML training tunes hyperparameters for you. BigQuery ML does not allow for hyperparameter tuning. Cloud TPUs are accelerators you can use to train large deep learning models.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform']
    },
    {
        'question':
        "You have created a Compute Engine instance with an attached GPU but the GPU is not used when you train a Tensorflow model. What might you do to ensure the GPU can be used for training your models?",
        'tags': [2, 'udemy'],
        'options': {
            'A': "Install GPU drivers",
            'B': "Use Pytorch instead of Tensorflow",
            'C': "Grant the Editor basic role to the VM service account",
            'D': "Update Python 2.7 on the VM"
        },
        'answers': ['A'],
        'explanation':
        "GPU drivers need to be installed if they are not installed already when using GPUs. Deep Learning VM images have GPU drivers installed but if you don't use an image with GPU drivers installed, you will need to install them. Using Pytorch instead of Tensorflow will require work to recode and Pytorch would not be able to use GPUs either if the drivers are not installed. Updating Python will not address the problem of missing drivers. Granting a new role to the service account of the VM will not address the need to install GPU drivers.",
        'references': [
            'https://cloud.google.com/compute/docs/gpus/install-drivers-gpu']
    },
    {
        'question':
        "A financial services company wants to implement a chatbot service to help direct customers to the best customer support team for their questions. What GCP service would you recommend?",
        'tags': [3, 'udemy'],
        'options': {
            'A': "Text-to-Speech API",
            'B': "Speech-to-Text API",
            'C': "AutoML Tables",
            'D': "Dialogflow"
        },
        'answers': ['D'],
        'explanation':
        "Dialogflow is a service for creating conversational user interfaces.\n"
        "Speech-to-Text converts spoken words to written words.\n"
        "Text-to-Speech converts text words to human voice-like sound.\n"
        "AutoML Tables is a machine learning service for structured data.",
        'references': [
            'https://cloud.google.com/dialogflow/docs']
    },
    {
        'question':
        "You lead a team of machine learning engineers working for an IoT startup. You need to create a machine learning model to predict the likelihood of a device failure in manufacturing environments. The device generates a stream of metrics every 60 seconds. The metrics include 2 categorical values, 7 integer values, and 1 floating point value. The floating point value ranges from 0 to 100. For the purposes of the model, the floating point value is more precise than needed. Mapping that value to a feature with possible values \"high\", \"medium\", and \"low\" is sufficient. What feature engineering technique would you use to transform the floating point value to high, medium, or low?",
        'tags': [4, 'udemy'],
        'options': {
            'A': "L1 Regularization",
            'B': "Normalization",
            'C': "Bucketing",
            'D': "L2 Regularization"
        },
        'answers': ['C'],
        'explanation':
        "The correct answer is bucketing. In this case, values from 0 to 33 could be low, 34 to 66 could be medium, and values greater than 66 could be high\n." 
        "Regularization is the limiting of information captured by a model to prevent overfishing\n" 
        "L1 and L2 are two examples of regularization techniques\n."
        "Normalization is a transformation that scales numeric values to the range 0 to 1.",
        'references': []
    },
    {
        'question':
        "You have trained a machine learning model. After training is complete, the model scores high on accuracy and F1 score when measured using training data; however, when validation data is used, the accuracy and F1 score l are much lower. What is the likely cause of this problem?",
        'tags': [5, 'udemy'],
        'options': {
            'A': "Overfitting",
            'B': "Underfitting",
            'C': "Insufficiently complex model",
            'D': "The learning rate is too small"
        },
        'answers': ['A'],
        'explanation':
        "This is an example of overfitting because the model has not generalized form the training data. Underfitting would have resulted in poor performance with training data. Insufficiently complex models can lead to underfitting but not overfitting. A small learning rate will lead to longer training times but would not cause the described problem.",
        'references': []
    },
    {
        'question':
        "You are building a machine learning model using random forests. You haven't achieved the precision and recall you would like. What hyperparameter or hyperparameters would you try adjusting to improve accuracy?",
        'tags': [6, 'udemy'],
        'options': {
            'A': "Number of trees only",
            'B': "Number of trees and depth of trees",
            'C': "Number of clusters",
            'D': "Learning rate"
        },
        'answers': ['B'],
        'explanation':
        "",
        'references': []
    },
    {
        'question':
        "A logistics analyst wants to build a machine learning model to predict the number of units of a product that will need to be shipped to stores over the next 30 days. The features they will use are all stored in a relational database. The business analyst is familiar with reporting tools but not programming in general. What service would you recommend the analyst use to build a model?",
        'tags': [7, 'udemy'],
        'options': {
            'A': "Spark ML",
            'B': "AutoML Tables",
            'C': "Bigtable ML",
            'D': "TensorFlow"
        },
        'answers': ['B'],
        'explanation':
        "AutoML Tables uses structured data to build models with little input from users. Spark ML and Tensorflow is suitable for modelers with programming skills. There is no Bigtable ML but BigQuery ML is a managed service for building machine learning models in BigQuery using SQL.",
        'references': []
    },
    {
        'question':
        "When testing a regression model to predict the selling price of houses. After several iterations of model building, you note that small changes in a few features can lead to large differences in the output. This is an example of what kind of problem?",
        'tags': [8, 'udemy'],
        'options': {
            'A': "Low variance",
            'B': "High variance",
            'C': "Low bias",
            'D': "High bias"
        },
        'answers': ['B'],
        'explanation':
        "This is an example of high variance. High bias occurs when relationships are missed. Low bias and low variance are desired in ML models and are not a problem.",
        'references': []
    },
    {
        'question':
        "You are an ML engineer with a startup building machine learning models for the pharmaceutical industry. You are currently developing a deep learning machine learning model to predict the toxicity of drug candidates. The training data set consists of a large number of chemical and physical attributes and there is a large number of instances. Training takes several days on an n2 type Compute Engine virtual machine. What would you recommend to reduce the training time without compromising the quality of the model?",
        'tags': [9, 'udemy'],
        'options': {
            'A': "Use TPUs",
            'B': "Randomly sample 20% of the training set and train on that smaller data set",
            'C': "Increase the machine size to make more memory available",
            'D': "Increase the machine size to make more CPUs available"
        },
        'answers': ['A'],
        'explanation':
        "TPUs are designed to accelerate the dominant computation in deep learning model training. Using a smaller data set by sampling would reduce training time but would likely compromise the quality of the model. Increasing CPUs would improve performance but not as much or as cost-effectively as TPUs. Increasing memory may reduce training time if memory is constrained but it will not decrease training time as much as using a TPU.",
        'references': []
    },
    {
        'question':
        "You want to evaluate a classification model using the True Positive Rate and the False Positive Rate. You want to view a graph showing the performance of the model at all classification thresholds. What evaluation metric would you use?",
        'tags': [10, 'udemy'],
        'options': {
            'A': "Area under the ROC curve (AUC)",
            'B': "Precision",
            'C': "F1 Score",
            'D': "L2 Regularization"
        },
        'answers': ['A'],
        'explanation':
        "Area under the ROC curve (AUC) is a graph of True Positive and False Positive rates.\n"
        "Precision is a measure of the quality of positive predictions.\n"
        "F1 Score is a harmonic mean of precision and recall.\n"
        "L2 Regularization is a technique to prevent overfitting.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc'
        ]
    },
    {
        'question':
        "You are building a machine learning model and during the data preparation stage, you preform normalization and standardization using the full data set. You then split the full data set into training, validation, and testing data sets. What problem could be introduced by performing the steps in the order described?",
        'tags': [11, 'udemy'],
        'options': {
            'A': "Regularization",
            'B': "Data leakage",
            'C': "Introduction of bias",
            'D': "Imbalanced classes"
        },
        'answers': ['B'],
        'explanation':
        "This is an example of data leakage because you are making additional data available during training that wold is not available when running predictions, in this case, additional information is used to perform normalization and standardization. Regularization is a technique to prevent overfitting. No bias is introduced and there is no indication that classes are imbalanced.",
        'references': []
    },
    {
        'question':
        "A simple model based on hand-coded heuristics or a simple algorithms such as a linear model is often built early in the model training process. What is the purpose of such as model?",
        'tags': [12, 'udemy'],
        'options': {
            'A': "It provides a baseline for the minimum performance to expect in an ML model",
            'B': "It provides the maximum expected performance in an ML model",
            'C': "It provides a measure of the likelihood of underfitting",
            'D': "It provides a measure of the likelihood of overfitting"
        },
        'answers': ['A'],
        'explanation':
        "It provides a baseline for the minimum performance to expect in an ML model. Simple models do not provide indication of maximum performance. A simple model could underfit and would be expected. Simple models are not likely to overfit.",
        'references': []
    },
    {
        'question':
        "What characteristics of feature values do we try to find when using descriptive statistics for data exploration?",
        'tags': [13, 'udemy'],
        'options': {
            'A': "Central tendency only",
            'B': "Spread of values only",
            'C': "Central tendency and spread of values",
            'D': "Likelihood to contribute to a prediction"
        },
        'answers': ['C'],
        'explanation':
        "Descriptive statistics are used to measure both central tendency and the spread of values. The likelihood of contributing to a prediction is not measured until after a model is created.",
        'references': []
    },
    {
        'question':
        "You are building a classification model to detect fraud in credit card transactions. When exploring the training data set you notice that 2% of instances are examples of fraudulent transactions and 98% are legitimate transactions. This is an example of what kind of data set?",
        'tags': [14, 'udemy'],
        'options': {
            'A': "An imbalanced data set",
            'B': "A standardized data set",
            'C': "A normalized data set",
            'D': "A marginalized data set"
        },
        'answers': ['A'],
        'explanation':
        "This is an imbalnaced data set because one class has significantly more instances than the others. Standardization and normalization are techniques for preparing the data set. There is no such thing as a marginalized data set in machine learning.",
        'references': []
    },
    {
        'question':
        "Which of the following techniques can be used when working with imbalanced data sets?",
        'tags': [15, 'udemy'],
        'options': {
            'A': "Collecting more data",
            'B': "Resampling",
            'C': "Generating synthetic data using an algorithm such as SMOTE",
            'D': "All of the abovoe"
        },
        'answers': ['D'],
        'explanation':
        "Collecting more data, generating synthetic data, and resampling are all techniques for working with imbalanced data sets.",
        'references': []
    },
    {
        'question':
        "A team of machine learning engineers is training an image recognition model to detect defects in manufactured parts. The team has a data set of 10,000 images but wants to train with at least 30,000 images. They do not have time to wait for an additional set of 20,000 images to be collected on the factory floor. What type of technique could they use to produce a data set with 30,000 images?",
        'tags': [16, 'udemy'],
        'options': {
            'A': "Normalization",
            'B': "Data augmentation",
            'C': "Data leakage",
            'D': "Imbalanced classes"
        },
        'answers': ['B'],
        'explanation':
        "Data augmentation is a set of techniques for artificially increasing the number of instances in a data set by manipulating other instances. Normalization is a data preparation technique. Data leakage is the use of data in training that is not available during prediction and is unwanted. Imbalanced classes is not a technique for expanding the size of a dataset.",
        'references': []
    },
    {
        'question':
        "You are using distributed training with TensorFlow. What type of server stores parameters and coordinates shared model state across workers?",
        'tags': [17, 'udemy'],
        'options': {
            'A': "Parameter servers",
            'B': "State servers",
            'C': "Evaluators",
            'D': "Primary replica"
        },
        'answers': ['A'],
        'explanation':
        "Parameter servers store model parameters and share state. Evaluators evaluate models while primary replicas manage other nodes. There is no state servers.",
        'references': []
    },
    {
        'question':
        "A dataset includes multiple categorical values. You want to train a deep learning neural network using the data set. Which of the following would be an appropriate data encoding scheme?",
        'tags': [18, 'udemy'],
        'options': {
            'A': "One-hot encoding",
            'B': "Categorical encoding",
            'C': "Regularization",
            'D': "Normalization"
        },
        'answers': ['A'],
        'explanation':
        "One-hot encoding is an appropriate encoding technique to map categorical values to a bit vector. Categorical values themselves are not suitable input to a deep learning network. Regularization is is used to prevent overfitting. Normalization is a data preparation operation.",
        'references': []
    },
    {
        'question':
        "A dataset you are using has categorical values mapped to integer values, such as red to 1, blue to 2, and green to 3. What kind of encoding scheme is this?",
        'tags': [19, 'udemy'],
        'options': {
            'A': "One-hot encoding",
            'B': "Feature hashing",
            'C': "Ordinal encoding",
            'D': "Data augmentation"
        },
        'answers': ['C'],
        'explanation':
        "This is an example of ordinal encoding. One-hot encoding maps to a bit vector with only one bit set to one. Feature hashing applies a hash function to compute a representation. Data augmentation is not an encoding scheme, it is a set of techniques for increasing the size of a data set.",
        'references': []
    },
    {
        'question':
        "Which of the following are ways bias can be introduced in a machine learning model? (Choose 2)",
        'tags': [20, 'udemy'],
        'options': {
            'A': "Biased data distribution",
            'B': "Proxy variables",
            'C': "Data leakage",
            'D': "Data augmentation",
            'E': "Normalization"
        },
        'answers': ['A', 'B'],
        'explanation':
        "Biased data distributions and proxy variables can introduce bias in a machine model. Data leakage can cause problems but is not likely to introduce bias that isn't already in the data set. Data augmentation can continue to represent bias in a data set but does not introduce new bias. Normalization is a data preparation operations.",
        'references': []
    },
    {
        'question':
        "A machine learning engineer detects non-linear relationships between two variables in a dataset. The dataset is relatively small and it is expensive to acquire new examples. What can the machine learning engineer do to increase the performance of the model with respect to the non-linear relationship detected?",
        'tags': [21, 'udemy'],
        'options': {
            'A': "Use a deep learning network",
            'B': "Use regularization",
            'C': "Create a feature cross",
            'D': "Use data leakage"
        },
        'answers': ['C'],
        'explanation':
        "A feature cross could capture the non-linear relationship. A deep learning network can also learn non-linear relationships but they require large volumes of data. Regularization is a set of techniques for preventing overfitting. Data leakage is unwanted in a machine learning model.",
        'references': []
    },
    {
        'question':
        "You have a dataset with more features than you believe you need to train a model. You would like to measure how well two numerical values linearly correlate so you can eliminate one of them if they highly correlate. What statistical test would you use?",
        'tags': [22, 'udemy'],
        'options': {
            'A': "Pearson's Correlation",
            'B': "ANOVA",
            'C': "Kendall's Rank Coefficient",
            'D': "Chi-Squared Test"
        },
        'answers': ['A'],
        'explanation':
        "* The Pearson's Correlation is used for measuring the linear correlation between two variables.\n"
        "* ANOVA is used to measure the difference among means.\n"  
        "* Kendall's Rank Coefficient is used for measuring numeric and categorical correlations.\n"
        "* The Chi-Squared test is used for measuring the correlation between categorical values.",
        'references': []
    },
    {
        'question':
        "You have a dataset with more features than you believe you need to train a model. You would like to measure how well two categorical values linearly correlate so you can eliminate one of them if they highly correlate. What statistical test would you use?",
        'tags': [23, 'udemy'],
        'options': {
            'A': "Pearson's Correlation",
            'B': "ANOVA",
            'C': "Chi-Squared Test",
            'D': "Kendall's Rank Coefficient"
        },
        'answers': ['C'],
        'explanation':
        "* The Chi-Squared test is used for measuring the correlation between categorical values.\n"
        "* Pearson's Correlation is used for measuring the linear correlation between two variables.\n"
        "* ANOVA is used to measure the difference among means.\n"
        "* Kendall's Rank Coefficient is used for measuring numeric and categorical correlations.",
        'references': []
    },
    {
        'question':
        "Which of the following types of pre-built containers are available in Vertex AI? (Choose 3)",
        'tags': [24, 'udemy'],
        'options': {
            'A': "TensorFlow Optimized Runtime",
            'B': "Theano",
            'C': "Hadoop Mahout",
            'D': "XGBoost",
            'E': "Scikit-Learn"
        },
        'answers': ['A', 'D', 'E'],
        'explanation':
        "TensorFlow Optimized Runtime, XGBoost, and Scikit-Learn are available in Vertex AI pre-built containers. Hadoop Mahout and Theano are machine learning platforms but not available as pre-built containers.",
        'references': []
    },
    {
        'question':
        "Which of the following are required of a custom container used with Vertex AI? (Choose 2)",
        'tags': [25, 'udemy'],
        'options': {
            'A': "Support for health checks and liveliness checks",
            'B': "Request and response message size may be no more than 10 MB",
            'C': "Running an HTTP server",
            'D': "Include GPU drivers",
            'E': "Include support for TPUs or GPUs"
        },
        'answers': ['A', 'C'],
        'explanation':
        "Custom container images running in Vertex AI must have an HTTP server as well as support health checks and liveliness checks. Request and response message sizes must be 1.5MB or less. Support for GPUs or TPUs is not required.",
        'references': []
    },
    {
        'question':
        "You are training large deep learning networks in Kubernetes Engine and want to use a cost-effective accelerator. You do not need high precision floating point operations. What would you choose?",
        'tags': [26, 'udemy'],
        'options': {
            'A': "GPUs",
            'B': "TPUs",
            'C': "ASICs",
            'D': "CPUs"
        },
        'answers': ['B'],
        'explanation':
        "",
        'references': []
    },
    {
        'question':
        "Several datasets you use for training ML models have missing data. You consider deleting rows with missing data. In which case would you not want to delete instances with missing data?",
        'tags': [27, 'udemy'],
        'options': {
            'A': "When a significant portion of the instances are missing data",
            'B': "When a small number of instances are missing data",
            'C': "When instances are missing data for more than one feature",
            'D': "when instances are missing data for more than three features"
        },
        'answers': ['A'],
        'explanation':
        "You would not want to delete instance with missing data when a significant portion of the instances are missing data because you would lose many instances. When a small number of instance are missing data, removing those instances would not adversely affect results. Since all data for all features are removed when removing a row with any missing data, the number of features with missing data does not impact the final results.",
        'references': []
    },
    {
        'question':
        "When is it appropriate to use the Last Observed Value Carried Forward strategy for missing data?",
        'tags': [28, 'udemy'],
        'options': {
            'A': "When working with time series data",
            'B': "When working with categorical data and a small number of values",
            'C': "When overfitting is a high risk",
            'D': "When underfitting is a high risk"
        },
        'answers': ['A'],
        'explanation':
        "The Last Observed Value Carried Forward strategy works well with time series data. Categorical values with a small number of possible values is not a good candidate since the previous value may have not relation to next instance in the data set. The technique is irrelevant to overfitting or underfitting.",
        'references': []
    },
    {
        'question':
        "Which of the following are examples of hyperparameters?",
        'tags': [29, 'udemy'],
        'options': {
            'A': "Maximum depth of a decision tree only",
            'B': "Number of layers in a deep learning network only",
            'C': "Learning rate of gradient descent",
            'D': "All of the above"
        },
        'answers': ['D'],
        'explanation':
        "These are all examples of hyperparameters to machine learning models.",
        'references': []
    },
    {
        'question':
        "You are validating a machine learning model and have decided you need to further tune hyperparamets. You would like to try analyze multiple hyperparameter combinations in parallel. Which of the following techniques could you use?",
        'tags': [30, 'udemy'],
        'options': {
            'A': "Grid search and Bayesian search",
            'B': "Random search and Grid search",
            'C': "Bayesian search only",
            'D': "Random search only"
        },
        'answers': ['B'],
        'explanation':
        "Random search and grid search can both be applied in parallel. Bayesian search is a sequential method for searching hyperparameter combinations.",
        'references': []
    },
    {
        'question':
        "You spend a lot of time tuning hyperparameters by manually testing combinations of hyperparameters. You want to automate the process and use a technique that can learn from previous evaluations of other hyperparameter combinations. What algorithm would you use?",
        'tags': [31, 'udemy'],
        'options': {
            'A': "Grid search",
            'B': "Data augmentation",
            'C': "Bayesian search",
            'D': "Random search"
        },
        'answers': ['C'],
        'explanation':
        "Bayesian search uses knowledge from previous evaluations when selecting new hyperparameter values. Grid search and random search are used for hyperparameter tuning but do not use prior knowledge. Data augmentation is not used for searching hyperparameters.",
        'references': []
    },
    {
        'question':
        "A dataset has been labeled by a crowd-sourced group of labelers. You want to evaluate the quality of the labeling process. You randomly select a group of labeled instances and find several are mislabled. You want to find other instances that are similar to the mislabeled instances. What kind of algorithm would you use to find similar instances?",
        'tags': [32, 'udemy'],
        'options': {
            'A': "Approximate Nearest Neighbor",
            'B': "XGBoost",
            'C': "Random Forest",
            'D': "Gradient descent"
        },
        'answers': ['A'],
        'explanation':
        "Approximate Nearest Neighbor algorithms use clustering to group similar instances and would be the correct choice. XGBoost and Random Forest are not clustering algorithms and would not be as good a choice as a clustering algorithm. Gradient descent is a technique used to optimize weights in deep learning.",
        'references': []
    },
    {
        'question':
        "A company is migrating a machine learning model that is currently being served on premises to Google Cloud. The model runs in Spark ML. You have been asked to recommend a way to migrate the service with the least disruption in service and minimal effort. The company does not want to manage infrastructure if possible and prefers to use managed services. What would you recommend?",
        'tags': [33, 'udemy'],
        'options': {
            'A': "BigQuery ML",
            'B': "Cloud Dataproc",
            'C': "Cloud Dataflow",
            'D': "Cloud Data Studio"
        },
        'answers': ['B'],
        'explanation':
        "Cloud Dataproc is a managed Spark/Hadoop service and would be a good choice. BigQuery supports BigQuery ML but that would require re-implmenting the model. Cloud Dataflow is a managed service for batch and stream processing. Cloud Data Studio is a visualization tool.",
        'references': []
    },
    {
        'question':
        "A group of data analysts know SQL and want to build machine learning models using data stored on premises in relational databases. They want to load the data into the cloud and use a cloud-based service for machine learning. They want to build models as quickly as possible and use them for problems in classification, forecasting, and recommendations. They do not want to program in Python or Java. What Google Cloud service would you recommend?",
        'tags': [34, 'udemy'],
        'options': {
            'A': "Cloud Dataproc",
            'B': "Cloud Dataflow",
            'C': "BigQuery ML",
            'D': "Bigtable"
        },
        'answers': ['C'],
        'explanation':
        "BigQuery ML uses SQL to create and serve machine learning models and dose not require programming in a language such as Python or Java. Cloud Dataflow is for data processing, not machine learning. Cloud Dataproc could be used for machine learning but requires programming in Java, Python or other programming languages. Bigtable does not support machine learning directly in the service.",
        'references': []
    },
    {
        'question':
        "What feature representation is used when training machine learning models using text or image data?",
        'tags': [35, 'udemy'],
        'options': {
            'A': "Feature vectors",
            'B': "Lists of categorical values",
            'C': "2-dimensional arrays",
            'D': "3-dimensional arrays"
        },
        'answers': ['A'],
        'explanation':
        "Feature vectors are the standard way of inputting data to a machine learning algorithm. Lists of categorical values are not accessible to many machine learning algorithms. 2-dimensional and 3-dimensional arrays are mapped to 1-dimensional feature vectors before submitting data to the machine learning training algorithm.",
        'references': []
    },
    {
        'question':
        "An IoT company has developed a TensorFlow deep learning model to detect anomalies in machine sensor readings. The model will be deployed to edge devices. Machine learning engineers want to reduce the model size without significantly reducing the quality of the model. What technique could they use?",
        'tags': [36, 'udemy'],
        'options': {
            'A': "ANOVA",
            'B': "Quantization",
            'C': "Data augmentation",
            'D': "Bucketing"
        },
        'answers': ['B'],
        'explanation':
        "Quantization is a technique for reducing model size without reducing quality.\n"
        "ANOVA is a statistical test for comparing the means of two or more populations.\n"
        "Data augmentation is used to create new training instances based on existing instances.\n"
        "Bucketing is a technique of mapping feature values into a smaller set of values.",
        'references': []
    },
    {
        'question':
        "You have created a machine learning model to identify defective parts in an image. Users will send images to an endpoint used to serve the model. You want to follow Google Cloud recommendations. How would you encode the image when making a request of the prediction service?",
        'tags': [37, 'udemy'],
        'options': {
            'A': "CSV",
            'B': "Avro",
            'C': "base64",
            'D': "Capacitor format"
        },
        'answers': ['C'],
        'explanation':
        "Base64 is the recommended encoding for images. CSV and Avro are file formats for structured data. Capacitor format is used by BigQuery to store data in compressed, columnar format.",
        'references': []
    },
    {
        'question':
        "You are making a large number of predictions using an API endpoint. Several of the services making requests could send batches of requests instead of individual requests to the endpoint. How could you improve the efficiency of serving predictions?",
        'tags': [38, 'udemy'],
        'options': {
            'A': "Use batches with a large batch size to take advantage of vectorization",
            'B': "Vertically scale the API server",
            'C': "Train with additional data to improve accuracy",
            'D': "Release re-trained models more frequently"
        },
        'answers': ['A'],
        'explanation':
        "Using batches with large batch size will take advantage of vectorization and improve efficiency. Vertically scaling will increase throughput but using the API and single requests will still use more compute resources than using batch processing. Training with additional data or re-training more frequently will not change serving efficiency.",
        'references': []
    },
    {
        'question':
        "Which component of the Vertex AI provides for the orchestration of machine learning operations in Vertex AI?",
        'tags': [39, 'udemy'],
        'options': {
            'A': "Vertex AI Prediction",
            'B': "Vertex AI Pipelines",
            'C': "Vertex AI Experiments",
            'D': "Vertex AI Workbench"
        },
        'answers': ['B'],
        'explanation':
        "Vertex AI Pipelines provides orchestration in Vertex AI. Vertex AI Prediction is for serving models, Vertex AI Experiments is for tracking training experiments, and Vertex AI Workbench provides managed and user managed notebooks for development.",
        'references': []
    },
    {
        'question':
        "A team of researchers have built a TensorFlow model for predicting near-term weather changes. They are using TPUs but are not achieving the throughput they would like. Which of the following might improve the efficiency of processing?",
        'tags': [40, 'udemy'],
        'options': {
            'A': "Using the tf.data API to maximize the efficiency of data pipelines using GPUs and TPUs",
            'B': "Use distributed XGBoost",
            'C': "Use early stopping",
            'D': "Scale up CPUs before scaling out the number of CPUs"
        },
        'answers': ['A'],
        'explanation':
        "Using the tf.data API to maximize the efficiency of data pipelines using GPUs and TPUs is the correct answer. XGBoost is another machine learning platform and will not improve the efficiency of a TensorFlow model. Early stopping is an optimization for training, not serving. Scaling up CPUs or adding more CPUs will not significantly change the efficiency of using GPUs or TPUs.",
        'references': []
    },
    {
        'question':
        "Managed data sets in Vertex AI provided which of the following benefits?",
        'tags': [41, 'udemy'],
        'options': {
            'A': "Manage data sets in a central location only",
            'B': "Managed data sets in a central location and create labels and annotations only",
            'C': "Managed data sets in a central location, create labels and annotations, and apply enhanced predefined IAM roles only",
            'D': "Managed data sets in a central location, create labels and annotations, apply enhanced predefined IAM roles, and track the lineage of models"
        },
        'answers': ['B'],
        'explanation':
        "The correct answer is managed data sets in a central location and create labels and annotations only. There are no enhanced predefined roles for Vertex AI datasets.",
        'references': []
    },
    {
        'question':
        "Which of the following are options for tabluar datasets in Vertex AI Datasets?",
        'tags': [42, 'udemy'],
        'options': {
            'A': "CSV files only",
            'B': "CSV files and BigQuery tables and views",
            'C': "CSv files, BigQuery tables and views, and Bigtable tables",
            'D': "CSV files, BigQuery tables and views, and Avro files"
        },
        'answers': ['B'],
        'explanation':
        "Vetext AI Datasets support CSV files and BigQuery tables and views for tabular data. Bigtable tables and Avro files are not supported.",
        'references': []
    },
    {
        'question':
        "A team of reviewers is analyzing a training data set for sensitive information that should not be used when training models. Which of the following are types of sensitive information that should be removed from the training set?",
        'tags': [43, 'udemy'],
        'options': {
            'A': "Credit card numbers",
            'B': "Government ID numbers",
            'C': "Purchase history",
            'D': "Faces in images",
            'E': "Customer segment identifier"
        },
        'answers': ['A', 'B', 'D'],
        'explanation':
        "Credit card numbers, government ID numbers, and faces in images are all examples of sensitive information. Purchase history and customer segment identifiers are not sensitive information.",
        'references': []
    },
    {
        'question':
        "Which of the follwoing techniques can be used to mask sensitive data?",
        'tags': [44, 'udemy'],
        'options': {
            'A': "Substitution cipher",
            'B': "Tokenization",
            'C': "Data augmentation",
            'D': "Regularization",
            'E': "Principal component analysis"
        },
        'answers': ['A', 'B', 'E'],
        'explanation':
        "Substitution cipher, tokenization, and principal component analysis can all be used to mask sensitive data. Data augmentation is used to increase the size of training sets. Regularization is used to prevent overfitting.",
        'references': []
    },
    {
        'question':
        "Which of the following is a type of security risk to machine learning model?",
        'tags': [45, 'udemy'],
        'options': {
            'A': "Data poisoning",
            'B': "Missing data",
            'C': "Inconsistent labeling",
            'D': "Insufficently agreed upon objectives"
        },
        'answers': ['A'],
        'explanation':
        "Data poisoning is a security risk associated with an attacker compromising the training process in order to train the model to behave in ways the attacker wants. Missing data and inconsistent data are data risks that can compromise a model but they are not security risks. Insufficiently agreed upon objectives is a process risk but not a security risk.",
        'references': []
    },
    {
        'question':
        "You are training a classifier using XGBoost in Vertex AI. Training is proceeding slower than expected so you add GPUs to your training server. There is no noticeable difference in the training time. Why is this?",
        'tags': [46, 'udemy'],
        'options': {
            'A': "GPUs are only useful for improving serving efficiency",
            'B': "TPUs should have been used instead",
            'C': "GPUs are not used with XGBoost in Vertex AI",
            'D': "You did not install GPU drivers on the server"
        },
        'answers': ['C'],
        'explanation':
        "Using TPUs would not improve performance either. GPUs are useful for improving training performance. Vertex AI manages images used for training and serving so there is no need to manually install GPU drivers.",
        'references': []
    },
    {
        'question':
        "Aerospace engineers are building a model to predict turbulence and impact on a new airplane wing design. They have large, multi-dimensional data sets. What file format would you recommend they use for training data?",
        'tags': [47, 'udemy'],
        'options': {
            'A': "Parquet",
            'B': "Petastorm",
            'C': "ORC",
            'D': "CSV"
        },
        'answers': ['B'],
        'explanation':
        "Petastorm is designed for multi-dimensional data. Parquet and ORC are both columnar formats and could be used but Petastorm is a better option. CSV is inefficient for large data sets.",
        'references': []
    },
    {
        'question':
        "You would like to use a nested file format for training data that will be used with TensorFlow. You would like to use the most efficient format. Which of the following would you choose?",
        'tags': [48, 'udemy'],
        'options': {
            'A': "JSON",
            'B': "XML",
            'C': "CSV",
            'D': "TFRecords"
        },
        'answers': ['D'],
        'explanation':
        "TFRecords is based on protobuf, a binary nested file format and optimized for TensorFlow. JSON and XML are plain text formats and not as efficient as TFRecord. CSV is not a nested file format.",
        'references': []
    },
    {
        'question':
        "A robotics developer has created a machine learning model to detect unripe apples in images. Robots use this information to remove unripe apples from a conveyor belt. The engineers who developed this model are using it as a starting model for training a model to detect unripe pears. This is an example of what kind of learning?",
        'tags': [49, 'udemy'],
        'options': {
            'A': "Unsupervised learning",
            'B': "Regression",
            'C': "Reinforcement learning",
            'D': "Transfer learning"
        },
        'answers': ['D'],
        'explanation':
        "This is an example of transfer learning\n."
        "Unsupervised learning uses data sets without labels.\n"
        "Regression models predict a continuous value.\n"
        "Reinforcement learning uses feedback from the environment to learn.",
        'references': []
    },
    {
        'question':
        "A retailer has deployed a machine learning model to predict when a customer is likely to abandon a shopping cart. A MLOps engineer notices that the feature data distribution in production deviates from feature data distribution in the latest training data set. This is an example of what kind of problem?",
        'tags': [50, 'udemy'],
        'options': {
            'A': "Skew",
            'B': "Drift",
            'C': "Data leakage",
            'D': "Underfitting"
        },
        'answers': ['A'],
        'explanation':
        "Skew is the problem of feature data distribution in production deviating from feature data distribution in training data.\n"
        "Drift occurs when feature data distribution in production changes significantly over time.\n"
        "Data leakage is a problem in training when data not available when making predictions is used in training.\n"
        "Underfitting occurs when a model does not perform well even on training data set because the model is unable to learn.",
        'references': []
    },
    # Packt - Journey to Become a Google Cloud Machine Learning Engineer (2022) - Dr. Logan Song (30 questions)
    {
        'question':
        "Space Y is launching its hundredth satellite to build its StarSphere network. They have designed an accurate orbit (launching speed/time/and so on) for it based on the existing 99 satellite orbits to cover the Earth’s scope. What’s the best solution to forecast the position of the 100 satellites after the hundredth launch?",
        'tags': [1, 'packt'],
        'options': {
            'A': "Use ML algorithms and train ML models to forecast",
            'B': "Use neural networks to train the model to forecast",
            'C': "Use physical laws and actual environmental data to model and forecast",
            'D': "Use a linear regression model to forecast"
        },
        'answers': ['C'],
        'explanation':
        "When we start, science modeling will be our first choice since it builds the most accurate model based on science and natural laws.",
        'references': [
            'Section Is ML the best solution? in Chapter 3, Preparing for ML Development']
    },
    {
        'question':
        "A financial company is building an ML model to detect credit card fraud based on their historical dataset, which contains 20 positives and 4,990 negatives.\n\n"
        
        "Due to the imbalanced classes, the model training is not working as desired. What’s the best way to resolve this issue?",
        'tags': [2, 'packt'],
        'options': {
            'A': "Data augmentation",
            'B': "Early stopping",
            'C': "Downsampling and upweighting",
            'D': "Regularization"
        },
        'answers': ['C'],
        'explanation':
        "When the data is imbalanced, it will be very difficult to train the ML model and get good forecasts",
        'references': [
            'Section Data sampling and balancing in Chapter 3, Preparing for ML Development']
    },
    {
        'question':
        "A chemical manufacturer is using a GCP ML pipeline to detect real-time sensor anomalies by queuing the inputs and analyzing and visualizing the data. Which one will you choose for the pipeline?",
        'tags': [3, 'packt'],
        'options': {
            'A': "Dataproc | Vertex AI | BQ",
            'B': "Dataflow | AutoML | Cloud SQL",
            'C': "Dataflow | Vertex AI | BQ",
            'D': "Dataproc | AutoML | Bigtable"
        },
        'answers': ['C'],
        'explanation':
        "Dataflow is based on parallel data processing and works better if your data has no implementation with Spark or Hadoop. BQ is great for analyzing and visualizing data",
        'references': []
    },
    {
        'question':
        "A real estate company, Zeellow, does great business buying and selling properties in the United States. Over the past few years, they have accumulated a big amount of historical data for US houses. \n\n" 
        
        "Zeellow is using ML training to predict housing prices, and they retrain the models every month by integrating new data. The company does not want to write any code in the ML process. What method best suits their needs?",
        'tags': [4, 'packt'],
        'options': {
            'A': "AutoML Tabular",
            'B': "BigQuery ML",
            'C': "Vertex AI",
            'D': "AutoML classification"
        },
        'answers': ['A'],
        'explanation':
        "AutoML serves the purpose of no coding during the ML process, and this is a structured data ML problem",
        'references': []
    },
    {
        'question':
        "The data scientist team is building a deep learning model for a customer support center of a big Enterprise Resource Planning (ERP) company, which has many ERP products and modules. The DL model will input customers’ chat texts and categorize them into products before routing them to the corresponding team. The company wants to minimize the model development time and data preprocessing time. What strategy/platform should they choose?",
        'tags': [5, 'packt'],
        'options': {
            'A': "Vertex AI",
            'B': "AutoML",
            'C': "NLP API",
            'D': "Vertex AI Custom notebooks"
        },
        'answers': ['B'],
        'explanation':
        "AutoML is the best choice to minimize the model development time and data preprocessing time",
        'references': []
    },
    {
        'question':
        "A real estate company, Zeellow, does great business buying and selling properties in the United States. Over the past few years, they have accumulated a big amount of historical data for US houses. \n\n"
        
        "Zeellow wants to use ML to forecast future sales by leveraging their historical sales data. The historical data is stored in cloud storage. You want to rapidly experiment with all the available data. How should you build and train your model?",
        'tags': [6, 'packt'],
        'options': {
            'A': "Load data into BigQuery and use BigQuery ML",
            'B': "Convert the data into CSV and use AutoML Tables",
            'C': "Convert the data into TFRecords and use TensorFlow",
            'D': "Convert and refactor the data into CSV format and use the built-in XGBoost library"
        },
        'answers': ['A'],
        'explanation':
        "BQ and BQML are the best options to experiment quickly with all the structured datasets stored in cloud storage.",
        'references': []
    },
    {
        'question':
        "A real estate company, Zeellow, uses ML to forecast future sales by leveraging their historical data. New data is coming in every week, and Zeellow needs to make sure the model is continually retrained to reflect the marketing trend. What should they do with the historical data and new data?",
        'tags': [7, 'packt'],
        'options': {
            'A': "Only use the new data for retraining",
            'B': "Update the datasets weekly with new data",
            'C': "Update the datasets with new data when model evaluation metrics do not meet the required criteria",
            'D': "Update the datasets monthly with new data"
        },
        'answers': ['C'],
        'explanation':
        "We need to retrain the model when the performance metrics do not meet the requirements.",
        'references': []
    },
    {
        'question':
        "A real estate company, Zeellow, uses ML to forecast future sales by leveraging their historical data. Their data science team trained and deployed a DL model in production half a year ago. Recently, the model is suffering from performance issues due to data distribution changes.\n\n"
        
        "The team is working on a strategy for model retraining. What is your suggestion?",
        'tags': [8, 'packt'],
        'options': {
            'A': "Monitor data skew and retrain the model",
            'B': "Retrain the model with fewer model features",
            'C': "Retrain the model to fix overfitting",
            'D': "Retrain the model with new data coming in every month"
        },
        'answers': ['A'],
        'explanation':
        "Model retraining is based on data value skews, which are significant changes in the statistical properties of data. When data skew is detected, this means that data patterns are changing, and we need to retrain the model to capture these changes.",
        'references': [
            'https://developers.google.com/machine-learning/guides/rules-of-ml/#rule_37_measure_trainingserving_skew']
    },
    {
        'question':
        "Recent research has indicated that when a certain kind of cancer, X, is developed in a human liver, there are usually other symptoms that can be identified as objects Y and Z from CT scan images. A hospital is using this research to train ML models with a label map of (X, Y, Z) on CT images. What cost functions should be used in this case?",
        'tags': [9, 'packt'],
        'options': {
            'A': "Binary cross-entropy",
            'B': "Categorical cross-entropy",
            'C': "Sparse categorical cross-entropy",
            'D': "Dense categorical cross-entropy"
        },
        'answers': ['B'],
        'explanation':
        "Categorical entropy is better to use when you want to prevent the model from giving more importance to a certain class – the same as the one-hot encoding idea.\n"
        "Sparse categorical entropy is more optimal when your classes are mutually exclusive (for example, when each sample belongs exactly to one class)",
        'references': []
    },
    {
        'question':
        "The data science team in your company has built a DNN model to forecast the sales value for an automobile company, based on historical data. As a Google ML Engineer, you need to verify that the features selected are good enough for the ML model",
        'tags': [10, 'packt'],
        'options': {
            'A': "Train the model with L1 regularization and verify that the loss is constant",
            'B': "Train the model with no regularization and verify that the loss is constant",
            'C': "Train the model with L2 regularization and verify that the loss is decreasing",
            'D': "Train the model with no regularization and verify that the loss is close to zero"
        },
        'answers': ['D'],
        'explanation':
        "",
        'references': [
            'Section Regularization in Chapter 4, Developing and Deploying ML Models']
    },
    {
        'question':
        "The data science team in your company has built a DNN model to forecast the sales value for a real estate company, based on historical data. As a Google ML Engineer, you find that the model has over 300 features and that you wish to remove some features that are not contributing to the target. What will you do?",
        'tags': [11, 'packt'],
        'options': {
            'A': "Use Explainable AI to understand the feature contributions and reduce the non-contributing ones.",
            'B': "Use L1 regularization to reduce features.",
            'C': "Use L2 regularization to reduce features.",
            'D': "Drop a feature at a time, train the model, and verify that it does not degrade the model. Remove these features."
        },
        'answers': ['A'],
        'explanation':
        "Explainable AI is one of the ways to understand which features are contributing and which ones are not",
        'references': []
    },
    {
        'question':
        "The data science team in your company has built a DNN model to forecast the sales value for a real estate company, based on historical data. They found that the model fits the training dataset well, but not the validation dataset. What would you do to improve the model?",
        'tags': [12, 'packt'],
        'options': {
            'A': "Apply a dropout parameter of 0.3 and decrease the learning rate by a factor of 10",
            'B': "Apply an L2 regularization parameter of 0.3 and decrease the learning rate by a factor of 10",
            'C': "Apply an L1 regularization parameter of 0.3 and increase the learning rate by a factor of 10",
            'D': "Tune the hyperparameters to optimize the L2 regularization and dropout parameters"
        },
        'answers': ['D'],
        'explanation':
        "The correct answer would be fitting to the general case",
        'references': []
    },
    {
        'question':
        "You are building a DL model for a customer service center. The model will input customers’ chat text and analyze their sentiments. What algorithm should be used for the model?",
        'tags': [13, 'packt'],
        'options': {
            'A': "MLP",
            'B': "Regression",
            'C': "CNN",
            'D': "RNN"
        },
        'answers': ['D'],
        'explanation':
        "Since text processing for sentiment analysis needs to process sequential data (time series), the best option is Recurrent Neural Networks (RNNs).",
        'references': []
    },
    {
        'question':
        "A health insurance company scans customers' hand-filled claim forms and stores them in Google Cloud Storage buckets in real time. They use ML models to recognize the handwritten texts. Since the claims may contain Personally Identifiable Information (PII), company policies require only authorized persons to access the information. What’s the best way to store and process this streaming data?",
        'tags': [14, 'packt'],
        'options': {
            'A': "Create two buckets and label them as sensitive and non-sensitive. Store data in the non-sensitive bucket first. Periodically scan it using the DLP API and move the sensitive data to the sensitive bucket.",
            'B': "Create one bucket to store the data. Only allow the ML service account access to it.",
            'C': "Create three buckets – quarantine, sensitive, and non-sensitive. Store all the data in the quarantine bucket first. Then, periodically scan it using the DLP API and move the data to either the sensitive or non-sensitive bucket.",
            'D': "Create three buckets – quarantine, sensitive, and non-sensitive. Store all the data in the quarantine bucket first. Then, once the file has been uploaded, trigger the DLP API to scan it, and move the data to either the sensitive or non-sensitive bucket."
        },
        'answers': ['D'],
        'explanation':
        "",
        'references': []
    },
    # {
    #     'question':
    #     'A real estate company, Zeellow, uses ML to forecast future sales by leveraging their historical data. The recent model training was able to achieve the desired forecast accuracy objective, but it took the data science team a long time. They want to decrease the training time without affecting the achieved model accuracy. What hyperparameter should the team adjust?",
    #     'tags': [15, 'packt'],
    #     'options': {
    #         'A': "Learning rate",
    #         'B': "Epochs",
    #         'C': "Scale tier",
    #         'D': "Batch size'
    #     },
    #     'answers': ['C'],
    #     'explanation':
    #     'Changing the other three parameters will change the model’s prediction accuracy.",
    #     'references': []
    # },
    {
        'question':
        "The data science team has built a DNN model to monitor and detect defective products using the images from the assembly line of an automobile manufacturing company. As a Google ML Engineer, you need to measure the performance of the ML model for the test dataset/images. Which of the following would you choose?",
        'tags': [16, 'packt'],
        'options': {
            'A': "The AUC value",
            'B': "The recall value",
            'C': "The precision value",
            'D': "The TP value"
        },
        'answers': ['A'],
        'explanation':
        "The AUC value measures how well the predictions are ranked rather than their absolute values. It is a classification threshold invariant and thus is the best way to measure the model’s performance.",
        'references': []
    },
    # {
    #     'question':
    #     'The data science team has built a DL model to monitor and detect defective products using the images from the assembly line of an automobile manufacturing company. Over time, the team has built multiple model versions in Vertex AI. As a Google ML Engineer, how will you compare the model versions?",
    #     'tags': [17, 'packt'],
    #     'options': {
    #         'A': "Compare the mean average precision for the model versions",
    #         'B': "Compare the model loss functions on the training dataset",
    #         'C': "Compare the model loss functions on the validation dataset",
    #         'D': "Compare the model loss functions on the testing dataset'
    #     },
    #     'answers': ['A'],
    #     'explanation':
    #     'It measures how well the different model versions perform over time: deploy your model as a model version and then create an evaluation job for that version. By comparing the mean average precision across the model versions, you can find the best performer.",
    #     'references': [
    #         'https://cloud.google.com/ai-platform/prediction/docs/continuous-evaluation/view-metrics#compare_mean_average_precision_across_models'
    #     ]
    # },
    {
        'question':
        "The data science team is building a recommendation engine for an e-commerce website using ML models to increase its business revenue, based on users’ similarities. What model would you choose?",
        'tags': [18, 'packt'],
        'options': {
            'A': "Collaborative filtering",
            'B': "Regression",
            'C': "Classification",
            'D': "Content-based filtering"
        },
        'answers': ['A'],
        'explanation':
        "Collaborative filtering uses similarities between users to provide recommendations",
        'references': [
            'https://developers.google.com/machine-learning/recommendation/overview/candidate-generation']
    },
    {
        'question':
        "The data science team is building a fraud-detection model for a credit card company, whose objective is to detect as much fraud as possible and avoid as many false alarms as possible. What confusion matrix index would you maximize for this model performance evaluation?",
        'tags': [19, 'packt'],
        'options': {
            'A': "Precision",
            'B': "Recall",
            'C': "The area under the PR curve",
            'D': "The area under the ROC curve"
        },
        'answers': ['C'],
        'explanation':
        "You want to maximize both precision and recall (maximize the area under the PR curve).",
        'references': [
            'https://machinelearningmastery.com/roc-curves-andprecision-recall-curves-for-imbalanced-classification/']
    },
    {
        'question':
        "The data science team is building a data pipeline for an auto manufacturing company, whose objective is to integrate all the data sources that exist in their on-premise facilities, via a codeless data ETL interface. What GCP service will you use?",
        'tags': [20, 'packt'],
        'options': {
            'A': "Dataproc",
            'B': "Dataflow",
            'C': "Dataprep",
            'D': "Data Fusion"
        },
        'answers': ['D'],
        'explanation':
        "Data Fusion is the best choice for data integration with a codeless interface",
        'references': [
            'https://cloud.google.com/data-fusion/docs/concepts/overview#using_the_code-free_web_ui']
    },
    {
        'question':
        "The data science team has built a TensorFlow model in BigQuery for a real estate company, whose objective is to integrate all their data models into the new Google Vertex. What’s the best strategy?",
        'tags': [21, 'packt'],
        'options': {
            'A': "Export the model from BigQuery ML",
            'B': "Register the BQML model to Vertex AI",
            'C': "Import the model into Vertex AI",
            'D': "Use Vertex AI as the middle stage"
        },
        'answers': ['B'],
        'explanation':
        "Vertex AI allows you to register a BQML model in it",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/managing-models-vertex']
    },
    {
        'question':
        "A real estate company, Zeellow, uses ML to forecast future house sale prices by leveraging their historical data. The data science team needs to build a model to predict US house sale prices based on the house location (US city-specific) and house type. What strategy is the best for feature engineering in this case?",
        'tags': [22, 'packt'],
        'options': {
            'A': "One feature cross: [latitude X longitude X housetype]",
            'B': "Two feature crosses: [binned latitude X binned housetype] and [binned longitude X binned housetype]",
            'C': "Three separate binned features: [binned latitude], [binned longitude], [binned housetype]",
            'D': "One feature cross: [binned latitude X binned longitude X binned housetype]"
        },
        'answers': ['D'],
        'explanation':
        "Crossing binned latitude with binned longitude enables the model to learn city-specific effects on house types. It prevents a change in latitude from producing the same result as a change in longitude",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/feature-crosses/check-your-understanding']
    },
    {
        'question':
        "A health insurance company scans customer’s hand-filled claim forms and stores them in Google Cloud Storage buckets in real time. The data scientist team has developed an AI documentation model to digitize the images. By the end of each day, the submitted forms need to be processed automatically. The model is ready for deployment. What strategy should the team use to process the forms?",
        'tags': [23, 'packt'],
        'options': {
            'A': "Vertex AI batch prediction",
            'B': "Vertex AI online prediction",
            'C': "Vertex AI ML pipeline prediction",
            'D': "Cloud Run to trigger prediction"
        },
        'answers': ['A'],
        'explanation':
        "We need to run the process at the end of each day, which implies batch processing",
        'references': []
    },
    {
        'question':
        "A real estate company, Zeellow, uses GCP ML to forecast future house sale prices by leveraging their historical data. Their data science team has about 30 members and each member has developed multiple versions of models using Vertex AI customer notebooks. What’s the best strategy to manage these different models and different versions developed by the team members?",
        'tags': [24, 'packt'],
        'options': {
            'A': "Set up IAM permissions to allow each member access to their notebooks, models, and versions",
            'B': "Create a GCP project for each member for clean management",
            'C': "Create a map from each member to their GCP resources using BQ",
            'D': "Apply label/tags to the resources when they’re created for scalable inventory/cost/access management"
        },
        'answers': ['D'],
        'explanation':
        "Resource tagging/labeling is the best way to manage ML resources for medium/big data science teams",
        'references': [
            'https://cloud.google.com/resource-manager/docs/tags/tags-creating-and-managing']
    },
    {
        'question':
        "Starbucks is an international coffee shop selling multiple products A, B, C… at different stores (1, 2, 3… using one-hot encoding and location binning). They are building stores and want to leverage ML models to predict product sales based on historical data (A1 is the data for product A sales at store 1). Following the best practices of splitting data into a training subset, validation subset, and testing subset, how should the data be distributed into these subsets?",
        'tags': [25, 'packt'],
        'options': {
            'A': "Distribute data randomly across the subsets:\n* Training set: [A1, B2, F1, E2, ...]\n* Testing set: [A2, C3, D2, F4, ...]\n* Validation set: [B1, C1, D9, C2...]",
            'B': "Distribute products randomly across the subsets:\n* Training set: [A1, A2, A3, E1, E2, ...]\n* Testing set: [B1, B2, C1, C2, C3, ...]\n* Validation set: [D1, D2, F1, F2, F3, ...]",
            'C': "Distribute stores randomly across subsets:\n* Training set: [A1, B1, C1, ...]\n* Testing set: [A2, C2, F2, ...]\n* Validation set: [D3, A3, C3, ...]",
            'D': "Aggregate the data groups by the cities where the stores are allocated and distribute cities randomly across subsets"
        },
        'answers': ['B'],
        'explanation':
        "If we divided things up at the product level so that the given products were only in the training subset, the validation subset, or the testing subset, the model would find it more difficult to get high accuracy on the validation since it would need to focus on the product characteristics/qualities",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/18th-century-literature']
    },
    {
        'question':
        "You are building a DL model with Keras that looks as follows:\n" 
        "model = tf.keras.sequential\n"
        "model.add(df.keras.layers.Dense(128,activation='relu',input_shape=(200, )))\n"
        "model.add(df.keras.layers.Dropout(rate=0.25))\n"
        "model.add(df.keras.layers.Dense(4,activation='relu'))\n"
        "model.add(df.keras.layers.Dropout(rate=0.25))\n"
        "model.add(Dense(2))\n\n"

        "How many trainable weights does this model have?",
        'tags': [26, 'packt'],
        'options': {
            'A': "200x128+128x4+4x2",
            'B': "200x128+128x4+2",
            'C': "200x128+129x4+5x2",
            'D': "200x128x0.25+128x4x0.25+4x2"
        },
        'answers': ['D'],
        'explanation':
        "",
        'references': []
    },
    {
        'question':
        "The data science team is building a DL model for a customer support center of a big ERP company, which has many ERP products and modules. The company receives over a million customer service calls every day and stores them in GCS. The call data must not leave the region in which the call originated and no PII can be stored/analyzed. The model will analyze calls for customer sentiments. How should you design a data pipeline for call processing, analyzing, and visualizing?",
        'tags': [27, 'packt'],
        'options': {
            'A': "GCS -> Speech2Text -> DLP -> BigQuery",
            'B': "GCS -> Pub/Sub -> Speech2Text -> DLP -> Datastore",
            'C': "GCS -> Speech2Text -> DLP -> BigTable",
            'D': "GCS -> Speech2Text -> DLP -> Cloud SQL"
        },
        'answers': ['A'],
        'explanation':
        "BigQuery is the best tool here to analyze and visualize",
        'references': []
    },
    {
        'question':
        "The data science team is building an ML model to monitor and detect defective products using the images from the assembly line of an automobile manufacturing company, which does not have reliable Wi-Fi near the assembly line. As a Google ML Engineer, you need to reduce the amount of time spent by quality control inspectors utilizing the model’s fast defect detection. Your company wants to implement the new ML model as soon as possible. Which model should you use?",
        'tags': [28, 'packt'],
        'options': {
            'A': "AutoML Vision",
            'B': "AutoML Vision Edge mobile-versatile-1",
            'C': "AutoML Vision Edge mobile-low-latency-1",
            'D': "AutoML Vision Edge mobile-high-accuracy-1"
        },
        'answers': ['C'],
        'explanation':
        "The question asks for a quick inspection time and prioritizes latency reduction",
        'references': [
            'https://cloud.google.com/vision/automl/docs/train-edge']
    },
    {
        'question':
        "A national hospital is leveraging Google Cloud and a cell phone app to build an ML model to forecast heart attacks based on age, gender, exercise, heart rate, blood pressure, and more. Since the health data is highly sensitive personal information and cannot be stored in cloud databases, how should you train and deploy the ML model?",
        'tags': [29, 'packt'],
        'options': {
            'A': "IoT with data encryption",
            'B': "Federated learning",
            'C': "Encrypted BQML",
            'D': "DLP API"
        },
        'answers': ['B'],
        'explanation':
        "With federated learning, all the data is collected, and the model is trained with algorithms across multiple decentralized edge devices such as cell phones or websites, without exchanging them",
        'references': []
    },
    ## whizlabs - Google Cloud Certified Professional Machine Learning Engineer - Practice Test 1
    {
        'question':
        "An industrial company wants to improve its quality system. It has developed its own deep neural network model with Tensorflow to identify the semi-finished products to be discarded with images taken from the production lines in the various production phases. During training, your custom model converges, but the tests are giving unsatisfactory results.\n"
        "What do you think might be the problem, and how could you proceed to fix it (pick 3)?",
        'tags': [1, 'whizlabs1'],
        'options': {
            'A': "You have used too few examples, you need to re-train with a larger set of images",
            'B': "You have to change the type of algorithm and use XGBoost",
            'C': "You have an overfitting problem",
            'D': "Decrease your Learning Rate hyperparameter",
            'E': "the model is too complex, you have to regularize the model and then make it simpler",
            'F': "Use L2 Ridge Regression"
        },
        'answers': ['C', 'E', 'F'],
        'explanation': 
        "* When you have a different trend between training and validation, you have an overfitting problem. More data may help you, but you have to simplify the model first.\n"
        "* The problem is not with the algorithm but is within feature management.\n"
        "* Decreasing the Learning Rate hyperparameter is useless. The model converges in training.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization', 'https://developers.google.com/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg']
    },
    {
        'question':
        "You need to develop and train a model capable of analyzing snapshots taken from a moving vehicle and detecting if obstacles arise. Your work environment is Vertex AI.\n"
        "Which technique or algorithm do you think is best to use?",
        'tags': [2, 'whizlabs1'],
        'options': {
            'A': "TabNet algorithm with TensorFlow",
            'B': "A linear learner with Tensorflow Estimator API",
            'C': "XGBoost with BigQueryML",
            'D': "TensorFlow Object Detection API"
        },
        'answers': ['D'],
        'explanation': 
        "TensorFlow Object Detection API is designed to identify and localize multiple objects within an image. So it is the best solution.\n"
        "* TabNet is used with tabular data, not images. It is a neural network that chooses the best features at each decision step in such a way that the model is optimized simpler.\n"
        "* linear learner is not suitable for images too. It can be applied to regression and classification predictions.\n"
        "* BigQueryML is designed for structured data, not images.",
        'references': [
            'https://github.com/tensorflow/models/tree/master/research/object_detection', 
            'https://cloud.google.com/vertex-ai/docs/training/training', 
            # 'https://cloud.google.com/ai-platform/training/docs', 
            'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/img/kites_detections_output.jpg'
        ]
    },
    {
        'question':
        "Your team works on a smart city project with wireless sensor networks and a set of gateways for transmitting sensor data. You have to cope with many design choices. You want, for each of the problems under study, to find the simplest solution.\n"
        "For example, it is necessary to decide on the placement of nodes so that the result is the most economical and inclusive. An algorithm without data tagging must be used.\n"
        "Which of the following choices do you think is the most suitable?",
        'tags': [3, 'whizlabs1'],
        'options': {
            'A': "K-means",
            'B': "Q-learning",
            'C': "K-Nearest Neighbors",
            'D': "Support Vector Machine(SVM)"
        },
        'answers': ['B'],
        'explanation':
        "Q-learning is an RL Reinforcement Learning algorithm. RL provides a software agent that evaluates possible solutions through a progressive reward in repeated attempts. It does not need to provide labels. But it requires a lot of data and several trials and the possibility to evaluate the validity of each attempt.\n"
        "The main RL algorithms are deep Q-network (DQN) and deep deterministic policy gradient (DDPG).\n"
        "* K-means is an unsupervised learning algorithm used for clustering problems. It is useful when you have to create similar groups of entities. So, even if there is no need to label data, it is not suitable for our scope.\n"
        "* K-NN is a supervised classification algorithm, therefore, labeled. New classifications are made by finding the closest known examples.\n"
        "* SVM is a supervised ML algorithm, too. K-NN distances are computed. These distances are not between data points, but with a hyper-plane, that better divides different classifications."
        ,
        'references': [
            'https://www.vebuso.com/2020/05/a-practical-application-of-k-nearest-neighbours-analysis-i/', 'https://towardsdatascience.com/reinforcement-learning-101-e24b50e1d292']
    },
    {
        'question':
        "The purpose of your current project is the recognition of genuine or forged signatures on checks and documents against regular signatures already stored by the Bank. There is obviously a very low incidence of fake signatures. The system must recognize which customer the signature belongs to and whether the signature is identified as genuine or skilled forged.\n"
        "What kind of ML model do you think is best to use?",
        'tags': [4, 'whizlabs1'],
        'options': {
            'A': "Binary logistic regression",
            'B': "Matrix Factorization",
            'C': "Convolutional Neural Networks",
            'D': "Multiclass logistic regression"
        },
        'answers': ['C'],
        'explanation':
        "A Convolutional Neural Network is a Deep Neural Network in which the layers are made up of processed sections of the source image. This technique allows you to simplify images and highlight shapes and features regardless of the physical position in which they may be found.\n"
        "* Binary logistic regression deals with a classification problem that may result in true or false, like with spam emails. The issue here is far more complex.\n"
        "* Matrix Factorization is used in recommender systems, like movies on Netflix. It is based on a user-item (movie) interaction matrix and the problem of reducing dimensionality.\n"
        "* Multiclass logistic regression deals with a classification problem with multiple solutions, fixed and finite classes. It is an extension of binary logistic regression with basically the same principles with the assumption of several independent variables. But in image recognition problems, the best results are achieved with CNN because they are capable of finding and relating patterns positioned in different ways on the images.",
        'references': [
            'https://towardsdatascience.com/convolution-neural-networks-a-beginners-guide-implementing-a-mnist-hand-written-digit-8aa60330d022', 'https://research.google.com/pubs/archive/42455.pdf']
    },
    {
        'question':
        "The purpose of your current project is the recognition of genuine or forged signatures on checks and documents against regular signatures already stored by the Bank. There is obviously a very low incidence of fake signatures. The system must recognize which customer the signature belongs to and whether the signature is identified as genuine or skilled forged.\n"
        "Which of the following technical specifications can't you use with CNN?",
        'tags': [5, 'whizlabs1'],
        'options': {
            'A': "Kernel Selection",
            'B': "Feature Cross",
            'C': "Stride ",
            'D': "Max pooling layer"
        },
        'answers': ['B'],
        'explanation': 
        "A cross of functions is a dome that creates new functions by multiplying (crossing) two or more functions.\n"
        "It has proved to be an important technique and is also used to introduce non-linearity to the model. We don't need it in our case.\n"
        "Filters or kernels are a computation on a sub-matrix of pixels.\n"
        "Stride is obtained by sliding the kernel by 1 pixel.\n"
        "A Max pooling layer is created taking the max value of a small region. It is used for simplification.\n"
        "Dropout is also for simplification or regularization. It randomly zeroes some of the matrix values in order to find out what can be discarded with minor loss (and no overfitting)",
        'references': [
            'https://towardsdatascience.com/convolution-neural-networks-a-beginners-guide-implementing-a-mnist-hand-written-digit-8aa60330d022']
    },
    {
        'question':
        "Your client has a large e-commerce Website that sells sports goods and especially scuba diving equipment. It has a seasonal business and has collected many sales data from its structured ERP and market trend databases. It wants to predict the demand of its customers both to increase business and improve logistics processes.\n"
        "Which of the following types of models and techniques should you focus on to obtain results quickly and with minimum effort?",
        'tags': [6, 'whizlabs1'],
        'options': {
            'A': "Custom Tensorflow model with an autoencoder neural network",
            'B': "Bigquery ML ARIMA",
            'C': "BigQuery Boosted Tree",
            'D': "BigQuery Linear regression"
        },
        'answers': ['B'],
        'explanation':
        "We need to manage time-series data. Bigquery ML ARIMA_PLUS can manage time-series forecasts. The model automatically handles anomalies, seasonality, and holidays.\n"
        "* A custom Tensorflow model needs more time and effort. Moreover, an autoencoder is a type of artificial neural network that is used in the case of unlabeled data (unsupervised learning). The autoencoder is an excellent system for generalization and therefore to reduce dimensionality, training the network to ignore insignificant data (\"noise\") is not our scope.\n"
        "* Boosted Tree is an ensemble of Decision Trees, so not suitable for time series.\n"
        "* Linear Regression cuts off seasonality. It is not what the customer wants.",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/arima-single-time-series-forecasting-tutorial', 
            'https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-time-series', 
            'https://cloud.google.com/bigquery-ml/docs/introduction']
    },
    {
        'question':
        "Your team is designing a fraud detection system for a major Bank. The requirements are:\n"
        "* Various banking applications will send transactions to the new system in real-time and in standard/normalized format.\n"
        "* The data will be stored in real-time with some statistical aggregations.\n"
        "* An ML model will be periodically trained for outlier detection.\n"
        "* The ML model will issue the probability of fraud for each transaction.\n"
        "* It is preferable to have no labeling and as little software development as possible.\n"
        "Which products would you choose (pick 3)?",
        'tags': [7, 'whizlabs1'],
        'options': {
            'A': "Dataprep",
            'B': "Dataproc",
            'C': "Dataflow Flex",
            'D': "Pub/Sub",
            'E': "Composer",
            'F': "BigQuery",
            'G': "BigTable"
        },
        'answers': ['C', 'D', 'F'],
        'explanation': 
        "The Optimal procedure to achieve the goal is:\n"
        "* Pub / Sub to capture the data stream"
        "* Dataflow Flex to aggregate and extract insights in real-time in BigQuery"
        "* BigQuery ML to create the models\n"
        "All the other solutions' usage will be sub-optimal and will need more effort. Practice with this lab for a detailed experience.\n"
        "For any further detail:",
        'references': [
            'https://cloud.google.com/solutions/building-anomaly-detection-dataflow-bigqueryml-dlp', 
            'https://cloud.google.com/architecture/detecting-anomalies-in-financial-transactions']
    },
    {
        'question':
        "Your team is designing a fraud detection system for a major Bank. The requirements are:\n"
        "* Various banking applications will send transactions to the new system in real-time and in standard/normalized format.\n"
        "* The data will be stored in real-time with some statistical aggregations.\n"
        "* An ML model will be periodically trained for outlier detection.\n"
        "* The ML model will issue the probability of fraud for each transaction.\n"
        "* It is preferable to have no labeling and as little software development as possible.\n"
        "Which products would you choose (pick 2)?",
        'tags': [8, 'whizlabs1'],
        'options': {
            'A': "K-means",
            'B': "Decision Tree",
            'C': "Random Forest",
            'D': "Matrix Factorization",
            'E': "Boosted Tree - XGBoost"
        },
        'answers': ['A', 'E'],
        'explanation': 
        "The k-means clustering is a mathematical and statistical method on numerical vectors that divides ann observes k clusters. Each example belongs to the cluster with the closest mean (cluster centroid).\n"
        "In ML, it is an unsupervised classification method and is widely used to detect unusual or outlier movements. For these reasons, it is one of the main methods for fraud detection.\n"
        "But it is not the only method because not all frauds are linked to strange movements. There may be other factors.\n"
        "XGBoost, which as you can see from the figure, is an evolution of the decision trees, has recently been widely used in this field and has had many positive results.\n"
        "It is an open-source project and this is the description from its Github page:\n"
        "XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solves many data science problems in a fast and accurate way. The same code runs on major distributed environments (Kubernetes, Hadoop, SGE, MPI, Dask) and can solve problems beyond billions of examples.\n"
        "* Decision Tree and Random Forest are suboptimal because of just Decision Trees.\n"
        "* Matrix Factorization is for recommender systems. So, it predicts the preference of an item based on the experience of other users. Not suitable for us.",
        'references': [
            'https://cloud.google.com/solutions/building-anomaly-detection-dataflow-bigqueryml-dlp',
            'https://cloud.google.com/architecture/detecting-anomalies-in-financial-transactions',
            'https://medium.com/@adityakumar24jun/xgboost-algorithm-the-new-king-c4a64ea677bf'
        ]
    },
    {
        'question': 
        "In your company, you train and deploy several ML models with Tensorflow. You use on-prem servers, but you often find it challenging to manage the most expensive training and control and update the models. You are looking for a system that can handle all these tasks.\n"
        "Which solutions can you adopt (Select TWO)?",
        'tags': [9, 'whizlabs1'],
        'options': {
            'A': "Kubeflow to run on Google Kubernetes Engine",
            'B': "Vertex AI",
            'C': "Use Scikit-Learn that is simple and powerful",
            'D': "Use SageMaker managed services"
        },
        'answers': ['A', 'B'],
        'explanation': 
        "Kubeflow Pipelines is an open-source platform designed specifically for creating and deploying ML workflows based on Docker containers.\n"
        "Their main features:\n"
        "* Using packaged templates in Docker images in a K8s environment\n"
        "* Manage your various tests/experiments\n"
        "* Simplifying the orchestration of ML pipelines\n"
        "* Reuse components and pipelines\n\n"
        "Vertex AI is an integrated suite of ML services that:\n"
        "* Train an ML model both without code (Auto ML) and with custom\n"
        "* Evaluate and tune a model\n"
        "* Deploy models\n"
        "* Manage prediction: Batch, Online and monitoring\n"
        "* Manage model versions: workflows and retraining\n"
        "* Manage the complete model maintenance cycle\n\n"
        "* Scikit-learn is an ML platform with many standard algorithms easy and immediate to use. TensorFlow (from the official doc) is an end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art into ML, and developers easily build and deploy ML-powered applications.\n\n"
        "So, there are 2 different platforms, even if there is Scikit Flow that integrates the two.\n"
        "Scikit-learn doesn't manage ML Pipelines.\n"
        "* SageMaker is an AWS ML product.",
        'references': [
            'https://cloud.google.com/ai-platform/training/docs/tensorflow-2',
            'https://cloud.google.com/vertex-ai'
        ]
    },
    {
        'question':
        "You have an NLP model for your company's Customer Care and Support Office. This model evaluates the general satisfaction of customers on the main categories of services offered and has always provided satisfactory performances.\n"
        "You have recently expanded the range of your services and want to refine / update your model. You also want to activate procedures that automate these processes.\n"
        "Which choices among the following do you prefer in the Cloud GCP?",
        'tags': [10, 'whizlabs1'],
        'options': {
            'A': "You don't need to change anything. If the model is well made and has no overfitting, it will be able to handle anything.",
            'B': "Retrain using information from the last week of work only.",
            'C': "Add examples with new product data and still regularly re-train and evaluate new models.",
            'D': "Make a separate model with new product data and create the model ensemble."
        },
        'answers': ['C'],
        'explanation': 
        "Creating and using templates is not a one-shot activity. But, like most processes, it is an ongoing one, because the underlying factors can vary over time.\n"
        "Therefore, you need to continuously monitor the processes and retrain the model also on newer data, if you find that the frequency distributions of the data vary from the original configuration. It may also be necessary or desirable to create a new model.\n"
        "Generally, a periodic schedule is adopted every month or week.\n"
        "For this very reason, all the other answers are not exact.",
        'references': [
            'https://cloud.google.com/ai-platform/pipelines/docs',
            'https://medium.com/kubeflow/automated-model-retraining-with-kubeflow-pipelines-691a5f211701'
        ]
    },
    {
        'question':
        "Your company is designing a series of models aimed at optimal customer care management.\n"
        "For this purpose, all written and voice communications with customers are recorded so that they can be classified and managed.\n"
        "The problem is that Clients often provide private information that cannot be distributed and disclosed.\n"
        "Which of the following techniques can you use (pick 3)?",
        'tags': [11, 'whizlabs1'],
        'options': {
            'A': "Cloud Data Loss Prevention API (DLP)",
            'B': "CNN - Convolutional Neural Network",
            'C': "Cloud Speech API",
            'D': "Cloud Vision API"
        },
        'answers': ['A', 'C', 'D'],
        'explanation': 
        "Cloud Data Loss Prevention is a managed service specially designed to discover sensitive data automatically that may be protected. It could be used for personal codes, credit card numbers, addresses and any private contact details, etc.\n"
        "Cloud Speech API is useful if you have audio recordings as it is a speech-to-text service.\n"
        "Cloud Vision API has a built-in text-detection service. So you can get text from images.\n"
        "* A Convolutional Neural Network is a Deep Neural Network in which the layers are made up of processed sections of the source image. So, it is a successful method for image and shape classification.",
        'references': [
            'https://cloud.google.com/architecture/sensitive-data-and-ml-datasets']
    },
    {
        'question':
        "Your team is working for a major apparel company that is developing an online business with significant investments.\n"
        "The company adopted Analytics-360. So, it can achieve a lot of data on the activities of its customers and on the interest of the various commercial initiatives of the websites, such as (from Google Analytics-360):\n"
        "* Average bounce rate per dimension\n"
        "* Average number of product page views by purchaser type\n"
        "* Average number of transactions per purchaser\n"
        "* Average amount of money spent per session\n"
        "* Sequence of hits (pathing analysis)\n"
        "* Multiple custom dimensions at hit or session level\n"
        "* Average number of user interactions before purchase\n"
        "The first thing management wants is to categorize customers to determine which types are more likely to buy.\n"
        "Subsequently, further models will be created to incentivize the most interesting customers better and boost sales.\n"
        "You have a lot of work to do and you want to start quickly. What techniques do you use in this first phase (pick 2)?",
        'tags': [12, 'whizlabs1'],
        'options': {
            'A': "BigQuery e BigQueryML",
            'B': "Cloud Storage con AVRO",
            'C': "Vertex AI TensorBoard",
            'D': "Binary Classification",
            'E': "K-means",
            'F': "KNN",
            'G': "Deep Neural Network"
        },
        'answers': ['A', 'E'],
        'explanation': 
        "It is necessary to create different groups of customers based on purchases and their characteristics for these requirements.\n"
        "We are in the field of unsupervised learning. BigQuery is already set up both for data acquisition and for training, validation and use of this kind of model.\n"
        "The K-means model in BigQuery ML uses a technique called clustering. Clustering is a statistical technique that allows, in our case, to classify customers with similar behaviors for marketing automatically.\n"
        "All the other answers address more complex and more cumbersome solutions.\n"
        "Vertex AI TensorBoard is suitable to set up visualizations for ML experiments.\n"
        "Furthermore, while the others are all supervised, we do not have ready-made solutions, but we want the model to provide us with the required categories.",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/kmeans-tutorial',
            'https://cloud.google.com/architecture/building-k-means-clustering-model'
        ]
    },
    {
        'question':
        "Your team prepared a custom model with Tensorflow that forecasts, based on diagnostic images, which cases need more analysis and medical support.\n"
        "The accuracy of the model is very high. But when it is deployed in production, the medical staff is very dissatisfied.\n"
        "What is the most likely motivation?",
        'tags': [13, 'whizlabs1'],
        'options': {
            'A': "Logistic regression with a classification threshold too high",
            'B': "DNN Model with overfitting",
            'C': "DNN Model with underfitting",
            'D': "You have to perform feature crosses"
        },
        'answers': ['A'],
        'explanation': 
        "When there is an imbalance between true and false ratios in binary classification, it is necessary to modify the classification threshold so that the most probable errors are those with minor consequences. In our case, it is better to be wrong with a healthy person than with a sick one.\n"
        "Accuracy is the number of correct predictions on the total of predictions done.\n"
        "Let’s imagine that we have 100 predictions, and 95 of them are correct. That is 95%. It looks almost perfect.\n"
        "But we assume that the system has foreseen 94 true negative cases and only one true positive case, and one case of false positive, and 4 cases of false negative.\n"
        "So, the model predicted 98 healthy when they were 95 and 2 suspected cases when they were 5.\n"
        "The problem is that sick patients are, luckily, a minimal percentage. But it is important that they are intercepted. So, our model failed because it correctly identified only 1 case out of the total of 5 real positives that is 20% (recall). It also identified 2 positives, one of which was negative, i.e. 50% (precision).\n"
        "It's not good at all.\n"
        "* Precision: Rate of correct positive identifications\n"
        "* Recall: Rate of real positives correctly identified\n"
        "To calibrate the result, we need to change the threshold we use to decide between positive and negative. The model does not return 0 and 1 but a value between 0 and 1 (sigmoid activation function). In our case, we have to choose a threshold lower than 0.5 to classify it as positive. In this way, we risk carrying out further investigations on the healthy but being able to treat more sick patients. It is definitely the desired result.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall'
        ]
    },
    {
        'question':
        "You work in a company that has acquired an advanced consulting services company. Management wants to analyze all past important projects and key customer relationships. The consulting company does not have an application that manages this data in a structured way but is certified for the quality of its services. All its documents follow specific rules.\n"
        "It was decided to acquire structured information on projects, areas of expertise and customers through the analysis of these documents.\n"
        "You're looking for ML methodologies that make this process quicker and easier.\n"
        "What are the better choices in GCP?",
        'tags': [14, 'whizlabs1'],
        'options': {
            'A': "Cloud Vision",
            'B': "Cloud Natural Language API",
            'C': "Document AI",
            'D': "AutoML Natural Language"
        },
        'answers': ['C'],
        'explanation': 
        "Document AI is the ideal broad-spectrum solution. It is a service that gives a complete solution with computer vision and OCR, NLP and data management. It allows you to extract and structure information automatically from documents. It can also enrich them with the Google Knowledge Graph to verify company names, addresses, and telephone numbers to draw additional or updated information.\n"
        "All other answers are incorrect because their functions are already built into Document AI.",
        'references': [
            'https://cloud.google.com/document-ai',
            'https://cloud.google.com/vision/pricing',
            'https://cloud.google.com/natural-language/pricing',
            'https://cloud.google.com/natural-language/automl/pricing'
        ]
    },
    {
        'question': 
            "Your customer has an online dating platform that, among other things, analyzes the degree of affinity between the various people. Obviously, it already uses ML models and uses, in particular, XGBoost, the gradient boosting decision tree algorithm, and is obtaining excellent results.\n"
            "All its development processes follow CI / CD specifications and use Docker containers. The requirement is to classify users in various ways and update models frequently, based on new parameters entered into the platform by the users themselves.\n"
            "So, the problem you are called to solve is how to optimize frequently re-trained operations with an optimized workflow system. Which solution among these proposals can best solve your needs?",
        'tags': [15, 'whizlabs1'],
        'options': {
            'A': "Deploy the model on BigQuery ML and setup a job",
            'B': "Use Kubeflow Pipelines to design and execute your workflow",
            'C': "Use Vertex AI",
            'D': "Orchestrate activities with Google Cloud Workflows",
            'E': "Develop procedures with Pub/Sub and Cloud Run",
            'F': "Schedule processes with Cloud Composer"
        },
        'answers': ['B'],
        'explanation': 
            "Kubeflow Pipelines is the ideal solution because it is a platform designed specifically for creating and deploying ML workflows based on Docker containers. So, it is the only answer that meets all requirements.\n\n"
            "The main functions of Kubeflow Pipelines are:\n"
            "* Using packaged templates in Docker images in a K8s environment\n"
            "* Manage your various tests/experiments\n"
            "* Simplifying the orchestration of ML pipelines\n"
            "* Reuse components and pipelines\n"
            "It is within the Kubeflow ecosystem, which is the machine learning toolkit for Kubernetes\n"
            "Vertex AI Model Monitoring is useful for detecting if the model is no longer suitable for your needs.\n"
            "Creating ML workflows is possible with Vertex AI Pipelines.\n"
            "The other answers may be partially correct but do not resolve all items or need to add more coding.",
        'references': [
            'https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/',
            'https://www.kubeflow.org/docs/started/kubeflow-overview/'
        ]
    },
    {
        'question':
        "You have an ML model designed for an industrial company that provides the correct price to buy goods based on a series of elements, such as the quantity requested, the level of quality and other specific variables for different types of products.\n"
        "You have built a linear regression model that works well but whose performance you want to optimize.\n"
        "Which of these techniques could you use?",
        'tags': [16, 'whizlabs1'],
        'options': {
            'A': "Clipping",
            'B': "Log scaling",
            'C': "Z-score",
            'D': "Scaling to a range",
            'E': "All of them"
        },
        'answers': ['E'],
        'explanation': 
        "Feature clipping eliminates outliers that are too high or too low.\n"
        "Scaling means transforming feature values into a standard range, from 0 and 1 or sometimes -1 to +1. It's okay when you have an even distribution between minimum and maximum.\n"
        "When you don't have a fairly uniform distribution, you can instead use Log Scaling which can compress the data range: x1 = log (x)\n"
        "Z-Score is similar to scaling, but uses the deviation from the mean divided by the standard deviation, which is the classic index of variability. So, it gives how many standard deviations each value is away from the mean.\n"
        "All these methods maintain the differences between values, but limit the range. So the computation is lighter.",
        'references': [
            'https://developers.google.com/machine-learning/data-prep/transform/normalization'
        ]
    },
    {
        'question':
        "You are starting to operate as a Data Scientist and are working on a deep neural network model with Tensorflow to optimize customer satisfaction for after-sales services to create greater client loyalty.\n"
        "You are doing Feature Engineering, and your focus is to minimize bias and increase accuracy. Your coordinator has told you that by doing so you risk having problems. He explained to you that, in addition to the bias, you must consider another factor to be optimized. Which one?",
        'tags': [17, 'whizlabs1'],
        'options': {
            'A': "Blending",
            'B': "Learning Rate",
            'C': "Feature Cross",
            'D': "Bagging",
            'E': "Variance"
        },
        'answers': ['E'],
        'explanation':
        "The variance indicates how much function f (X) can change with a different training dataset. Obviously, different estimates will correspond to different training datasets, but a good model should reduce this gap to a minimum.\n"
        "The bias-variance dilemma is an attempt to minimize both bias and variance.\n"
        "The bias error is the non-estimable part of the learning algorithm. The higher it is, the more underfitting there is.\n"
        "Variance is the sensitivity to differences in the training set. The higher it is, the more overfitting there is.\n"
        "* Blending indicates an ensemble of ML models.\n"
        "* Learning Rate is a hyperparameter in neural networks.\n"
        "* Feature Cross is the method for obtaining new features by multiplying other ones.\n"
        "* Bagging is an ensemble method like Blending.",
        'references': [
            'https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff']
    },
    {
        'question':
        "Your company supplies environmental management services and has a network of sensors that acquire information uploaded to the Cloud to be pre-processed and managed with some ML models with dynamic dashboards used by customers.\n"
        "Periodically, the models are retrained and re-deployed, with a rather complex pipeline on VM clusters:\n"
        "* New data is streamed from Dataflow\n"
        "* Data is transformed through aggregations and normalizations (z-scores)\n"
        "* The model is periodically retrained and evaluated\n"
        "* New Docker images are created and stored\n"
        "You want to simplify the pipeline as much as possible and use fully managed or even serverless services as far as you can.\n"
        "Which do you choose from the following services?",
        'tags': [18, 'whizlabs1'],
        'options': {
            'A': "Kubeflow",
            'B': "Vertex AI custom training",
            'C': "BigQuery and BigQuery ML",
            'D': "TFX"
        },
        'answers': ['C'],
        'explanation': 
        "BigQuery and BigQueryML are powerful services for data analysis and machine learning.\n"
        "They are fully serverless services that can process petabytes of data in public and private datasets and even data stored in files.\n"
        "BigQuery works with standard SQL and has a CLI interface: bq.\n"
        "You can use BigQuery jobs to automate and schedule tasks and operations.\n"
        "With BigQueryML, you can train models with a rich set of algorithms with data already stored in the Cloud. You may perform feature engineering and hyperparameter tuning and export a BigQuery ML model to a Docker image as required.\n"
        "All other services are useful in ML pipelines, but they aren't that easy and ready to use.\n"
        "With Vertex AI you can use AutoML training and custom training in the same environment.\n"
        "It's a managed but not a serverless service, especially for custom training.\n"
        "It obviously has a rich set of features for managing ML pipelines.",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/export-model-tutorial',
            'https://cloud.google.com/bigquery/docs/jobs-overview',
            'https://cloud.google.com/bigquery/',
            'https://cloud.google.com/vertex-ai/docs/training/custom-training'
        ]
    },
    {
        'question':
        "Your company runs an e-commerce site. You produced static deep learning models with Tensorflow that process Analytics-360 data. They have been in production for some time. Initially, they gave you excellent results, but gradually, the accuracy has progressively decreased. You retrained the models with the new data and solved the problem.\n"
        "At this point, you want to automate the process using the Google Cloud environment. Which of these solutions allows you to quickly reach your goal?",
        'tags': [19, 'whizlabs1'],
        'options': {
            'A': "Cluster Compute Engine and KubeFlow",
            'B': "GKE and TFX",
            'C': "GKE and KubeFlow",
            'D': "Vertex AI Pipelines and TensorFlow Extended TFX"
        },
        'answers': ['D'],
        'explanation':
        "TFX is a platform that allows you to create scalable production ML pipelines for TensorFlow projects, therefore Kubeflow.\n"
        "It, therefore, allows you to manage the entire life cycle seamlessly from modeling, training, and validation, up to production start-up and management of the inference service.\n"
        "Vertex AI Pipelines can run pipelines built using TFX:\n"
        "* You can configure a Cluster\n"
        "* Select basic parameters and click create\n"
        "* You get your Kubeflow and Kubernetes launched\n"
        "All the other answers are correct, but not optimal for a quick and managed solution.",
        'references': [
            'https://cloud.google.com/ai-platform/pipelines/docs',
            'https://developers.google.com/machine-learning/crash-course/production-ml-systems',
            'https://www.tensorflow.org/tfx/guide',
            'https://www.youtube.com/watch?v=Mxk4qmO_1B4']
    },
    {
        'question':
        "You have a Linear Regression model for the optimal management of supplies to a sales network based on a large number of different driving factors. You want to simplify the model to make it more efficient and faster. Your first goal is to synthesize the features without losing the information content that comes from them.\n"
        "Which of these is the best technique?",
        'tags': [20, 'whizlabs1'],
        'options': {
            'A': "Feature Crosses",
            'B': "Principal component analysis (PCA)",
            'C': "Embeddings",
            'D': "Functional Data Analysis"
        },
        'answers': ['B'],
        'explanation': 
        "Principal component analysis is a technique to reduce the number of features by creating new variables obtained from linear combinations or mixes of the original variables, which can then replace them but retain most of the information useful for the model. In addition, the new features are all independent of each other.\n"
        "The new variables are called principal components.\n"
        "A linear model is assumed as a basis. Therefore, the variables are independent of each other.\n"
        "* Feature Crosses are for the same objective, but they add non-linearity.\n"
        "* Embeddings, which transform large sparse vectors into smaller vectors are used for categorical data.\n"
        "* Functional Data Analysis has the goal to cope with complexity, but it is used when it is possible to substitute features with functions- not our case.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/embeddings/categorical-input-data',
            'https://builtin.com/data-science/step-step-explanation-principal-component-analysis',
            'https://en.wikipedia.org/wiki/Principal_component_analysis'
        ]
    },
    {
        'question':
        "TerramEarth is a company that builds heavy equipment for mining and agriculture.\n"
        "During maintenance services for vehicles produced by TerramEarth at the service centers, information relating to their use is downloaded. Every evening, this data flows into the data center, is consolidated and sent to the Cloud.\n"
        "TerramEarth has an ML model that predicts component failures and optimizes the procurement of spare parts for service centers to offer customers the highest level of service. TerramEarth wants to automate the redevelopment and distribution process every time it receives a new file.\n"
        "What is the best service to start the process?",
        'tags': [21, 'whizlabs1'],
        'options': {
            'A': "Cloud Storage trigger with Cloud Functions",
            'B': "Cloud Scheduler every night",
            'C': "Pub/Sub",
            'D': "Cloud Run and Cloud Build"
        },
        'answers': ['A'],
        'explanation': 
        "Files are received from Cloud Storage, which has native triggers for all the events related to its file management.\n"
        "So, we may start a Cloud Function that may activate any Cloud Service as soon as the file is received.\n"
        "Cloud Storage triggers may also activate a Pub/Sub notification, just a little more complex.\n"
        "It is the simplest and most direct solution of all the answers.",
        'references': [
            'https://cloud.google.com/functions/docs/calling/storage',
            'https://cloud.google.com/blog/products/gcp/cloud-storage-introduces-cloud-pub-sub-notifications'
        ]
    },
    {
        'question':
        "You work in a major banking institution. The Management has decided to rapidly launch a bank loan service, as the Government has created a series of “first home” facilities for the younger population.\n"
        "The goal is to carry out the automatic management of the required documents (certificates, origin documents, legal information) so that the practice can be built and verified automatically using the data and documents provided by customers and can be managed in a short time and with the minimum contribution of the scarce specialized personnel.\n"
        "Which of these GCP services can you use?",
        'tags': [22, 'whizlabs1'],
        'options': {
            'A': "Dialogflow",
            'B': "Document AI",
            'C': "Cloud Natural Language API",
            'D': "AutoML"
        },
        'answers': ['B'],
        'explanation': 
        "Document AI is the perfect solution because it is a complete service for the automatic understanding of documents and their management.\n"
        "It integrates computer natural language processing, OCR, and vision and can create pre-trained templates aimed at intelligent document administration.\n"
        "* Dialogflow is for speech Dialogs, not written documents.\n"
        "* NLP is integrated into Document AI.\n"
        "* functions like AutoML are integrated into Document AI, too."
        ,
        'references': [
            'https://cloud.google.com/document-ai',
            'https://cloud.google.com/solutions/lending-doc-ai',
            'https://www.qwiklabs.com/focuses/12733?&parent=catalog',
            'https://cloud.google.com/automl',
            'https://cloud.google.com/blog/products/ai-machine-learning/google-cloud-announces-document-ai-platform'
        ]
    },
    {
        'question':
        "Your company does not have an excellent ML experience. They want to start with a service that is as smooth, simple and managed as possible. The idea is to use BigQuery ML. Therefore, you are considering whether it can cover all the functionality you need.\n"
        "Which of the following features are not present in BigQuery ML natively?",
        'tags': [23, 'whizlabs1'],
        'options': {
            'A': "Exploratory data analysis",
            'B': "Feature selection",
            'C': "Model building",
            'D': "Training",
            'E': "Hyperparameter tuning",
            'F': "Automatic deployment and serving"
        },
        'answers': ['F'],
        'explanation': 
        "BigQuery is perfect for Analytics. So, exploratory data analysis and feature selection are simple and very easy to perform with the power of SQL and the ability to query petabytes of data.\n"
        "BigQuery ML offers all other features except automatic deployment and serving.\n"
        "BigQuery ML can simply export a model (packaged in a container image) to Cloud Storage.",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-export-model',
            'https://cloud.google.com/blog/products/data-analytics/automl-tables-now-generally-available-bigquery-ml'
        ]
    },
    {
        'question':
        "Your client has an e-commerce site for commercial spare parts for cars with competitive prices. It started with the small car sector but is continually adding products. Since 80% of them operate in a B2B market, he wants to ensure that his customers are encouraged to use the new products that he gradually offers on the site quickly and profitably.\n"
        "Which GCP service can be valuable in this regard and in what way?",
        'tags': [24, 'whizlabs1'],
        'options': {
            'A': "Create a Tensorflow model using Matrix factorization",
            'B': "Use Recommendations AI",
            'C': "Import the Product Catalog",
            'D': "Record / Import User events"
        },
        'answers': ['B'],
        'explanation':
        "Recommendations AI is a ready-to-use service for all the requirements shown in the question. You don’t need to create models, tune, train, all that is done by the service with your data. Also, the delivery is automatically done, with high-quality recommendations via web, mobile, email. So, it can be used directly on websites during user sessions.\n"
        "* Create a Tensorflow model using Matrix factorization could be OK, but it needs a lot of work.\n"
        "* Import the Product Catalog and Record / Import User events deal only with data management, not creating recommendations.",
        'references': [
            'https://cloud.google.com/retail/recommendations-ai/docs/create-models',
            'https://cloud.google.com/recommendations%C2%A0'
        ]
    },
    {
        'question':
        "Your client has an e-commerce site for commercial spare parts for cars with competitive prices. It started with the small car sector but is continually adding products. Since 80% of them operate in a B2B market, he wants to ensure that his customers are encouraged to use the new products that he gradually offers on the site quickly and profitably.\n"
        "You decided on Recommendations AI. What specific recommendation model type is not useful for new products?",
        'tags': [25, 'whizlabs1'],
        'options': {
            'A': "Others You May Like",
            'B': "Frequently Bought Together",
            'C': "Recommended for You",
            'D': "Recently Viewed"
        },
        'answers': ['D'],
        'explanation': 
        "The \"Recently Viewed\" recommendation is not for new products, and it is not a recommendation either.\n"
        "It provides the list of products the user has recently viewed, starting with the last.",
        'references': [
            'https://cloud.google.com/retail/recommendations-ai/docs/placements#oyml',
            'https://cloud.google.com/retail/recommendations-ai/docs/placements#fbt',
            'https://cloud.google.com/retail/recommendations-ai/docs/placements#rfy',
            'https://cloud.google.com/retail/recommendations-ai/docs/placements#rv'
        ]
    },
    {
        'question':
        "Your business makes excellent use of ML models. Many of these were developed with Tensorflow. But lately, you've been making good use of AutoML to make your design work leaner, faster, and more efficient.\n"
        "You are looking for an environment that organizes and manages training, validation and tuning, and updating models with new data, distribution and monitoring in production.\n"
        "Which of these do you think is the best solution?",
        'tags': [26, 'whizlabs1'],
        'options': {
            'A': "Deploy Tensorflow on Kubernetes",
            'B': "Leverage Kubeflow Pipelines",
            'C': "Adopt Vertex AI: custom tooling and pipelines",
            'D': "Migrate all models to BigQueryML with AutoML",
            'E': "Migrate all models to AutoML Tables"
        },
        'answers': ['C'],
        'explanation': 
        "Vertex AI combines AutoML, custom models and ML pipeline management through to production.\n"
        "Vertex AI integrates many GCP ML services, especially AutoML and custom training, and includes many different tools to help you in every step of the ML workflow.\n"
        "So, Vertex AI offers two strategies for model training: AutoML and Personalized training.\n"
        "Machine learning operations (MLOps) is the practice of using DevOps for machine learning (ML).\n"
        "DevOps strategies automate the release of code changes and control of systems, resulting in greater security and less time to get systems up and running.\n"
        "All the other solutions are suitable for production. But, given these requirements, Vertex AI, with the AutoML solution's strong inclusion, is the best and the most productive one.\n",
        'references': [
            'https://cloud.google.com/vertex-ai/docs',
            'https://cloud.google.com/vertex-ai/docs/pipelines/introduction',
            'https://codelabs.developers.google.com/codelabs/vertex-ai-custom-models#1'
        ]
    },
    {
        'question':
        "You are a junior Data Scientist. You need to create a multi-class classification Machine Learning model with Keras Sequential model API. In particular, your model must indicate the main categories of a text.\n"
        "Which of the following techniques should not be used?",
        'tags': [27, 'whizlabs1'],
        'options': {
            'A': "Feedforward Neural Network",
            'B': "N-grams for tokenize text",
            'C': "K-means",
            'D': "Softmax function",
            'E': "Pre-trained embeddings",
            'F': "Dropout layer",
            'G': "Categorical cross-entropy"
        },
        'answers': ['C'],
        'explanation':
        "The answers identify the main techniques to be used for a multi-class classification Machine Learning model. For more details, see the step-by-step example.\n"
        "The only unsuitable element is K-means clustering, one of the most popular unsupervised machine learning algorithms. Therefore, it is out of this scope.\n"
        "* Feedforward Neural Network is a kind of DNN, widely used for many applications.\n"
        "* N-grams for tokenizing text is a contiguous sequence of items (usually words) in NLP.\n"
        "* Softmax is an activation function for multi-class classification.\n"
        "* Embeddings are used for reducing high-dimensional tensors, so categories, too.\n"
        "* The Dropout layer is used for regularization, eliminating input features randomly.\n"
        "* Categorical cross-entropy is a loss function for multi-class classification.",
        'references': [
            'https://developers.google.com/machine-learning/guides/text-classification/',
            'https://en.wikipedia.org/wiki/N-gram',
            'https://en.wikipedia.org/wiki/K-means_clustering',
            'https://en.wikipedia.org/wiki/Multilayer_perceptron',
            'https://developers.google.com/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg'
        ]
    },
    {
        'question':
        "You work for a digital publishing website with an excellent technical and cultural level, where you have both famous authors and unknown experts who express ideas and insights.\n"
        "You, therefore, have an extremely demanding audience with strong interests that can be of various types.\n"
        "Users have a small set of articles that they can read for free every month. Then they need to sign up for a paid subscription.\n"
        "You have been asked to prepare an ML training model that processes user readings and article preferences. You need to predict trends and topics that users will prefer.\n"
        "But when you train your DNN with Tensorflow, your input data does not fit into RAM memory.\n"
        "What can you do in the simplest way?",
        'tags': [28, 'whizlabs1'],
        'options': {
            'A': "Use tf.data.Dataset",
            'B': "Use a queue with tf.train.shuffle_batch",
            'C': "Use pandas.DataFrame",
            'D': "Use a NumPy array"
        },
        'answers': ['A'],
        'explanation': 
        "The tf.data.Dataset allows you to manage a set of complex elements made up of several inner components.\n"
        "It is designed to create efficient input pipelines and to iterate over the data for their processing.\n"
        "These iterations happen in streaming. So, they work even if the input matrix is very large and doesn’t fit in memory.\n"
        "* A queue with tf.train.shuffle_batch is far more complex, even if it is feasible.\n"
        "* A pandas.DataFrame and a NumPy array work in real memory, so they don’t solve the problem at all.",
        'references': [
            'https://www.tensorflow.org/api_docs/python/tf/data/Dataset',
            'https://www.kaggle.com/jalammar/intro-to-data-input-pipelines-with-tf-data'
        ]
    },
    {
        'question':
        "TerramEarth is a company that builds heavy equipment for mining and agriculture.\n"
        "It is developing a series of ML models for different activities: manufacturing, procurement, logistics, marketing, customer service and vehicle tracking.\n"
        "TerramEarth uses Google Vertex AI and wants to scale training and inference processes in a managed way.\n"
        "It is necessary to forecast whether a vehicle, based on the data collected during the maintenance service, has risks of failures in the next six months in order to recommend an extraordinary service operation.\n"
        "Which kind of technology/model should you advise using?",
        'tags': [29, 'whizlabs1'],
        'options': {
            'A': "Feedforward Neural Network",
            'B': "Convolutional Neural Network",
            'C': "Recurrent Neural Network",
            'D': "Transformers",
            'E': "Reinforcement Learning",
            'F': "GAN Generative Adversarial Network",
            'G': "Autoencoder and self-encoder"
        },
        'answers': ['A'],
        'explanation': 
        "Feedforward neural networks are the classic example of neural networks. In fact, they were the first and most elementary type of artificial neural network. Feedforward neural networks are mainly used for supervised learning when the data, mainly numerical, to be learned is neither time-series nor sequential (such as NLP).\n"
        "These networks do not have any loops or loops in the network. Information moves in one direction only, forward, from the input nodes, through the hidden nodes (if any) and to the output nodes.\n"
        "All the other techniques are more complex and suitable for different applications (images, NLP, recommendations). Following a brief explanation of all of them.\n"
        "* The convolutional neural network (CNN) is a type of artificial neural network extensively used for image recognition and classification. It uses the convolutional layers, that is, the reworking of sets of pixels by running filters on the input pixels.\n"
        "* A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.\n"
        "* A transformer is a deep learning model that can give different importance to each part of the input data.\n"
        "* Reinforcement Learning provides a software agent that evaluates possible solutions through a progressive reward in repeated attempts. It does not need to provide labels. But it requires a lot of data and several trials, and the possibility to evaluate the validity of each attempt.\n"
        "* GAN is a special class of machine learning frameworks used for the automatic generation of facial images.\n"
        "* Autoencoder is a neural network aimed to transform and learn with a compressed representation of raw data.",
        'references': [
            'https://en.wikipedia.org/wiki/Feedforward_neural_network']
    },
    {
        'question':
        "You work for a large retail company. You are preparing a marketing model. The model will have to make predictions based on the historical and analytical data of the e-commerce site (analytics-360). In particular, customer loyalty and remarketing possibilities should be studied. You work on historical tabular data. You want to quickly create an optimal model, both from the point of view of the algorithm used and the tuning and life cycle of the model.\n"
        "What are the two best services you can use?",
        'tags': [30, 'whizlabs1'],
        'options': {
            'A': "AutoML Tables",
            'B': "BigQuery ML",
            'C': "Vertex AI",
            'D': "GKE"
        },
        'answers': ['A', 'C'],
        'explanation': 
        "AutoML Tables can select the best model for your needs without having to experiment.\n"
        "The architectures currently used (they are added at the same time) are:\n"
        "* Linear\n"
        "* Feedforward deep neural network\n"
        "* Gradient Boosted Decision Tree\n"
        "* AdaNet\n"
        "* Ensembles of various model architectures\n\n"
        "In addition, AutoML Tables automatically performs feature engineering tasks, too, such as:\n"
        "* Normalization\n"
        "* Encoding and embeddings for categorical features.\n"
        "* Timestamp columns management (important in our case)\n\n"
        "So, it has special features for time columns: for example, it can correctly split the input data into training, validation and testing.\n"
        "With Vertex AI you can use both AutoML training and custom training in the same environment.\n"
        "* BigQuery ML is wrong because AutoML Tables has additional automated feature engineering and is integrated into Vertex AI\n"
        "* GKE doesn’t supply all the ML features of Vertex AI. It is an advanced K8s managed environment",
        'references': [
            'https://cloud.google.com/automl-tables/docs/features',
            'https://cloud.google.com/vertex-ai/docs/pipelines/introduction',
            'https://cloud.google.com/automl-tables/docs/beginners-guide'
        ]
    },
    {
        'question':
        "TerramEarth is a company that builds heavy equipment for mining and agriculture.\n"
        "It is developing a series of ML models for different activities: manufacturing, procurement, logistics, marketing, customer service and vehicle tracking. TerramEarth uses Google Cloud Vertex AI and wants to scale training and inference processes in a managed way.\n"
        "During the maintenance service, snapshots of the various components of the vehicle will be taken. Your new model should be able to determine both the degree of deterioration and any breakages or possible failures. Which kind of technology/model should you advise using?",
        'tags': [31, 'whizlabs1'],
        'options': {
            'A': "Feedforward Neural Network",
            'B': "Convolutional Neural Network",
            'C': "Recurrent Neural Network",
            'D': "Transformers",
            'E': "Reinforcement Learning",
            'F': "GAN Generative Adversarial Network",
            'G': "Autoencoder and self-encoder"
        },
        'answers': ['B'],
        'explanation': 
        "The convolutional neural network (CNN) is a type of artificial neural network extensively used for image recognition and classification. It uses the convolutional layers, that is, the reworking of sets of pixels by running filters on the input pixels.\n"
        "All the other technologies are not specialized for images."
        "Feedforward neural networks are the classic example of neural networks. In fact, they were the first and most elementary type of artificial neural network. Feedforward neural networks are mainly used for supervised learning when the data, mainly numerical, to be learned is neither time-series nor sequential (such as NLP).\n"
        "* A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.\n"
        "* A transformer is a deep learning model that can give different importance to each part of the input data.\n"
        "* Reinforcement Learning provides a software agent that evaluates possible solutions through a progressive reward in repeated attempts. It does not need to provide labels. But it requires a lot of data and several trials, and the possibility to evaluate the validity of each attempt.\n"
        "* GAN is a special class of machine learning frameworks used for the automatic generation of facial images.\n"
        "* Autoencoder is a neural network aimed to transform and learn with a compressed representation of raw data.",
        'references': [
            'https://en.wikipedia.org/wiki/Convolutional_neural_network']
    },
    {
        'question':
        "You work for a video game company. Your management came up with the idea of creating a game in which the characteristics of the characters were taken from those of the human players. You have been asked to generate not only the avatars but also various visual expressions during the game actions.",
        'tags': [32, 'whizlabs1'],
        'options': {
            'A': "Feedforward Neural Network",
            'B': "Convolutional Neural Network",
            'C': "Recurrent Neural Network",
            'D': "Transformers",
            'E': "Reinforcement Learning",
            'F': "GAN Generative Adversarial Network",
            'G': "Autoencoder and self-encoder"
        },
        'answers': ['F'],
        'explanation': 
        "GAN is a special class of machine learning frameworks used for the automatic generation of facial images.\n"
        "GAN can create new characters from the provided images.\n"
        "It is also used with photographs and can generate new photos that look authentic.\n"
        "It is a kind of model highly specialized for this task. So, it is the best solution.\n"
        "* Feedforward neural networks are the classic example of neural networks. In fact, they were the first and most elementary type of artificial neural network.\n"
        "Feedforward neural networks are mainly used for supervised learning when the data, mainly numerical, to be learned is neither time-series nor sequential (such as NLP).\n"
        "* The convolutional neural network (CNN) is a type of artificial neural network extensively used for image recognition and classification. It uses the convolutional layers, that is, the reworking of sets of pixels by running filters on the input pixels.\n"
        "* A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.\n"
        "* A transformer is a deep learning model that can give different importance to each part of the input data.\n"
        "* Reinforcement Learning provides a software agent that evaluates possible solutions through a progressive reward in repeated attempts. It does not need to provide labels. But it requires a lot of data and several trials, and the possibility to evaluate the validity of each attempt.\n"
        "* Autoencoder is a neural network aimed to transform and learn with a compressed representation of raw data.",
        'references': [
            'https://en.wikipedia.org/wiki/Generative_adversarial_network',
            'https://developer.nvidia.com/blog/photo-editing-generative-adversarial-networks-2/'
        ]
    },
    {
        'question':
        "You work for a digital publishing website with an excellent technical and cultural level, where you have both famous authors and unknown experts who express ideas and insights. You, therefore, have an extremely demanding audience with strong interests of various types. Users have a small set of articles that they can read for free every month; they need to sign up for a paid subscription.\n"
        "You aim to provide your audience with pointers to articles that they will indeed find of interest to themselves.\n"
        "Which of these models can be useful to you?",
        'tags': [33, 'whizlabs1'],
        'options': {
            'A': "Hierarchical Clustering",
            'B': "Autoencoder and self-encoder",
            'C': "Convolutional Neural Network",
            'D': "Collaborative filtering using Matrix Factorization"
        },
        'answers': ['D'],
        'explanation': 
        "Collaborative filtering works on the idea that a user may like the same things of the people with similar profiles and preferences.\n"
        "So, exploiting the choices of other users, the recommendation system makes a guess and can advise people on things not yet been rated by them.\n"
        "* Hierarchical Clustering creates clusters using a hierarchical tree. It may be effective, but it is heavy with lots of data, like in our example.\n"
        "* Autoencoder and self-encoder are useful when you need to reduce the number of variables under consideration for the model, therefore for dimensionality reduction.\n"
        "* Convolutional Neural Network is used for image classification.",
        'references': [
            'https://en.wikipedia.org/wiki/Collaborative_filtering',
            'https://www.youtube.com/playlist?list=PLQY2H8rRoyvy2MiyUBz5RWZr5MPFkV3qz'
        ]
    },
    # {
    #     'question':
    #     'You work for a video game company. Your management came up with the idea of creating a game in which the characteristics of the characters were taken from those of the human players. You have been asked to generate not only the avatars but also the various visual expressions during the game actions. You are working with GAN - Generative Adversarial Network models, but the training is intensive and time-consuming.\n"
    #     'You want to increase the power of your training quickly, but your management wants to keep costs down.\n"
    #     'What solutions could you adopt (pick 3)?",
    #     'tags': [34, 'whizlabs1'],
    #     'options': {
    #         'A': "Use preemptible Cloud TPU",
    #         'B': "Use AI Platform with TPUs",
    #         'C': "Use the Cloud TPU Profiler TensorBoard plugin",
    #         'D': "Use one Compute Engine Cloud TPU VM and install TensorFlow'
    #     },
    #     'answers': ['A', 'B', 'C'],
    #     'explanation': 
    #     'All these solutions are ideal for increasing power and speed at the right cost for your training.\n"
    #     'You may use preemptible Cloud TPU (70% cheaper) for your fault-tolerant machine learning workloads.\n"
    #     'You may use TPUs in the AI Platform because TensorFlow APIs and custom templates can allow the managed environment to use TPUs and GPUs using scale tiers\n"
    #     'You may optimize your workload using the Profiler with TensorBoard.\n"
    #     'TensorBoard is a visual tool for ML experimentation for Tensorflow\n"
    #     "There are AI Platform Deep Learning VM Image types. So, you don't have to install your own ML tools and libraries and you can use managed services that help you with more productivity and savings",
    #     'references': [
    #         'https://storage.googleapis.com/nexttpu/index.html'
    #         'https://cloud.google.com/ai-platform/training/docs/using-tpus'
    #     ]
    # },
    {
        'question':
        "TerramEarth is a company that builds heavy equipment for mining and agriculture. During maintenance services for vehicles produced by TerramEarth at the service centers, information relating to their use is collected together with administrative and billing data. All this information goes through a data pipeline process that you are asked to automate in the fastest and most managed way, possibly without code.\n"
        "Which service do you advise?",
        'tags': [35, 'whizlabs1'],
        'options': {
            'A': "Cloud Dataproc",
            'B': "Cloud Dataflow",
            'C': "Cloud Data Fusion",
            'D': "Cloud Dataprep"
        },
        'answers': ['C'],
        'explanation': 
        "Cloud Data Fusion is a managed service for quickly building data pipelines and ETL processes. It is based on the open-source CDAP project and therefore is portable to any environment.\n"
        "It has a visual interface that allows you to create codeless data pipelines as required.\n"
        "* Cloud Dataproc is the managed Hadoop service. So, it could manage data pipelines but in a non-serverless and complex way.\n"
        "* Dataflow is more complex, too, even though it has more functionality, such as batch and stream data processing with the same code.\n"
        "* Cloud Dataprep is for cleaning, exploration and preparation, and is used primarily for ML processes.",
        'references': [
            'https://cloud.google.com/data-fusion',
            'https://www.youtube.com/watch?v=kehG0CJw2wo'
        ]
    },
    {
        'question':
        "You are starting to operate as a Data Scientist and are working on a model of prizes optimization with products with a lot of categorical features. You don’t know how to deal with them. Your manager told you that you had to encode them in a limited set of numbers.\n"
        "Which of the following methods will not help you with this task?",
        'tags': [36, 'whizlabs1'],
        'options': {
            'A': "Ordinal Encoding",
            'B': "One-Hot Encoding",
            'C': "Sigmoids",
            'D': "Embeddings",
            'E': "Feature Crosses"
        },
        'answers': ['C'],
        'explanation': 
        "Sigmoids are the most common activation functions (logistic function) for binary classification. There is nothing to do with categorical variables.\n"
        "* Ordinal encoding simply creates a correspondence between each unique category with an integer.\n"
        "* One-hot encoding creates a sparse matrix with values (0 and 1, see the picture) that indicate the presence (or absence) of each possible value.\n"
        "* Embeddings are often used with texts and in Natural Language Processing (NLP) and address the problem of complex categories linked together.\n"
        "* Feature crosses creates a new feature created by joining or multiplying multiple variables to add further predictive capabilities, such as transforming the geographic location of properties into a region of interest.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/embeddings/categorical-input-data',
            'https://developers.google.com/machine-learning/crash-course/feature-crosses/crossing-one-hot-vectors',
            'https://www.kaggle.com/alexisbcook/categorical-variables'
        ]
    },
    {
        'question':
        "Your company operates an innovative auction site for furniture from all times. You have to create a series of ML models that allow you, starting from the photos, to establish the period, style and type of the piece of furniture depicted.\n"
        "Furthermore, the model must be able to determine whether the furniture is interesting and require it to be subject to a more detailed estimate. You want Google Cloud to help you reach this ambitious goal faster.\n"
        "Which of the following services do you think is the most suitable?",
        'tags': [37, 'whizlabs1'],
        'options': {
            'A': "AutoML Vision Edge",
            'B': "Vision AI",
            'C': "Video AI",
            'D': "AutoML Vision"
        },
        'answers': ['D'],
        'explanation': 
        "Vision AI uses pre-trained models trained by Google. This is powerful, but not enough.\n"
        "But AutoML Vision lets you train models to classify your images with your own characteristics and labels. So, you can tailor your work as you want.\n"
        "* AutoML Vision Edge is for local devices\n"
        "* Video AI manages videos, not pictures. It can extract metadata from any streaming video, get insights in a far shorter time, and let trigger events.",
        'references': [
            'https://cloud.google.com/vision/automl/docs/edge-quickstart',
            'https://cloud.google.com/vision/automl/docs/beginners-guide',
            'https://cloud.google.com/natural-language/',
            'https://cloud.google.com/automl',
            'https://www.youtube.com/watch?v=hUzODH3uGg0'
        ]
    },
    {
        'question':
        "You are a junior Data Scientist, and you need to create a new classification Machine Learning model with Tensorflow. You have a limited set of data on which you build your model. You know the rule to create training, test and validation datasets, but you're afraid you won't have enough to make something satisfying.\n"
        "Which solution is the best one?",
        'tags': [38, 'whizlabs1'],
        'options': {
            'A': "Use Cross-Validation",
            'B': "All data for learning",
            'C': "Split data between Training and Test",
            'D': "Split data between Training and Test and Validation"
        },
        'answers': ['A'],
        'explanation': 
        "Cross-validation involves running our modeling process on various subsets of data, called \"folds\".\n"
        "Obviously, this creates a computational load. Therefore, it can be prohibitive in very large datasets, but it is great when you have small datasets.\n"
        "* All data for learning is the best way to obtain overfitting.\n"
        "* Split data between Training and Test and Split data between Training and Test and Validation are wrong because with small datasets cross-validation achieves far better results.",
        'references': [
            'https://developers.google.com/machine-learning/glossary?hl=en#cross-validation',
            'https://www.kaggle.com/alexisbcook/cross-validation'
        ]
    },
    {
        'question':
        "You are a Data Scientist, and you work in a large organization. A fellow programmer, who is working on a project with Dataflow, asked you what the GCP techniques support the computer's ability to entertain almost human dialogues and if these techniques can be used individually.\n"
        "Which of the following choices do you think is wrong?",
        'tags': [39, 'whizlabs1'],
        'options': {
            'A': "Speech to Text",
            'B': "Polly",
            'C': "Cloud NLP",
            'D': "Text to Speech",
            'E': "Speech Synthesis Markup Language (SSML) "
        },
        'answers': ['B'],
        'explanation': 
        "Amazon Polly is a text-to-speech service from AWS, not GCP.\n"
        "* Speech to Text can convert voice to written text.\n"
        "* Cloud Natural Language API can understand text meanings such as syntax, feelings, content, entities and can create classifications.\n"
        "* Text to Speech can convert written text to voice.\n"
        "* Speech Synthesis Markup Language (SSML) is not a service but a language used in Text-to-Speech requests. It gives you the ability to indicate how you want to format the audio, pauses, how to read acronyms, dates, times, abbreviations and so on. Really, it is useful for getting closer to human dialogue.",
        'references': [
            'https://cloud.google.com/speech-to-text',
            'https://cloud.google.com/text-to-speech/docs/basics',
            'https://cloud.google.com/text-to-speech/docs/ssml',
            'https://cloud.google.com/natural-language/'
        ]
    },
    {
        'question':
        "You are working on a deep neural network model with Tensorflow on a cluster of VMs for a Bank. Your model is complex, and you work with huge datasets with complex matrix computations.\n"
        "You have a big problem: your training jobs last for weeks. You are not going to deliver your project in time.\n"
        "Which is the best solution that you can adopt?",
        'tags': [40, 'whizlabs1'],
        'options': {
            'A': "Cloud TPU",
            'B': "Nvidia GPU",
            'C': "Intel CPU",
            'D': "AMD CPU"
        },
        'answers': ['A'],
        'explanation': 
        "Given these requirements, it is the best solution.\n"
        "GCP documentation states that the use of TPUs is advisable with models that:\n"
        "* use TensorFlow\n"
        "* need training for weeks or months\n"
        "* have huge matrix computations\n"
        "* have deals with big datasets and effective batch sizes\n\n"
        "So, Cloud TPU is better than Nvidia GPU, while Intel CPU and AMD CPU are wrong because the CPUs turned out to be inadequate for our purpose.",
        'references': [
            'https://cloud.google.com/tpu/docs/tpus',
            'https://cloud.google.com/tpu/docs/how-to'
        ]
    },
    {
        'question':
        "You are working with a Linear Regression model for an important Financial Institution. Your model has many independent variables. You discovered that you could not achieve good results because many variables are correlated. You asked for advice from an experienced Data scientist that explains what you can do.\n"
        "Which techniques or algorithms did he advise to use (pick 3)?",
        'tags': [41, 'whizlabs1'],
        'options': {
            'A': "Multiple linear regression with MLE",
            'B': "Partial Least Squares",
            'C': "Principal components",
            'D': "Maximum Likelihood Estimation",
            'E': "Multivariate Multiple Regression"
        },
        'answers': ['B', 'C', 'E'],
        'explanation': 
        "If you have many independent variables, some of which are correlated with each other. You have multicollinearity; therefore, you cannot use classical linear regression.\n"
        "Partial Least Squares and Principal components create new variables that are uncorrelated.\n"
        "Partial Least Squares method uses projected new variables using functions.\n"
        "The main PCA components reduce the variables while maintaining their variance. Hence, the amount of variability contained in the original characteristics.\n"
        "Multivariate regression finds out ways to explain how different elements in variables react together to changes.\n"
        "* Multiple linear regression is an OLS Ordinary Least Square method.\n"
        "* Maximum Likelihood Estimation requires independence for variables, too. Maximum Likelihood Estimation finds model parameter values with probability, maximizing the likelihood of seeing the examples given the model.",
        'references': [
            'https://towardsdatascience.com/partial-least-squares-f4e6714452a',
            'https://en.wikipedia.org/wiki/Partial_least_squares_regression',
            'https://towardsdatascience.com/maximum-likelihood-estimation-984af2dcfcac',
            'https://en.wikipedia.org/wiki/Partial_least_squares_regression',
            'https://www.mygreatlearning.com/blog/introduction-to-multivariate-regression/',
            'https://colab.research.google.com/github/kaustubholpadkar/Predicting-House-Price-using-Multivariate-Linear-Regression/blob/master/Multivariate_Linear_Regression_Python.ipynb',
            'https://en.wikipedia.org/wiki/Polynomial_regression'
        ]
    },
    # {
    #     'question':
    #     'You are using Vertex AI, with a series of demanding training jobs. So, you want to use TPUs instead of CPUs. You are not using Docker images or custom containers.\n"
    #     'What is the simplest configuration to indicate if you do not have particular needs to customize in the YAML configuration file?",
    #     'tags': [42, 'whizlabs1'],
    #     'options': {
    #         'A': "Use scale-tier to BASIC_TPU",
    #         'B': "Set Master-machine-type",
    #         'C': "Set Worker-machine-type",
    #         'D': "Set parameterServerType'
    #     },
    #     'answers': ['A'],
    #     'explanation':
    #     'Vertex AI lets you perform distributed training and serving with accelerators (TPUs and GPUs).\n"
    #     'You usually must specify the number and types of machines you need for master and worker VMs. But you can also use scale tiers that are predefined cluster specifications.\n"
    #     'In our case,\n"
    #     'scale-tier=BASIC_TPU\n"
    #     'covers all the given requirements.\n"
    #     "* The other options are wrong because it is not the easiest way. Moreover, workerType, parameterServerType, evaluatorType, workerCount, parameterServerCount, and evaluatorCount for jobs use custom containers and for TensorFlow jobs.",
    #     'references': [
    #         'https://cloud.google.com/ai-platform/training/docs/machine-types#scale_tiers",
    #         'https://cloud.google.com/ai-platform/training/docs",
    #         'https://cloud.google.com/tpu/docs/tpus'
    #     ]
    # },
    {
        'question':
        "You are training a set of modes that should be simple, using regression techniques. During training, your model seems to work. But the tests are giving unsatisfactory results. You discover that you have several wrongs and missing data. You need a tool that helps you cope with them.\n"
        "Which of the following problems is not related to Data Validation?",
        'tags': [43, 'whizlabs1'],
        'options': {
            'A': "Omitted values.",
            'B': "Categories",
            'C': "Duplicate examples.",
            'D': "Bad labels.",
            'E': "Bad feature values"
        },
        'answers': ['B'],
        'explanation': 
        "Categories are not related to Data Validation. Usually, they are categorical, string variables that in ML usually are mapped in a numerical set before training.\n"
        "* Omitted values are a problem because they may change fundamental statistics like average, for example.\n"
        "* Duplicate examples may change fundamental statistics, too.\n"
        "For example, we may have duplicates when a program loops and creates the same data several times.\n"
        "* Having bad labels (with supervised learning) or bad features means obtaining a bad model.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/representation/cleaning-data'
        ]
    },
    {
        'question':
        "You are a junior Data Scientist, and you are being interviewed for a new job.\n"
        "A senior Data Scientist asked you:\n"
        "Which metric for classification models evaluation gives you the percentage of real spam email that was recognized correctly?\n"
        "What was the exact answer to this question?",
        'tags': [44, 'whizlabs1'],
        'options': {
            'A': "Precision",
            'B': "Recall",
            'C': "Accuracy",
            'D': "F-Score"
        },
        'answers': ['B'],
        'explanation': 
        "Recall indicates how true positives were recalled (found).\n"
        "* Precision is the metric that shows the percentage of true positives related to all your positive class predictions.\n"
        "* Accuracy is the percentage of correct predictions on all outcomes.\n"
        "* F1 score is the harmonic mean between precision and recall.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall',
            'https://en.wikipedia.org/wiki/F-score',
            'https://en.wikipedia.org/wiki/Precision_and_recall#/media/File:Precisionrecall.svg'
        ]
    },
    {
        'question':
        "You are working on an NLP model. So, you are dealing with words and sentences, not numbers. Your problem is to categorize these words and make sense of them. Your manager told you that you have to use embeddings.\n"
        "Which of the following techniques are not related to embeddings?",
        'tags': [45, 'whizlabs1'],
        'options': {
            'A': "Count Vector",
            'B': "TF-IDF Vector",
            'C': "Co-Occurrence Matrix",
            'D': "CoVariance Matrix"
        },
        'answers': ['D'],
        'explanation': 
        "Covariance matrices are square matrices with the covariance between each pair of elements.\n"
        "It measures how much the change of one with respect to another is related.\n"
        "All the others are embeddings:\n"
        "* A Count Vector gives a matrix with the count of every single word in every example. 0 if no occurrence. It is okay for small vocabularies.\n"
        "* TF-IDF vectorization counts words in the entire experiment, not a single example or sentence.\n"
        "* Co-Occurrence Matrix puts together words that occur together. So, it is more useful for text understanding.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/embeddings/categorical-input-data',
            'https://developers.google.com/machine-learning/crash-course/feature-crosses/crossing-one-hot-vectors',
            'https://www.wikiwand.com/en/Covariance_matrix',
            'https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/',
            'https://towardsdatascience.com/5-things-you-should-know-about-covariance-26b12a0516f1'
        ]
    },
    {
        'question':
        "Your company operates an innovative auction site for furniture from all times. You have to create a series of ML models that allow you to establish the period, style and type of the piece of furniture depicted starting from the photos. Furthermore, the model must be able to determine whether the furniture is interesting and require it to be subject to a more detailed estimate. You created the model, but your manager said that he wants to supply this service to mobile users when they go to the fly markets.\n"
        "Which of the following services do you think is the most suitable?",
        'tags': [46, 'whizlabs1'],
        'options': {
            'A': "AutoML Vision Edge",
            'B': "Vision AI",
            'C': "Video AI",
            'D': "AutoML Vision"
        },
        'answers': ['A'],
        'explanation': 
        "AutoML Vision Edge lets your model be deployed on edge devices and, therefore, mobile phones, too.\n"
        "All the other answers refer to Cloud solutions; so, they are wrong.\n"
        "Vision AI uses pre-trained models trained by Google.\n"
        "AutoML Vision lets you train models to classify your images with your own characteristics and labels; so, you can tailor your work as you want.\n"
        "Video AI manages videos, not pictures. It can extract metadata from any streaming video, get insights in a far shorter time, and let trigger events.",
        'references': [
            'https://cloud.google.com/vision/automl/docs/edge-quickstart',
            'https://cloud.google.com/vision/automl/docs/beginners-guide',
            'https://firebase.google.com/docs/ml/automl-image-labeling'
        ]
    },
    {
        'question':
        "You are training a set of modes that should be simple, using regression techniques. During training, your model seems to work. But the tests are giving unsatisfactory results. You discover that you have several missing data. You need a tool that helps you cope with them.\n"
        "WhichGCP product would you choose?",
        'tags': [47, 'whizlabs1'],
        'options': {
            'A': "Dataproc",
            'B': "Dataprep",
            'C': "Dataflow",
            'D': "Data Fusion"
        },
        'answers': ['B'],
        'explanation': 
        "Dataprep is a serverless service that lets you examine clean and correct structured and unstructured data.\n"
        "So, it is fully compliant with our requirements.\n"
        "* Dataproc is a managed Spark and Hadoop service. Therefore, it is for BigData processing.\n"
        "* Cloud Dataflow is a managed service to run Apache Beam-based data pipeline, both batch and streaming.\n"
        "* Data Fusion is for data pipelines too. But it is visual and simpler, and it integrates multiple data sources to produce new data.",
        'references': [
            'https://cloud.google.com/dataprep',
            'https://docs.trifacta.com/display/dp/',
            'https://developers.google.com/machine-learning/crash-course/representation/cleaning-data'
        ]
    },
    {
        'question':
        "In your company you use Tensorflow and Keras as main libraries for Machine Learning and your data is stored in disk files, so block storage.\n"
        "Recently there has been the migration of all the management computing systems to Google Cloud and management has requested that the files should be stored in Cloud Storage and that the tabular data should be stored in BigQuery and pre-processed with Dataflow.\n"
        "Which of the following techniques is NOT suitable for accessing tabular data as required?",
        'tags': [48, 'whizlabs1'],
        'options': {
            'A': "BigQuery Python client library",
            'B': "BigQuery I/O Connector",
            'C': "tf.data.Iterator",
            'D': "tf.data.dataset reader"
        },
        'answers': ['C'],
        'explanation': 
        "tf.data.Iterator is used for enumerating elements in a Dataset, using Tensorflow API, so it is not suitable for accessing tabular data.\n"
        "* Python BigQuery client library allows you to get BigQuery data in Panda, so it's definitely useful in this environment.\n"
        "* BigQuery I/O Connector is used by Dataflow (Apache Beam) for reading Data for transformation and processing, as required.\n"
        "* tf.data.dataset reader is wrong because you must first access the data using the tf.data.dataset reader for BigQuery.",
        'references': [
            'https://cloud.google.com/architecture/ml-on-gcp-best-practices#store-tabular-data-in-bigquery',
            'https://cloud.google.com/bigquery/docs/bigquery-storage-python-pandas',
            'https://beam.apache.org/documentation/io/built-in/google-bigquery/'
        ]
    },
    {
        'question':
        "You are a junior Data Scientist. You are working with a linear regression model with sklearn.\n"
        "Your outcome model presented a good R-square - coefficient of determination, but the final results were poor.\n"
        "When you asked for advice, your mentor laughed and said that you failed because of the Anscombe Quartet problem.\n"
        "What are the other possible problems described by the famous Anscombe Quartet?",
        'tags': [49, 'whizlabs1'],
        'options': {
            'A': "Not linear relation between independent and dependent variables",
            'B': "Outliers that change the result",
            'C': "Correlation among variables",
            'D': "Uncorrect Data"
        },
        'answers': ['A', 'B'],
        'explanation': 
        "The most common problems are:\n"
        "* Not linear relation and\n"
        "* Outliers\n"
        "Correlation and incorrect data prevent the model from working, but they do not give good theoretical results.",
        'references': [
            'https://en.wikipedia.org/wiki/Anscombe%27s_quartet',
            'https://www.r-bloggers.com/2015/01/k-means-clustering-is-not-a-free-lunch/'
        ]
    },
    {
        'question':
        "You are working on a deep neural network model with Tensorflow. Your model is complex, and you work with very large datasets full of numbers.\n"
        "You want to increase performances. But you cannot use further resources.\n"
        "You are afraid that you are not going to deliver your project in time.\n"
        "Your mentor said to you that normalization could be a solution.\n"
        "Which of the following choices do you think is not for data normalization?",
        'tags': [50, 'whizlabs1'],
        'options': {
            'A': "Scaling to a range",
            'B': "Feature Clipping",
            'C': "Z-test",
            'D': "log scaling",
            'E': "Z-score"
        },
        'answers': ['C'],
        'explanation': 
        "z-test is not correct because it is a statistic that is used to prove if a sample mean belongs to a specific population. For example, it is used in medical trials to prove whether a new drug is effective or not.\n"
        "* Scaling to a range converts numbers into a standard range ( 0 to 1 or -1 to 1).\n"
        "* Feature Clipping caps all numbers outside a certain range.\n"
        "* Log Scaling uses the logarithms instead of your values to change the shape. This is possible because the log function preserves monotonicity.\n"
        "* Z-score is a variation of scaling: the resulting number is divided by the standard deviations. It is aimed at obtaining distributions with mean = 0 and std = 1.",
        'references': [
            'https://developers.google.com/machine-learning/data-prep/transform/transform-numeric',
            'https://developers.google.com/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg'
        ]
    },
    {
        'question':
        "Your team is designing a financial analysis model for a major Bank.\n"
        "The requirements are:\n"
        "* Various banking applications will send transactions to the new system both in real-time and in batch in standard/normalized format\n"
        "* The data will be stored in a repository\n"
        "* Structured Data will be trained and retrained\n"
        "* Labels are drawn from the data.\n"
        "You need to prepare the model quickly and decide to use Auto ML for structured Data.\n"
        "Which GCP Services could you use (Select THREE)?",
        'tags': [51, 'whizlabs1'],
        'options': {
            'A': "AutoML Tables",
            'B': "Tensorflow Extended",
            'C': "BigQuery ML",
            'D': "Vertex AI"
        },
        'answers': ['A', 'C', 'D'],
        'explanation': 
        "Auto ML Tables is aimed to automatically build and deploy models on your data in the fastest way possible.\n"
        "It is integrated within BigQuery ML and is available in the unified Vertex AI.\n"
        "* But Tensorflow Extended is for deploying production ML pipelines, and it doesn't have any AutoML Services",
        'references': [
            'https://cloud.google.com/automl-tables/docs/beginners-guide',
            'https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-automl',
            'https://cloud.google.com/vertex-ai/docs/beginner/beginners-guide#text'
        ]
    },
    {
        'question':
        "You are starting to operate as a Data Scientist and are working on a deep neural network model with Tensorflow to optimize the level of customer satisfaction for after-sales services with the goal of creating greater client loyalty.\n"
        "You have to follow the entire lifecycle: model development, design, and training, testing, deployment, and retraining.\n"
        "You are looking for UI tools that can help you work and solve all issues faster.\n"
        "Which solutions can you adopt (pick 3)?",
        'tags': [52, 'whizlabs1'],
        'options': {
            'A': "Tensorboard",
            'B': "Jupyter notebooks",
            'C': "KFServing",
            'D': "Kubeflow UI",
            'E': "Vertex AI"
        },
        'answers': ['A', 'B', 'D'],
        'explanation': 
        "Tensorboard is aimed at model creation and experimentation:\n"
        "* Profiling\n"
        "* Monitoring metrics, weights, biases\n"
        "* Examine model graph\n"
        "* Working with embeddings\n\n"
        "Jupyter notebooks are a wonderful tool to develop, experiment, and deploy. You may have the latest data science and machine learning frameworks with them.\n"
        "The Kubeflow UIs is for ML pipelines and includes visual tools for:\n"
        "* Pipelines dashboards\n"
        "* Hyperparameter tuning\n"
        "* Artifact Store\n"
        "* Jupyter notebooks\n\n"
        "KFServing is an open-source library for Kubernetes that enables serverless inferencing. It works with TensorFlow, XGBoost, scikit-learn, PyTorch, and ONNX to solve issues linked to production model serving. So, no UI.\n"
        "With Vertex AI you can use AutoML training and custom training in the same environment.",
        'references': [
            'https://www.tensorflow.org/tensorboard',
            'https://www.kubeflow.org/docs/components/kfserving/kfserving/',
            'https://cloud.google.com/vertex-ai/docs/pipelines/visualize-pipeline',
            'https://www.kubeflow.org/docs/components/central-dash/overview/'
        ]
    },
    {
        'question':
        "You work for an industrial company that wants to improve its quality system. It has developed its own deep neural network model with Tensorflow to identify the semi-finished products to be discarded with images taken from the production lines in the various production phases.\n"
        "You work on this project. You need to deal with input data that is binary (images) and made by CSV files.\n"
        "You are looking for the most convenient way to import and manage this type of data.\n"
        "Which is the best solution that you can adopt?",
        'tags': [53, 'whizlabs1'],
        'options': {
            'A': "tf.RaggedTensor",
            'B': "Tf.quantization",
            'C': "tf.train.Feature",
            'D': "tf.TFRecordReader"
        },
        'answers': ['D'],
        'explanation': 
        "The TFRecord format is efficient for storing a sequence of binary and not-binary records using Protocol buffers for serialization of structured data.\n"
        "* RaggedTensor is a tensor with ragged dimensions, that is with different lengths like this: [[6, 4, 7, 4], [], [8, 12, 5], [9], []]\n"
        "* Quantization is aimed to reduce CPU and TPU GCP latency, processing, and power.\n"
        "* tf.train is a feature for Graph-based Neural Structured model training",
        'references': [
            'https://www.tensorflow.org/tutorials/load_data/tfrecord'
        ]
    },
    {
        'question':
        "You work for an industrial company that wants to improve its quality system. It has developed its own deep neural network model with Tensorflow to identify the semi-finished products to be discarded with images taken from the production lines in the various production phases.\n"
        "You need to monitor the performance of your models and let them go faster.\n"
        "Which is the best solution that you can adopt?",
        'tags': [54, 'whizlabs1'],
        'options': {
            'A': "TF Profiler",
            'B': "TF function",
            'C': "TF Trace",
            'D': "TF Debugger",
            'E': "TF Checkpoint",
        },
        'answers': ['A'],
        'explanation': 
        "TensorFlow Profiler is a tool for checking the performance of your TensorFlow models and helping you to obtain an optimized version.\n" 
        "In TensorFlow 2, the default is eager execution. So, one-off operations are faster, but recurring ones may be slower. So, you need to optimize the model.\n"
        "* TF function is a transformation tool used to make graphs out of your programs. It helps to create performant and portable models but is not a tool for optimization.\n"
        "* TF tracing lets you record TensorFlow Python operations in a graph.\n"
        "* TF debugging is for Debugger V2 and creates a log of debug information.\n"
        "* Checkpoints catch the value of all parameters in a serialized SavedModel format.",
        'references': [
            'https://www.tensorflow.org/guide/profiler',
            'https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile',
            'https://www.tensorflow.org/tensorboard/debugger_v2',
            'https://www.tensorflow.org/guide/checkpoint'
        ]
    },
    {
        'question':
        "You work for an important Banking group.\n"
        "The purpose of your current project is the automatic and smart acquisition of data from documents and modules of different types.\n"
        "You work on big datasets with a lot of private information that cannot be distributed and disclosed.\n"
        "You are asked to replace sensitive data with specific surrogate characters.\n"
        "Which of the following techniques do you think is best to use?",
        'tags': [55, 'whizlabs1'],
        'options': {
            'A': "Format-preserving encryption",
            'B': "K-anonymity",
            'C': "Replacement",
            'D': "Masking"
        },
        'answers': ['D'],
        'explanation': 
        "Masking replaces sensitive values with a given surrogate character, like hash (#) or asterisk (*).\n"
        "* Format-preserving encryption (FPE) encrypts in the same format as the plaintext data.\n"
        "For example, a 16-digit credit card number becomes another 16-digit number.\n"
        "* k-anonymity is a way to anonymize data in such a way that it is impossible to identify person-specific information. Still, you maintain all the information contained in the record.\n"
        "* Replacement just substitutes a sensitive element with a specified value.",
        'references': [
            'https://en.wikipedia.org/wiki/Data_masking',
            'https://en.wikipedia.org/wiki/K-anonymity',
            'https://www.mysql.com/it/products/enterprise/masking.html'
        ]
    },
    ## whizlabs - Google Cloud Certified Professional Machine Learning Engineer - Practice Test 2
    {
        'question':
        "You have a customer ranking ML model in production for an e-commerce site; the model used to work very well.\n"
        "You use GCP managed services, specifically Vertex AI.\n"
        "Suddenly, there is a sensible degradation in the quality of the inferences. You perform various checks, but the model seems to be perfectly fine.\n"
        "Finally, you control the input data and notice that the frequency distributions have changed for a specific feature.\n"
        "Which GCP service can be helpful for you to manage features in a more organized way?",
        'tags': [1, 'whizlabs2'],
        'options': {
            'A': "Regularization against overfitting",
            'B': "Feature Store",
            'C': "Hyperparameters tuning",
            'D': "Model Monitoring"
        },
        'answers': ['B'],
        'explanation': 
        "Feature engineering means transforming input data, often strings, into a feature vector.\n"
        "Lots of effort is spent in mapping categorical values in the best way: we have to convert strings to numeric values. We have to define a vocabulary of possible values, usually mapped to integer values.\n"
        "We remember that in an ML model everything must be translated into numbers. Therefore it is easy to run into problems of this type.\n"
        "Vertex Feature Store is a service to organize and store ML features through a central store.\n"
        "This allows you to share and optimize ML features important for the specific environment and to reuse them at any time.\n"
        "All these translate into the greater speed of the creation of ML services. But these also allow minimizing problems such as processing skew, which occurs when the distribution of data in production is different from that of training, often due to errors in the organization of the features.\n"
        "For example, Training-serving skew may happen when your training data uses a different unit of measure than prediction requests.\n"
        "So, Training-serving skew happens when you generate your training data differently than you generate the data you use to request predictions. For example, if you use an average value, and for training purposes, you average over 10 days, but you average over the last month when you request prediction.\n"
        "* Regularization against overfitting and Hyperparameters tuning are wrong because the model is OK. So both Regularization against overfitting and Hyperparameters are tuned.\n"
        "* Monitor is suitable for Training-serving skew prevention, not organization.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/representation/feature-engineering',
            'https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-vertex-feature-store-with-structured-data',
            'https://cloud.google.com/blog/topics/developers-practitioners/kickstart-your-organizations-ml-application-development-flywheel-vertex-feature-store'
        ]
    },
    {
        'question':
        "You have a customer ranking ML model in production for an e-commerce site; the model used to work very well. You use GCP managed services, specifically Vertex AI. Suddenly there is a sensible degradation in the quality of the inferences. You perform various checks, but the model seems to be perfectly fine.\n"
        "Which of the following methods could you use to avoid such problems?",
        'tags': [2, 'whizlabs2'],
        'options': {
            'A': "Regularization against overfitting",
            'B': "Feature Store",
            'C': "Hyperparameters tuning",
            'D': "Model Monitoring"
        },
        'answers': ['D'],
        'explanation': 
        "Input data to ML models may change over time. This can be a serious problem, as performance will obviously degrade.\n"
        "To avoid this, it is necessary to monitor the quality of the forecasts continuously.\n"
        "Vertex Model Monitoring has been designed just for this.\n"
        "The main goal is to cope with feature skew and drift detection.\n"
        "For skew detection, it looks at and compares the feature's values distribution in the training data.\n"
        "For drift detection, it looks at and compares the feature's values distribution in the production data.\n"
        "It uses two main methods:"
        "* Jensen-Shannon divergence for numerical features.\n"
        "* L-infinity distance for categorical features. More details can be found here\n"
        "* Regularization against overfitting and Hyperparameters tuning are wrong because the model is OK. So both Regularization against overfitting and Hyperparameters are tuned.\n"
        "* Feature Store is suitable for feature organization, not for data skew prevention.",
        'references': [
            'https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence',
            'https://en.wikipedia.org/wiki/Chebyshev_distance',
            'https://github.com/tensorflow/data-validation/blob/master/g3doc/anomalies.md',
            'https://cloud.google.com/vertex-ai/docs/model-monitoring/overview'
        ]
    },
    {
        'question':
        "Your company runs an e-commerce site. You produced static deep learning models with Tensorflow that process Analytics-360 data. They have been in production for some time. Initially, they gave you excellent results, but then gradually, the accuracy has decreased.\n"
        "You are using Compute Engine and GKE. You decided to use a library that lets you have more control over all processes, from development up to production.\n"
        "Which tool is the best one for your needs?",
        'tags': [3, 'whizlabs2'],
        'options': {
            'A': "TFX",
            'B': "Vertex AI",
            'C': "SageMaker",
            'D': "Kubeflow"
        },
        'answers': ['A'],
        'explanation': 
        "TensorFlow Extended (TFX) is a set of open-source libraries to build and execute ML pipelines in production. Its main functions are:\n"
        "* Metadata management\n"
        "* Model validation\n"
        "* Deployment\n"
        "* Production execution.\n"
        "The libraries can also be used individually.\n"
        "* Vertex AI is an integrated suite of ML managed products, and you are looking for a library.\n\n"
        "Vertex AI's main functions are:\n"
        "* Train custom and AutoML models\n"
        "* Evaluate and tune models\n"
        "* Deploy models\n"
        "* Manage prediction: Batch, Online and monitoring\n"
        "* Manage model versions: workflows and retraining\n"
        "Sagemaker is a managed product in AWS, not GCP.\n"
        "Kubeflow Pipelines don’t deal with production control.\n"
        "Kubeflow Pipelines is an open-source platform designed specifically for creating and deploying ML workflows based on Docker containers.\n"
        "Their main features:\n"
        "* Using packaged templates in Docker images in a K8s environment\n"
        "* Manage your various tests / experiments\n"
        "* Simplifying the orchestration of ML pipelines\n"
        "* Reuse components and pipelines",
        'references': [
            'https://www.tensorflow.org/tfx'
        ]
    },
    {
        'question':
        "Your company runs a big retail website. You develop many ML models for all the business activities. You migrated to Google Cloud when you were using Vertex AI. Your models are developed with PyTorch, TensorFlow and BigQueryML.\n"
        "You also use BigTable and CloudSQL, and of course Cloud Storage. In many cases, the same data is used for multiple models and projects. And your data is continuously updated, sometimes in streaming mode.\n"
        "Which is the best way to organize the input data?",
        'tags': [4, 'whizlabs2'],
        'options': {
            'A': "Dataflow per Data Transformation sia in streaming che batch",
            'B': "CSV",
            'C': "BigQuery",
            'D': "Datasets",
            'E': "BigTable"
        },
        'answers': ['D'],
        'explanation': 
        "Vertex AI integrates the following elements:\n"
        "* Datasets: data, metadata and annotations, structured or unstructured. For all kinds of libraries.\n"
        "* Training pipelines to build an ML model\n"
        "* ML models, imported or created in the environment\n"
        "* Endpoints for inference\n\n"
        "Because Datasets are suitable for all kinds of libraries, it is a useful abstraction for this requirement.\n"
        "* Dataflow deals with Data Pipelines and is not a way to access and organize data.\n"
        "* CSV is just a data format, and an ML Dataset is made with data and metadata dealing with many different formats.\n"
        "* BigQuery and BigTable are just one of the ways in which you can store data. Moreover, BigTable is not currently supported for data store for Vertex datasets.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/datasets/datasets',
            'https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets',
            'https://codelabs.developers.google.com/codelabs/vertex-ai-custom-code-training'
        ]
    },
    {
        'question':
        "You are a Data Scientist and working on a project with PyTorch. You need to save the model you are working on because you have to cope with an urgency. You, therefore, need to resume your work later.\n"
        "What command will you use for this operation?",
        'tags': [5, 'whizlabs2'],
        'options': {
            'A': "callbacks.ModelCheckpoint (keras)",
            'B': "save",
            'C': "model.fit",
            'D': "train.Checkpoint TF"
        },
        'answers': ['B'],
        'explanation': 
        "PyTorch is a popular library for deep learning that you can leverage using GPUs and CPUs.\n"
        "When you have to save a model for resuming training, you have to record both models and updated buffers and parameters in a checkpoint.\n"
        "A checkpoint is an intermediate dump of a model’s entire internal state (its weights, current learning rate, etc.) so that the framework can resume the training from that very point.\n"
        "In other words, you train for a few iterations, then evaluate the model, checkpoint it, then fit some more. When you are done, save the model and deploy it as normal.\n"
        "To save checkpoints, you must use torch.save() to serialize the dictionary of all your state data,\n"
        "In order to reload, the command is torch.load().\n"
        "* ModelCheckpoint is used with keras.\n"
        "* model.fit is used to fit a model in scikit-learn best.\n"
        "* train.Checkpoint is used with Tensorflow.",
        'references': [
            'https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html',
            'https://towardsdatascience.com/ml-design-pattern-2-checkpoints-e6ca25a4c5fe'
        ]
    },
    {
        'question':
        "You are a Data Scientist. You are going to develop an ML model with Python. Your company adopted GCP and Vertex AI, but you need to work with your developing tools.\n"
        "What are you going to do (Select TWO)?",
        'tags': [6, 'whizlabs2'],
        'options': {
            'A': "Use an Emulator",
            'B': "Work with the Console",
            'C': "Create a service account key",
            'D': "Set the environment variable named GOOGLE_APPLICATION_CREDENTIALS"
        },
        'answers': ['C', 'D'],
        'explanation': 
        "Client libraries are used by developers for calling the Vertex AI API in their code.\n"
        "The client libraries reduce effort and boilerplate code.\n"
        "The correct procedure is:\n"
        "* Enable the Vertex AI API & Prediction and Compute Engine APIs.\n"
        "* Enable the APIs\n"
        "* Create/Use a Service account and a service account key\n"
        "* Set the environment variable named GOOGLE_APPLICATION_CREDENTIALS\n\n"
        "* Use an Emulator is wrong because there isn’t a specific Emulator for using the SDK\n"
        "* Work with the Console is wrong because it was asked to create a local work environment.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/start/client-libraries#python',
            'https://cloud.google.com/ai-platform/training/docs/python-client-library'
        ]
    },
    {
        'question': 
            "You are working with Vertex AI, the managed ML Platform in GCP. You are dealing with custom training and you are looking and studying the job progresses during the training service lifecycle.\n"
            "Which of the following states are not correct?",
        'tags': [7, 'whizlabs2'],
        'options': {
            'A': "JOB_STATE_ACTIVE",
            'B': "JOB_STATE_RUNNING",
            'C': "JOB_STATE_QUEUED",
            'D': "JOB_STATE_ENDED"
        },
        'answers': ['A'],
        'explanation': 
            "This is a brief description of the lifecycle of a custom training service.\n"
            "Queueing a new job\n"
            "When you create a CustomJob or HyperparameterTuningJob, the job is in the JOB_STATE_QUEUED.\n"
            "When a training job starts, Vertex AI schedules as many workers according to configuration, in parallel.\n"
            "So Vertex AI starts running code as soon as a worker becomes available.\n"
            "When all the workers are available, the job state will be: JOB_STATE_RUNNING.\n"
            "A training job ends successfully when its primary replica exits with exit code 0.\n"
            "Therefore all the other workers will be stopped. The state will be: JOB_STATE_ENDED.\n"
            "So JOB_STATE_ACTIVE is wrong simply because this state doesn’t exist. All the other answers are correct.\n"
            "Each replica in the training cluster is given a single role or task in distributed training. For example:\n"
            "Primary replica: Only one replica, whose main task is to manage the workers.\n"
            "Worker(s): Replicas that do part of the work.\n"
            "Parameter server(s): Replicas that store model parameters (optional).\n"
            "Evaluator(s): Replicas that evaluate your model (optional).",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/training/custom-training',
            'https://cloud.google.com/vertex-ai/docs/training/distributed-training'
        ]
    },
    {
        'question':
        "You work as a Data Scientist for a major banking institution that recently completed the first phase of migration in GCP.\n"
        "You now have to work in the GCP Managed Platform for ML. You need to deploy a custom model with Vertex AI so that it will be available for online predictions.\n"
        "Which is the correct procedure (Select TWO)?",
        'tags': [8, 'whizlabs2'],
        'options': {
            'A': "Save the model in a Docker container",
            'B': "Set a VM with a GPU processor",
            'C': "Use TensorFlow Serving",
            'D': "Create an endpoint and deploy to that endpoint"
        },
        'answers': ['A', 'D'],
        'explanation': 
        "Vertex AI Prediction can serve prediction deploying custom or pre-built containers on N1 Compute Engine Instances.\n"
        "You create an \"endpoint object\" for your model and then you can deploy the various versions of your model.\n"
        "Its main elements are given below:\n"
        "Custom or Pre-built containers\n"
        "Model\n"
        "Vertex AI Prediction uses an architectural paradigm that is based on immutable instances of models and model versions.\n"
        "Regional endpoint\n"
        "The endpoint is the object that will be equipped with all the resources needed for online predictions and it is the target for your model deployments.\n"
        "* You don’t need to set any specific VM. You will point out the configuration and Vertex will manage everything.\n"
        "* TensorFlow Serving is used under the hood, but you don’t need to call their functions explicitly.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container',
            'https://www.tensorflow.org/tfx/guide/serving',
            'https://cloud.google.com/vertex-ai/docs/general/deployment'
        ]
    },
    {
        'question':
        "You work as a Data Scientist in a Startup. You want to create an optimized input pipeline to increase the performance of training sessions, avoiding GPUs and TPUs as much as possible because they are expensive.\n"
        "Which technique or algorithm do you think is best to use?",
        'tags': [9, 'whizlabs2'],
        'options': {
            'A': "Caching",
            'B': "Prefetching",
            'C': "Parallelizing data",
            'D': "All of the above"
        },
        'answers': ['D'],
        'explanation': 
        "GPUs and TPUs can greatly increase the performance of training sessions, but an optimized input pipeline is likewise important.\n"
        "The tf.data API provides these functions:\n"
        "Prefetching\n"
        "tf.data.Dataset.prefetch: while the execution of a training pass, the data for the next pass is read.\n"
        "Parallelizing data transformation\n"
        "The tf.data API offers the map function for the tf.data.Dataset.map transformation.\n"
        "This transformation can be parallelized across multiple cores with the num_parallel_calls option.\n"
        "Sequential and parallel interleave\n"
        "tf.data.Dataset.interleave offers the possibility of interleaving and allowing multiple datasets to execute in parallel (num_parallel_calls).\n"
        "Caching\n"
        "tf.data.Dataset.cache allows you to cache a dataset increasing performance.",
        'references': [
            'https://www.tensorflow.org/guide/data_performance'
        ]
    },
    {
        'question':
        "You are working on a new model together with your client, a large financial institution. The data you are dealing with contains PII (Personally Identifiable Information) contents.\n\n"
        "You face 2 different sets of problems:\n"
        "* Transform data to hide personal information you don't need\n"
        "* Protect your work environment because certain combinations of personal data are useful for your model and you need to keep them\n"
        "What are the solutions offered by GCP that it is advisable to use (choose 2)?",
        'tags': [10, 'whizlabs2'],
        'options': {
            'A': "Cloud Armor security policies",
            'B': "Cloud HSM",
            'C': "Cloud Data Loss Prevention",
            'D': "Network firewall rules",
            'E': "VPC service-controls"
        },
        'answers': ['C', 'E'],
        'explanation': 
        "Cloud Data Loss Prevention is a service that can discover, conceal and mask personal information in data.\n"
        "VPC service-controls is a service that lets you build a security perimeter that is not accessible from outside; in this way data exfiltration dangers are greatly mitigated. It is a network security service that helps protect data in a Virtual Private Cloud (VPC) in a multi-tenant environment.\n"
        "* Cloud Armor is a security service at the edge against attacks like DDoS.\n"
        "* Cloud HSM is a service for cryptography based on special and certified hardware and software\n"
        "* Network firewall rules are a set of rules that deny or block network traffic in a VPC, just network rules. VPC service-controls lets you define control at a more granular level, with context-aware access, suitable for multi-tenant environments like this one.",
        'references': [
            'https://cloud.google.com/vpc-service-controls',
            'https://cloud.google.com/dlp'
        ]
    },
    {
        'question':
        "You are a junior Data Scientist and working on a deep neural network model with Tensorflow to optimize the level of customer satisfaction for after-sales services to create greater client loyalty.\n"
        "You are struggling with your model (learning rates, hidden layers and nodes selection) for optimizing processing and letting it converge in the fastest way.\n"
        "What is your problem in ML language?",
        'tags': [11, 'whizlabs2'],
        'options': {
            'A': "Cross Validation",
            'B': "Regularization",
            'C': "Hyperparameter tuning",
            'D': "drift detection management"
        },
        'answers': ['C'],
        'explanation': 
        "ML training Manages three main data categories:\n"
        "* Training data is also called examples or records. It is the main input for model configuration and, in supervised learning, presents labels, that are the correct answers based on past experience. Input data is used to build the model but will not be part of the model.\n"
        "* Parameters are instead the variables to be found to solve the riddle. They are part of the final model and they make the difference among similar models of the same type.\n"
        "* Hyperparameters are configuration variables that influence the training process itself: Learning rate, hidden layers number, number of epochs, regularization, batch size are all examples of hyperparameters.\n"
        "Hyperparameters tuning is made during the training job and used to be a manual and tedious process, made by running multiple trials with different values.\n"
        "The time required to train and test a model can depend upon the choice of its hyperparameters.\n"
        "With Vertex AI you just need to prepare a simple YAML configuration without coding.\n"
        "* Cross Validation is related to the input data organization for training, test and validation.\n"
        "* Regularization is related to feature management and overfitting.\n"
        "* Drift management is when data distribution changes and you have to adjust the model.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview',
            'https://cloud.google.com/blog/products/ai-machine-learning/hyperparameter-tuning-cloud-machine-learning-engine-using-bayesian-optimization'
        ]
    },
    {
        'question':
        "You work for an important organization. Your manager tasked you with a new classification model with lots of data drawn from the company Data Lake.\n"
        "The big problem is that you don’t have the labels for all the data, but you have very little time to complete the task for only a subset of it.\n"
        "Which of the following services could help you?",
        'tags': [12, 'whizlabs2'],
        'options': {
            'A': "Vertex Data Labeling",
            'B': "Mechanical Turk",
            'C': "GitLab ML",
            'D': "Tag Manager"
        },
        'answers': ['A'],
        'explanation': 
        "In supervised learning, the correctness of label data, together with the quality of all your training data, is utterly important for the resulting model and the quality of the future predictions.\n"
        "If you cannot have your data correctly labeled, you may request professional people to complete your training data.\n"
        "GCP has a service for this: Vertex AI data labeling. Human labelers will prepare correct labels following your directions.\n"
        "You have to set up a data labeling job with:\n"
        "* The dataset\n"
        "* A list, vocabulary of the possible labels\n"
        "* An instructions document for the professional people\n\n"
        "* Mechanical Turk is an Amazon service.\n"
        "* GitLab is a DevOps lifecycle tool.\n"
        "* Tag Manager is in the Google Analytics ecosystem.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/datasets/data-labeling-job'
        ]
    },
    {
        'question':
        "Your company runs an e-commerce site. You manage several deep learning models with Tensorflow that process Analytics-360 data, and they have been in production for some time. The modeling is made essentially with customers and orders data. You need to classify many business outcomes.\n"
        "Your Manager realized that different teams in different projects used to deal with the same features based on the same data differently. The problem arose when models drifted unexpectedly over time.\n"
        "You have to advise your Manager on the best strategy. Which of the following do you choose (pick 2)?",
        'tags': [13, 'whizlabs2'],
        'options': {
            'A': "Each group classifies their features and sends them to the other teams",
            'B': "For each model of the different features store them in Cloud Storage",
            'C': "Search for features in Cloud Storage and reuse them",
            'D': "Search the Vertex Feature Store for features that are the same",
            'E': "Insert or update the features in Vertex Feature Store accordingly"
        },
        'answers': ['D', 'E'],
        'explanation': 
        "The best strategy is to use the Vertex Feature Store.\n"
        "Vertex Feature Store is a service to organize and store ML features through a central store.\n"
        "This allows you to share and optimize ML features important for the specific environment and to reuse them at any time.\n"
        "Here is the typical procedure for using the Feature Store:\n"
        "* Check out the Vertex Feature Store for Features that you can reuse or use as a template.\n"
        "* If you don't find a Feature that fits perfectly, create or modify an existing one.\n"
        "* Update or insert features of your work in the Vertex Feature Store.\n"
        "* Use them in training work.\n"
        "* Sets up a periodic job to generate feature vocabulary data and optionally updates the Vertex Feature Store\n\n"
        "* Each group classifies their features and sends them to the other teams creates confusion and doesn't solve the problem.\n"
        "* For each model of the different features store them in Cloud Storage and search for features in Cloud Storage and reuse them will not avoid feature definition overlapping. Cloud Storage is not enough for identifying different features.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/representation/feature-engineering\n',
            'https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-vertex-feature-store-with-structured-data\n',
            'https://cloud.google.com/blog/topics/developers-practitioners/kickstart-your-organizations-ml-application-development-flywheel-vertex-feature-store'
        ]
    },
    {
        'question':
        "You are starting to operate as a Data Scientist. You speak with your mentor who asked you to prepare a simple model with a nonparametric Machine Learning algorithm of your choice. The problem is that you don’t know the difference between parametric and nonparametric algorithms. So you looked for it.\n"
        "Which of the following methods are nonparametric?",
        'tags': [14, 'whizlabs2'],
        'options': {
            'A': "Simple Neural Networks",
            'B': "K-Nearest Neighbors",
            'C': "Decision Trees",
            'D': "Logistic Regression"
        },
        'answers': ['B', 'C'],
        'explanation': 
        "The non-parametric method refers to a method that does not assume any distribution with any function with parameters.\n"
        "K-nearest neighbor is a simple supervised algorithm for both classification and regression problems.\n"
        "You begin with data that is already classified. A new example will be set by looking at the k nearest classified points. Number k is the most important hyperparameter.\n"
        "A decision tree has a series of tests inside a flowchart-like structure. So, no mathematical functions to solve.\n"
        "In the case of both Neural Networks and Logistic Regression, you have to figure out the parameters of a specific function that best fit the data.",
        'references': [
            'https://towardsdatascience.com/all-machine-learning-algorithms-you-should-know-in-2021-2e357dd494c7',
            'https://towardsdatascience.com/k-nearest-neighbors-knn-algorithm-23832490e3f4',
            'https://machinelearningmastery.com/parametric-and-nonparametric-machine-learning-algorithms/'
        ]
    },
    {
        'question':
        "As a Data Scientist, you are involved in various projects in an important retail company. You prefer to use, whenever possible, simple and easily explained algorithms. Where you can't get satisfactory results, you adopt more complex and sophisticated methods. Your manager told you that you should try ensemble methods. Intrigued, you are documented.",
        'tags': [15, 'whizlabs2'],
        'options': {
            'A': "Random Forests",
            'B': "DCN",
            'C': "Decision Tree",
            'D': "XGBoost",
            'E': "Gradient Boost"
        },
        'answers': ['A', 'D', 'E'],
        'explanation': 
        "Ensemble learning is performed by multiple learning algorithms working together for higher predictive performance.\n"
        "Examples of Ensemble learning are: Random forests, AdaBoost, gradient boost, and XGBoost.\n"
        "Two main concepts for combining algorithms;\n"
        "* Bootstrap sampling uses random samples and selects the best of them.\n"
        "* Bagging when you put together selected random samples to achieve a better result\n\n"
        "Random forests are made with multiple decision trees, random sampling, a subset of variables and optimization techniques at each step (voting the best models).\n\n"
        "AdaBoost is built with multiple decision trees, too, with the following differences:\n"
        "* It creates stumps, that is, trees with only one node and two leaves.\n"
        "* Stumps with less error win.\n"
        "* Ordering is built in such a way as to reduce errors.\n\n"
        "Gradient Boost is built with multiple decision trees, too, with the following differences from AdaBoost;\n"
        "* Trees instead stumps\n"
        "* It uses a loss function to minimize errors.\n"
        "* Trees are selected to predict the difference from actual values\n\n"
        "XGBoost is currently very popular. It is similar to Gradient Boost with the following differences:\n"
        "* Leaf nodes pruning, that is regularization in order to keep the best ones for generalization\n"
        "* Newton Boosting instead of gradient descent, so math-based and faster\n"
        "* Correlation between trees reduction with an additional randomization parameter\n"
        "* Optimized algorithm for tree penalization\n\n"
        "* Deep and Cross Networks are a new kind of Neural Networks. Decision Trees are flowchart like with a series of tests on the nodes. So both of them use one kind of method.",
        'references': [
            'https://towardsdatascience.com/all-machine-learning-algorithms-you-should-know-in-2021-2e357dd494c7'
        ]
    },
    {
        'question': 
            "Your team works for an international company with Google Cloud, and you develop, train and deploy several ML models with Tensorflow. You use many tools and techniques and you want to make your work leaner, faster, and more efficient.\n"
            "You would like engineer-to-engineer assistance from both Google Cloud and Google’s TensorFlow teams."
            "How is it possible? Which service?",
        'tags': [16, 'whizlabs2'],
        'options': {
            'A': "Vertex AI",
            'B': "Kubeflow",
            'C': "Tensorflow Enterprise",
            'D': "TFX"
        },
        'answers': ['C'],
        'explanation': 
        "The TensorFlow Enterprise is a distribution of the open-source platform for ML, linked to specific versions of TensorFlow, tailored for enterprise customers.\n"
        "It is free but only for big enterprises with a lot of services in GCP. it is prepackaged and optimized for usage with containers and VMs.\n"
        "It works in Google Cloud, from VM images to managed services like GKE and Vertex AI.\n"
        "The TensorFlow Enterprise library is integrated in the following products:\n\n"
        "* Deep Learning VM Images\n"
        "* Deep Learning Containers\n"
        "* Notebooks\n"
        "* Vertex AI Training\n"
        "It is ready for automatic provisioning and scaling with any kind of processor.\n"
        "It has a premium level of support from Google.\n"
        "* Vertex AI is a managed service without the kind of support required"
        "* Kubeflow and TFX are wrong because they are open source libraries with standard support from the community",
        'references': [
            'https://cloud.google.com/tensorflow-enterprise/docs/overview'
        ]
    },
    {
        'question':
        "Your team works for a startup company with Google Cloud. You develop, train and deploy several ML models with Tensorflow. You use data in Parquet format and need to manage it both in input and output. You want the smoothest solution without adding infrastructure and keeping costs down.\n"
        "Which one of the following options do you follow?",
        'tags': [17, 'whizlabs2'],
        'options': {
            'A': "Cloud Dataproc",
            'B': "TensorFlow I/O",
            'C': "Dataflow Flex Template",
            'D': "BigQuery to TFRecords"
        },
        'answers': ['B'],
        'explanation': 
        "TensorFlow I/O is a set of useful file formats, Dataset, streaming, and file system types management not available in TensorFlow's built-in support, like Parquet.\n"
        "So the integration will be immediate without any further costs or data transformations.\n"
        "Apache Parquet is an open-source column-oriented data storage format born in the Apache Hadoop environment but supported in many tools and used for data analysis.\n"
        "* Cloud Dataproc is the managed Hadoop service in GCP. It uses Parquet but not Tensorflow out of the box. Furthermore, it’d be an additional cost.\n"
        "* Dataflow Flex Template and BigQuery to TFRecords are wrong because there will be an additional cost and additional data transformations.",
        'references': [
            'https://www.tensorflow.org/io',
            'https://towardsdatascience.com/data-formats-for-training-in-tensorflow-parquet-petastorm-feather-and-more-e55179eeeb72'
        ]
    },
    {
        'question':
        "You are starting to operate as a Data Scientist and speaking with your mentor who asked you to prepare a simple model with a lazy learning algorithm.\n"
        "The problem is that you don’t know the meaning of lazy learning; so you looked for it.\n"
        "Which of the following methods uses lazy learning?",
        'tags': [18, 'whizlabs2'],
        'options': {
            'A': "Naive Bayes",
            'B': "K-Nearest Neighbors",
            'C': "Logistic Regression",
            'D': "Simple Neural Networks",
            'E': "Semi-supervised learning"
        },
        'answers': ['A', 'B'],
        'explanation': 
        "Lazy learning means that the algorithm only stores the data of the training part without learning a function. The stored data will then be used for the evaluation of a new query point.\n"
        "K-nearest neighbor is a simple supervised algorithm for both classification and regression problems.\n"
        "You begin with data that is already classified. A new example will be set by looking at the k nearest classified points. Number k is the most important hyperparameter.\n"
        "Naive Bayes is a classification algorithm. The features have to be independent. It requires a small amount of training data.\n"
        "* With Neural Networks and Logistic Regression you have to train the model and figure out the parameters of a specific function that best fit the data before the inference.\n"
        "* Semi-supervised learning is a family of classification algorithms with labeled and unlabeled data and methods to organize examples based on similarities and clustering. They have to set up a model and find parameters with training jobs.",
        'references': [
            'https://towardsdatascience.com/all-machine-learning-algorithms-you-should-know-in-2021-2e357dd494c7',
            'https://towardsdatascience.com/k-nearest-neighbors-knn-algorithm-23832490e3f4',
            'https://machinelearningmastery.com/parametric-and-nonparametric-machine-learning-algorithms/',
            'https://en.wikipedia.org/wiki/Lazy_learning'
        ]
    },
    {
        'question':
        "Your company traditionally deals with the statistical analysis of data. The services have been integrated with ML models for forecasting for some years, but analyzes and simulations of all kinds are carried out.\n"
        "So you are using two types of tools. But you have been told that it is possible to have more levels of integration between traditional statistical methodologies and those more related to AI / ML processes.\n"
        "Which tool is the best one for your needs?",
        'tags': [19, 'whizlabs2'],
        'options': {
            'A': "TensorFlow Hub",
            'B': "TensorFlow Probability",
            'C': "TensorFlow Enterprise",
            'D': "TensorFlow Statistics"
        },
        'answers': ['B'],
        'explanation': 
        "TensorFlow Probability is a Python library for statistical analysis and probability, which can be processed on TPU and GPU.\n"
        "TensorFlow Probability main features are:\n"
        "* Probability distributions and differentiable and injective (one to one) functions.\n"
        "* Tools for deep probabilistic models building.\n"
        "* Inference and Simulation methods support: Markov chain, Monte Carlo.\n"
        "* Optimizers such as Nelder-Mead, BFGS, and SGLD.\n"
        "All the other answers are wrong because they don’t deal with traditional statistical methodologies.",
        'references': [
            'https://www.tensorflow.org/probability'
        ]
    },
    {
        'question':
        "Your team works for an international company with Google Cloud. You develop, train and deploy different ML models. You use a lot of tools and techniques and you want to make your work leaner, faster and more efficient.\n"
        "Now you have the problem that you have to create a model for recognizing photographic images related to collaborators and consultants. You have to do it quickly, and it has to be an R-CNN model. You don't want to start from scratch. So you are looking for something that can help you and that can be optimal for the GCP platform.\n"
        "Which of these tools do you think can help you?",
        'tags': [20, 'whizlabs2'],
        'options': {
            'A': "TensorFlow-hub",
            'B': "GitHub",
            'C': "GCP Marketplace Solutions",
            'D': "BigQueryML Open"
        },
        'answers': ['A'],
        'explanation': 
        "TensorFlow Hub is ready to use repository of trained machine learning models.\n"
        "It is available for reusing advanced trained models with minimal code.\n"
        "The ML models are optimized for GCP.\n"
        "* GitHub is public and for any kind of code.\n"
        "* GCP Marketplace Solutions is a solution that lets you select and deploy software packages from vendors.\n"
        "* BigQueryML Open is related to Open Data.",
        'references': [
            'https://www.tensorflow.org/hub'
        ]
    },
    {
        'question':
        "You work in a large company that produces luxury cars. The following models will have a control unit capable of collecting data on mileage and technical status to allow intelligent management of maintenance by both the customer and the service centers.\n"
        "Every day a small batch of data will be sent that will be collected and processed in order to provide customers with the management of their vehicle health and push notifications in case of important messages.\n"
        "Which GCP products are the most suitable for this project (pick 3)?",
        'tags': [21, 'whizlabs2'],
        'options': {
            'A': "Pub/Sub",
            'B': "DataFlow",
            'C': "Dataproc",
            'D': "Firebase Messaging"
        },
        'answers': ['A', 'B', 'D'],
        'explanation': 
        "The best products are:\n"
        "* Pub/Sub for technical data messages\n"
        "* DataFlow for data management both in streaming and in batch mode\n"
        "* Firebase Messaging for push notifications\n\n"
        "DataFlow manages data pipelines directed acyclic graphs (DAG) of transformations (PTransforms) on data (PCollections).\n"
        "The same pipeline can activate multiple PTransforms.\n"
        "All the processing can be performed both in batch and in streaming mode.\n"
        "So, in our case of streaming data, Dataflow can:\n"
        "* Serialize input data\n"
        "* Preprocess and transform data\n"
        "* Call the inference function\n"
        "* Get the results and postprocess them\n\n"
        "Dataproc is the managed Apache Hadoop environment for big data analysis usually for batch processing.",
        'references': [
            'https://cloud.google.com/architecture/processing-streaming-time-series-data-overview',
            'https://cloud.google.com/blog/products/data-analytics/ml-inference-in-dataflow-pipelines',
            'https://github.com/GoogleCloudPlatform/dataflow-sample-applications/tree/master/timeseries-streaming'
        ]
    },
    {
        'question':
        "Your company does not have a great ML experience. Therefore they want to start with a service that is as smooth, simple and managed as possible.\n"
        "The idea is to use BigQuery ML. Therefore you are considering whether it can cover all the functionality you need. Various projects start with the design and set up various models using various techniques and algorithms in your company.\n"
        "Which of these techniques/algorithms is not supported by BigQuery ML?",
        'tags': [22, 'whizlabs2'],
        'options': {
            'A': "Wide-and-Deep DNN models",
            'B': "ARIMA",
            'C': "Ensamble Boosted Model",
            'D': "CNN"
        },
        'answers': ['D'],
        'explanation': 
        "The convolutional neural network (CNN) is a type of artificial neural network extensively used especially for image recognition and classification. It uses the convolutional layers, that is, the reworking of sets of pixels by running filters on the input pixels.\n"
        "It is not supported because it is specialized for images.\n"
        "The other answers are wrong because they are all supported by BigQuery ML.\n"
        "Following the list of the current models and techniques.\n"
        "* Linear regression\n"
        "* Binary logistic regression\n"
        "* Multiclass logistic regression\n"
        "* K-means clustering\n"
        "* Matrix Factorization\n"
        "* Time series\n"
        "* Boosted Tree\n"
        "* Deep Neural Network (DNN)\n"
        "* AutoML Tables\n"
        "* TensorFlow model importing\n"
        "* Autoencoder\n"
        "MODEL_TYPE = { 'LINEAR_REG' | 'LOGISTIC_REG' | 'KMEANS' | 'PCA' | 'MATRIX_FACTORIZATION' | 'AUTOENCODER' | 'TENSORFLOW' | 'AUTOML_REGRESSOR' | 'AUTOML_CLASSIFIER' | 'BOOSTED_TREE_CLASSIFIER' | 'BOOSTED_TREE_REGRESSOR' | 'DNN_CLASSIFIER' | 'DNN_REGRESSOR' | 'DNN_LINEAR_COMBINED_CLASSIFIER' | 'DNN_LINEAR_COMBINED_REGRESSOR' | 'ARIMA_PLUS' }",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/introduction'
        ]
    },
    {
        'question':
        "Your team is working on a great number of ML projects.\n"
        "You need to appropriately collect and transform data and then create and tune your ML models.\n"
        "In a second moment, these procedures will be inserted in an MLOps flow and therefore will have to be automated and be as simple as possible.\n"
        "What are the methodologies / services recommended by Google (pick 3)?\n",
        'tags': [23, 'whizlabs2'],
        'options': {
            'A': "Dataflow",
            'B': "BigQuery",
            'C': "Tensorflow",
            'D': "Cloud Fusion",
            'E': "Dataprep"
        },
        'answers': ['A', 'B', 'C'],
        'explanation': 
        "Dataflow is an optimal solution for compute-intensive preprocessing operations because it is a fully managed autoscaling service for batch and streaming data processing.\n"
        "BigQuery is a strategic tool for GCP. BigData at scale, machine learning, preprocessing with plain SQL are all important factors.\n"
        "TensorFlow has many tools for data preprocessing and transformation operations.\n"
        "Main techniques are aimed to feature engineering (crossed_column, embedding_column, bucketized_column) and data transformation (tf.Transform library).\n"
        "* Cloud Fusion is for ETL with a GUI, so with limited programming.\n"
        "* Dataprep is a tool for visual data cleaning and preparation.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/datasets/datasets',
            'https://cloud.google.com/architecture/data-preprocessing-for-ml-with-tf-transform-pt1',
            'https://cloud.google.com/architecture/data-preprocessing-for-ml-with-tf-transform-pt1#where_to_do_preprocessing',
            'https://cloud.google.com/blog/topics/developers-practitioners/architect-your-data-lake-google-cloud-data-fusion-and-composer'
        ]
    },
    {
        'question':
        "Your team is preparing a Deep Neural Network custom model with Tensorflow in Vertex AI that forecasts, based on diagnostic images, medical diagnoses. It is a complex and demanding job. You want to get help from GCP for hyperparameter tuning.\n"
        "What are the parameters that you must indicate (Select TWO)?",
        'tags': [24, 'whizlabs2'],
        'options': {
            'A': "learning_rate",
            'B': "parameterServerType",
            'C': "scaleTier",
            'D': "num_hidden_layers"
        },
        'answers': ['A', 'D'],
        'explanation':
        "With Vertex AI, it is possible to create a hyperparameter tuning job for LINEAR_REGRESSION and DNN.\n"
        "You can choose many parameters. But in the case of DNN, you have to use a hyperparameter named learning_rate.\n"
        "The ConditionalParameterSpec object lets you add hyperparameters to a trial when the value of its parent hyperparameter matches a condition that you specify (added automatically) and the number of hidden layers, that is num_hidden_layers.\n"
        "* scaleTier and parameterServerType are parameters for infrastructure set up for a training job.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview',
            'https://youtu.be/8hZ_cBwNOss'
        ]
    },
    {
        'question':
        "Your team needs to create a model for managing security in restricted areas of campus.\n"
        "Everything that happens in these areas is filmed. Instead of having a physical surveillance service, the videos must be managed by a model capable of intercepting unauthorized people and vehicles, especially at particular times.\n"
        "What are the GCP services that allow you to achieve all this with minimal effort?",
        'tags': [25, 'whizlabs2'],
        'options': {
            'A': "AI Infrastructure",
            'B': "Cloud Video Intelligence AI",
            'C': "AutoML Video Intelligence Classification",
            'D': "Vision AI"
        },
        'answers': ['C'],
        'explanation': 
        "AutoML Video Intelligence is a service that allows you to customize the pre-trained Video intelligence GCP system according to your specific needs.\n"
        "In particular, AutoML Video Intelligence Object Tracking allows you to identify and locate particular entities of interest to you with your specific tags.\n"
        "* AI Infrastructure allows you to manage hardware configurations for ML systems and, in particular, the processors used to accelerate machine learning workloads.\n"
        "* Cloud Video Intelligence AI is a pre-configured and ready-to-use service, therefore not configurable for specific needs.\n"
        "* Vision AI is for images and not video.",
        'references': [
            'https://cloud.google.com/video-intelligence/automl/object-tracking/docs/index-object-tracking',
            'https://cloud.google.com/video-intelligence/automl/docs/beginners-guide'
        ]
    },
    {
        'question': 
        "Your client has a large e-commerce Website that sells sports goods and especially scuba diving equipment.\n"
        "It has a seasonal business and has collected a lot of sales data from its structured ERP and market trend databases.\n"
        "It wants to predict the demand of its customers both to increase business and improve logistics processes.\n"
        "What managed and fast-to-use GCP products can be used for these types of models (pick 2)?",
        'tags': [26, 'whizlabs2'],
        'options': {
            'A': "Auto ML",
            'B': "BigQuery ML",
            'C': "KubeFlow",
            'D': "TFX"
        },
        'answers': ['A', 'B'],
        'explanation': 
        "We speak clearly of X. Obviously, we have in GCP the possibility to use a large number of models and platforms. But the fastest and most immediate modes are with Auto ML and BigQuery ML; both support quick creation and fine-tuning of templates.\n"
        "* KubeFlow and TFX are open-source libraries that work with Tensorflow. So, they are not managed and so simple.\n\n"
        "Moreover, they can work in an environment outside GCP that is a big advantage, but it is not in our requirements.\n"
        "Kubeflow is a system for deploying, scaling and managing complex Tensorflow systems on Kubernetes.\n"
        "TFX is a platform that allows you to create scalable production ML pipelines for TensorFlow projects.",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create#model_type',
            'https://ai.googleblog.com/2020/12/using-automl-for-time-series-forecasting.html'
        ]
    },
    {
        'question':
        "You are consulting a CIO of a big firm regarding organization and cost optimization for his company's ML projects in GCP.\n"
        "He asked: “How can I get the most from ML services and the least costs?”\n"
        "What are the best practices recommended by Google in this regard?",
        'tags': [27, 'whizlabs2'],
        'options': {
            'A': "Use Notebooks as ephemeral instances",
            'B': "Set up an automatic shutdown routine",
            'C': "Use Preemptible VMs per long-running interrumpible tasks",
            'D': "Get monitoring alerts about GPU usage",
            'E': "All of the above"
        },
        'answers': ['E'],
        'explanation': 
        "* Notebooks are used for a limited time, but they reserve VM and other resources. So you have to treat them as ephemeral instances, not as long-living ones.\n"
        "* You can configure an automatic shutdown routine when your instance is idle, saving money.\n"
        "* Preemptible VMs are far cheaper than normal instances and are OK for long-running (batch) large experiments.\n"
        "* You can set up the GPU metrics reporting script; it is important because GPU is expensive.",
        'references': [
            'https://medium.com/kovalevskyi-viacheslav/aiplatform-notebooks-and-vms-auto-shutdown-on-idle-dd94ed3d4724',
            'https://cloud.google.com/compute/docs/instances/preemptible',
            'https://cloud.google.com/compute/docs/gpus/monitor-gpus#setup-script',
            'https://cloud.google.com/solutions/machine-learning/best-practices-for-ml-performance-cost'
        ]
    },
    {
        'question':
        "Your team is working with a great number of ML projects, especially with Tensorflow.\n"
        "You have to prepare a demo for the Manager and Stakeholders. You are certain that they will ask you about the understanding of the classification and regression mechanism. You’d like to show them an interactive demo with some cool interference.\n"
        "Which of these tools is best for all of this?",
        'tags': [28, 'whizlabs2'],
        'options': {
            'A': "Tensorboard",
            'B': "Tableau",
            'C': "What-If Tool",
            'D': "Looker",
            'E': "LIT"
        },
        'answers': ['C'],
        'explanation': 
        "The What-If Tool (WIT) is an open-source tool that lets you visually understand classification and regression ML models.\n"
        "It lets you see data points distributions with different shapes and colors and interactively try new inferences.\n"
        "Moreover, it shows which features affect your model the most, together with many other characteristics.\n"
        "All without code.\n"
        "* Tensorboard provides visualization and tooling needed for experiments, not for explaining inference. You can access the What-If Tool from Tensorboard.\n"
        "* Tableau and Looker are graphical tools for data reporting.\n"
        "* LIT is for NLP models.",
        'references': [
            'https://www.tensorflow.org/tensorboard/what_if_tool'
        ]
    },
    {
        'question':
        "Your team is working with a great number of ML projects, especially with Tensorflow.\n"
        "You recently prepared an NLP model that works well and is about to be rolled out in production.\n"
        "You have to prepare a demo for the Manager and Stakeholders for your new system of text and sentiment interpretation. You are certain that they will ask you for explanations and understanding about how a software may capture human feelings. You’d like to show them an interactive demo with some cool interference.\n"
        "Which of these tools is best for all of this?",
        'tags': [29, 'whizlabs2'],
        'options': {
            'A': "Tensorboard",
            'B': "Tableau",
            'C': "What-If Tool",
            'D': "Looker",
            'E': "LIT"
        },
        'answers': ['E'],
        'explanation': 
        "The Language Interpretability Tool (LIT) is an open-source tool developed specifically to explain and visualize NLP natural language processing models.\n"
        "It is similar to the What-If tool, which instead targets classification and regression models with structured data.\n"
        "It offers visual explanations of the model's predictions and analysis with metrics, tests and validations.\n"
        "* Tensorboard provides visualization and tooling needed for experiments, not for explaining inference. You can access the What-If Tool from Tensorboard.\n"
        "* Tableau and Looker are graphical tools for data reporting.\n"
        "* What-If Tool is for classification and regression models with structured data.",
        'references': [
            'https://pair-code.github.io/lit/',
            'https://www.tensorflow.org/tensorboard/what_if_tool'
        ]
    },
    {
        'question': 
            "Your team is working with a great number of ML projects, especially with Tensorflow.\n"
            "You recently prepared a DNN model for image recognition that works well and is about to be rolled out in production.\n"
            "Your manager asked you to demonstrate the inner workings of the model.\n"
            "It is a big problem for you because you know that it is working well but you don’t have the explainability of the model.\n"
            "Which of these techniques could help you?",
        'tags': [30, 'whizlabs2'],
        'options': {
            'A': "Integrated Gradient",
            'B': "LIT",
            'C': "WIT",
            'D': "PCA"
        },
        'answers': ['A'],
        'explanation':
            "Integrated Gradient is an explainability technique for deep neural networks which gives info about what contributes to the model’s prediction.\n"
            "Integrated Gradient works highlight the feature importance. It computes the gradient of the model’s prediction output regarding its input features without modification to the original model.\n"
            "In the picture, you can see that it tunes the inputs and computes attributions so that it can compute the feature importances for the input image.\n"
            "You can use tf.GradientTape to compute the gradients\n"
            "* LIT is only for NLP models\n"
            "* What-If Tool is only for classification and regression models with structured data.\n"
            "* Principal component analysis (PCA) transforms and reduces the number of features by creating new variables, from linear combinations of the original variables.\n"
            "The new features will be all independent of each other.",
        'references': [
            'https://www.tensorflow.org/overview',
            'https://towardsdatascience.com/understanding-deep-learning-models-with-integrated-gradients-24ddce643dbf'
        ]
    },
    {
        'question':
        "You are working on a linear regression model with data stored in BigQuery. You have a view with many columns. You want to make some simplifications for your work and avoid overfitting. You are planning to use regularization. You are working with Bigquery ML and preparing the query for model training. You need an SQL statement that allows you to have all fields in the view apart from the label.\n"
        "Which one do you choose?",
        'tags': [31, 'whizlabs2'],
        'options': {
            'A': "ROLLUP",
            'B': "UNNEST",
            'C': "EXCEPT",
            'D': "LAG"
        },
        'answers': ['C'],
        'explanation': 
        "SQL and Bigquery are powerful tools for querying and manipulating structured data.\n"
        "EXCEPT gives all rows or fields on the left side except the one coming from the right side of the query.\n"
        "Example:\n"
        "SELECT\n"
        "* EXCEPT(mylabel) myvalue AS label\n\n"
        "* ROLLUP is a group function for subtotals.\n"
        "* UNNEST gives the elements of a structured file.\n"
        "* LAG returns the field value on a preceding row.",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-hyperparameter-tuning',
            'https://cloud.google.com/bigquery-ml/docs/hyperparameter-tuning-tutorial'
        ]
    },
    {
        'question':
        "Your team is preparing a multiclass logistic regression model with tabular data.\n"
        "The environment is Vertex AI with Auto ML, and your data is stored in a CSV file in Cloud Storage.\n"
        "AutoML can perform transformations on the data to make the most of it.\n"
        "Which of the following types of transformations are you not allowed, based on your requirements?",
        'tags': [32, 'whizlabs2'],
        'options': {
            'A': "Categorical",
            'B': "Text",
            'C': "Timestamp",
            'D': "Array",
            'E': "Number"
        },
        'answers': ['D'],
        'explanation': 
        "With complex data like Arrays and Structs, transformations are available only by using BigQuery, which supports them natively.\n"
        "All the other kinds of data are also supported for CSV files, as stated in the referred documentation.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/datasets/data-types-tabular',
            'https://cloud.google.com/vertex-ai/docs/datasets/data-types-tabular#compound_data_types'
        ]
    },
    {
        'question':
        "You are a junior Data Scientist, and you work in a Governmental Institution.\n"
        "You are preparing data for a linear regression model for Demographic research. You need to choose and manage the correct feature.\n"
        "Your input data is in BigQuery.\n"
        "You know very well that you have to avoid multicollinearity and optimize categories. So you need to group some features together and create macro categories.\n"
        "In particular, you have to join country and language in one variable and divide data between 5 income classes.\n"
        "Which ones of the following options can you use (pick 2)?",
        'tags': [33, 'whizlabs2'],
        'options': {
            'A': "FEATURE_CROSS",
            'B': "ARRAY_CONCAT",
            'C': "QUANTILE_BUCKETIZE",
            'D': "ST_AREA"
        },
        'answers': ['A', 'C'],
        'explanation': 
        "A feature cross is a new feature that joins two or more input features together. (The term cross comes from cross product.) Usually, numeric new features are created by multiplying two or more other features.\n"
        "QUANTILE_BUCKETIZE groups a continuous numerical feature into categories with the bucket name as the value based on quantiles.\n"
        "Example: ML.FEATURE_CROSS STRUCT(country,    language) AS origin)\n"
        " and ML.QUANTILE_BUCKETIZE → income_class\n\n"
        "* ARRAY_CONCAT joins one or more arrays (number or strings) into a single array.\n"
        "* ST_AREA returns the number of square meters covered by a GEOGRAPHY area.",
        'references': [
            'https://towardsdatascience.com/assumptions-of-linear-regression-fdb71ebeaa8b',
            'https://cloud.google.com/bigquery-ml/docs/bigqueryml-transform'
        ]
    },
    {
        'question':
        "You are a junior Data Scientist and you need to create a multi-class classification Machine Learning model with Keras Sequential model API.\n"
        "You have been asked which activation function to use.\n"
        "Which of the following do you choose?",
        'tags': [34, 'whizlabs2'],
        'options': {
            'A': "ReLU",
            'B': "Softmax",
            'C': "SIGMOID",
            'D': "TANH"
        },
        'answers': ['B'],
        'explanation': 
        "Softmax is for multi-class classification what Sigmoid is for logistic regression. Softmax assigns decimal probabilities to each class so that their sum is 1.\n"
        "* ReLU (Rectified Linear Unit): half rectified. f(z) is zero when z is less than zero and f(z) is equal to z when z. It returns one value\n"
        "* Sigmoid is for logistic regression and therefore returns one value from 0 to 1.\n"
        "* Tanh or hyperbolic tangent is like sigmoid but returns one value from -1 to 1.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax?hl=en',
            'https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax'
        ]
    },
    {
        'question':
        "Your team is working on a great number of ML projects for an international consulting firm.\n"
        "The management has decided to store most of the data to be used for ML models in BigQuery.\n"
        "The motivation is that BigQuery allows for preprocessing and transformations easily and with standard SQL. It is highly structured; so it offers efficiency, integration and security.\n"
        "Your team must create and modify code to directly access BigQuery data for building models in different environments.\n"
        "What are the tools you can use (pick 3)?",
        'tags': [35, 'whizlabs2'],
        'options': {
            'A': "Tf.data.dataset",
            'B': "BigQuery Omni",
            'C': "BigQuery Python client library",
            'D': "BigQuery I/O Connector"
        },
        'answers': ['A', 'C', 'D'],
        'explanation': 
        "tf.data.dataset reader for BigQuery is the way to connect directly to BigQuery from TensorFlow or Keras.\n"
        "BigQuery I/O Connector is the way to connect directly to BigQuery from Dataflow.\n"
        "For any other framework, you can use BigQuery Python client library\n"
        "* BigQuery Omni is a multi-cloud analytics solution. You can access from BigQuery data across Google Cloud, Amazon Web Services (AWS), and Azure.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets',
            'https://cloud.google.com/bigquery/docs/bigquery-storage-python-pandas',
            'https://beam.apache.org/documentation/io/built-in/google-bigquery/',
            'https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets'
        ]
    },
    {
        'question':
        "Your team has prepared a Multiclass logistic regression model with tabular data in the Vertex AI with Auto ML environment. Everything went very well. You appreciated the convenience of the platform and AutoML.\n"
        "What other types of models can you implement with AutoML (Pick 3)?",
        'tags': [36, 'whizlabs2'],
        'options': {
            'A': "Image Data",
            'B': "Text Data",
            'C': "Cluster Data",
            'D': "Video Data"
        },
        'answers': ['A', 'B', 'D'],
        'explanation': 
        "AutoML on Vertex AI can let you build a code-free model. You have to provide training data.\n"
        "The types of models that AutoML on Vertex AI can build are created with image data, tabular data, text data, and video data.\n"
        "All the detailed information is at the link:\n"
        "* Cluster Data may be related to unsupervised learning; that is not supported by Auto ML.",
        'references': [
            'https://cloud.google.com/vision/automl/docs/beginners-guide',
            'https://cloud.google.com/vertex-ai/docs/start/automl-model-types'
        ]
    },
    {
        'question':
        "With your team, you have to decide the strategy for implementing an online forecasting model in production. This template needs to work with both a web interface as well as DialogFlow and Google Assistant. A lot of requests are expected.\n"
        "You are concerned that the final system is not efficient and scalable enough. You are looking for the simplest and most managed GCP solution.\n"
        "Which of these can be the solution?",
        'tags': [37, 'whizlabs2'],
        'options': {
            'A': "Vertex AI online prediction",
            'B': "GKE e TensorFlow",
            'C': "VMs and Autoscaling Groups with Application LB",
            'D': "Kubeflow"
        },
        'answers': ['A'],
        'explanation': 
        "The Vertex AI prediction service is fully managed and automatically scales machine learning models in the cloud.\n"
        "The service supports both online prediction and batch prediction.\n"
        "GKE e TensorFlow, VMs and Autoscaling Groups with Application LB are not managed services.\n"
        "Kubeflow is not a managed service. It is used in Vertex AI and lets you deploy ML systems in various environments.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/predictions/online-predictions-custom-models',
            'https://cloud.google.com/vertex-ai/docs/predictions/configure-compute'
        ]
    },
    {
        'question':
        "You work in a medium-sized company as a developer and data scientist and use the managed ML platform, Vertex AI.\n"
        "You have updated an Auto ML model and want to deploy it to production. But you want to maintain both the old and the new version at the same time. The new version should only serve a small portion of the traffic.\n"
        "What can you do (Select TWO)?",
        'tags': [38, 'whizlabs2'],
        'options': {
            'A': "Save the model in a Docker container image",
            'B': "Deploy on the same endpoint",
            'C': "Update the Traffic split percentage",
            'D': "Create a Canary Deployment with Cloud Build"
        },
        'answers': ['B', 'C'],
        'explanation': 
        "The correct procedure is:\n"
        "* Deploy your model to an existing endpoint.\n"
        "* Update the Traffic split percentage in such a way that all of the percentages add up to 100%.\n\n"
        "You don’t have to create a Docker container image with AutoML.\n"
        "Canary Deployment with Cloud Build is a procedure used in CI/CD pipelines. There is no need in such a managed environment.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-console'
        ]
    },
    {
        'question':
        "You and your team are working for a large consulting firm. You are preparing an NLP ML model to classify customer support needs and to assess the degree of satisfaction. The texts of the various communications are stored in different storage.\n"
        "What types of storage should you avoid in the managed environment of GCP ML, such as Vertex AI (Select TWO)?",
        'tags': [39, 'whizlabs2'],
        'options': {
            'A': "Cloud Storage",
            'B': "BigQuery",
            'C': "Filestore",
            'D': "Block Storage"
        },
        'answers': ['C', 'D'],
        'explanation': 
        "Google advises avoiding data storage for ML in block storage, like persistent disks or NAS like Filestore.\n"
        "They are more difficult to manage than Cloud Storage or BigQuery.\n"
        "Likewise, it is strongly discouraged to read data directly from databases such as Cloud SQL. So, it is strongly recommended to store data in BigQuery and Cloud Storage.\n"
        "Similarly, avoid reading data directly from databases like Cloud SQL.",
        'references': [
            'https://cloud.google.com/architecture/ml-on-gcp-best-practices#avoid-storing-data-in-block-storage',
            'https://cloud.google.com/bigquery/docs/loading-data',
            'https://cloud.google.com/blog/products/ai-machine-learning/google-cloud-launches-vertex-ai-unified-platform-for-mlops'
        ]
    },
    {
        'question':
        "You are working with Vertex AI, the managed ML Platform in GCP.\n"
        "You want to leverage Explainable AI to understand which are the most essential features and how they influence the model.\n"
        "For what kind of model may you use Vertex Explainable AI (pick 3)?",
        'tags': [40, 'whizlabs2'],
        'options': {
            'A': "AutoML tables",
            'B': "Image Classification",
            'C': "DNN",
            'D': "Decision Tree"
        },
        'answers': ['A', 'B', 'C'],
        'explanation': 
        "Deep Learning is known to give little comprehension about how a model works in detail.\n"
        "Vertex Explainable AI helps to detect it, both for classification and regression tasks. So these functions are useful for testing, tuning, finding biases and thus improving the process.\n"
        "You can get explanations from Vertex Explainable AI both for online and batch inference but only regarding these ML models:\n"
        "* Structured data models (AutoML, classification and regression)\n"
        "* Custom-trained models with tabular data and images\n\n"
        "In the Evaluate section, you can find these insights in the Google Cloud Console (Feature importance graph).\n"
        "* Decision Tree Models are explainable without any sophisticated tool for enlightenment.\n\n"
        "It uses three methods for feature attributions:\n"
        "* sampled Shapley: Uses scores for each feature and their permutations\n"
        "* integrated gradextension of the integrated gradients method creates a saliency map with overlapping regions of the image (like in the picture)",
        'references': [
            'https://cloud.google.com/resources/mlops-whitepaper',
            'https://cloud.google.com/vertex-ai/docs/explainable-ai/overview'
        ]
    },
    {
        'question': 
            "You work as a Data Scientist in a Startup and you work with several project with Python and Tensorflow;\n"
            "You need to increase the performance of the training sessions and you already use caching and prefetching.\n"
            "So now you want to use GPUs, but in a single machine, for cost reduction and experimentations.\n"
            "Which of the following is the correct strategy?",
        'tags': [41, 'whizlabs2'],
        'options': {
            'A': "tf.distribute.MirroredStrategy",
            'B': "tf.distribute.TPUStrategy",
            'C': "tf.distribute.MultiWorkerMirroredStrategy",
            'D': "tf.distribute.OneDeviceStrategy"
        },
        'answers': ['A'],
        'explanation':
            "tf.distribute.Strategy is an API explicitly for training distribution among different processors and machines.\n"
            "tf.distribute.MirroredStrategy lets you use multiple GPUs in a single VM, with a replica for each CPU.\n"
            "* tf.distribute.TPUStrategy let you use TPUs, not GPUs\n"
            "* tf.distribute.MultiWorkerMirroredStrategy is for multiple machines\n"
            "* tf.distribute.OneDeviceStrategy, like the default strategy, is for a single device, so a single virtual CPU.",
        'references': [
            'https://www.tensorflow.org/guide/distributed_training',
            'https://www.tensorflow.org/guide/intro_to_graphs',
            'https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html'
        ]
    },
    {
        'question':
        "You work as a junior Data Scientist in a Startup and work with several projects with Python and Tensorflow in Vertex AI. You deployed a new model in the test environment and detected some problems that are puzzling you.\n"
        "An experienced colleague of yours asked for the logs. You found out that there is no logging information available. What kind of logs do you need and how do you get them (pick 2)?",
        'tags': [42, 'whizlabs2'],
        'options': {
            'A': "You need to Use Container logging",
            'B': "You need to Use Access logging",
            'C': "You can enable logs dynamically",
            'D': "You have to undeploy and redeploy"
        },
        'answers': ['A', 'D'],
        'explanation':
        "In Vertex AI, you may enable or avoid logs for prediction. When you want to change, you must undeploy and redeploy.\n"
        "There are two types of logs:\n"
        "* Container logging, which logs data from the containers hosting your model; so these logs are essential for problem solving and debugging.\n"
        "* Access logging, which logs accesses and latency information.\n\n"
        "Therefore, you need Container logging.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/predictions/online-prediction-logging'
        ]
    },
    {
        'question':
        "You are a junior Data Scientist working on a logistic regression model to break down customer text messages into two categories: important / urgent and unimportant / non-urgent.\n"
        "You want to find a metric that allows you to evaluate your model for how well it separates the two classes. You are interested in finding a method that is scale invariant and classification threshold invariant.\n"
        "Which of the following is the optimal methodology?",
        'tags': [43, 'whizlabs2'],
        'options': {
            'A': "Log Loss",
            'B': "One-hot encoding",
            'C': "ROC- AUC",
            'D': "Mean Square Error",
            'E': "Mean Absolute Error"
        },
        'answers': ['C'],
        'explanation': 
        "The ROC curve (receiver operating characteristic curve) is a graph showing the behavior of the model with positive guesses at different classification thresholds.\n"
        "It plots and relates each others two different values:\n"
        "* TPR: true positives / all actual positives\n"
        "* FPR: false positives / all actual negatives\n\n"
        "The AUC (Area Under the Curve) index is the area under the ROC curve and indicates the capability of a binary classifier to discriminate between two categories. Being a probability, it is always a value between 0 and 1. Hence it is a scale invariant.\n"
        "It provides divisibility between classes. So it is independent of the chosen threshold value; in other words, it is threshold-invariant.\n"
        "When it is equal, it is 0.5 indicating that the model randomly foresees the division between two classes, similar to what happens with heads and tails when tossing coins.\n"
        "* Log Loss is a loss function used especially for logistic regression; it measures loss. So it is highly dependent on threshold values.\n"
        "* One-hot encoding is a method used in feature engineering for obtaining better regularization and independence.\n"
        "* Mean Square Error is the most frequently used loss function used for linear regression. It takes the square of the difference between predictions and real values.\n"
        "* Mean Absolute Error is a loss function, too. It takes the absolute value of the difference between predictions and actual outcomes.",
        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc'
        ]
    },
    {
        'question':
        "You work as a junior Data Scientist in a consulting company and work with several projects with Tensorflow. You prepared and tested a new model, and you are optimizing it before deploying it in production. You asked for advice from an experienced colleague of yours. He said that it is not advisable to deploy the model in eager mode.\n"
        "What can you do (pick 3)?",
        'tags': [44, 'whizlabs2'],
        'options': {
            'A': "Configure eager_execution=no",
            'B': "Use graphs",
            'C': "Use tf.function decoration function",
            'D': "Create a new tf.Graph"
        },
        'answers': ['B', 'C', 'D'],
        'explanation': 
        "When you develop and test a model, the eager mode is really useful because it lets you execute operations one by one and facilitate debugging.\n"
        "But when in production, it is better to use graphs, which are data structures with Tensors and integrated computations Python independent. In this way, they can be deployed on different devices (like mobiles) and are optimizable.\n"
        "To do that, you have to use tf.function decoration function for a new tf.Graph creation.\n"
        "* There is no such parameter as eager_execution = no. Using graphs instead of eager execution is more complex than that.",
        'references': [
            'https://www.tensorflow.org/guide/function',
            'https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Eager_Execution_Enabled.ipynb'
        ]
    },
    {
        'question':
        "In your company, you train and deploy several ML models with Tensorflow. You use on-prem servers, but you often find it challenging to manage the most expensive training.\n"
        "Checking and updating models create additional difficulties. You are undecided whether to use Vertex Pipelines and Kubeflow Pipelines. You wonder if starting from Kubeflow, you can later switch to a more automated and managed system like Vertex AI.\n"
        "Which of these answers are correct (pick 4)?",
        'tags': [45, 'whizlabs2'],
        'options': {
            'A': "Kubeflow pipelines and Vertex Pipelines are incompatible",
            'B': "You may use Kubeflow Pipelines written with DSL in Vertex AI",
            'C': "Kubeflow pipelines work only in GCP",
            'D': "Kubeflow pipelines may work in any environment",
            'E': "Kubeflow pipelines may use Kubernetes persistent volume claims (PVC)",
            'F': "Vertex Pipelines can use Cloud Storage FUSE"
        },
        'answers': ['B', 'D', 'E', 'F'],
        'explanation': 
        "Vertex AI Pipelines is a managed service in GCP.\n"
        "* Kubeflow Pipelines is an open-source tool based on Kubernetes and Tensorflow for any environment.\n"
        "* Vertex AI support code written with Kubeflow Pipelines SDK v2 domain-specific language (DSL).\n\n"
        "Like any workflow in Kubernetes, access to persistent data is performed with Volumes and Volume Claims.\n"
        "Vertex Pipelines can use Cloud Storage FUSE. So Vertex AI can leverage Cloud Storage buckets like file systems on Linux or macOS.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#compare',
            'https://cloud.google.com/storage/docs/gcs-fuse',
            'https://cloud.google.com/vertex-ai'
        ]
    },
    {
        'question':
        "Your company runs a big retail website. You develop many ML models for all the business activities.\n"
        "You migrated to Google Cloud. Your models are developed with PyTorch, TensorFlow, and BigQueryML. You also use BigTable and CloudSQL, and Cloud Storage, of course. You need to use input tabular data in CSV format. You are working with Vertex AI.\n"
        "How do you manage them in the best way (pick 2)?",
        'tags': [46, 'whizlabs2'],
        'options': {
            'A': "Vertex AI manage any CSV automatically, no operations needed",
            'B': "You have to setup an header and column names may have only alphanumeric character and underscore",
            'C': "Vertex AI cannot handle CSV files",
            'D': "Delimiter must be a comma",
            'E': "You can import only a file max 10GB"
        },
        'answers': ['B', 'D'],
        'explanation': 
        "Vertex AI manages CSV files automatically. But you need to have headers only with alphanumeric characters and underscores with commas as delimiters.\n"
        "You can import multiple files, each one max 10GB.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/datasets/prepare-tabular#csv',
            'https://cloud.google.com/vertex-ai/docs/datasets/datasets'
        ]
    },
    {
        'question':
        "Your company is a Financial Institution. You develop many ML models for all the business activities. You migrated to Google Cloud. Your models are developed with PyTorch, TensorFlow, and BigQueryML.\n"
        "You are now working on an international project with other partners. You need to use the Vertex AI. You are asking experts which the capabilities of this managed suite of services are.\n"
        "Which elements are integrated into Vertex AI?",
        'tags': [47, 'whizlabs2'],
        'options': {
            'A': "Training environments and MLOps",
            'B': "Training Pipelines, Datasets, Custom tooling, AutoML, Models Management and inference environments (endpoints)",
            'C': "Deployment environments",
            'D': "Training Pipelines and Datasets for data sources"
        },
        'answers': ['B'],
        'explanation': 
        "Vertex AI covers all the activities and functions listed: from Training Pipelines (so MLOps), to Data Management (Datasets), custom models and Auto ML models management, custom tooling and libraries deployment and monitoring.\n"
        "So, all the other answers are wrong because they cover only a subset of Vertex functionalities.",
        'references': [
            'https://cloud.google.com/vertex-ai',
            'https://codelabs.developers.google.com/codelabs/vertex-ai-custom-code-training'
        ]
    },
    {
        'question':
        "You are a junior data scientist working on a logistic regression model to break down customer text messages into important/urgent and important / not urgent. You want to use the best loss function that you can use to determine your model's performance.\n"
        "Which of the following is the optimal methodology?",
        'tags': [48, 'whizlabs2'],
        'options': {
            'A': "Log Loss",
            'B': "Mean Square Error",
            'C': "Mean Absolute Error",
            'D': "Mean Bias Error",
            'E': "Softmax"
        },
        'answers': ['A'],
        'explanation': 
        "With a logistic regression model, the optimal loss function is the log loss.\n"
        "The intuitive explanation is that when you want to emphasize the loss of bigger mistakes, you need to find a way to penalize such differences.\n"
        "In this case, it is often used the square loss. But in the case of probabilistic values (between 0 and 1), the squaring decreases the values; it does not make them bigger.\n"
        "On the other hand, with a logarithmic transformation, the process is reversed: decimal values get bigger.\n"
        "In addition, logarithmic transformations do not modify the minimum and maximum characteristics (monotonic functions).\n"
        "These are some of the reasons why they are widely used in ML.\n"
        "Pay attention to the difference between loss function and ROC/AUC, which is useful as a measure of how well the model can discriminate between two categories.\n"
        "You may have two models with the same AUC but different losses.\n"
        "* Mean Square Error, as explained, would penalize higher errors.\n"
        "* Mean Absolute Error takes the absolute value of the difference between predictions and actual outcomes. So, it would not empathize higher errors.\n"
        "* Mean Bias Error takes just the value of the difference between predictions and actual outcomes. So, it compensates positive and negative differences between predicted/actual values. It is used to calculate the average bias in the model.\n"
        "* softmax is used in multi-class classification models which is clearly not suitable in the case of a binary-class logarithmic loss.",
        'references': [
            'https://www.kaggle.com/dansbecker/what-is-log-loss',
            'https://developers.google.com/machine-learning/crash-course/logistic-regression/model-training',
            'https://en.wikipedia.org/wiki/Monotonic_function',
            'https://datawookie.dev/blog/2015/12/making-sense-of-logarithmic-loss/'
        ]
    },
    {
        'question':
        "You have just started working as a junior Data Scientist in a Startup. You are involved in several projects with Python and Tensorflow in Vertex AI.\n"
        "You are starting to get interested in MLOps and are trying to understand the different processes involved.\n"
        "You have prepared a checklist, but inside there is a service that has nothing to do with MLOps.\n"
        "Which one?",
        'tags': [49, 'whizlabs2'],
        'options': {
            'A': "CI/CD",
            'B': "Source Control Tools",
            'C': "Data Pipelines",
            'D': "CDN",
            'E': "Artifact Registry, Container Registry"
        },
        'answers': ['D'],
        'explanation': 
        "Cloud CDN is the service that caches and delivers static content from the closest locations (edge locations) to customers to accelerate web and mobile applications. This is a very important service for the Cloud but out of scope for MLOps.\n"
        "MLOps covers all processes related to ML models; experimentation, preparation, testing, deployment and above all continuous integration and delivery.\n"
        "The MLOps environment is designed to provide (some of) the following:\n"
        "* Environment for testing and experimentation\n"
        "* Source control, like Github\n"
        "* CI/CD Continuous integration/continuous delivery\n"
        "* Container registry: custom Docker images management\n"
        "* Feature Stores\n"
        "* Training services\n"
        "* Metadata repository\n"
        "* Artifacts repository\n"
        "* ML pipelines orchestrators\n"
        "* Data warehouse/ storage and scalable data processing for batch and streaming data.\n"
        "* Prediction service both batch and online.\n"
        "So, all the other answers describe MLOps functionalities.",        
        'references': [
            'https://cloud.google.com/architecture/setting-up-mlops-with-composer-and-mlflow',
            'https://mlflow.org/',
            'https://cloud.google.com/composer/docs'
        ]
    },
    {
        'question':
        "You are working with Vertex AI, the managed ML Platform in GCP.\n"
        "You want to leverage Vertex Explainable AI to understand the most important features and how they influence the model.\n"
        "Which three methods does Vertex AI leverage for feature attributions?",
        'tags': [50, 'whizlabs2'],
        'options': {
            'A': "sampled Shapley",
            'B': "integrated gradients",
            'C': "Maximum Likelihood",
            'D': "XRAI"
        },
        'answers': ['A', 'B', 'D'],
        'explanation':
        "Deep Learning is known to give little comprehension about how a model works in detail.\n"
        "Vertex Explainable AI helps to detect it, both for classification and regression tasks. So, these functions are useful for testing, tuning, finding biases and thus improving the process.\n"
        "It uses three methods for feature attributions:\n"
        "* sampled Shapley: Uses scores for each feature and their permutations\n"
        "* integrated gradients: computes the gradient of the features at different points, integrates them and computes the relative weights\n"
        "* XRAI is an optimization of the integrated gradients method\n\n"
        "* Maximum Likelihood is a probabilistic method for determining the parameters of a statistical distribution.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/explainable-ai/overview',
            'https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf'
        ]
    },
    {
        'question':
        "Your company produces and sells a lot of different products.\n"
        "You work as a Data Scientist. You train and deploy several ML models.\n"
        "Your manager just asked you to find a simple method to determine affinities between different products and categories to give sellers and applications a wider range of suitable offerings for customers.\n"
        "The method should give good results even without a great amount of data.\n"
        "Which of the following different techniques may help you better?",
        'tags': [51, 'whizlabs2'],
        'options': {
            'A': "One-hot encoding",
            'B': "Cosine Similarity",
            'C': "Matrix Factorization",
            'D': "PCA"
        },
        'answers': ['B'],
        'explanation':
        "In a recommendation system (like with the Netflix movies) it is important to discover similarities between products so that you may recommend a movie to another user because the different users like similar objects.\n"
        "So, the problem is to find similar products as a first step.\n"
        "You take two products and their characteristics (all transformed in numbers). So, you have two vectors.\n"
        "You may compute differences between vectors in the euclidean space. Geometrically, that means that they have different lengths and different angles.\n"
        "* One-hot encoding is a method used in feature engineering for obtaining better regularization and independence.\n"
        "* Matrix Factorization is correctly used in recommender systems. Still, it is used with a significant amount of data, and there is the problem of reducing dimensionality. So, for us, Cosine Similarity is a better solution.\n"
        "* Principal component analysis is a technique to reduce the number of features by creating new variables.",
        'references': [
            'https://wikipedia.org/wiki/Principal_component_analysis',
            'https://en.wikipedia.org/wiki/Cosine_similarity',
            'https://cloud.google.com/architecture/recommendations-using-machine-learning-on-compute-engine'
        ]
    },
    {
        'question':
        "Your company runs a big retail website. You develop many ML models for all the business activities.\n"
        "You migrated to Google Cloud. Your models are developed with PyTorch, TensorFlow and BigQueryML.\n"
        "You are now working on an international project with other partners.\n"
        "You need to let them use your Vertex AI dataset in Cloud Storage for a different organization.\n"
        "What can you do (pick 2)?",
        'tags': [52, 'whizlabs2'],
        'options': {
            'A': "Let them use your GCP Account",
            'B': "Exporting metadata and annotations in a JSONL file",
            'C': "Exporting metadata and annotations in a CSV file",
            'D': "Give access (Service account or signed URL) to the Cloud Storage file",
            'E': "Copy the data in a removable storage"
        },
        'answers': ['B', 'D'],
        'explanation':
        "You can export a Dataset; when you do that, no additional copies of data are generated. The result is only JSONL files with all the useful information, including the Cloud Storage files URIs.\n"
        "But you have to grant access to these Cloud Storage files with a Service account or a signed URL, if to be used outside GCP.\n"
        "* Let them use your GCP Account, Copy the data in a removable storage are wrong mainly for security reasons.\n"
        "* Annotations are written in JSON files.",
        'references': [
            'https://cloud.google.com/vertex-ai/docs/datasets/export-metadata-annotations',
            'https://cloud.google.com/vertex-ai/docs/datasets/datasets',
            'https://codelabs.developers.google.com/codelabs/vertex-ai-custom-code-training'
        ]
    },
    {
        'question':
        "You work as a junior Data Scientist in a consulting company, and you work with several ML projects.\n"
        "You need to properly collect and transform data and then work on your ML models. You want to identify the services for data transformation that are most suitable for your needs. You need automatic procedures triggered before training.\n"
        "What are the methodologies / services recommended by Google (pick 3)?",
        'tags': [53, 'whizlabs2'],
        'options': {
            'A': "Dataflow",
            'B': "BigQuery",
            'C': "Tensorflow",
            'D': "Cloud Composer"
        },
        'answers': ['A', 'B', 'C'],
        'explanation': 
        "Google primarily recommends BigQuery, because this service allows you to efficiently perform both data and feature engineering operations with SQL standard.\n"
        "In other words, it is suitable both to correct, divide and aggregate the data, and to process the features (fields) merging, normalizing and categorizing them in an easy way.\n"
        "In order to transform data in advanced mode, for example, with window-aggregation feature transformations in streaming mode, the solution is Dataflow.\n"
        "It is also possible to perform transformations on the data with Tensorflow (tf.transform), such as creating new features: crossed_column, embedding_column, bucketized_column.\n"
        "It is important to note that with Tensorflow these transformations become part of the model and will be integrated into the graph that will be produced when the SavedModel is created.\n"
        "Look at the summary table at this link for a complete overview.\n"
        "* Cloud Composer is often used in ML processes, but as a workflow tool, not for data transformation.",
        'references': [
            'https://cloud.google.com/architecture/data-preprocessing-for-ml-with-tf-transform-pt1#preprocessing_options_summary',
            'https://cloud.google.com/vertex-ai/docs/datasets/datasets',
            'https://cloud.google.com/composer',
            'https://cloud.google.com/architecture/setting-up-mlops-with-composer-and-mlflow'
        ]
    },
    {
        'question':
        "You just started working as a junior Data Scientist in a consulting Company. You are in a project team that is building a new model and you are experimenting. But the results are absolutely unsatisfactory because your data is dirty and needs to be modified.\n"
        "In particular, you have various fields that have no value or report NaN. Your expert colleague told you that you need to carry out a procedure that modifies them at the time of acquisition. What kind of functionalities do you need to provide (pick 3)?",
        'tags': [54, 'whizlabs2'],
        'options': {
            'A': "Delete all records that have a null/NaN value in any field",
            'B': "Compute Mean / Median for numeric measures",
            'C': "Replace Categories with the most frequent one",
            'D': "Use another ML model for missing values guess"
        },
        'answers': ['B', 'C', 'D'],
        'explanation': 
        "The most frequent methodologies have been listed.\n"
        "In the case of numerical values, substituting the mean generally does not distort the model (it depends on the underlying statistical distribution).\n"
        "In the case of categories, the most common method is to replace them with the more frequent values.\n"
        "There are often multiple categories in the data. So, in this way, the effect of the missing category is minimized, but the additional values of the current example are used.\n"
        "* The common practice is to delete records / examples that are completely wrong or completely lacking information (all null values).\n\n"
        "In all other cases, it is better to draw all the possible meanings from them.",
        'references': [
            'https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e',
            'https://cloud.google.com/architecture/data-preprocessing-for-ml-with-tf-transform-pt1'
        ]
    },
    {
        'question':
        "You just started working as a junior Data Scientist in a consulting Company.\n"
        "The job they gave you is to perform Data cleaning and correction so that they will later be used in the best possible way for creating and updating ML models.\n"
        "Data is stored in files of different formats.\n"
        "Which GCP service is best to help you with this business?",
        'tags': [55, 'whizlabs2'],
        'options': {
            'A': "BigQuery",
            'B': "Dataprep",
            'C': "Cloud Compose",
            'D': "Dataproc"
        },
        'answers': ['B'],
        'explanation':
        "Dataprep is an end-user service that allows you to explore, clean and prepare structured and unstructured data for many purposes, especially for machine learning.\n"
        "It is completely serverless. You don’t need to write code or procedures.",
        'references': [
            'BigQuery could obviously query and update data. But you need to preprocess data and prepare queries and procedures.',
            'Cloud Compose is for workflow management, not for Data preparation.',
            'Dataproc is a fully managed service for the Apache Hadoop environment.'
        ]
    },
    # Machine Learning Crash Course
    {
        'question':
        "Suppose you want to develop a supervised machine learning model to predict whether a given email is \"spam\" or \"not spam.\"\n" 
        "Which of the following statements are true? (pick 2)",
        'tags': [1, 'crash-course'],
        'options': {
            'A': "The labels applied to some examples might be unreliable.",
            'B': "We'll use unlabeled examples to train the model.",
            'C': "Emails not marked as \"spam\" or \"not spam\" are unlabeled examples.",
            'D': "Words in the subject header will make good labels."
        },
        'answers': ['A', 'B'],
        'explanation':
        
        "* (correct) The labels applied to some examples might be unreliable.\n"
        "Definitely. It's important to check how reliable your data is. The labels for this dataset probably come from email users who mark particular email messages as spam. Since most users do not mark every suspicious email message as spam, we may have trouble knowing whether an email is spam. Furthermore, spammers could intentionally poison our model by providing faulty labels.\n\n"
        
        "* (correct) Emails not marked as \"spam\" or \"not spam\" are unlabeled examples.\n"
        "Because our label consists of the values \"spam\" and \"not spam\", any email not yet marked as spam or not spam is an unlabeled example.\n\n"
        
        "* (wrong) We'll use unlabeled examples to train the model.\n"
        "We'll use labeled examples to train the model. We can then run the trained model against unlabeled examples to infer whether the unlabeled email messages are spam or not spam.\n\n"
        
        "* (wrong) Words in the subject header will make good labels.\n"
        "Words in the subject header might make excellent features, but they won't make good labels.",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/framing/check-your-understanding'
        ]
    },
    {
        'question':
        "Suppose an online shoe store wants to create a supervised ML model that will provide personalized shoe recommendations to users. That is, the model will recommend certain pairs of shoes to Marty and different pairs of shoes to Janet. The system will use past user behavior data to generate training data.\n" 
        "Which of the following statements are true? (pick 2)",
        'tags': [2, 'crash-course'],
        'options': {
            'A': "\"The user clicked on the shoe\'s description\" is a useful label.",
            'B': "\"Shoe beauty\" is a useful feature.",
            'C': "\"Shoes that a user adores\" is a useful label.",
            'D': "\"Shoe size\" is a useful feature."
        },
        'answers': ['A', 'D'],
        'explanation':
        
        "* (correct) \"The user clicked on the shoe\'s description\" is a useful label.\n"
        "Users probably only want to read more about those shoes that they like. Clicks by users is, therefore, an observable, quantifiable metric that could serve as a good training label. Since our training data derives from past user behavior, our labels need to derive from objective behaviors like clicks that strongly correlate with user preferences.\n\n"
        
        "* (correct) \"Shoe size\" is a useful feature.\n"
        "\"Shoe size\" is a quantifiable signal that likely has a strong impact on whether the user will like the recommended shoes. For example, if Marty wears size 9, the model shouldn\'t recommend size 7 shoes.\n\n",
        
        "* (wrong) \"Shoe beauty\" is a useful feature.\n"
        "Good features are concrete and quantifiable. Beauty is too vague a concept to serve as a useful feature. Beauty is probably a blend of certain concrete features, such as style and color. Style and color would each be better features than beauty.\n\n"
        
        "* (wrong) \"Shoes that a user adores\" is a useful label.\n"
        "Adoration is not an observable, quantifiable metric. The best we can do is search for observable proxy metrics for adoration."

        'references': [
            'https://developers.google.com/machine-learning/crash-course/framing/check-your-understanding'
        ]
    },
    {
        'question':
        "Imagine a linear model with 100 input features:\n"
        " * 10 are highly informative.\n"
        " * 90 are non-informative.\n"
        "Assume that all features have values between -1 and 1.\n" 
        "Which of the following statements are true? (pick 2)",
        'tags': [3, 'crash-course'],
        'options': {
            'A': "L2 regularization may cause the model to learn a moderate weight for some non-informative features.",
            'B': "L2 regularization will encourage most of the non-informative weights to be exactly 0.0.",
            'C': "L2 regularization will encourage many of the non-informative weights to be nearly (but not exactly) 0.0."
        },
        'answers': ['A', 'C'],
        'explanation':
        
        "* (correct) L2 regularization may cause the model to learn a moderate weight for some non-informative features.\n"
        "Surprisingly, this can happen when a non-informative feature happens to be correlated with the label. In this case, the model incorrectly gives such non-informative features some of the \"credit\" that should have gone to informative features.\n\n"
        
        "* (correct) L2 regularization will encourage many of the non-informative weights to be nearly (but not exactly) 0.0.\n"
        "Yes, L2 regularization encourages weights to be near 0.0, but not exactly 0.0.\n\n",
        
        "* (wrong) L2 regularization will encourage most of the non-informative weights to be exactly 0.0.\n"
        "L2 regularization does not tend to force weights to exactly 0.0. L2 regularization penalizes larger weights more than smaller weights. As a weight gets close to 0.0, L2 \"pushes\" less forcefully toward 0.0."
        
        'references': [
            'https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/check-your-understanding'
        ]
    },
    {
        'question':
        "Imagine a linear model with two strongly correlated features; that is, these two features are nearly identical copies of one another but one feature contains a small amount of random noise. If we train this model with L2 regularization, what will happen to the weights for these two features?",
        'tags': [4, 'crash-course'],
        'options': {
            'A': "One feature will have a large weight; the other will have a weight of almost 0.0.",
            'B': "One feature will have a large weight; the other will have a weight of exactly 0.0.",
            'C': "Both features will have roughly equal, moderate weights."
        },
        'answers': ['C'],
        'explanation':
        
        "* (correct) Both features will have roughly equal, moderate weights.\n"
        "L2 regularization will force the features towards roughly equivalent weights that are approximately half of what they would have been had only one of the two features been in the model.\n\n"
        
        "* (wrong) One feature will have a large weight; the other will have a weight of almost 0.0.\n"
        "L2 regularization penalizes large weights more than small weights. So, even if one weight started to drop faster than the other, L2 regularization would tend to force the bigger weight to drop more quickly than the smaller weight.\n\n"
        
        "* (wrong) One feature will have a large weight; the other will have a weight of exactly 0.0.\n"
        "L2 regularization rarely forces weights to exactly 0.0. By contrast, L1 regularization (discussed later) does force weights to exactly 0.0.",
        
        'references': [
            'https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/check-your-understanding'
        ]
    },
    {
        'question':
        "In which of the following scenarios would a high accuracy value suggest that the ML model is doing a good job?",
        'tags': [5, 'crash-course'],
        'options': {
            'A': "An expensive robotic chicken crosses a very busy road a thousand times per day. An ML model evaluates traffic patterns and predicts when this chicken can safely cross the street with an accuracy of 99.99%.",
            'B': "A deadly, but curable, medical condition afflicts .01% of the population. An ML model uses symptoms as features and predicts this affliction with an accuracy of 99.99%.",
            'C': "In the game of roulette, a ball is dropped on a spinning wheel and eventually lands in one of 38 slots. Using visual features (the spin of the ball, the position of the wheel when the ball was dropped, the height of the ball over the wheel), an ML model can predict the slot that the ball will land in with an accuracy of 4%."
        },
        'answers': ['C'],
        'explanation':
        
        "* (correct) In the game of roulette ...\n"
        "This ML model is making predictions far better than chance; a random guess would be correct 1/38 of the time—yielding an accuracy of 2.6%. Although the model\'s accuracy is \"only\" 4%, the benefits of success far outweigh the disadvantages of failure.\n\n"
        
        "* (wrong) An expensive robotic chicken ...\n"
        "A 99.99% accuracy value on a very busy road strongly suggests that the ML model is far better than chance. In some settings, however, the cost of making even a small number of mistakes is still too high. 99.99% accuracy means that the expensive chicken will need to be replaced, on average, every 10 days. (The chicken might also cause extensive damage to cars that it hits.)\n\n"
        
        "* (wrong) A deadly, but curable, medical ...\n"
        "Accuracy is a poor metric here. After all, even a \"dumb\" model that always predicts \"not sick\" would still be 99.99% accurate. Mistakenly predicting \"not sick\" for a person who actually is sick could be deadly.",
        
        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/check-your-understanding-accuracy-precision-recall'
        ]
    },
    {
        'question':
        "Consider a classification model that separates email into two categories: \"spam\" or \"not spam.\" If you raise the classification threshold, what will happen to precision?",
        'tags': [6, 'crash-course'],
        'options': {
            'A': "Definitely decrease.",
            'B': "Definitely increase.",
            'C': "Probably decrease.",
            'D': "Probably increase."
        },
        'answers': ['D'],
        'explanation':
        
        "* (correct) Probably increase.\n"
        "In general, raising the classification threshold reduces false positives, thus raising precision.\n\n",
        
        "* (wrong) Definitely increase\n"
        "Raising the classification threshold typically increases precision; however, precision is not guaranteed to increase monotonically as we raise the threshold."

        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/check-your-understanding-accuracy-precision-recall'
        ]
    },
    {
        'question':
        "Consider a classification model that separates email into two categories: \"spam\" or \"not spam.\" If you raise the classification threshold, what will happen to recall?",
        'tags': [7, 'crash-course'],
        'options': {
            'A': "Always stay constant",
            'B': "Always increase",
            'C': "Always decrease or stay the same"
        },
        'answers': ['C'],
        'explanation':
        
        "* (correct) Always decrease or stay the same\n"
        "Raising our classification threshold will cause the number of true positives to decrease or stay the same and will cause the number of false negatives to increase or stay the same. Thus, recall will either stay constant or decrease.\n\n"
        
        "* (wrong) Always stay constant\n"
        "Raising our classification threshold will cause the number of true positives to decrease or stay the same and will cause the number of false negatives to increase or stay the same. Thus, recall will either stay constant or decrease.\n\n"
        
        "* (wrong) Always increase\n"
        "Raising the classification threshold will cause both of the following:\n"
        " * The number of true positives will decrease or stay the same.\n"
        " * The number of false negatives will increase or stay the same.\n"
        "Thus, recall will never increase.\n\n",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/check-your-understanding-accuracy-precision-recall'
        ]
    },
    {
        'question':
        "Consider two models—A and B—that each evaluate the same dataset. Which one of the following statements is true?",
        'tags': [8, 'crash-course'],
        'options': {
            'A': "If Model A has better precision than model B, then model A is better.",
            'B': "If model A has better precision and better recall than model B, then model A is probably better.",
            'C': "If model A has better recall than model B, then model A is better."
        },
        'answers': ['B'],
        'explanation':
        
        "* (correct) If model A has better precision and better recall than model B, then model A is probably better."
        "In general, a model that outperforms another model on both precision and recall is likely the better model. Obviously, we'll need to make sure that comparison is being done at a precision / recall point that is useful in practice for this to be meaningful. For example, suppose our spam detection model needs to have at least 90% precision to be useful and avoid unnecessary false alarms. In this case, comparing one model at {20% precision, 99% recall} to another at {15% precision, 98% recall} is not particularly instructive, as neither model meets the 90% precision requirement. But with that caveat in mind, this is a good way to think about comparing models when using precision and recall.\n\n"
        
        "* (wrong) If Model A has better precision or recall than model B, then model A is better\n"
        "While better precision is good, it might be coming at the expense of a large reduction in recall. In general, we need to look at both precision and recall together, or summary metrics like AUC which we'll talk about next.",
        
        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/check-your-understanding-accuracy-precision-recall'
        ]
    },
    {
        'question':
        "How would multiplying all of the predictions from a given model by 2.0 (for example, if the model predicts 0.4, we multiply by 2.0 to get a prediction of 0.8) change the model's performance as measured by AUC?",
        'tags': [9, 'crash-course'],
        'options': {
            'A': "It would make AUC terrible, since the prediction values are now way off",
            'B': "No change. AUC only cares about relative prediction scores",
            'C': "It would make AUC better, because the prediction values are all farther apart"
        },
        'answers': ['B'],
        'explanation':
        
        "* (correct) No change. AUC only cares about relative prediction scores\n"
        "Yes, AUC is based on the relative predictions, so any transformation of the predictions that preserves the relative ranking has no effect on AUC. This is clearly not the case for other metrics such as squared error, log loss, or prediction bias.\n\n"
        
        "* (wrong) It would make AUC terrible, since the prediction values are now way off\n"
        "Interestingly enough, even though the prediction values are different (and likely farther from the truth), multiplying them all by 2.0 would keep the relative ordering of prediction values the same. Since AUC only cares about relative rankings, it is not impacted by any simple scaling of the predictions.\n\n"
        
        "* (wrong) It would make AUC better, because the prediction values are all farther apart\n"
        "The amount of spread between predictions does not actually impact AUC. Even a prediction score for a randomly drawn true positive is only a tiny epsilon greater than a randomly drawn negative, that will count that as a success contributing to the overall AUC score.",
        
        'references': [
            'https://developers.google.com/machine-learning/crash-course/classification/check-your-understanding-roc-and-auc'
        ]
    },
    {
        'question':
        "Imagine a linear model with 100 input features:\n"
        "* 10 are highly informative.\n"
        "* 90 are non-informative.\n"
        "Assume that all features have values between -1 and 1.\n" 
        "Which of the following statements are true? (pick 2)",
        'tags': [10, 'crash-course'],
        'options': {
            'A': "L1 regularization will encourage many of the non-informative weights to be nearly (but not exactly) 0.0.",
            'B': "L1 regularization may cause informative features to get a weight of exactly 0.0.",
            'C': "L1 regularization will encourage most of the non-informative weights to be exactly 0.0."
        },
        'answers': ['B', 'C'],
        'explanation':
        
        "* (correct) L1 regularization may cause informative features to get a weight of exactly 0.0.\n"
        "Be careful, L1 regularization may cause the following kinds of features to be given weights of exactly 0:\n"
        " * Weakly informative features.\n"
        " * Strongly informative features on different scales.\n"
        " * Informative features strongly correlated with other similarly informative features.\n\n"
        
        "* (correct) L1 regularization will encourage most of the non-informative weights to be exactly 0.0.\n"
        "L1 regularization of sufficient lambda tends to encourage non-informative weights to become exactly 0.0. By doing so, these non-informative features leave the model.\n\n"
        
        "* (wrong) L1 regularization will encourage many of the non-informative weights to be nearly (but not exactly) 0.0.\n"
        "In general, L1 regularization of sufficient lambda tends to encourage non-informative features to weights of exactly 0.0. Unlike L2 regularization, L1 regularization \"pushes\" just as hard toward 0.0 no matter how far the weight is from 0.0.",
        
        'references': [
            'https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/check-your-understanding'
        ]
    },
    {
        'question':
        "Imagine a linear model with 100 input features, all having values between -1 and 1:\n"
        "* 10 are highly informative.\n"
        "* 90 are non-informative.\n"
        "Which type of regularization will produce the smaller model?",
        'tags': [11, 'crash-course'],
        'options': {
            'A': "L2 regularization.",
            'B': "L1 regularization."
        },
        'answers': ['B'],
        'explanation':
        
        "* (correct) L1 regularization\n"
        "L1 regularization tends to reduce the number of features. In other words, L1 regularization often reduces the model size.\n\n"
        
        "* (wrong) L2 regularization\n"
        "L2 regularization rarely reduces the number of features. In other words, L2 regularization rarely reduces the model size.",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/check-your-understanding'
        ]
    },
    {
        'question':
        "Which one of the following statements is true of dynamic (online) training?",
        'tags': [12, 'crash-course'],
        'options': {
            'A': "The model stays up to date as new data arrives",
            'B': "Very little monitoring of training jobs needs to be done",
            'C': "Very little monitoring of input data needs to be done at inference time"
        },
        'answers': ['A'],
        'explanation':
        
        "* (correct) The model stays up to date as new data arrives.\n"
        "This is the primary benefit of online training—we can avoid many staleness issues by allowing the model to train on new data as it comes in.\n\n"
        
        "* (wrong) Very little monitoring of training jobs needs to be done.\n"
        "Actually, you must continuously monitor training jobs to ensure that they are healthy and working as intended. You'll also need supporting infrastructure like the ability to roll a model back to a previous snapshot in case something goes wrong in training, such as a buggy job or corruption in input data.\n\n"
        
        "* (wrong) Very little monitoring of input data needs to be done at inference time.\n"
        "Just like a static, offline model, it is also important to monitor the inputs to the dynamically updated models. We are likely not at risk for large seasonality effects, but sudden, large changes to inputs (such as an upstream data source going down) can still cause unreliable predictions.",
        
        'references': [
            'https://developers.google.com/machine-learning/crash-course/static-vs-dynamic-training/check-your-understanding'
        ]
    },
    {
        'question':
        "Which of the following statements are true about static (offline) training? (pick 2)",
        'tags': [13, 'crash-course'],
        'options': {
            'A': "Offline training requires less monitoring of training jobs than online training",
            'B': "You can verify the model before applying it in production",
            'C': "Very little monitoring of input data needs to be done at inference time",
            'D': "The model stays up to date as new data arrives"
        },
        'answers': ['A', 'B'],
        'explanation':
        
        "* (correct) Offline training requires less monitoring of training jobs than online training\n"
        "In general, monitoring requirements at training time are more modest for offline training, which insulates us from many production considerations. However, the more frequently you train your model, the higher the investment you'll need to make in monitoring. You'll also want to validate regularly to ensure that changes to your code (and its dependencies) don't adversely affect model quality.\n\n"
        
        "* (correct) You can verify the model before applying it in production\n"
        "Yes, offline training gives ample opportunity to verify model performance before introducing the model in production.\n\n"
        
        "* (wrong) Very little monitoring of input data needs to be done at inference time\n"
        "Counterintuitively, you do need to monitor input data at serving time. If the input distributions change, then our model's predictions may become unreliable. Imagine, for example, a model trained only on summertime clothing data suddenly being used to predict clothing buying behavior in wintertime.\n\n"
        
        "* (wrong) The model stays up to date as new data arrives.\n"
        "Actually, if we train offline, then the model has no way to incorporate new data as it arrives. This can lead to model staleness, if the distribution we are trying to learn from changes over time.",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/static-vs-dynamic-training/check-your-understanding'
        ]
    },
    {
        'question':
        "In offline inference, we make predictions on a big batch of data all at once. We then put those predictions in a look-up table for later use.\n" 
        "Which of the following are true of offline inference? (pick 3)",
        'tags': [14, 'crash-course'],
        'options': {
            'A': "We will be able to react quickly to changes in the world",
            'B': "We will need to carefully monitor our input signals over a long period of time",
            'C': "We must create predictions for all possible inputs",
            'D': "For a given input, we can serve a prediction more quickly than with online inference",
            'E': "After generating the predictions, we can verify them before applying them"
        },
        'answers': ['C', 'D', 'E'],
        'explanation':
        
        "* (correct) We must create predictions for all possible inputs\n"
        "Yes, we will have to make predictions for all possible inputs and store them into a cache or lookup table to use offline inference. This is one of the drawbacks of offline inference. We will only be able to serve a prediction for those examples that we already know about. This is fine if the set of things that we're predicting is limited, like all world cities or all items in a database table. But for freeform inputs like user queries that have a long tail of unusual or rare items, we would not be able to provide full coverage with an offline-inference system.\n\n"
        
        "* (correct) For a given input, we can serve a prediction more quickly than with online inference\n"
        "One of the great things about offline inference is that once the predictions have been written to some look-up table, they can be served with minimal latency. No feature computation or model inference needs to be done at request time\n\n"
        
        "* (correct) After generating the predictions, we can verify them before applying them\n"
        "This is indeed one useful thing about offline inference. We can sanity check and verify all of our predictions before they are used\n\n"
        
        "* (wrong) We will be able to react quickly to changes in the world\n"
        "No, this is a drawback of offline inference. We'll need to wait until a new set of predictions have been written to the look-up table before we can respond differently based on any changes in the world.\n\n"
        
        "* (wrong) We will need to carefully monitor our input signals over a long period of time\n"
        "This is the one case where we don't actually need to monitor input signals over a long period of time. This is because once the predictions have been written to a look-up table, we're no longer dependent on the input features. Note that any subsequent update of the model will require a new round of input verification.",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/static-vs-dynamic-inference/video-lecture'
        ]
    },
    {
        'question':
        "Dynamic (online) inference means making predictions on demand. That is, in online inference, we put the trained model on a server and issue inference requests as needed.\n" 
        "Which of the following are true of dynamic inference? (pick 2)",
        'tags': [15, 'crash-course'],
        'options': {
            'A': "When performing online inference, you do not need to worry about prediction latency (the lag time for returning predictions) as much as when performing offline inference.",
            'B': "You can do post-verification of predictions before they are used",
            'C': "You can provide predictions for all possible items",
            'D': "You must carefully monitor input signals."
        },
        'answers': ['C', 'D'],
        'explanation':
        
        "* (correct) You can provide predictions for all possible items\n"
        "Yes, this is a strength of online inference. Any request that comes in will be given a score. Online inference handles long-tail distributions (those with many rare items), like the space of all possible sentences written in movie reviews\n\n"
        
        "* (correct) You must carefully monitor input signals\n"
        "Yes. Signals could change suddenly due to upstream issues, harming our predictions\n"

        "* (wrong) When performing online inference ...\n"
        "Prediction latency is often a real concern in online inference. Unfortunately, you can't necessarily fix prediction latency issues by adding more inference servers\n\n"
        
        "* (wrong) You can do post-verification of predictions before they are used\n"
        "In general, it's not possible to do a post-verification of all predictions before they get used because predictions are being made on demand. You can, however, potentially monitor aggregate prediction qualities to provide some level of sanity checking, but these will signal fire alarms only after the fire has already spread.",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/static-vs-dynamic-inference/video-lecture'
        ]
    },
    {
        'question':
        "Which of the following models are susceptible to a feedback loop? (pick 3)",
        'tags': [16, 'crash-course'],
        'options': {
            'A': "A university-ranking model that rates schools in part by their selectivity—the percentage of students who applied that were admitted",
            'B': "A traffic-forecasting model that predicts congestion at highway exits near the beach, using beach crowd size as one of its features",
            'C': "An election-results model that forecasts the winner of a mayoral race by surveying 2% of voters after the polls have closed",
            'D': "A housing-value model that predicts house prices, using size (area in square meters), number of bedrooms, and geographic location as features",
            'E': "A face-attributes model that detects whether a person is smiling in a photo, which is regularly trained on a database of stock photography that is automatically updated monthly",
            'F': "A book-recommendation model that suggests novels its users may like based on their popularity (i.e., the number of times the books have been purchased)"
        },
        'answers': ['A', 'B', 'F'],
        'explanation':
        
        "* (correct) A university-ranking model ...\n"
        "The model's rankings may drive additional interest to top-rated schools, increasing the number of applications they receive. If these schools continue to admit the same number of students, selectivity will increase (the percentage of students admitted will go down). This will boost these schools' rankings, which will further increase prospective student interest, and so on…\n\n"
        
        "* (correct) A traffic-forecasting model ...\n"
        "Some beachgoers are likely to base their plans on the traffic forecast. If there is a large beach crowd and traffic is forecast to be heavy, many people may make alternative plans. This may depress beach turnout, resulting in a lighter traffic forecast, which then may increase attendance, and the cycle repeats.\n\n"
        
        "* (correct) A book-recommendation model ...\n"
        "Book recommendations are likely to drive purchases, and these additional sales will be fed back into the model as input, making it more likely to recommend these same books in the future.\n\n"
        
        "* (wrong) An election-results model ...\n"
        "If the model does not publish its forecast until after the polls have closed, it is not possible for its predictions to affect voter behavior.\n\n"
        
        "* (wrong) A housing-value model ...\n"
        "It is not possible to quickly change a house's location, size, or number of bedrooms in response to price forecasts, making a feedback loop unlikely. However, there is potentially a correlation between size and number of bedrooms (larger homes are likely to have more rooms) that may need to be teased apart\n\n"
        
        "* (wrong) A face-attributes model ...\n"
        "There is no feedback loop here, as model predictions don't have any impact on our photo database. However, versioning of our input data is a concern here, as these monthly updates could potentially have unforeseen effects on the model",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/data-dependencies/check-your-understanding'
        ]
    },
    {
        'question':
        "Which of the following model's predictions have been affected by selection bias? (pick 2)",
        'tags': [17, 'crash-course'],
        'options': {
            'A': "Engineers built a model to predict the likelihood of a person developing diabetes based on their daily food intake. The model was trained on 10,000 \"food diaries\" collected from a randomly chosen group of people worldwide representing a variety of different age groups, ethnic backgrounds, and genders. However, when the model was deployed, it had very poor accuracy. Engineers subsequently discovered that food diary participants were reluctant to admit the true volume of unhealthy foods they ate, and were more likely to document consumption of nutritious food than less healthy snacks.",
            'B': "A German handwriting recognition smartphone app uses a model that frequently incorrectly classifies ß (Eszett) characters as B characters, because it was trained on a corpus of American handwriting samples, mostly written in English.",
            'C': "Engineers at a company developed a model to predict staff turnover rates (the percentage of employees quitting their jobs each year) based on data collected from a survey sent to all employees. After several years of use, engineers determined that the model underestimated turnover by more than 20%. When conducting exit interviews with employees leaving the company, they learned that more than 80% of people who were dissatisfied with their jobs chose not to complete the survey, compared to a company-wide opt-out rate of 15%.",
            'D': "Engineers developing a movie-recommendation system hypothesized that people who like horror movies will also like science-fiction movies. When they trained a model on 50,000 users' watchlists, however, it showed no such correlation between preferences for horror and for sci-fi; instead it showed a strong correlation between preferences for horror and for documentaries. This seemed odd to them, so they retrained the model five more times using different hyperparameters. Their final trained model showed a 70% correlation between preferences for horror and for sci-fi, so they confidently released it into production."
        },
        'answers': ['B', 'C'],
        'explanation':
        
        "* (correct) A German handwriting recognition smartphone app ...\n"
        "This model was affected by a type of selection bias called coverage bias: the training data (American English handwriting) was not representative of the type of data provided by the model's target audience (German handwriting).\n\n"
        
        "* (correct) Engineers at a company developed a model ...\n"
        "This model was affected by a type of selection bias called non-response bias. People who were dissatisfied with their jobs were underrepresented in the training data set because they opted out of the company-wide survey at much higher rates than the entire employee population.\n\n"
        
        "* (wrong) Engineers developing a movie-recommendation system ...\n"
        "There is no evidence of selection bias, but this model may have instead been affected by experimenter's bias, as the engineers kept iterating on their model until it confirmed their preexisting hypothesis\n\n"
        
        "* (wrong) Engineers built a model to predict ...\n"
        "There is no selection bias in this model; participants who provided training data were a representative sampling of users and were chosen randomly. Instead, this model was affected by reporting bias. Ingestion of unhealthy foods was reported at a much lower frequency than true real-world occurrence.",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/fairness/check-your-understanding'
        ]
    },
    {
        'question':
        "Engineers are working on retraining this model to address inconsistencies in sarcasm-detection accuracy across age demographics, but the model has already been released into production.\n"
        "Which of the following stopgap strategies will help mitigate errors in the model's predictions? (pick 2)",
        'tags': [18, 'crash-course'],
        'options': {
            'A': "Restrict the model's usage to text messages sent by minors.",
            'B': "When the model predicts \"not sarcastic\" for text messages sent by minors, adjust the output so the model returns a value of \"unsure\" instead",
            'C': "Restrict the model's usage to text messages sent by adults",
            'D': "Adjust the model output so that it returns \"sarcastic\" for all text messages sent by minors, regardless of what the model originally predicted."
        },
        'answers': ['B', 'C'],
        'explanation':
        
        "* (correct) When the model predicts \"not sarcastic\" ...\n"
        "The precision rate for text messages sent by minors is high, which means that when the model predicts \"sarcastic\" for this group, it is nearly always correct.\n"
        "The problem is that recall is very low for minors; The model fails to identify sarcasm in approximately 50% of examples. Given that the model's negative predictions for minors are no better than random guesses, we can avoid these errors by not providing a prediction in these cases.\n\n"
        
        "* (correct) Restrict the model's usage to text messages sent by adults.\n"
        "The model performs well on text messages from adults (with precision and recall rates both above 90%), so restricting its use to this group will sidestep the systematic errors in classifying minors' text messages.\n\n"

        "* (wrong) Adjust the model output so that it returns ...\n"
        "Always predicting \"sarcastic\" for minors\' text messages would increase the recall rate from 0.497 to 1.0, as the model would no longer fail to identify any messages as sarcastic. However, this increase in recall would come at the expense of precision. All the true negatives would be changed to false positives, which would decrease the precision rate. So, adding this calibration would change the type of error but would not mitigate the magnitude of the error.\n\n"
        
        "* (wrong) Restrict the model's usage to text messages sent by minors.\n"
        "The systematic errors in this model are specific to text messages sent by minors. Restricting the model's use to the group more susceptible to error would not help.",

        'references': [
            'https://developers.google.com/machine-learning/crash-course/fairness/check-your-understanding'
        ]
    },
    {
        'question':
        "Command to create a bucket",
        'tags': ['gcp'],
        'options': {},
        'answers': [],
        'explanation':
        "gsutil mb -l $REGION gs://{$BUCKET_NAME}",
        'references': []
    },
    {
        'question':
        "Command to get Project ID",
        'tags': ['gcp'],
        'options': {},
        'answers': [],
        'explanation':
        "* gcloud config get-value project\n"
        "* gcloud config get-value core/project\n"
        "* gcloud info --format='value(config.project)'"
        "* gcloud config list --format 'value(core.project)'",
        'references': []
    },
    {
        'question':
        "Hardware accelerator types 'accelerator_type'",
        'tags': ['gcp'],
        'options': {},
        'answers': [],
        'explanation':
        "* ACCELERATOR_TYPE_UNSPECIFIED\n"
        "* NVIDIA_TESLA_K80\n"
        "* NVIDIA_TESLA_P100\n"
        "* NVIDIA_TESLA_V100\n"
        "* NVIDIA_TESLA_P4\n"
        "* NVIDIA_TESLA_T4",
        'references': [
            'https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomTrainingJob#google_cloud_aiplatform_CustomTrainingJob_run'
        ]
    },
    {
        'question':
        "Command to enable Vertex AI API in your project",
        'tags': ['gcp'],
        'options': {},
        'answers': [],
        'explanation':
        "gcloud --project PROJECT_ID services enable aiplatform.googleapis.com",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/managing-models-vertex'
        ]
    },
    {
        'question':
        "Command to grant Vertex AI Model Registry permission to your service account",
        'tags': ['gcp'],
        'options': {},
        'answers': [],
        'explanation':
        "gcloud projects add-iam-policy-binding PROJECT_ID --member=serviceAccount:YOUR_SERVICE_ACCOUNT --role=roles/aiplatform.admin --condition=None",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/managing-models-vertex'
        ]
    },
    {
        'question':
        "Command to grant Vertex AI Model Registry permission to your service account, if you are not owner of your project.",
        'tags': ['gcp'],
        'options': {},
        'answers': [],
        'explanation':
        "gcloud projects add-iam-policy-binding PROJECT_ID --member=user:YOUR_GCLOUD_ACCOUNT --role=roles/aiplatform.admin --condition=None",
        'references': [
            'https://cloud.google.com/bigquery-ml/docs/managing-models-vertex'
        ]
    },
    # handson-ml3
    {
        'question':
        "How would you define machine learning?",
        'tags': ['handson-ml3', 1],
        'options': {},
        'answers': [],
        'explanation':
        "Machine Learning is about building systems that can learn from data. Learning means getting better at some task, given some performance measure.",
        'references': []
    },
    {
        'question':
        "Can you name four types of applications where it shines?",
        'tags': ['handson-ml3', 2],
        'options': {},
        'answers': [],
        'explanation':
        "Machine Learning is great for:\n"
        " * complex problems for which we have no algorithmic solution\n"
        " * to replace long lists of hand-tuned rules\n"
        " * to build systems that adapt to fluctuating environments\n"
        " * to help humans learn (e.g., data mining).",
        'references': []
    },
    {
        'question':
        "What is a labeled training set?",
        'tags': ['handson-ml3', 3],
        'options': {},
        'answers': [],
        'explanation':
        "A labeled training set is a training set that contains the desired solution (a.k.a. a label) for each instance.",
        'references': []
    },
    {
        'question':
        "What are the two most common supervised tasks?",
        'tags': ['handson-ml3', 4],
        'options': {},
        'answers': [],
        'explanation':
        "Regression and classification.",
        'references': []
    },
    {
        'question':
        "Can you name four common unsupervised tasks?",
        'tags': ['handson-ml3', 5],
        'options': {},
        'answers': [],
        'explanation':
        " * clustering\n"
        " * visualization\n"
        " * dimensionality reduction\n"
        " * association rule learning",
        'references': []
    },
    {
        'question':
        "What type of algorithm would you use to allow a robot to walk in various unknown terrains?",
        'tags': ['handson-ml3', 6],
        'options': {},
        'answers': [],
        'explanation':
        "Reinforcement Learning is likely to perform best if we want a robot to learn to walk in various unknown terrains, since this is typically the type of problem that Reinforcement Learning tackles. It might be possible to express the problem as a supervised or semi-supervised learning problem, but it would be less natural.",
        'references': []
    },
    {
        'question':
        "What type of algorithm would you use to segment your customers into multiple groups?",
        'tags': ['handson-ml3', 7],
        'options': {},
        'answers': [],
        'explanation':
        " * If you don't know how to define the groups, then you can use a clustering algorithm (unsupervised learning) to segment your customers into clusters of similar customers.\n"
        " * If you know what groups you would like to have, then you can feed many examples of each group to a classification algorithm (supervised learning), and it will classify all your customers into these groups.",
        'references': []
    },
    {
        'question':
        "Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?",
        'tags': ['handson-ml3', 8],
        'options': {},
        'answers': [],
        'explanation':
        "Spam detection is a typical supervised learning problem: the algorithm is fed many emails along with their labels (spam or not spam).",
        'references': []
    },
    {
        'question':
        "What is an online learning system?",
        'tags': ['handson-ml3', 9],
        'options': {},
        'answers': [],
        'explanation':
        "An online learning system can learn incrementally, as opposed to a batch learning system. This makes it capable of adapting rapidly to both changing data and autonomous systems, and of training on very large quantities of data.",
        'references': []
    },
    {
        'question':
        "What is out-of-core learning?",
        'tags': ['handson-ml3', 10],
        'options': {},
        'answers': [],
        'explanation':
        "Out-of-core algorithms can handle vast quantities of data that cannot fit in a computer's main memory. An out-of-core learning algorithm chops the data into mini-batches and uses online learning techniques to learn from these mini-batches.",
        'references': []
    },
    {
        'question':
        "What type of algorithm relies on a similarity measure to make predictions?",
        'tags': ['handson-ml3', 11],
        'options': {},
        'answers': [],
        'explanation':
        "An instance-based learning system learns the training data by heart; then, when given a new instance, it uses a similarity measure to find the most similar learned instances and uses them to make predictions.",
        'references': []
    },
    {
        'question':
        "What is the difference between a model parameter and a model hyperparameter?",
        'tags': ['handson-ml3', 12],
        'options': {},
        'answers': [],
        'explanation':
        "A model has one or more model parameters that determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances. A hyperparameter is a parameter of the learning algorithm itself, not of the model (e.g., the amount of regularization to apply).",
        'references': []
    },
    {
        'question':
        "What do model-based algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?",
        'tags': ['handson-ml3', 13],
        'options': {},
        'answers': [],
        'explanation':
        "Model-based learning algorithms search for an optimal value for the model parameters such that the model will generalize well to new instances. We usually train such systems by minimizing a cost function that measures how bad the system is at making predictions on the training data, plus a penalty for model complexity if the model is regularized. To make predictions, we feed the new instance's features into the model's prediction function, using the parameter values found by the learning algorithm.",
        'references': []
    },
    {
        'question':
        "Can you name four of the main challenges in machine learning?",
        'tags': ['handson-ml3', 14],
        'options': {},
        'answers': [],
        'explanation':
        "Some of the main challenges in Machine Learning are:\n"
        " * the lack of data\n"
        " * poor data quality\n"
        " * nonrepresentative data\n"
        " * uninformative features\n"
        " * excessively simple models that underfit the training data\n"
        " * excessively complex models that overfit the data",
        'references': []
    },
    {
        'question':
        "If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?",
        'tags': ['handson-ml3', 15],
        'options': {},
        'answers': [],
        'explanation':
        "If a model performs great on the training data but generalizes poorly to new instances, the model is likely overfitting the training data. Possible solutions to overfitting are:\n"
        " * getting more data\n"
        " * simplifying the model (selecting a simpler algorithm, reducing the number of parameters or features used, or regularizing the model)\n"
        " * reducing the noise in the training data.",
        'references': []
    },
    {
        'question':
        "What is a test set, and why would you want to use it?",
        'tags': ['handson-ml3', 16],
        'options': {},
        'answers': [],
        'explanation':
        "A test set is used to estimate the generalization error that a model will make on new instances, before the model is launched in production.",
        'references': []
    },
    {
        'question':
        "What is the purpose of a validation set?",
        'tags': ['handson-ml3', 17],
        'options': {},
        'answers': [],
        'explanation':
        "A validation set is used to compare models. It makes it possible to select the best model and tune the hyperparameters.",
        'references': []
    },
    {
        'question':
        "What is the train-dev set, when do you need it, and how do you use it?",
        'tags': ['handson-ml3', 18],
        'options': {},
        'answers': [],
        'explanation':
        "The train-dev set is used when there is a risk of mismatch between the training data and the data used in the validation and test datasets (which should always be as close as possible to the data used once the model is in production). The train-dev set is a part of the training set that's held out (the model is not trained on it). The model is trained on the rest of the training set, and evaluated on both the train-dev set and the validation set. If the model performs well on the training set but not on the train-dev set, then the model is likely overfitting the training set. If it performs well on both the training set and the train-dev set, but not on the validation set, then there is probably a significant data mismatch between the training data and the validation + test data, and you should try to improve the training data to make it look more like the validation + test data.",
        'references': []
    },
    {
        'question':
        "What can go wrong if you tune hyperparameters using the test set?",
        'tags': ['handson-ml3', 19],
        'options': {},
        'answers': [],
        'explanation':
        "If you tune hyperparameters using the test set, you risk overfitting the test set, and the generalization error you measure will be optimistic (you may launch a model that performs worse than you expect).",
        'references': []
    },
]
