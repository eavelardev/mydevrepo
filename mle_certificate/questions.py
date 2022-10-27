# Packt - Journey to Become a Google Cloud Machine Learning Engineer (2022) - Dr. Logan Song (30 questions)
# https://www.packtpub.com/product/journey-to-become-a-google-cloud-machine-learning-engineer/9781803233727

# Cloud OnAir: Machine Learning Certification

# Professional Machine Learning Engineer Sample Questions (20 questions)
# https://docs.google.com/forms/d/e/1FAIpQLSeYmkCANE81qSBqLW0g2X7RoskBX9yGYQu-m1TtsjMvHabGqg/viewform

# 25 Free Questions – Google Cloud Certified Professional Machine Learning Engineer (25 questions)
# https://www.whizlabs.com/blog/gcp-professional-machine-learning-engineer-questions/

# Google GCP-PMLE Certification Exam Sample Questions (10 questions) (AI Platform)
# https://www.vmexam.com/google/google-gcp-pmle-certification-exam-sample-questions

# Google Professional Machine Learning Engineer Practice Exam (30 questions) (AI Platform)
# https://gcp-examquestions.com/course/google-professional-machine-learning-engineer-practice-exam/

# Google Professional Machine Learning Engineer Exam Actual Questions (AI Platform)
# https://www.examtopics.com/exams/google/professional-machine-learning-engineer/view/

question_format = [
    {
        'question': '',
        'number': 0,
        'options': {
            'A': '',
            'B': '',
            'C': '',
            'D': ''
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
]

questions = [
    # Packt - Journey to Become a Google Cloud Machine Learning Engineer (2022) - Dr. Logan Song
    # https://www.packtpub.com/product/journey-to-become-a-google-cloud-machine-learning-engineer/9781803233727
    {
        'question': 'Space Y is launching its hundredth satellite to build its StarSphere network. They have designed an accurate orbit (launching speed/time/and so on) for it based on the existing 99 satellite orbits to cover the Earth’s scope. What’s the best solution to forecast the position of the 100 satellites after the hundredth launch?\n',
        'number': 1,
        'options': {
            'A': 'Use ML algorithms and train ML models to forecast',
            'B': 'Use neural networks to train the model to forecast',
            'C': 'Use physical laws and actual environmental data to model and forecast',
            'D': 'Use a linear regression model to forecast'
        },
        'answers': ['C'],
        'explanation': 'When we start, science modeling will be our first choice since it builds the most accurate model based on science and natural laws.',
        'references': ['Section Is ML the best solution? in Chapter 3, Preparing for ML Development']
    },
    {
        'question': 'A financial company is building an ML model to detect credit card fraud based on their historical dataset, which contains 20 positives and 4,990 negatives.\n\n'
        
        'Due to the imbalanced classes, the model training is not working as desired. What’s the best way to resolve this issue?\n',
        'number': 2,
        'options': {
            'A': 'Data augmentation',
            'B': 'Early stopping',
            'C': 'Downsampling and upweighting',
            'D': 'Regularization'
        },
        'answers': ['C'],
        'explanation': 'When the data is imbalanced, it will be very difficult to train the ML model and get good forecasts',
        'references': ['Section Data sampling and balancing in Chapter 3, Preparing for ML Development']
    },
    {
        'question': 'A chemical manufacturer is using a GCP ML pipeline to detect real-time sensor anomalies by queuing the inputs and analyzing and visualizing the data. Which one will you choose for the pipeline?\n',
        'number': 3,
        'options': {
            'A': 'Dataproc | Vertex AI | BQ',
            'B': 'Dataflow | AutoML | Cloud SQL',
            'C': 'Dataflow | Vertex AI | BQ',
            'D': 'Dataproc | AutoML | Bigtable'
        },
        'answers': ['C'],
        'explanation': 'Dataflow is based on parallel data processing and works better if your data has no implementation with Spark or Hadoop. BQ is great for analyzing and visualizing data',
        'references': []
    },
    {
        'question': 'A real estate company, Zeellow, does great business buying and selling properties in the United States. Over the past few years, they have accumulated a big amount of historical data for US houses. \n\n' 
        
        'Zeellow is using ML training to predict housing prices, and they retrain the models every month by integrating new data. The company does not want to write any code in the ML process. What method best suits their needs?\n',
        'number': 4,
        'options': {
            'A': 'AutoML Tabular',
            'B': 'BigQuery ML',
            'C': 'Vertex AI',
            'D': 'AutoML classification'
        },
        'answers': ['A'],
        'explanation': 'AutoML serves the purpose of no coding during the ML process, and this is a structured data ML problem',
        'references': []
    },
    {
        'question': 'The data scientist team is building a deep learning model for a customer support center of a big Enterprise Resource Planning (ERP) company, which has many ERP products and modules. The DL model will input customers’ chat texts and categorize them into products before routing them to the corresponding team. The company wants to minimize the model development time and data preprocessing time. What strategy/platform should they choose?\n',
        'number': 5,
        'options': {
            'A': 'Vertex AI',
            'B': 'AutoML',
            'C': 'NLP API',
            'D': 'Vertex AI Custom notebooks'
        },
        'answers': ['B'],
        'explanation': 'AutoML is the best choice to minimize the model development time and data preprocessing time',
        'references': []
    },
    {
        'question': 'A real estate company, Zeellow, does great business buying and selling properties in the United States. Over the past few years, they have accumulated a big amount of historical data for US houses. \n\n'
        
        'Zeellow wants to use ML to forecast future sales by leveraging their historical sales data. The historical data is stored in cloud storage. You want to rapidly experiment with all the available data. How should you build and train your model?\n',
        'number': 6,
        'options': {
            'A': 'Load data into BigQuery and use BigQuery ML',
            'B': 'Convert the data into CSV and use AutoML Tables',
            'C': 'Convert the data into TFRecords and use TensorFlow',
            'D': 'Convert and refactor the data into CSV format and use the built-in XGBoost library'
        },
        'answers': ['A'],
        'explanation': 'BQ and BQML are the best options to experiment quickly with all the structured datasets stored in cloud storage.',
        'references': []
    },
    {
        'question': 'A real estate company, Zeellow, uses ML to forecast future sales by leveraging their historical data. New data is coming in every week, and Zeellow needs to make sure the model is continually retrained to reflect the marketing trend. What should they do with the historical data and new data?\n',
        'number': 7,
        'options': {
            'A': 'Only use the new data for retraining',
            'B': 'Update the datasets weekly with new data',
            'C': 'Update the datasets with new data when model evaluation metrics do not meet the required criteria',
            'D': 'Update the datasets monthly with new data'
        },
        'answers': ['C'],
        'explanation': 'We need to retrain the model when the performance metrics do not meet the requirements.',
        'references': []
    },
    {
        'question': 'A real estate company, Zeellow, uses ML to forecast future sales by leveraging their historical data. Their data science team trained and deployed a DL model in production half a year ago. Recently, the model is suffering from performance issues due to data distribution changes.\n\n'
        
        'The team is working on a strategy for model retraining. What is your suggestion?\n',
        'number': 8,
        'options': {
            'A': 'Monitor data skew and retrain the model',
            'B': 'Retrain the model with fewer model features',
            'C': 'Retrain the model to fix overfitting',
            'D': 'Retrain the model with new data coming in every month'
        },
        'answers': ['A'],
        'explanation': 'Model retraining is based on data value skews, which are significant changes in the statistical properties of data. When data skew is detected, this means that data patterns are changing, and we need to retrain the model to capture these changes.',
        'references': ['https://developers.google.com/machine-learning/guides/rules-of-ml/#rule_37_measure_trainingserving_skew']
    },
    {
        'question': 'Recent research has indicated that when a certain kind of cancer, X, is developed in a human liver, there are usually other symptoms that can be identified as objects Y and Z from CT scan images. A hospital is using this research to train ML models with a label map of (X, Y, Z) on CT images. What cost functions should be used in this case?\n',
        'number': 9,
        'options': {
            'A': 'Binary cross-entropy',
            'B': 'Categorical cross-entropy',
            'C': 'Sparse categorical cross-entropy',
            'D': 'Dense categorical cross-entropy'
        },
        'answers': ['B'],
        'explanation': 'Categorical entropy is better to use when you want to prevent the model from giving more importance to a certain class – the same as the one-hot encoding idea. Sparse categorical entropy is more optimal when your classes are mutually exclusive (for example, when each sample belongs exactly to one class)',
        'references': []
    },
    {
        'question': 'The data science team in your company has built a DNN model to forecast the sales value for an automobile company, based on historical data. As a Google ML Engineer, you need to verify that the features selected are good enough for the ML model\n',
        'number': 10,
        'options': {
            'A': 'Train the model with L1 regularization and verify that the loss is constant',
            'B': 'Train the model with no regularization and verify that the loss is constant',
            'C': 'Train the model with L2 regularization and verify that the loss is decreasing',
            'D': 'Train the model with no regularization and verify that the loss is close to zero'
        },
        'answers': ['D'],
        'explanation': '',
        'references': ['Section Regularization in Chapter 4, Developing and Deploying ML Models']
    },
    {
        'question': 'The data science team in your company has built a DNN model to forecast the sales value for a real estate company, based on historical data. As a Google ML Engineer, you find that the model has over 300 features and that you wish to remove some features that are not contributing to the target. What will you do?\n',
        'number': 11,
        'options': {
            'A': 'Use Explainable AI to understand the feature contributions and reduce the non-contributing ones.',
            'B': 'Use L1 regularization to reduce features.',
            'C': 'Use L2 regularization to reduce features.',
            'D': 'Drop a feature at a time, train the model, and verify that it does not degrade the model. Remove these features.'
        },
        'answers': ['A'],
        'explanation': 'Explainable AI is one of the ways to understand which features are contributing and which ones are not',
        'references': []
    },
    {
        'question': 'The data science team in your company has built a DNN model to forecast the sales value for a real estate company, based on historical data. They found that the model fits the training dataset well, but not the validation dataset. What would you do to improve the model?\n',
        'number': 12,
        'options': {
            'A': 'Apply a dropout parameter of 0.3 and decrease the learning rate by a factor of 10',
            'B': 'Apply an L2 regularization parameter of 0.3 and decrease the learning rate by a factor of 10',
            'C': 'Apply an L1 regularization parameter of 0.3 and increase the learning rate by a factor of 10',
            'D': 'Tune the hyperparameters to optimize the L2 regularization and dropout parameters'
        },
        'answers': ['D'],
        'explanation': 'The correct answer would be fitting to the general case',
        'references': []
    },
    {
        'question': 'You are building a DL model for a customer service center. The model will input customers’ chat text and analyze their sentiments. What algorithm should be used for the model?\n',
        'number': 13,
        'options': {
            'A': 'MLP',
            'B': 'Regression',
            'C': 'CNN',
            'D': 'RNN'
        },
        'answers': ['D'],
        'explanation': 'Since text processing for sentiment analysis needs to process sequential data (time series), the best option is Recurrent Neural Networks (RNNs).',
        'references': []
    },
    {
        'question': "A health insurance company scans customers' hand-filled claim forms and stores them in Google Cloud Storage buckets in real time. They use ML models to recognize the handwritten texts. Since the claims may contain Personally Identifiable Information (PII), company policies require only authorized persons to access the information. What’s the best way to store and process this streaming data?",
        'number': 14,
        'options': {
            'A': 'Create two buckets and label them as sensitive and non-sensitive. Store data in the non-sensitive bucket first. Periodically scan it using the DLP API and move the sensitive data to the sensitive bucket.',
            'B': 'Create one bucket to store the data. Only allow the ML service account access to it.',
            'C': 'Create three buckets – quarantine, sensitive, and non-sensitive. Store all the data in the quarantine bucket first. Then, periodically scan it using the DLP API and move the data to either the sensitive or non-sensitive bucket.',
            'D': 'Create three buckets – quarantine, sensitive, and non-sensitive. Store all the data in the quarantine bucket first. Then, once the file has been uploaded, trigger the DLP API to scan it, and move the data to either the sensitive or non-sensitive bucket.'
        },
        'answers': ['D'],
        'explanation': '',
        'references': []
    },
    {
        'question': 'A real estate company, Zeellow, uses ML to forecast future sales by leveraging their historical data. The recent model training was able to achieve the desired forecast accuracy objective, but it took the data science team a long time. They want to decrease the training time without affecting the achieved model accuracy. What hyperparameter should the team adjust?\n',
        'number': 15,
        'options': {
            'A': 'Learning rate',
            'B': 'Epochs',
            'C': 'Scale tier',
            'D': 'Batch size'
        },
        'answers': ['C'],
        'explanation': 'Changing the other three parameters will change the model’s prediction accuracy.',
        'references': []
    },
    {
        'question': 'The data science team has built a DNN model to monitor and detect defective products using the images from the assembly line of an automobile manufacturing company. As a Google ML Engineer, you need to measure the performance of the ML model for the test dataset/images. Which of the following would you choose?\n',
        'number': 16,
        'options': {
            'A': 'The AUC value',
            'B': 'The recall value',
            'C': 'The precision value',
            'D': 'The TP value'
        },
        'answers': ['A'],
        'explanation': 'The AUC value measures how well the predictions are ranked rather than their absolute values. It is a classification threshold invariant and thus is the best way to measure the model’s performance.',
        'references': []
    },
    {
        'question': 'The data science team has built a DL model to monitor and detect defective products using the images from the assembly line of an automobile manufacturing company. Over time, the team has built multiple model versions in Vertex AI. As a Google ML Engineer, how will you compare the model versions?\n',
        'number': 17,
        'options': {
            'A': 'Compare the mean average precision for the model versions',
            'B': 'Compare the model loss functions on the training dataset',
            'C': 'Compare the model loss functions on the validation dataset',
            'D': 'Compare the model loss functions on the testing dataset'
        },
        'answers': ['A'],
        'explanation': 'It measures how well the different model versions perform over time: deploy your model as a model version and then create an evaluation job for that version. By comparing the mean average precision across the model versions, you can find the best performer.',
        'references': ['https://cloud.google.com/ai-platform/prediction/docs/continuous-evaluation/view-metrics#compare_mean_average_precision_across_models']
    },
    {
        'question': 'The data science team is building a recommendation engine for an e-commerce website using ML models to increase its business revenue, based on users’ similarities. What model would you choose?\n',
        'number': 18,
        'options': {
            'A': 'Collaborative filtering',
            'B': 'Regression',
            'C': 'Classification',
            'D': 'Content-based filtering'
        },
        'answers': ['A'],
        'explanation': 'Collaborative filtering uses similarities between users to provide recommendations',
        'references': ['https://developers.google.com/machine-learning/recommendation/overview/candidate-generation']
    },
    {
        'question': 'The data science team is building a fraud-detection model for a credit card company, whose objective is to detect as much fraud as possible and avoid as many false alarms as possible. What confusion matrix index would you maximize for this model performance evaluation?\n',
        'number': 19,
        'options': {
            'A': 'Precision',
            'B': 'Recall',
            'C': 'The area under the PR curve',
            'D': 'The area under the ROC curve'
        },
        'answers': ['C'],
        'explanation': 'You want to maximize both precision and recall (maximize the area under the PR curve).',
        'references': ['https://machinelearningmastery.com/roc-curves-andprecision-recall-curves-for-imbalanced-classification/']
    },
    {
        'question': 'The data science team is building a data pipeline for an auto manufacturing company, whose objective is to integrate all the data sources that exist in their on-premise facilities, via a codeless data ETL interface. What GCP service will you use?\n',
        'number': 20,
        'options': {
            'A': 'Dataproc',
            'B': 'Dataflow',
            'C': 'Dataprep',
            'D': 'Data Fusion'
        },
        'answers': ['D'],
        'explanation': 'Data Fusion is the best choice for data integration with a codeless interface',
        'references': ['https://cloud.google.com/data-fusion/docs/concepts/overview#using_the_code-free_web_ui']
    },
    {
        'question': 'The data science team has built a TensorFlow model in BigQuery for a real estate company, whose objective is to integrate all their data models into the new Google Vertex. What’s the best strategy?\n',
        'number': 21,
        'options': {
            'A': 'Export the model from BigQuery ML',
            'B': 'Register the BQML model to Vertex AI',
            'C': 'Import the model into Vertex AI',
            'D': 'Use Vertex AI as the middle stage'
        },
        'answers': ['B'],
        'explanation': 'Vertex AI allows you to register a BQML model in it',
        'references': ['https://cloud.google.com/bigquery-ml/docs/managingmodels-vertex']
    },
    {
        'question': 'A real estate company, Zeellow, uses ML to forecast future house sale prices by leveraging their historical data. The data science team needs to build a model to predict US house sale prices based on the house location (US city-specific) and house type. What strategy is the best for feature engineering in this case?\n',
        'number': 22,
        'options': {
            'A': 'One feature cross: [latitude X longitude X housetype]',
            'B': 'Two feature crosses: [binned latitude X binned housetype] and [binned longitude X binned housetype]',
            'C': 'Three separate binned features: [binned latitude], [binned longitude], [binned housetype]',
            'D': 'One feature cross: [binned latitude X binned longitude X binned housetype]'
        },
        'answers': ['D'],
        'explanation': 'Crossing binned latitude with binned longitude enables the model to learn city-specific effects on house types. It prevents a change in latitude from producing the same result as a change in longitude',
        'references': ['https://developers.google.com/machine-learning/crashcourse/feature-crosses/check-your-understanding']
    },
    {
        'question': 'A health insurance company scans customer’s hand-filled claim forms and stores them in Google Cloud Storage buckets in real time. The data scientist team has developed an AI documentation model to digitize the images. By the end of each day, the submitted forms need to be processed automatically. The model is ready for deployment. What strategy should the team use to process the forms?\n',
        'number': 23,
        'options': {
            'A': 'Vertex AI batch prediction',
            'B': 'Vertex AI online prediction',
            'C': 'Vertex AI ML pipeline prediction',
            'D': 'Cloud Run to trigger prediction'
        },
        'answers': ['A'],
        'explanation': 'We need to run the process at the end of each day, which implies batch processing',
        'references': []
    },
    {
        'question': 'A real estate company, Zeellow, uses GCP ML to forecast future house sale prices by leveraging their historical data. Their data science team has about 30 members and each member has developed multiple versions of models using Vertex AI customer notebooks. What’s the best strategy to manage these different models and different versions developed by the team members?\n',
        'number': 24,
        'options': {
            'A': 'Set up IAM permissions to allow each member access to their notebooks, models, and versions',
            'B': 'Create a GCP project for each member for clean management',
            'C': 'Create a map from each member to their GCP resources using BQ',
            'D': 'Apply label/tags to the resources when they’re created for scalable inventory/cost/access management'
        },
        'answers': ['D'],
        'explanation': 'Resource tagging/labeling is the best way to manage ML resources for medium/big data science teams',
        'references': ['https://cloud.google.com/resource-manager/docs/tags/tags-creating-and-managing']
    },
    {
        'question': 'Starbucks is an international coffee shop selling multiple products A, B, C… at different stores (1, 2, 3… using one-hot encoding and location binning). They are building stores and want to leverage ML models to predict product sales based on historical data (A1 is the data for product A sales at store 1). Following the best practices of splitting data into a training subset, validation subset, and testing subset, how should the data be distributed into these subsets?\n',
        'number': 25,
        'options': {
            'A': 'Distribute data randomly across the subsets:\n* Training set: [A1, B2, F1, E2, ...]\n* Testing set: [A2, C3, D2, F4, ...]\n* Validation set: [B1, C1, D9, C2...]',
            'B': 'Distribute products randomly across the subsets:\n* Training set: [A1, A2, A3, E1, E2, ...]\n* Testing set: [B1, B2, C1, C2, C3, ...]\n* Validation set: [D1, D2, F1, F2, F3, ...]',
            'C': 'Distribute stores randomly across subsets:\n* Training set: [A1, B1, C1, ...]\n* Testing set: [A2, C2, F2, ...]\n* Validation set: [D3, A3, C3, ...]',
            'D': 'Aggregate the data groups by the cities where the stores are allocated and distribute cities randomly across subsets'
        },
        'answers': ['B'],
        'explanation': 'If we divided things up at the product level so that the given products were only in the training subset, the validation subset, or the testing subset, the model would find it more difficult to get high accuracy on the validation since it would need to focus on the product characteristics/qualities',
        'references': ['https://developers.google.com/machine-learning/crashcourse/18th-century-literature']
    },
    {
        'question': 'You are building a DL model with Keras that looks as follows:\n' 
        'model = tf.keras.sequential\n'
        "model.add(df.keras.layers.Dense(128,activation='relu',input_shape=(200, )))\n"
        'model.add(df.keras.layers.Dropout(rate=0.25))\n'
        "model.add(df.keras.layers.Dense(4,activation='relu'))\n"
        'model.add(df.keras.layers.Dropout(rate=0.25))\n'
        'model.add(Dense(2))\n\n'

        'How many trainable weights does this model have?\n',
        'number': 26,
        'options': {
            'A': '200x128+128x4+4x2',
            'B': '200x128+128x4+2',
            'C': '200x128+129x4+5x2',
            'D': '200x128x0.25+128x4x0.25+4x2'
        },
        'answers': ['D'],
        'explanation': '',
        'references': []
    },
    {
        'question': 'The data science team is building a DL model for a customer support center of a big ERP company, which has many ERP products and modules. The company receives over a million customer service calls every day and stores them in GCS. The call data must not leave the region in which the call originated and no PII can be stored/analyzed. The model will analyze calls for customer sentiments. How should you design a data pipeline for call processing, analyzing, and visualizing?\n',
        'number': 27,
        'options': {
            'A': 'GCS -> Speech2Text -> DLP -> BigQuery',
            'B': 'GCS -> Pub/Sub -> Speech2Text -> DLP -> Datastore',
            'C': 'GCS -> Speech2Text -> DLP -> BigTable',
            'D': 'GCS -> Speech2Text -> DLP -> Cloud SQL'
        },
        'answers': ['A'],
        'explanation': 'BigQuery is the best tool here to analyze and visualize',
        'references': []
    },
    {
        'question': 'The data science team is building an ML model to monitor and detect defective products using the images from the assembly line of an automobile manufacturing company, which does not have reliable Wi-Fi near the assembly line. As a Google ML Engineer, you need to reduce the amount of time spent by quality control inspectors utilizing the model’s fast defect detection. Your company wants to implement the new ML model as soon as possible. Which model should you use?\n',
        'number': 28,
        'options': {
            'A': 'AutoML Vision',
            'B': 'AutoML Vision Edge mobile-versatile-1',
            'C': 'AutoML Vision Edge mobile-low-latency-1',
            'D': 'AutoML Vision Edge mobile-high-accuracy-1'
        },
        'answers': ['C'],
        'explanation': 'The question asks for a quick inspection time and prioritizes latency reduction',
        'references': ['https://cloud.google.com/vision/automl/docs/train-edge']
    },
    {
        'question': 'A national hospital is leveraging Google Cloud and a cell phone app to build an ML model to forecast heart attacks based on age, gender, exercise, heart rate, blood pressure, and more. Since the health data is highly sensitive personal information and cannot be stored in cloud databases, how should you train and deploy the ML model?\n',
        'number': 29,
        'options': {
            'A': 'IoT with data encryption',
            'B': 'Federated learning',
            'C': 'Encrypted BQML',
            'D': 'DLP API'
        },
        'answers': ['B'],
        'explanation': 'With federated learning, all the data is collected, and the model is trained with algorithms across multiple decentralized edge devices such as cell phones or websites, without exchanging them',
        'references': []
    },
    {
        'question': 'You are an ML engineer at a media company. You need to build an ML model to analyze video content frame by frame, identify objects, and alert users if there is inappropriate content. Which Google Cloud products should you use to build this project?\n',
        'number': 30,
        'options': {
            'A': 'Pub/Sub, Cloud Functions, and Cloud Vision API',
            'B': 'Pub/Sub, Cloud IoT, Dataflow, Cloud Vision API, and Cloud Logging',
            'C': 'Pub/Sub, Cloud Functions, Video Intelligence API, and Cloud Logging',
            'D': 'Pub/Sub, Cloud Functions, AutoML Video Intelligence, and Cloud Logging'
        },
        'answers': ['C'],
        'explanation': '',
        'references': []
    },
    # Cloud OnAir: Machine Learning Certification
    {
        'question': 'You need to write a generic test to verify wheter Deep Neural Network (DNN) models automatically released by your team have a sufficient number of parameters to learn the task for which they were built. What should you do?\n',
        'number': 1,
        'options': {
            'A': 'Train the model for a few iterations, and check for NaN values.',
            'B': 'Train the model for a few iterations, and verify that the loss is constant.',
            'C': 'Train a simple linear model, and determine if the DNN model outperforms it.',
            'D': 'Train the model with no regularization, and verify that the loss function is close to zero.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    # Professional Machine Learning Engineer Sample Questions
    # https://docs.google.com/forms/d/e/1FAIpQLSeYmkCANE81qSBqLW0g2X7RoskBX9yGYQu-m1TtsjMvHabGqg/viewform
    {
        'question': 'You are developing a proof of concept for a real-time fraud detection model. After undersampling the training set to achieve a 50% fraud rate, you train and tune a tree classifier using area under the curve (AUC) as the metric, and then calibrate the model. You need to share metrics that represent your model’s effectiveness with business stakeholders in a way that is easily interpreted. Which approach should you take?\n',
        'number': 1,
        'options': {
            'A': 'Calculate the AUC on the holdout dataset at a classification threshold of 0.5, and report true positive rate, false positive rate, and false negative rate.',
            'B': 'Undersample the minority class to achieve a 50% fraud rate in the holdout set. Plot the confusion matrix at a classification threshold of 0.5, and report precision and recall.',
            'C': 'Select all transactions in the holdout dataset. Plot the area under the receiver operating characteristic curve (AUC ROC), and report the F1 score for all available thresholds.',
            'D': 'Select all transactions in the holdout dataset. Plot the precision-recall curve with associated average precision, and report the true positive rate, false positive rate, and false negative rate for all available thresholds.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'Your organization’s marketing team wants to send biweekly scheduled emails to customers that are expected to spend above a variable threshold. This is the first machine learning (ML) use case for the marketing team, and you have been tasked with the implementation. After setting up a new Google Cloud project, you use Vertex AI Workbench to develop model training and batch inference with an XGBoost model on the transactional data stored in Cloud Storage. You want to automate the end-to-end pipeline that will securely provide the predictions to the marketing team, while minimizing cost and code maintenance. What should you do?\n',
        'number': 2,
        'options': {
            'A': 'Create a scheduled pipeline on Vertex AI Pipelines that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending.',
            'B': ' Create a scheduled pipeline on Cloud Composer that accesses the data from Cloud Storage, copies the data to BigQuery, uses BigQuery ML to perform training and batch prediction, and outputs a table in BigQuery with customer emails and expected spending.',
            'C': 'Create a scheduled notebook on Vertex AI Workbench that accesses the data from Cloud Storage, performs training and batch prediction on the managed notebook instance, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending.',
            'D': 'Create a scheduled pipeline on Cloud Composer that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and sends an email to the marketing team’s Gmail group email with an attachment that contains an encrypted list of all customer emails and expected spending.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You have developed a very large network in TensorFlow Keras that is expected to train for multiple days. The model uses only built-in TensorFlow operations to perform training with high-precision arithmetic. You want to update the code to run distributed training using tf.distribute.Strategy and configure a corresponding machine instance in Compute Engine to minimize training time. What should you do?\n',
        'number': 3,
        'options': {
            'A': 'Select an instance with an attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add MirroredStrategy to the code, and create the model in the strategy’s scope with batch size dependent on the number of replicas.',
            'B': 'Create an instance group with one instance with attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add TF_CONFIG and MultiWorkerMirroredStrategy to the code, create the model in the strategy’s scope, and set up data autosharing.',
            'C': ' Create a TPU virtual machine, and gradually scale up the machine type until the optimal execution time is reached. Add TPU initialization at the start of the program, define a distributed TPUStrategy, and create the model in the strategy’s scope with batch size and training steps dependent on the number of TPUs.',
            'D': 'Create a TPU node, and gradually scale up the machine type until the optimal execution time is reached. Add TPU initialization at the start of the program, define a distributed TPUStrategy, and create the model in the strategy’s scope with batch size and training steps dependent on the number of TPUs.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You developed a tree model based on an extensive feature set of user behavioral data. The model has been in production for 6 months. New regulations were just introduced that require anonymizing personally identifiable information (PII), which you have identified in your feature set using the Cloud Data Loss Prevention API. You want to update your model pipeline to adhere to the new regulations while minimizing a reduction in model performance. What should you do?\n',
        'number': 4,
        'options': {
            'A': 'Redact the features containing PII data, and train the model from scratch.',
            'B': 'Mask the features containing PII data, and tune the model from the last checkpoint.',
            'C': 'Use key-based hashes to tokenize the features containing PII data, and train the model from scratch.',
            'D': 'Use deterministic encryption to tokenize the features containing PII data, and tune the model from the last checkpoint.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You set up a Vertex AI Workbench instance with a TensorFlow Enterprise environment to perform exploratory data analysis for a new use case. Your training and evaluation datasets are stored in multiple partitioned CSV files in Cloud Storage. You want to use TensorFlow Data Validation (TFDV) to explore problems in your data before model tuning. You want to fix these problems as quickly as possible. What should you do?\n',
        'number': 5,
        'options': {
            'A': '1. Use TFDV to generate statistics, and use Pandas to infer the schema for the training dataset that has been loaded from Cloud Storage. 2. Visualize both statistics and schema, and manually fix anomalies in the dataset’s schema and values.',
            'B': '1. Use TFDV to generate statistics and infer the schema for the training and evaluation datasets that have been loaded from Cloud Storage by using URI. 2. Visualize statistics for both datasets simultaneously to fix the datasets’ values, and fix the training dataset’s schema after displaying it together with anomalies in the evaluation dataset.',
            'C': '1. Use TFDV to generate statistics, and use Pandas to infer the schema for the training dataset that has been loaded from Cloud Storage. 2. Use TFRecordWriter to convert the training dataset into a TFRecord. 3. Visualize both statistics and schema, and manually fix anomalies in the dataset’s schema and values.',
            'D': '1. Use TFDV to generate statistics and infer the schema for the training and evaluation datasets that have been loaded with Pandas. 2. Use TFRecordWriter to convert the training and evaluation datasets into TFRecords. 3. Visualize statistics for both datasets simultaneously to fix the datasets’ values, and fix the training dataset’s schema after displaying it together with anomalies in the evaluation dataset.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You have developed a simple feedforward network on a very wide dataset. You trained the model with mini-batch gradient descent and L1 regularization. During training, you noticed the loss steadily decreasing before moving back to the top at a very sharp angle and starting to oscillate. You want to fix this behavior with minimal changes to the model. What should you do?\n',
        'number': 6,
        'options': {
            'A': 'Shuffle the data before training, and iteratively adjust the batch size until the loss improves.',
            'B': 'Explore the feature set to remove NaNs and clip any noisy outliers. Shuffle the data before retraining.',
            'C': 'Switch from L1 to L2 regularization, and iteratively adjust the L2 penalty until the loss improves.',
            'D': 'Adjust the learning rate to exponentially decay with a larger decrease at the step where the loss jumped, and iteratively adjust the initial learning rate until the loss improves.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You trained a neural network on a small normalized wide dataset. The model performs well without overfitting, but you want to improve how the model pipeline processes the features because they are not all expected to be relevant for the prediction. You want to implement changes that minimize model complexity while maintaining or improving the model’s offline performance. What should you do?\n',
        'number': 7,
        'options': {
            'A': 'Keep the original feature set, and add L1 regularization to the loss function.',
            'B': 'Use principal component analysis (PCA), and select the first n components that explain 99% of the variance.',
            'C': 'Perform correlation analysis. Remove features that are highly correlated to one another and features that are not correlated to the target.',
            'D': 'Ensure that categorical features are one-hot encoded and that continuous variables are binned, and create feature crosses for a subset of relevant features.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You trained a model in a Vertex AI Workbench notebook that has good validation RMSE. You defined 20 parameters with the associated search spaces that you plan to use for model tuning. You want to use a tuning approach that maximizes tuning job speed. You also want to optimize cost, reproducibility, model performance, and scalability where possible if they do not affect speed. What should you do?\n',
        'number': 8,
        'options': {
            'A': 'Set up a cell to run a hyperparameter tuning job using Vertex AI Vizier with val_rmse specified as the metric in the study configuration.',
            'B': ' Using a dedicated Python library such as Hyperopt or Optuna, configure a cell to run a local hyperparameter tuning job with Bayesian optimization.',
            'C': 'Refactor the notebook into a parametrized and dockerized Python script, and push it to Container Registry. Use the UI to set up a hyperparameter tuning job in Vertex AI. Use the created image and include Grid Search as an algorithm.',
            'D': 'Refactor the notebook into a parametrized and dockerized Python script, and push it to Container Registry. Use the command line to set up a hyperparameter tuning job in Vertex AI. Use the created image and include Random Search as an algorithm where maximum trial count is equal to parallel trial count.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You trained a deep model for a regression task. The model predicts the expected sale price for a house based on features that are not guaranteed to be independent. You want to evaluate your model by defining a baseline approach and selecting an evaluation metric for comparison that detects high variance in the model. What should you do?\n',
        'number': 9,
        'options': {
            'A': 'Use a heuristic that predicts the mean value as the baseline, and compare the trained model’s mean absolute error against the baseline.',
            'B': 'Use a linear model trained on the most predictive features as the baseline, and compare the trained model’s root mean squared error against the baseline.',
            'C': 'Determine the maximum acceptable mean absolute percentage error (MAPE) as the baseline, and compare the model’s MAPE against the baseline.',
            'D': 'Use a simple neural network with one fully connected hidden layer as the baseline, and compare the trained model’s mean squared error against the baseline.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You designed a 5-billion-parameter language model in TensorFlow Keras that used autotuned tf.data to load the data in memory. You created a distributed training job in Vertex AI with tf.distribute.MirroredStrategy, and set the large_model_v100 machine for the primary instance. The training job fails with the following error:\n\n' 
        
        '“The replica 0 ran out of memory with a non-zero status of 9.”\n\n' 
        
        'You want to fix this error without vertically increasing the memory of the replicas. What should you do?\n',
        'number': 10,
        'options': {
            'A': 'Keep MirroredStrategy. Increase the number of attached V100 accelerators until the memory error is resolved.',
            'B': 'Switch to ParameterServerStrategy, and add a parameter server worker pool with large_model_v100 instance type.',
            'C': 'Switch to tf.distribute.MultiWorkerMirroredStrategy with Reduction Server. Increase the number of workers until the memory error is resolved.',
            'D': 'Switch to a custom distribution strategy that uses TF_CONFIG to equally split model layers between workers. Increase the number of workers until the memory error is resolved.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You need to develop an online model prediction service that accesses pre-computed near-real-time features and returns a customer churn probability value. The features are saved in BigQuery and updated hourly using a scheduled query. You want this service to be low latency and scalable and require minimal maintenance. What should you do?\n',
        'number': 11,
        'options': {
            'A': '1. Configure a Cloud Function that exports features from BigQuery to Memorystore. 2. Use Memorystore to perform feature lookup. Deploy the model as a custom prediction endpoint in Vertex AI, and enable automatic scaling.',
            'B': '1. Configure a Cloud Function that exports features from BigQuery to Memorystore. 2. Use a custom container on Google Kubernetes Engine to deploy a service that performs feature lookup from Memorystore and performs inference with an in-memory model.',
            'C': '1. Configure a Cloud Function that exports features from BigQuery to Vertex AI Feature Store. 2. Use the online service API from Vertex AI Feature Store to perform feature lookup. Deploy the model as a custom prediction endpoint in Vertex AI, and enable automatic scaling.',
            'D': '1. Configure a Cloud Function that exports features from BigQuery to Vertex AI Feature Store. 2. Use a custom container on Google Kubernetes Engine to deploy a service that performs feature lookup from Vertex AI Feature Store’s online serving API and performs inference with an in-memory model.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You are logged into the Vertex AI Pipeline UI and noticed that an automated production TensorFlow training pipeline finished three hours earlier than a typical run. You do not have access to production data for security reasons, but you have verified that no alert was logged in any of the ML system’s monitoring systems and that the pipeline code has not been updated recently. You want to debug the pipeline as quickly as possible so you can determine whether to deploy the trained model. What should you do?\n',
        'number': 12,
        'options': {
            'A': 'Navigate to Vertex AI Pipelines, and open Vertex AI TensorBoard. Check whether the training regime and metrics converge.',
            'B': 'Access the Pipeline run analysis pane from Vertex AI Pipelines, and check whether the input configuration and pipeline steps have the expected values.',
            'C': 'Determine the trained model’s location from the pipeline’s metadata in Vertex ML Metadata, and compare the trained model’s size to the previous model.',
            'D': 'Request access to production systems. Get the training data’s location from the pipeline’s metadata in Vertex ML Metadata, and compare data volumes of the current run to the previous run.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You recently developed a custom ML model that was trained in Vertex AI on a post-processed training dataset stored in BigQuery. You used a Cloud Run container to deploy the prediction service. The service performs feature lookup and pre-processing and sends a prediction request to a model endpoint in Vertex AI. You want to configure a comprehensive monitoring solution for training-serving skew that requires minimal maintenance. What should you do?\n',
        'number': 13,
        'options': {
            'A': 'Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses email to send alerts. When an alert is received, use the console to diagnose the issue.',
            'B': 'Update the model hosted in Vertex AI to enable request-response logging. Create a Data Studio dashboard that compares training data and logged data for potential training-serving skew and uses email to send a daily scheduled report.',
            'C': 'Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses Cloud Logging to send alerts. Set up a Cloud Function to initiate model retraining that is triggered when an alert is logged.',
            'D': 'Update the model hosted in Vertex AI to enable request-response logging. Schedule a daily DataFlow Flex job that uses Tensorflow Data Validation to detect training-serving skew and uses Cloud Logging to send alerts. Set up a Cloud Function to initiate model retraining that is triggered when an alert is logged.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You have a historical data set of the sale price of 10,000 houses and the 10 most important features resulting from principal component analysis (PCA). You need to develop a model that predicts whether a house will sell at one of the following equally distributed price ranges: 200-300k, 300-400k, 400-500k, 500-600k, or 600-700k. You want to use the simplest algorithmic and evaluative approach. What should you do?\n',
        'number': 14,
        'options': {
            'A': 'Define a one-vs-one classification task where each price range is a categorical label. Use F1 score as the metric.',
            'B': 'Define a multi-class classification task where each price range is a categorical label. Use accuracy as the metric.',
            'C': 'Define a regression task where the label is the sale price represented as an integer. Use mean absolute error as the metric.',
            'D': 'Define a regression task where the label is the average of the price range that corresponds to the house sale price represented as an integer. Use root mean squared error as the metric.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You downloaded a TensorFlow language model pre-trained on a proprietary dataset by another company, and you tuned the model with Vertex AI Training by replacing the last layer with a custom dense layer. The model achieves the expected offline accuracy; however, it exceeds the required online prediction latency by 20ms. You want to optimize the model to reduce latency while minimizing the offline performance drop before deploying the model to production. What should you do?\n',
        'number': 15,
        'options': {
            'A': 'Apply post-training quantization on the tuned model, and serve the quantized model.',
            'B': 'Use quantization-aware training to tune the pre-trained model on your dataset, and serve the quantized model.',
            'C': 'Use pruning to tune the pre-trained model on your dataset, and serve the pruned model after stripping it of training variables.',
            'D': 'Use clustering to tune the pre-trained model on your dataset, and serve the clustered model after stripping it of training variables.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You developed a model for a classification task where the minority class appears in 10% of the data set. You ran the training on the original imbalanced data set and have checked the resulting model performance. The confusion matrix indicates that the model did not learn the minority class. You want to improve the model performance while minimizing run time and keeping the predictions calibrated. What should you do?\n',
        'number': 16,
        'options': {
            'A': 'Update the weights of the classification function to penalize misclassifications of the minority class.',
            'B': 'Tune the classification threshold, and calibrate the model with isotonic regression on the validation set.',
            'C': 'Upsample the minority class in the training set, and update the weight of the upsampled class by the same sampling factor.',
            'D': 'Downsample the majority class in the training set, and update the weight of the downsampled class by the same sampling factor.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You have a dataset that is split into training, validation, and test sets. All the sets have similar distributions. You have sub-selected the most relevant features and trained a neural network in TensorFlow. TensorBoard plots show the training loss oscillating around 0.9, with the validation loss higher than the training loss by 0.3. You want to update the training regime to maximize the convergence of both losses and reduce overfitting. What should you do?\n',
        'number': 17,
        'options': {
            'A': 'Decrease the learning rate to fix the validation loss, and increase the number of training epochs to improve the convergence of both losses.',
            'B': 'Decrease the learning rate to fix the validation loss, and increase the number and dimension of the layers in the network to improve the convergence of both losses.',
            'C': 'Introduce L1 regularization to fix the validation loss, and increase the learning rate and the number of training epochs to improve the convergence of both losses.',
            'D': 'Introduce L2 regularization to fix the validation loss, and increase the number and dimension of the layers in the network to improve the convergence of both losses.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You recently used Vertex AI Prediction to deploy a custom-trained model in production. The automated re-training pipeline made available a new model version that passed all unit and infrastructure tests. You want to define a rollout strategy for the new model version that guarantees an optimal user experience with zero downtime. What should you do?\n',
        'number': 18,
        'options': {
            'A': 'Release the new model version in the same Vertex AI endpoint. Use traffic splitting in Vertex AI Prediction to route a small random subset of requests to the new version and, if the new version is successful, gradually route the remaining traffic to it.',
            'B': 'Release the new model version in a new Vertex AI endpoint. Update the application to send all requests to both Vertex AI endpoints, and log the predictions from the new endpoint. If the new version is successful, route all traffic to the new application.',
            'C': 'Deploy the current model version with an Istio resource in Google Kubernetes Engine, and route production traffic to it. Deploy the new model version, and use Istio to route a small random subset of traffic to it. If the new version is successful, gradually route the remaining traffic to it.',
            'D': 'Install Seldon Core and deploy an Istio resource in Google Kubernetes Engine. Deploy the current model version and the new model version using the multi-armed bandit algorithm in Seldon to dynamically route requests between the two versions before eventually routing all traffic over to the best-performing version.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You trained a model for sentiment analysis in TensorFlow Keras, saved it in SavedModel format, and deployed it with Vertex AI Predictions as a custom container. You selected a random sentence from the test set, and used a REST API call to send a prediction request. The service returned the error:\n'
        
        "“Could not find matching concrete function to call loaded from the SavedModel. Got: Tensor('inputs:0', shape=(None,), dtype=string). Expected: TensorSpec(shape=(None, None), dtype=tf.int64, name='inputs')”." 
        
        'You want to update the model’s code and fix the error while following Google-recommended best practices. What should you do?\n',
        'number': 19,
        'options': {
            'A': 'Combine all preprocessing steps in a function, and call the function on the string input before requesting the model’s prediction on the processed input.',
            'B': 'Combine all preprocessing steps in a function, and update the default serving signature to accept a string input wrapped into the preprocessing function call.',
            'C': 'Create a custom layer that performs all preprocessing steps, and update the Keras model to accept a string input followed by the custom preprocessing layer.',
            'D': ' Combine all preprocessing steps in a function, and update the Keras model to accept a string input followed by a Lambda layer wrapping the preprocessing function.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    {
        'question': 'You used Vertex AI Workbench user-managed notebooks to develop a TensorFlow model. The model pipeline accesses data from Cloud Storage, performs feature engineering and training locally, and outputs the trained model in Vertex AI Model Registry. The end-to-end pipeline takes 10 hours on the attached optimized instance type. You want to introduce model and data lineage for automated re-training runs for this pipeline only while minimizing the cost to run the pipeline. What should you do?\n',
        'number': 20,
        'options': {
            'A': 
                '1. Use the Vertex AI SDK to create an experiment for the pipeline runs, and save metadata throughout the pipeline.\n'
                '2. Configure a scheduled recurring execution for the notebook.\n'
                '3. Access data and model metadata in Vertex ML Metadata.',
            'B': 
                '1. Use the Vertex AI SDK to create an experiment, launch a custom training job in Vertex training service with the same instance type configuration as the notebook, and save metadata throughout the pipeline.\n'
                '2. Configure a scheduled recurring execution for the notebook.\n'
                '3. Access data and model metadata in Vertex ML Metadata.',
            'C': 
                '1. Create a Cloud Storage bucket to store metadata.\n'
                '2. Write a function that saves data and model metadata by using TensorFlow ML Metadata in one time-stamped subfolder per pipeline run.\n'
                '3. Configure a scheduled recurring execution for the notebook. 4. Access data and model metadata in Cloud Storage.',
            'D': 
                '1. Refactor the pipeline code into a TensorFlow Extended (TFX) pipeline.\n'
                '2. Load the TFX pipeline in Vertex AI Pipelines, and configure the pipeline to use the same instance type configuration as the notebook.\n'
                '3. Use Cloud Scheduler to configure a recurring execution for the pipeline. 4. Access data and model metadata in Vertex AI Pipelines.'
        },
        'answers': [],
        'explanation': '',
        'references': []
    },
    ## 25 Free Questions – Google Cloud Certified Professional Machine Learning Engineer
    {
        'question': 'Your team works on a smart city project with wireless sensor networks and a set of gateways for transmitting sensor data. You have to cope with many design choices. You want, for each of the problems under study, to find the simplest solution.\n'
        'For example, it is necessary to decide on the placement of nodes so that the result is the most economical and inclusive. An algorithm without data tagging must be used.\n'
        'Which of the following choices do you think is the most suitable?\n',
        'number': 1,
        'options': {
            'A': 'K-means',
            'B': 'Q-learning',
            'C': 'K-Nearest Neighbors',
            'D': 'Support Vector Machine(SVM)'
        },
        'answers': ['B'],
        'explanation': 'Q-learning is an RL Reinforcement Learning algorithm. RL provides a software agent that evaluates possible solutions through a progressive reward in repeated attempts. It does not need to provide labels. But it requires a lot of data and several trials and the possibility to evaluate the validity of each attempt.\n'
        'The main RL algorithms are deep Q-network (DQN) and deep deterministic policy gradient (DDPG).\n'
        '* K-means is an unsupervised learning algorithm used for clustering problems. It is useful when you have to create similar groups of entities. So, even if there is no need to label data, it is not suitable for our scope.\n'
        '* K-NN is a supervised classification algorithm, therefore, labeled. New classifications are made by finding the closest known examples.\n'
        '* SVM is a supervised ML algorithm, too. K-NN distances are computed. These distances are not between data points, but with a hyper-plane, that better divides different classifications.'
        ,
        'references': []
    },
    {
        'question': 'Your client has an e-commerce site for commercial spare parts for cars with competitive prices. It started with the small car sector but is continually adding products. Since 80% of them operate in a B2B market, he wants to ensure that his customers are encouraged to use the new products that he gradually offers on the site quickly and profitably.\n'
        'Which GCP service can be valuable in this regard and in what way?\n',
        'number': 2,
        'options': {
            'A': 'Create a Tensorflow model using Matrix factorization',
            'B': 'Use Recommendations AI',
            'C': 'Import the Product Catalog',
            'D': 'Record / Import User events'
        },
        'answers': ['B'],
        'explanation': 'Recommendations AI is a ready-to-use service for all the requirements shown in the question. You don’t need to create models, tune, train, all that is done by the service with your data. Also, the delivery is automatically done, with high-quality recommendations via web, mobile, email. So, it can be used directly on websites during user sessions.\n'
        '* Create a Tensorflow model using Matrix factorization could be OK, but it needs a lot of work.\n'
        '* Import the Product Catalog and Record / Import User events deal only with data management, not creating recommendations.',
        'references': []
    },
    {
        'question': 'You are working on an NLP model. So, you are dealing with words and sentences, not numbers. Your problem is to categorize these words and make sense of them. Your manager told you that you have to use embeddings.\n'
        'Which of the following techniques are not related to embeddings?\n',
        'number': 3,
        'options': {
            'A': 'Count Vector',
            'B': 'TF-IDF Vector',
            'C': 'Co-Occurrence Matrix',
            'D': 'CoVariance Matrix'
        },
        'answers': ['D'],
        'explanation': 
        'Covariance matrices are square matrices with the covariance between each pair of elements. It measures how much the change of one with respect to another is related.\n'
        'All the others are embeddings:\n'
        '* A Count Vector gives a matrix with the count of every single word in every example. 0 if no occurrence. It is okay for small vocabularies.\n'
        '* TF-IDF vectorization counts words in the entire experiment, not a single example or sentence.\n'
        '* Co-Occurrence Matrix puts together words that occur together. So, it is more useful for text understanding.',
        'references': []
    },
    {
        'question': 'You are a junior Data Scientist and are working on a deep neural network model with Tensorflow to optimize the level of customer satisfaction for after-sales services with the goal of creating greater client loyalty.\n'
        'You are struggling with your model (learning rates, hidden layers and nodes selection) for optimizing processing and to let it converge in the fastest way.\n'
        'Which is your problem, in ML language?\n',
        'number': 4,
        'options': {
            'A': 'Cross-Validation',
            'B': 'Regularization',
            'C': 'Hyperparameter tuning',
            'D': 'drift detection management'
        },
        'answers': ['C'],
        'explanation': 'ML training Manages three main data categories:\n\n'
        '* Training data also called examples or records. It is the main input for model configuration and, in supervised learning, presents labels, that is the correct answers based on past experience. Input data is used to build the model but will not be part of the model.\n'
        '* Parameters are Instead the variables to be found to solve the riddle. They are part of the final model and they make the difference among similar models of the same type.\n' 
        '* Hyperparameters are configuration variables that influence the training process itself: Learning rate, hidden layers number, number of epochs, regularization, batch size are all examples of hyperparameters.\n'
        'Hyperparameters tuning is made during the training job and used to be a manual and tedious process, made by running multiple trials with different values.\n'
        'The time required to train and test a model can depend upon the choice of its hyperparameters.\n'
        'With Vertex AI you just need to prepare a simple YAML configuration without coding.\n'
        '* Cross Validation is related to the input data organization for training, test, and validation\n'
        '* Regularization is related to feature management and overfitting\n'
        '* drift management is when data distribution changes and you have to adjust the model',
        'references': []
    },
    {
        'question': 'You work in a major banking institution. The Management has decided to rapidly launch a bank loan service, as the Government has created a series of “first home” facilities for the younger population.\n'
        'The goal is to carry out the automatic management of the required documents (certificates, origin documents, legal information) so that the practice can be built and verified automatically using the data and documents provided by customers and can be managed in a short time and with the minimum contribution of the scarce specialized personnel.\n'
        'Which of these GCP services can you use?\n',
        'number': 5,
        'options': {
            'A': 'Dialogflow',
            'B': 'Document AI',
            'C': 'Cloud Natural Language API',
            'D': 'AutoML'
        },
        'answers': ['B'],
        'explanation': 'Document AI is the perfect solution because it is a complete service for the automatic understanding of documents and their management.\n'
        'It integrates computer natural language processing, OCR, and vision and can create pre-trained templates aimed at intelligent document administration.\n'
        '* Dialogflow is for speech Dialogs, not written documents.\n'
        '* NLP is integrated into Document AI.\n'
        '* functions like AutoML are integrated into Document AI, too.'
        ,
        'references': []
    },
    {
        'question': 'You work for a large retail company. You are preparing a marketing model. The model will have to make predictions based on the historical and analytical data of the e-commerce site (analytics-360). In particular, customer loyalty and remarketing possibilities should be studied. You work on historical tabular data. You want to quickly create an optimal model, both from the point of view of the algorithm used and the tuning and life cycle of the model.\n' 'What are the two best services you can use?\n',
        'number': 6,
        'options': {
            'A': 'AutoML Tables',
            'B': 'BigQuery ML',
            'C': 'Vertex AI',
            'D': 'GKE'
        },
        'answers': ['A', 'C'],
        'explanation': 'AutoML Tables can select the best model for your needs without having to experiment.\n'
        'The architectures currently used (they are added at the same time) are:\n'
        '* Linear\n'
        '* Feedforward deep neural network\n'
        '* Gradient Boosted Decision Tree\n'
        '* AdaNet\n'
        '* Ensembles of various model architectures\n'
        'In addition, AutoML Tables automatically performs feature engineering tasks, too, such as:\n'
        '* Normalization\n'
        '* Encoding and embeddings for categorical features.\n'
        '* Timestamp columns management (important in our case)\n'
        'So, it has special features for time columns: for example, it can correctly split the input data into training, validation and testing.\n'
        'With Vertex AI you can use both AutoML training and custom training in the same environment.\n'
        '* BigQuery ML is wrong because AutoML Tables has additional automated feature engineering and is integrated into Vertex AI\n'
        '* GKE doesn’t supply all the ML features of Vertex AI. It is an advanced K8s managed environment\n',
        'references': []
    },
    {
        'question': 'Your company operates an innovative auction site for furniture from all times. You have to create a series of ML models that allow you, starting from the photos, to establish the period, style and type of the piece of furniture depicted.\n'
        'Furthermore, the model must be able to determine whether the furniture is interesting and require it to be subject to a more detailed estimate. You want Google Cloud to help you reach this ambitious goal faster.\n'
        'Which of the following services do you think is the most suitable?\n',
        'number': 7,
        'options': {
            'A': 'AutoML Vision Edge',
            'B': 'Vision AI',
            'C': 'Video AI',
            'D': 'AutoML Vision'
        },
        'answers': ['D'],
        'explanation': 'Vision AI uses pre-trained models trained by Google. This is powerful, but not enough.\n'
        'But AutoML Vision lets you train models to classify your images with your own characteristics and labels. So, you can tailor your work as you want.\n'
        '* AutoML Vision Edge is for local devices\n'
        '* Video AI manages videos, not pictures. It can extract metadata from any streaming video, get insights in a far shorter time, and let trigger events.',
        'references': []
    },
    {
        'question': 'You are using an Vertex AI, and you are working with a series of demanding training jobs. So, you want to use TPUs instead of CPUs. You are not using Docker images or custom containers.\n'
        'What is the simplest configuration to indicate if you do not have particular needs to customize in the YAML configuration file?\n',
        'number': 8,
        'options': {
            'A': 'Use scale-tier to BASIC_TPU',
            'B': 'Set Master-machine-type',
            'C': 'Set Worker-machine-type',
            'D': 'Set parameterServerType'
        },
        'answers': ['A'],
        'explanation': 'Vertex AI lets you perform distributed training and serving with accelerators (TPUs and GPUs).\n'
        'You usually must specify the number and types of machines you need for master and worker VMs. But you can also use scale tiers that are predefined cluster specifications.\n'
        'In our case,\n'
        'scale-tier=BASIC_TPU\n'
        'covers all the given requirements.\n'
        'The other options are wrong because it is not the easiest way. Moreover, workerType, parameterServerType, evaluatorType, workerCount, parameterServerCount, and evaluatorCount for jobs use custom containers and for TensorFlow jobs.',
        'references': []
    },
    {
        'question': 'You work for an industrial company that wants to improve its quality system. It has developed its own deep neural network model with Tensorflow to identify the semi-finished products to be discarded with images taken from the production lines in the various production phases.\n'
        'You need to monitor the performance of your models and let them go faster.\n'
        'Which is the best solution that you can adopt?\n',
        'number': 9,
        'options': {
            'A': 'TF Profiler',
            'B': 'TF function',
            'C': 'TF Trace',
            'D': 'TF Debugger',
            'E': 'TF Checkpoint',
        },
        'answers': ['A'],
        'explanation': 'TensorFlow Profiler is a tool for checking the performance of your TensorFlow models and helping you to obtain an optimized version.\n' 
        'In TensorFlow 2, the default is eager execution. So, one-off operations are faster, but recurring ones may be slower. So, you need to optimize the model.\n'
        '* TF function is a transformation tool used to make graphs out of your programs. It helps to create performant and portable models but is not a tool for optimization.\n'
        '* TF tracing lets you record TensorFlow Python operations in a graph.\n'
        '* TF debugging is for Debugger V2 and creates a log of debug information.\n'
        '* Checkpoints catch the value of all parameters in a serialized SavedModel format.',
        'references': []
    },
    {
        'question': 'Your team needs to create a model for managing security in restricted areas of a campus.\n'
        'Everything that happens in these areas is filmed and, instead of having a physical surveillance service, the videos must be managed by a model capable of intercepting unauthorized people and vehicles, especially at particular times.\n'
        'What are the GCP services that allow you to achieve all this with minimal effort?\n',
        'number': 10,
        'options': {
            'A': 'AI Infrastructure',
            'B': 'Cloud Video Intelligence AI',
            'C': 'AutoML Video Intelligence Classification',
            'D': 'Vision AI'
        },
        'answers': ['C'],
        'explanation': 'AutoML Video Intelligence is a service that allows you to customize the pre-trained Video intelligence GCP system according to your specific needs.\n'
        'In particular, AutoML Video Intelligence Object Tracking allows you to identify and locate particular entities of interest to you with your specific tags.\n'
        '* AI Infrastructure allows you to manage hardware configurations for ML systems and in particular the processors used to accelerate machine learning workloads.\n'
        '* Cloud Video Intelligence AI is a pre-configured and ready-to-use service, therefore not configurable for specific needs\n'
        '* Vision AI is for images and not video.',
        'references': []
    },
    {
        'question': 'With your team you have to decide the strategy for implementing an online forecasting model in production.\n'
        'This template needs to work with both a web interface as well as DialogFlow and Google Assistant and a lot of requests are expected.\n'
        'You are concerned that the final system is not efficient and scalable enough, and you are looking for the simplest and most managed GCP solution.\n'
        'Which of these can be the solution?\n',
        'number': 11,
        'options': {
            'A': 'Vertex AI Prediction',
            'B': 'GKE e TensorFlow',
            'C': 'VMs and Autoscaling Groups with Application LB',
            'D': 'Kubeflow'
        },
        'answers': ['A'],
        'explanation': 'The Vertex AI Prediction service is fully managed and automatically scales machine learning models in the cloud\n'
        'The service supports both online prediction and batch prediction.\n'
        '* GKE e TensorFlow and VMs and Autoscaling Groups with Application LB are wrong because they are not managed services\n'
        '* Kubeflow is not a managed service, it is used into Vertex AI and let you to deploy ML systems to various environments',
        'references': []
    },
    {
        'question': 'ou work for a digital publishing website with an excellent technical and cultural level, where you have both famous authors and unknown experts who express ideas and insights.\n'
        'You, therefore, have an extremely demanding audience with strong interests that can be of various types.\n'
        'Users have a small set of articles that they can read for free every month. Then they need to sign up for a paid subscription.\n'
        'You have been asked to prepare an ML training model that processes user readings and article preferences. You need to predict trends and topics that users will prefer.\n'
        'But when you train your DNN with Tensorflow, your input data does not fit into RAM memory.\n'
        'What can you do in the simplest way?\n',
        'number': 12,
        'options': {
            'A': 'Use tf.data.Dataset',
            'B': 'Use a queue with tf.train.shuffle_batch',
            'C': 'Use pandas.DataFrame',
            'D': 'Use a NumPy array'
        },
        'answers': ['A'],
        'explanation': 'The tf.data.Dataset allows you to manage a set of complex elements made up of several inner components.\n'
        'It is designed to create efficient input pipelines and to iterate over the data for their processing.\n'
        'These iterations happen in streaming. So, they work even if the input matrix is very large and doesn’t fit in memory.\n'
        '* A queue with tf.train.shuffle_batch is far more complex, even if it is feasible.\n'
        '* A pandas.DataFrame and a NumPy array work in real memory, so they don’t solve the problem at all.',
        'references': []
    },
    {
        'question': 'You are working on a deep neural network model with Tensorflow. Your model is complex, and you work with very large datasets full of numbers.\n'
        'You want to increase performances. But you cannot use further resources.\n'
        'You are afraid that you are not going to deliver your project in time.\n'
        'Your mentor said to you that normalization could be a solution.\n'
        'Which of the following choices do you think is not for data normalization?\n',
        'number': 13,
        'options': {
            'A': 'Scaling to a range',
            'B': 'Feature Clipping',
            'C': 'Z-test',
            'D': 'log scaling',
            'E': 'Z-score'
        },
        'answers': ['C'],
        'explanation': 'z-test is not correct because it is a statistic that is used to prove if a sample mean belongs to a specific population. For example, it is used in medical trials to prove whether a new drug is effective or not.\n'
        '* Scaling to a range converts numbers into a standard range ( 0 to 1 or -1 to 1).\n'
        '* Feature Clipping caps all numbers outside a certain range.\n'
        '* Log Scaling uses the logarithms instead of your values to change the shape. This is possible because the log function preserves monotonicity.\n'
        '* Z-score is a variation of scaling: the resulting number is divided by the standard deviations. It is aimed at obtaining distributions with mean = 0 and std = 1.',
        'references': []
    },
    {
        'question': 'You need to develop and train a model capable of analyzing snapshots taken from a moving vehicle and detecting if obstacles arise. Your work environment is a Vertex AI.\n'
        'Which technique or algorithm do you think is best to use?\n',
        'number': 14,
        'options': {
            'A': 'TabNet algorithm with TensorFlow',
            'B': 'A linear learner with Tensorflow Estimator API',
            'C': 'XGBoost with BigQueryML',
            'D': 'TensorFlow Object Detection API'
        },
        'answers': ['D'],
        'explanation': 'TensorFlow Object Detection API is designed to identify and localize multiple objects within an image. So it is the best solution.\n'
        '* TabNet is used with tabular data, not images. It is a neural network that chooses the best features at each decision step in such a way that the model is optimized simpler.\n'
        '* linear learner is not suitable for images too. It can be applied to regression and classification predictions.\n'
        '* BigQueryML is designed for structured data, not images.',
        'references': []
    },
    {
        'question': 'You are starting to operate as a Data Scientist and are working on a deep neural network model with Tensorflow to optimize customer satisfaction for after-sales services to create greater client loyalty.\n'
        'You are doing Feature Engineering, and your focus is to minimize bias and increase accuracy. Your coordinator has told you that by doing so you risk having problems. He explained to you that, in addition to the bias, you must consider another factor to be optimized. Which one?\n',
        'number': 15,
        'options': {
            'A': 'Blending',
            'B': 'Learning Rate',
            'C': 'Feature Cross',
            'D': 'Bagging',
            'E': 'Variance'
        },
        'answers': ['E'],
        'explanation': 'The variance indicates how much function f (X) can change with a different training dataset. Obviously, different estimates will correspond to different training datasets, but a good model should reduce this gap to a minimum.\n'
        'The bias-variance dilemma is an attempt to minimize both bias and variance.\n'
        'The bias error is the non-estimable part of the learning algorithm. The higher it is, the more underfitting there is.\n'
        'Variance is the sensitivity to differences in the training set. The higher it is, the more overfitting there is.\n'
        '* Blending indicates an ensemble of ML models.\n'
        '* Learning Rate is a hyperparameter in neural networks.\n'
        '* Feature Cross is the method for obtaining new features by multiplying other ones.\n'
        '* Bagging is an ensemble method like Blending.',
        'references': []
    },
    {
        'question': 'You have a Linear Regression model for the optimal management of supplies to a sales network based on a large number of different driving factors. You want to simplify the model to make it more efficient and faster. Your first goal is to synthesize the features without losing the information content that comes from them.\n'
        'Which of these is the best technique?\n',
        'number': 16,
        'options': {
            'A': 'Feature Crosses',
            'B': 'Principal component analysis (PCA)',
            'C': 'Embeddings',
            'D': 'Functional Data Analysis'
        },
        'answers': ['B'],
        'explanation': 'Principal component analysis is a technique to reduce the number of features by creating new variables obtained from linear combinations or mixes of the original variables, which can then replace them but retain most of the information useful for the model. In addition, the new features are all independent of each other.\n'
        'The new variables are called principal components.\n'
        'A linear model is assumed as a basis. Therefore, the variables are independent of each other.\n'
        '* Feature Crosses are for the same objective, but they add non-linearity.\n'
        '* Embeddings, which transform large sparse vectors into smaller vectors are used for categorical data.\n'
        '* Functional Data Analysis has the goal to cope with complexity, but it is used when it is possible to substitute features with functions- not our case.',
        'references': []
    },
    {
        'question': 'You work for a digital publishing website with an excellent technical and cultural level, where you have both famous authors and unknown experts who express ideas and insights. You, therefore, have an extremely demanding audience with strong interests of various types. Users have a small set of articles that they can read for free every month; they need to sign up for a paid subscription.\n'
        'You aim to provide your audience with pointers to articles that they will indeed find of interest to themselves.\n'
        'Which of these models can be useful to you?\n',
        'number': 17,
        'options': {
            'A': 'Hierarchical Clustering',
            'B': 'Autoencoder and self-encoder',
            'C': 'Convolutional Neural Network',
            'D': 'Collaborative filtering using Matrix Factorization'
        },
        'answers': ['D'],
        'explanation': 'Collaborative filtering works on the idea that a user may like the same things of the people with similar profiles and preferences.\n'
        'So, exploiting the choices of other users, the recommendation system makes a guess and can advise people on things not yet been rated by them.\n'
        '* Hierarchical Clustering creates clusters using a hierarchical tree. It may be effective, but it is heavy with lots of data, like in our example.\n'
        '* Autoencoder and self-encoder are useful when you need to reduce the number of variables under consideration for the model, therefore for dimensionality reduction.\n'
        '* Convolutional Neural Network is used for image classification.',
        'references': []
    },
    {
        'question': 'You work for an important Banking group.\n'
        'The purpose of your current project is the automatic and smart acquisition of data from documents and modules of different types.\n'
        'You work on big datasets with a lot of private information that cannot be distributed and disclosed.\n'
        'You are asked to replace sensitive data with specific surrogate characters.\n'
        'Which of the following techniques do you think is best to use?\n',
        'number': 18,
        'options': {
            'A': 'Format-preserving encryption',
            'B': 'K-anonymity',
            'C': 'Replacement',
            'D': 'Masking'
        },
        'answers': ['D'],
        'explanation': 'Masking replaces sensitive values with a given surrogate character, like hash (#) or asterisk (*).\n'
        '* Format-preserving encryption (FPE) encrypts in the same format as the plaintext data.\n'
        'For example, a 16-digit credit card number becomes another 16-digit number.\n'
        '* k-anonymity is a way to anonymize data in such a way that it is impossible to identify person-specific information. Still, you maintain all the information contained in the record.\n'
        '* Replacement just substitutes a sensitive element with a specified value.',
        'references': []
    },
    {
        'question': 'Your company traditionally deals with statistical analysis on data. The services have been integrated for some years with ML models for forecasting, but analyzes and simulations of all kinds are carried out.\n'
        'So you are using 2 types of tools but you have been told that it is possible to have more levels of integration between traditional statistical methodologies and those more related to AI / ML processes.\n'
        'Which tool is the best one for your needs?\n',
        'number': 19,
        'options': {
            'A': 'TensorFlow Hub',
            'B': 'TensorFlow Probability',
            'C': 'TensorFlow Enterprise',
            'D': 'TensorFlow Statistics'
        },
        'answers': ['B'],
        'explanation': 'TensorFlow Probability is a Python library for statistical analysis and probability, which can be processed on TPU and GPU, too.\n'
        'TensorFlow Probability main features are:\n'
        '* Probability distributions and differentiable and injective (one to one) functions.\n'
        '* Tools for deep probabilistic models building\n'
        '* Inference and Simulation methods support Markov chain, Monte Carlo.\n'
        '* Optimizers such as Nelder-Mead, BFGS, and SGLD.\n'
        '  All the other answers are wrong because they don’t deal with traditional statistical methodologies.',
        'references': []
    },
    {
        'question': 
            'Your customer has an online dating platform that, among other things, analyzes the degree of affinity between the various people. Obviously, it already uses ML models and uses, in particular, XGBoost, the gradient boosting decision tree algorithm, and is obtaining excellent results.\n'
            'All its development processes follow CI / CD specifications and use Docker containers. The requirement is to classify users in various ways and update models frequently, based on new parameters entered into the platform by the users themselves.\n'
            'So, the problem you are called to solve is how to optimize frequently re-trained operations with an optimized workflow system. Which solution among these proposals can best solve your needs?\n',
        'number': 20,
        'options': {
            'A': 'Deploy the model on BigQuery ML and setup a job',
            'B': 'Use Kubeflow Pipelines to design and execute your workflow',
            'C': 'Use Vertex AI',
            'D': 'Orchestrate activities with Google Cloud Workflows',
            'E': 'Develop procedures with Pub/Sub and Cloud Run',
            'F': 'Schedule processes with Cloud Composer'
        },
        'answers': ['B'],
        'explanation': 
            'Kubeflow Pipelines is the ideal solution because it is a platform designed specifically for creating and deploying ML workflows based on Docker containers. So, it is the only answer that meets all requirements.\n\n'
            'The main functions of Kubeflow Pipelines are:\n'
            '* Using packaged templates in Docker images in a K8s environment\n'
            '* Manage your various tests/experiments\n'
            '* Simplifying the orchestration of ML pipelines\n'
            '* Reuse components and pipelines\n'
            'It is within the Kubeflow ecosystem, which is the machine learning toolkit for Kubernetes\n'
            'The other answers may be partially correct but do not resolve all items or need to add more coding.',
        'references': []
    },
    {
        'question': 
            'You are working with Vertex AI, the managed ML Platform in GCP.\n'
            'You are dealing with custom training and you are looking and studying the job progresses during the training service lifecycle.\n'
            'Which of the following states are not correct?\n',
        'number': 21,
        'options': {
            'A': 'JOB_STATE_ACTIVE',
            'B': 'JOB_STATE_RUNNING',
            'C': 'JOB_STATE_QUEUED',
            'D': 'JOB_STATE_ENDED'
        },
        'answers': ['A'],
        'explanation': 
            'This is a brief description of the lifecycle of a custom training service.\n'
            'Queueing a new job'
            'When you create a CustomJob or HyperparameterTuningJob, the job is in the JOB_STATE_QUEUED.'
            'When a training job starts, Vertex AI schedules as many workers according to configuration, in parallel.'
            'So Vertex AI starts running code as soon as a worker becomes available.'
            'When all the workers are available, the job state will be: JOB_STATE_RUNNING.'
            'A training job ends successfully when its primary replica exits with exit code 0.'
            'Therefore all the other workers will be stopped. The state will be: JOB_STATE_ENDED.'
            'So A is wrong simply because this state doesn’t exist. All the other answers are correct.'
            'Each replica in the training cluster is given a single role or task in distributed training. For example:'
            'Primary replica: Only one replica, whose main task is to manage the workers.'
            'Worker(s): Replicas that do part of the work.'
            'Parameter server(s): Replicas that store model parameters (optional).'
            'Evaluator(s): Replicas that evaluate your model (optional).',
        'references': []
    },
    {
        'question': 
            'Your team works for an international company with Google Cloud, and you develop, train and deploy several ML models with Tensorflow. You use many tools and techniques and you want to make your work leaner, faster, and more efficient.\n'
            'You would like engineer-to-engineer assistance from both Google Cloud and Google’s TensorFlow teams.'
            'How is it possible? Which service?\n',
        'number': 22,
        'options': {
            'A': 'Vertex AI',
            'B': 'Kubeflow',
            'C': 'Tensorflow Enterprise',
            'D': 'TFX'
        },
        'answers': ['C'],
        'explanation': 
        'The TensorFlow Enterprise is a distribution of the open-source platform for ML, linked to specific versions of TensorFlow, tailored for enterprise customers.\n'
        'It is free but only for big enterprises with a lot of services in GCP. it is prepackaged and optimized for usage with containers and VMs.\n'
        'It works in Google Cloud, from VM images to managed services like GKE and Vertex AI.\n'
        'The TensorFlow Enterprise library is integrated in the following products:\n\n'
        '* Deep Learning VM Images\n'
        '* Deep Learning Containers\n'
        '* Notebooks\n'
        '* Vertex AI Training\n'
        '  It is ready for automatic provisioning and scaling with any kind of processor.\n'
        '  It has a premium level of support from Google.\n'
        '* Vertex AI is a managed service without the kind of support required'
        '* Kubeflow and TFX are wrong because they are open source libraries with standard support from the community',
        'references': []
    },
    {
        'question': 
            'You work for an important organization and your manager tasked you with a new classification model with lots of data drawn from the company Data Lake.\n'
            'The big problem is that you don’t have the labels for all the data, but for only a subset of it and you have very little time to complete the task.\n'
            'Which of the following services could help you?\n',
        'number': 23,
        'options': {
            'A': 'Vertex Data Labeling',
            'B': 'Mechanical Turk',
            'C': 'GitLab ML',
            'D': 'Tag Manager'
        },
        'answers': ['A'],
        'explanation': 
            'In supervised learning, the correctness of label data, together with the quality of all your training data is utterly important for the resulting model and the quality of the future predictions.\n'
            'If you cannot have your data correctly labeled you may request to have professional people that will complete your training data.\n'
            'GCP has a service for this: Vertex AI data labeling. Human labelers will prepare correct labels following your directions.\n'
            'You have to set up a data labeling job with:\n\n'
            'The dataset\n'
            '* Vertex Data Labeling list, vocabulary of the possible labels\n'
            '* An instructions document for the professional people\n'
            '* Mechanical Turk is an Amazon service\n'
            '* GitLab is a DevOps lifecycle tooln\n'
            '* Tag Manager is in the Google Analytics ecosystem',
        'references': []
    },
    {
        'question': 
            'Your team is working with a great number of ML projects, especially with Tensorflow.\n'
            'You recently prepared a DNN model for image recognition that works well and is about to be rolled out in production.\n'
            'Your manager asked you to demonstrate the inner workings of the model.\n'
            'It is a big problem for you because you know that it is working well but you don’t have the explainability of the model.\n'
            'Which of these techniques could help you?\n',
        'number': 24,
        'options': {
            'A': 'Integrated Gradient',
            'B': 'LIT',
            'C': 'WIT',
            'D': 'PCA'
        },
        'answers': ['A'],
        'explanation':
            'Integrated Gradient is an explainability technique for deep neural networks which gives info about what contributes to the model’s prediction.\n'
            'Integrated Gradient works highlight the feature importance. It computes the gradient of the model’s prediction output regarding its input features without modification to the original model.\n'
            'In the picture, you can see that it tunes the inputs and computes attributions so that it can compute the feature importances for the input image.\n'
            'You can use tf.GradientTape to compute the gradients\n'
            '* LIT is only for NLP models\n'
            '* What-If Tool is only for classification and regression models with structured data.\n'
            '* Principal component analysis (PCA) transforms and reduces the number of features by creating new variables, from linear combinations of the original variables.\n'
            'The new features will be all independent of each other.',
        'references': []
    },
    {
        'question': 
            'You work as a Data Scientist in a Startup and you work with several project with Python and Tensorflow;\n'
            'You need to increase the performance of the training sessions and you already use caching and prefetching.\n'
            'So now you want to use GPUs, but in a single machine, for cost reduction and experimentations.\n'
            'Which of the following is the correct strategy?\n',
        'number': 25,
        'options': {
            'A': 'tf.distribute.MirroredStrategy',
            'B': 'tf.distribute.TPUStrategy',
            'C': 'tf.distribute.MultiWorkerMirroredStrategy',
            'D': 'tf.distribute.OneDeviceStrategy'
        },
        'answers': ['A'],
        'explanation':
            'tf.distribute.Strategy is an API explicitly for training distribution among different processors and machines.\n'
            'tf.distribute.MirroredStrategy lets you use multiple GPUs in a single VM, with a replica for each CPU.\n'
            '* tf.distribute.TPUStrategy let you use TPUs, not GPUs\n'
            '* tf.distribute.MultiWorkerMirroredStrategy is for multiple machines\n'
            '* tf.distribute.OneDeviceStrategy, like the default strategy, is for a single device, so a single virtual CPU.',
        'references': []
    }
]