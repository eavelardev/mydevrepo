# Feature Engineering
https://www.cloudskillsboost.google/course_templates/11

## 0. Introduction
* ([Video](https://www.youtube.com/watch?v=YjZ4wrEs2B4) - Mar 3, 2022) ([Slide]()) Course introduction

## 1. Introduction to Vertex AI Feature Store
* ([Video](https://www.youtube.com/watch?v=05sOUYD1WpI) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1Ugl588Ndle8FZeIkt2htMYspaqyhGYfW)) Introduction
* ([Video](https://www.youtube.com/watch?v=HJNoOt9LAR0) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1UtKA2L-afilUXRLV1ZOqVVzzR2c8R8Gm)) Feature Store benefits
* ([Video](https://www.youtube.com/watch?v=cwTcaxi8tV0) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1V0UcjHrowLAvytwSsjr6D43B6YjaCAAe)) Feature Store terminology and concepts
* ([Video](https://www.youtube.com/watch?v=ViXZcbezSfc) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1V4PXd30LqMXk0tLIXsnHIKj-qDEH38LL)) The Feature Store data model
* ([Video](https://www.youtube.com/watch?v=Y50GQUvc2-Y) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1VAvnVV6cHEgiBY-zTSdyI1pfHLNNDsoJ)) Creating a Feature Store
* ([Video](https://www.youtube.com/watch?v=VrlpE49pARE) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1VO21L3xlBw4LJ4gt9KInA-gkifiVwF6f)) Serving features: Batch and online
* ([Video](https://www.youtube.com/watch?v=_9frJMLTvRQ) - Mar 3, 2022) ([Slide](https://docs.google.com/presentation/d/1VUOJwlmuTQMbl80RzLHBzBggpFzBxrhT)) (Optional) Lab intro: Using Feature Store
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438561/labs/319204)) Using Feature Store
    * training-data-analyst/courses/machine_learning/deepdive2/[feature_engineering](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/feature_engineering/solutions)/
        * `6_gapic_feature_store.ipynb` - [lab](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/6_gapic_feature_store.ipynb), [sol](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/solutions/6_gapic_feature_store.ipynb)
        * `7_get_started_with_feature_store.ipynb` - [lab](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/7_get_started_with_feature_store.ipynb), [sol](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/solutions/7_get_started_with_feature_store.ipynb)
    * vertex-ai-samples/notebooks/official/feature_store/[sdk-feature-store.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/feature_store/sdk-feature-store.ipynb)
    * vertex-ai-samples/notebooks/community/feature_store/[gapic-feature-store.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/feature_store/gapic-feature-store.ipynb)
    * fraudfinder/[03_feature_engineering_streaming.ipynb](https://github.com/GoogleCloudPlatform/fraudfinder/blob/main/03_feature_engineering_streaming.ipynb)

## 2. Raw Data to Features
* ([Video](https://www.youtube.com/watch?v=oOZ6Y6wNdPo) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1VhDI8Ukr2Bx_CdqoEn5HYRuWY9TVMZGi)) Introduction
* ([Video](https://www.youtube.com/watch?v=8r0V05FXaa4) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1Vku0WSswgVdoOl-a68hSEZgco4LuKaGL)) Overview of feature engineering
* ([Video](https://www.youtube.com/watch?v=Gaxi74SzEfg) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1Vp8iEW1NKstg3tXO4gxKSDC8s6XVYMuZ)) Raw data to features
* ([Video](https://www.youtube.com/watch?v=FLpXm3uVs68) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1VtuFvEzn440xpX50pz0KpzUZXrjXK3LB)) Good features versus bad features
* ([Video](https://www.youtube.com/watch?v=tz_cW3o8Hy0) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1W0z7b6a87-ajA7q7lkldYLLRJtRZ-Wp8)) Features should be known at prediction-time
* ([Video](https://www.youtube.com/watch?v=-kdyaTk_ACU) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1WDLNC3FaxdiqrhIYdi54R0NzQRazXTC3)) Features should be numeric
* ([Video](https://www.youtube.com/watch?v=lJCz_TwGIBU) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1WGjiUYW-7MHKfK8inPofBtcadKF8jiHQ)) Features should have enough examples
* ([Video](https://www.youtube.com/watch?v=hu_TisiCwf4) - Mar 16, 2022) ([Slide](https://docs.google.com/presentation/d/1WMe2Iytr1DJGi8WyEgT8yPpJT-SjUETz)) Bringing human insight
* ([Video](https://www.youtube.com/watch?v=gdJT0hkjk2I) - Mar 16, 2022) ([Slide](https://docs.google.com/presentation/d/1WVcGsljKzyrBMldC-efrctmiyRaFquUV)) Representing features

## 3. Feature Engineering
* ([Video](https://www.youtube.com/watch?v=XXbGma3UuRE) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1WXJ-L5S3RdaYOqyJXD9jT87vBcmjJx1i)) Introduction
* ([Video](https://www.youtube.com/watch?v=ZiPJyR8e06M) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1WaqmNopGHRk1Ry2TmxyzJEQob6XwFhsk)) Machine learning versus statistics
* ([Video](https://www.youtube.com/watch?v=7x5S37S5eVw) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1WfXd9znllhmLIyLzNw_2nhh2QJXappk3)) Basic feature engineering
* ([Video](https://www.youtube.com/watch?v=nYWwUQLwDA4) - Mar 16, 2022) ([Slide](https://docs.google.com/presentation/d/1WlsOsepJlOcFkd9A7-A1yNpLcLovb16Z)) Lab intro: Performing Basic Feature Engineering in BigQuery ML
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438561/labs/319222)) Performing Basic Feature Engineering in BQML
    * training-data-analyst/courses/machine_learning/deepdive2/[feature_engineering](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/feature_engineering/solutions)/
        * `1_bqml_basic_feat_eng_bqml.ipynb` - [lab](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/1_bqml_basic_feat_eng_bqml-lab.ipynb), [sol](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/solutions/1_bqml_basic_feat_eng.ipynb)
* ([Video](https://www.youtube.com/watch?v=Zo1d_07Y3lk) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1X1OTyJSmS-W1wSPWCU2JSsxI0wBfLDeo)) Advanced feature engineering: Feature crosses
* ([Video](https://www.youtube.com/watch?v=TG66HwjWQMc) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1X1e9_O7zv7irXeNihmSKrcaU1XiY57Gn)) Bucketize and Transform Functions
* ([Video](https://www.youtube.com/watch?v=2BdGFifwgnw) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1XDDcHPEN73eqLtZDnSFfRm80cxfWDg5p)) (Optional) Lab intro: Advanced Feature Engineering BigQuery ML
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438561/labs/319226)) Performing Advanced Feature Engineering in BQML
    * training-data-analyst/courses/machine_learning/deepdive2/[feature_engineering](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/feature_engineering/solutions)/
        * `2_bqml_adv_feat_eng.ipynb` - [lab](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/2_bqml_adv_feat_eng-lab.ipynb), [sol](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/solutions/2_bqml_adv_feat_eng.ipynb)
* ([Video](https://www.youtube.com/watch?v=o6Ez4deX49g) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1XTNOCYg3U3W9msWjKvpeIRkE0c2fQgQI)) Predict housing prices
* ([Video](https://www.youtube.com/watch?v=0oSnzcusc48) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1XVcwOZ4m5TiFnaoCYHTYp6Xn-f3vdhLp)) Estimate taxi fare
* ([Video](https://www.youtube.com/watch?v=PxmnL-AohaQ) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1Xg3kr63smaJB9nxFD5OcAvYs0rbq99gP)) Temporal and geolocation features
* ([Video](https://www.youtube.com/watch?v=AE18psmcNaw) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1XgxRQSrqgffezn78HbVbCNkWKQgZNgQ1)) Lab intro: Basic Feature Engineering in Keras
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438561/labs/319231)) Performing Basic Feature Engineering in Keras
    * training-data-analyst/courses/machine_learning/deepdive2/[feature_engineering](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/feature_engineering/solutions)/
        * `3_keras_basic_feat_eng.ipynb` - [lab](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/3_keras_basic_feat_eng-lab.ipynb), [sol](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/solutions/3_keras_basic_feat_eng.ipynb) (tf.feature_column)
* ([Video](https://www.youtube.com/watch?v=MfA0H0McTL4) - Mar 16, 2022) ([Slide](https://docs.google.com/presentation/d/1Xk3TEZ77MP6l0tvcE_g8gfJ5ApBMEQf8)) Lab intro: Advanced Feature Engineering in Keras
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438561/labs/319233)) Performing Advanced Feature Engineering in Keras
    * training-data-analyst/courses/machine_learning/deepdive2/[feature_engineering](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/feature_engineering/solutions)/
        * `4_keras_adv_feat_eng.ipynb` - [lab](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/4_keras_adv_feat_eng-lab.ipynb), [sol](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/solutions/4_keras_adv_feat_eng.ipynb) (tf.feature_column)

## 4. Preprocessing and Feature Creation
* ([Video](https://www.youtube.com/watch?v=k0lhL5LMCxo) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1Y0GH9u70MqHu5RYjxw1iTChpmE3bJ2LA)) Introduction
* ([Video](https://www.youtube.com/watch?v=Xr2D-lDz-B0) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1Y3ugeF3Hjq3Rv00DqBMBS6AX43BmLZcl)) Apache Beam and Dataflow
* ([Video](https://www.youtube.com/watch?v=5MAN4fserOQ) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1Y9RIZpxnrhMQwYNbq56ESgbWXrzknDL5)) Dataflow terms and concepts

## 5. Feature Crosses and TensorFlow Playground
* ([Video](https://www.youtube.com/watch?v=I1B_gNy2SIg) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1YHxbXfbmOCtBC9f3mwspEReVEni6ra5q)) Introduction
* ([Video](https://www.youtube.com/watch?v=h0bVHakJN_0) - Mar 4, 2022 - Hindi) ([Slide](https://docs.google.com/presentation/d/1YnqEq6eZE0Ig6XncanLuIZ0TktNbc0Ru)) What is a feature cross
* ([Video](https://www.youtube.com/watch?v=KeB3PzqO_fs) - Mar 4, 2022 - Hindi) ([Slide](https://docs.google.com/presentation/d/1ZHiz5TK3FxaxQXNgTnDGGdtgdUeWTBYj)) Discretization
* ([Video](https://www.youtube.com/watch?v=Fl9-Ma844Hw) - Mar 4, 2022 - Hindi) ([Slide](https://docs.google.com/presentation/d/1ZMhYkcREGCWzTnAREOb6GJFg-Ag1hheI)) Lab intro: TensorFlow Playground: Use feature crosses to create a good classifier
* ([Video](https://www.youtube.com/watch?v=_NxNrCOXnx8) - Mar 16, 2022 - Hindi) ([Slide](https://docs.google.com/presentation/d/1ZYCR-pn3T0SC5FZcJrckATY8d4ctvrU0)) Lab intro: TensorFlow Playground: Too much of a good thing

## 6. TensorFlow Transform
* ([Video](https://www.youtube.com/watch?v=eeY850mZnC0) - Dec 7, 2021 - Hindi) Introducing TensorFlow Transform
* ([Video](https://www.youtube.com/watch?v=2oZvmsMGGjE) - Mar 4, 2022) ([Slide](https://docs.google.com/presentation/d/1ZvBlT96Znh9APaoTRuc1x4hdPeZRTUFd)) TensorFlow Transform
* ([Video](https://www.youtube.com/watch?v=i6IuBfUd2aM) - Mar 4, 2022 - Hindi) ([Slide](https://docs.google.com/presentation/d/1ZwdPtXQR5RNVLT3UdZBvxiN7vCM_yOns)) Analyze phase
* ([Video](https://www.youtube.com/watch?v=AO9uZPOVHk8) - Mar 4, 2022 - Hindi) ([Slide](https://docs.google.com/presentation/d/1_B7bJD6oJuajbtjh8wpcJ0tB4ctM7GF-)) Transform phase
* ([Video](https://www.youtube.com/watch?v=0qq2LQR7aXM) - Mar 4, 2022 - Hindi) ([Slide](https://docs.google.com/presentation/d/1_DVhku3_4mPxQ9aYADNKKMy9YXhzIMLP)) Supporting serving
* ([Video](https://www.youtube.com/watch?v=PZQ7p3B3seQ) - Mar 4, 2022 - Hindi) ([Slide](https://docs.google.com/presentation/d/1_QShsPhR76EsyVPAtlJchTrlyhQgiIqc)) Lab Intro: Exploring tf.transform
* ([Lab](https://www.cloudskillsboost.google/course_sessions/2438561/labs/319254)) Exploring tf.transform
    * training-data-analyst/courses/machine_learning/deepdive2/[feature_engineering](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/feature_engineering/solutions)/
        * `5_tftransform_taxifare.ipynb` - [lab](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/5_tftransform_taxifare.ipynb), [sol](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/solutions/5_tftransform_taxifare.ipynb)

## 7. Summary
* ([Document](https://drive.google.com/open?id=1_ZI0-u__FDiFOE92d8xfHCDqIcnVxWFB)) Summary
* ([Document](https://drive.google.com/open?id=1_g5pbBuUhh_SSjVXvh0xVQrIYAl-N32U)) Resource: All quiz questions
* ([Document](https://drive.google.com/open?id=1VNa7qa8ou-7bnmBf0WcJCIrLYOeT6Hix)) Resource: All readings
* ([Document](https://drive.google.com/open?id=1VL5u6foHgh3BY8tKoi-IT_5QOp_NhH71)) Resource: All slides
