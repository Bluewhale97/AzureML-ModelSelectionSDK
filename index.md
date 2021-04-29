## Introduction

Automated Machine Learning enables you to try multiple algorithms and preprocessing transformations with your data. This, combined with scalable cloud-based compute makes it possible to find the best performing model for your data without the huge amount of time-consuming manual trial and error that would otherwise be required.

Azure Machine Learning includes support for automated machine learning through a visual interface in Azure Machine Learning studio for Enterprise edition workspaces only. You can use the Azure Machine Learning SDK to run automated machine learning experiments in either Basic or Enterprise edition workspaces.

In this article, we will discuss about how to use Azure machine learning's automated machine learning capabilities to determine the best performing algorithm for your data, use automated machine learning to preprocess data for training as well as run an automated machine learning experiment.

## 1. Automated machine learning tasks and algorithms

You can use different types of machine leanring algorithms in Azure machine leanring's automated machine learning, including classifcation, regression and time-series forecasting.

There are some commonly used algorithms within different types:

![image](https://user-images.githubusercontent.com/71245576/116622509-cedc9480-a912-11eb-8c37-16d2995e29e2.png)

By default, automated machine learning will randomly select from the full range of algorithms for the specified task. You can choose to block individual algorithms from being selected; which can be useful if you know that your data is not suited to a particular type of algorithm, or you have to comply with a policy that restricts the type of machine learning algorithms you can use in your organization.

## 2. Preprocessing and featurization

Automated machine learning applies scaling and normalization to numeric data automatically, helping prevent any large-scale features from dominating training. During an automated machine learning experiment, multiple scaling or normalization techniques will be applied.

You can choose to have automated machine learning apply preprocessing transformations, such as:

![image](https://user-images.githubusercontent.com/71245576/116622601-f92e5200-a912-11eb-9c05-f1859af2d34d.png)


## 3. Running automated machine learning experiments

To run an automated machine learning experiment, you can either use the user interface in Azure Machine Learning studio, or submit an experiment using the SDK.

To configure an automated machine learning experiment, you can set experiment options using the AutoMLConfig class, as shown in the following example.

```python
from azureml.train.automl import AutoMLConfig

automl_run_config = RunConfiguration(framework='python')
automl_config = AutoMLConfig(name='Automated ML Experiment',
                             task='classification',
                             primary_metric = 'AUC_weighted',
                             compute_target=aml_compute,
                             training_data = train_dataset,
                             validation_data = test_dataset,
                             label_column_name='Label',
                             featurization='auto',
                             iterations=12,
                             max_concurrent_iterations=4)
```
Automated machine learning is designed to enable you to simply bring your data, and have Azure Machine Learning figure out how best to train a model from it.

When using the Automated Machine Learning user interface in Azure Machine Learning studio, you can create or select an Azure Machine Learning dataset to be used as the input for your automated machine learning experiment.

When using the SDK to run an automated machine learning experiment, you can submit the data in the following ways: 

1. Specify a dataset or dataframe of training data that includes features and the label to be predicted. Optionally, specify a second validation data dataset or dataframe that will be used to validate the trained model. if this is not provided, Azure Machine Learning will apply cross-validation using the training data.
Alternatively:

2. Specify a dataset, dataframe, or numpy array of X values containing the training features, with a corresponding y array of label values. Optionally, specify X_valid and y_valid datasets, dataframes, or numpy arrays of X_valid values to be used for validation.

One of the most important settings you must specify is the primary_metric. This is the target performance metric for which the optimal model will be determined. Azure Machine Learning supports a set of named metrics for each type of task. To retrieve the list of metrics available for a particular task type, you can use the get_primary_metrics function as shown here:

```python
from azureml.train.automl.utilities import get_primary_metrics

get_primary_metrics('classification')
```


You can submit an automated machine learning experiment like any other SDK-based experiment.

```python
from azureml.core.experiment import Experiment

automl_experiment = Experiment(ws, 'automl_experiment')
automl_run = automl_experiment.submit(automl_config)
```
You can monitor automated machine learning experiment runs in Azure Machine Learning studio, or in the Jupyter Notebooks RunDetails widget.

In addition, you can easily identify the best run in Azure Machine Learning studio, and download or deploy the model it generated. To accomplish this programmatically with the SDK, you can use code like the following example:
```python
best_run, fitted_model = automl_run.get_output()
best_run_metrics = best_run.get_metrics()
for metric_name in best_run_metrics:
    metric = best_run_metrics[metric_name]
    print(metric_name, metric)
```

Automated machine learning uses scikit-learn pipelines to encapsulate preprocessing steps with the model. You can view the steps in the fitted model you obtained from the best run using the code above like this:
```python
for step_ in fitted_model.named_steps:
    print(step_)
```

## Reference

Build AI solutions with Azure Machine Learning, retrieved from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/




