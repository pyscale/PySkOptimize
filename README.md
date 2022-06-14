# PySkOptimize

<div style="width:200px; height:200px">

![GitHub Logo](./logo.png)

</div>

[![codecov](https://codecov.io/gh/pyscale/PySkOptimize/branch/main/graph/badge.svg?token=KR4L8HOCXV)](https://codecov.io/gh/pyscale/PySkOptimize)

## Introduction

Welcome to PySkOptimize!

The goal of this package is to simply enable Data Scientists, Machine Learning Engineers, or
the curious Software Engineer, to develop highly optimized models using low-code.  By using 
Pydantic, we were able to develop a methodology that will allow all to create `JSON` representations
of their Machine Learning Pipeline and the hyperparameters you want to optimize.  

As a note, it is imperative perform all feature engineering tasks, especially tasks that uses 
transformers from `sklearn.preprocessing`, to include in the `JSON` representation.  This enables
the representing pipeline to have all of required steps so that it is easy to have a finalized 
pipeline at the end.

One limitation is the inability to scale the tuning through the usage
of Apache Spark

## Installation

If you want to install a stable version, simply follow the command: `pip install pyskoptimize`.

If you want to install from source, please follow the directions under the **Developer Contribution** section.

## Developer Contribution

To install from local development, run the following commands in order:

    1. make init
    2. make install

To add any new packages, follow the poetry requirements and add
unit tests

## Tutorial

It is quite easy.  The documentation of the classes and their needed arguments
are found on https://www.pyscale.net/PySkOptimize/.  

First you develop a `JSON` file named `data.json` that resembles

```json
{
  "targetTransformer": {
    "name": "sklearn.preprocessing.PowerTransformer"
  },
  "model": {
    "name": "sklearn.linear_model.ElasticNet",
    "params": [
      {
        "name": "alpha",
        "low": 1e-16,
        "high": 1e16,
        "log_scale": true
      },
      {
        "name": "l1_ratio",
        "low": 1e-10,
        "high": 0.9999999999
      }
    ]
  },
  "scoring": "neg_mean_squared_error",
  "preprocess": [
    {
      "name": "featurePod1",
      "features": ["MedInc"],
      "pipeline": [
        {
          "name": "sklearn.preprocessing.KBinsDiscretizer",
          "params": [
            {
              "name": "n_bins",
              "lowInt": 2,
              "highInt": 99
            }
          ]
        }
      ]
    },
    {
      "name": "featurePod2",
      "features": ["HouseAge"],
      "pipeline": [
        {
          "name": "sklearn.preprocessing.KBinsDiscretizer",
          "params": [
            {
              "name": "n_bins",
              "lowInt": 2,
              "highInt": 99
            }
          ]
        }
      ]
    },
    {
      "name": "featurePod3",
      "features": ["AveRooms"],
      "pipeline": [
        {
          "name": "sklearn.preprocessing.KBinsDiscretizer",
          "params": [
            {
              "name": "n_bins",
              "lowInt": 2,
              "highInt": 99
            }
          ]
        }
      ]
    },
    {
      "name": "featurePod4",
      "features": ["Population"],
      "pipeline": [
        {
          "name": "sklearn.preprocessing.KBinsDiscretizer",
          "params": [
            {
              "name": "n_bins",
              "lowInt": 2,
              "highInt": 99
            }
          ]
        }
      ]
    },
    {
      "name": "featurePod5",
      "features": ["AveOccup"],
      "pipeline": [
        {
          "name": "sklearn.preprocessing.KBinsDiscretizer",
          "params": [
            {
              "name": "n_bins",
              "lowInt": 2,
              "highInt": 99
            }
          ]
        }
      ]
    },
    {
      "name": "featurePod6",
      "features": ["Latitude"],
      "pipeline": [
        {
          "name": "sklearn.preprocessing.KBinsDiscretizer",
          "params": [
            {
              "name": "n_bins",
              "lowInt": 2,
              "highInt": 99
            }
          ]
        }
      ]
    },
    {
      "name": "featurePod7",
      "features": ["Longitude"],
      "pipeline": [
        {
          "name": "sklearn.preprocessing.KBinsDiscretizer",
          "params": [
            {
              "name": "n_bins",
              "lowInt": 2,
              "highInt": 99
            }
          ]
        }
      ]
    }
  ]
}
```

Second, you create script that resembles

```python

from pyskoptimize.base import MLOptimizer

config = MLOptimizer.parse_file("data.json")

# lets assume you read in a dataset with the features and that dataset is df

df = ...

bayes_opt = config.to_bayes_opt()

bayes_opt.fit(df.drop("target"), df["target"])

```

That is it!  You are done!  You now have created a Machine Learning model
with highly optimized parameters, tuned to your specific metrics.  

### Vocabulary

One of the fundamental building blocks of this methodology lies within 
what we define:

    1. Preprocessing
    2. Post-processing
    3. Model (or estimator)
    4. Target Transformer

Each of these points are extremely important for an end-to-end model in industry.

The **Preprocessing** includes an array of transformations taken on arrays of features,
or a subset of those features.  For example, we want to standardize the price of an item
while extract features from the description of an item.  These would involve two separate 
transformations on two arrays of subset of features.  Instead of manually performing these
transformations separately, these can be embedded into the one singular object.  This is done
through the `sklearn.compose.ColumnTransformer` object.  However, for more complicated transformations,
you may need to develop a `sklearn.pipeline.Pipeline` of transformations for a singular subset of 
features.  Instead of developing the code specifically for that, just follow the example above in your
configuration file and PySkOptimize will take care of the rest.  

The **Post-processing** includes an array of transformations taken on preprocessed features.  This is 
similar to preprocessing in concept, but the application is different.

The **Model** or the **Estimator** is how you would classically define.  Use anything from `sklearn` or from
`xgboost.sklearn` or `lightgbm.sklearn`.  Just remember to include the full class path.

The **Target Transformer** enables the user to apply a transformation from `sklearn.preprocessing` to be 
applied onto the target variable(s) such that the model, followed by all preprocessing and post-processing, 
learns the transformed target space.  An example is performing the log transformation on the target and training
the model on the log-transformed space.  This is preferred when you know the transformed space helps to preserve
the original space, which is the case of predicting the price of a home since the value of a home is non-negative.

Another requirement is the `scoring` parameter.  This is to evaluate the candidate models throughout the search.  

