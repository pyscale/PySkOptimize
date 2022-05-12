.. PySkOptimize documentation master file, created by
   sphinx-quickstart on Wed May 11 16:57:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySkOptimize's documentation!
========================================

This software package focuses on enabling the generation of a Bayesian
Optimized model from a configuration file.

This is the first attempt of developing a low-code methodology for
Data Science & AI.

One limitation is the inability to scale the tuning through the usage
of Apache Spark

Example
-------

The following is an example JSON representation of a compatible configuration
to be used.

.. code-block:: json

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
     ],
     "postprocess": {
       "pipeline": [
         {
           "name": "sklearn.preprocessing.PolynomialFeatures",
           "params": [
             {
               "name": "interaction_only",
               "categories": [true]
             },
             {
               "name": "include_bias",
               "categories": [false]
             }
           ]
         },
         {
           "name": "sklearn.feature_selection.VarianceThreshold"
         }
       ]
     }
   }


.. toctree::
   :maxdepth: 2

   base



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
