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

One notable issue is the inability to use integer uniform distributions.

Currently, the only solution is to use the categories structure, with the
value of the categories being the list of all of the integers you want to consider.

More work will be done to resolve this.

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
                 "categories": [2, 3, 4, 8, 9, 16, 27, 32, 64, 81, 99]
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
                 "categories": [2, 3, 4, 8, 9, 16, 27, 32, 64, 81, 99]
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
                 "categories": [2, 3, 4, 8, 9, 16, 27, 32, 64, 81, 99]
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
                 "categories": [2, 3, 4, 8, 9, 16, 27, 32, 64, 81, 99]
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
                 "categories": [2, 3, 4, 8, 9, 16, 27, 32, 64, 81, 99]
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
                 "categories": [2, 3, 4, 8, 9, 16, 27, 32, 64, 81, 99]
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
                 "categories": [2, 3, 4, 8, 9, 16, 27, 32, 64, 81, 99]
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
               "name": "degree",
               "categories": [2, 3, 4, 5, 6]
             },
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
