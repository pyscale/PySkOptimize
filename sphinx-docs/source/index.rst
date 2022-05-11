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

The following is an example JSON representation of a compatible configuration
to be used.

.. code-block:: json

   {
     "targetTransformer": {
       "name": "sklearn.preprocessing.PowerTransformer"
     },
     "model": {
       "name": "sklearn.linear_model.Ridge",
       "params": [
         {
           "name": "alpha",
           "low": 1e-16,
           "high": 1e16,
           "log_scale": true
         }
       ]
     },
     "scoring": "neg_mean_squared_error",
     "preprocess": [
       {
         "name": "featurePod1",
         "features": [
           "MedInc",
           "HouseAge",
           "AveRooms",
           "Population",
           "AveOccup",
           "Latitude",
           "Longitude"
         ],
         "pipeline": [
           {
             "name": "sklearn.preprocessing.PowerTransformer"
           }
         ]
       }
     ]
   }


.. toctree::
   :maxdepth: 2

   base



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
