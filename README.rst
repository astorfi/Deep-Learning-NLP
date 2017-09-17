
*************************************************
Deep-Learning-World-Resources - `Project Page`_
*************************************************
.. _Project Page: http://tensorflow-world-resources.readthedocs.io/en/latest/



.. image:: https://travis-ci.org/astorfi/TensorFlow-World-Resources.svg?branch=master
    :target: https://travis-ci.org/astorfi/TensorFlow-World-Resources
.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/astorfi/TensorFlow-World-Resources/pulls
.. image:: https://badges.frapsoft.com/os/v2/open-source.svg?v=102
    :target: https://github.com/ellerbrock/open-source-badge/
.. image:: https://coveralls.io/repos/github/astorfi/TensorFlow-World-Resources/badge.svg?branch=master
    :target: https://coveralls.io/github/astorfi/TensorFlow-World-Resources?branch=master

.. image:: _img/mainpage/TensorFlow_World.gif

#################
Table of Contents
#################
.. contents::
  :local:
  :depth: 3

============
Introduction
============

The purpose of this project is to introduce a shortcut to developers and researcher
for finding useful resources about TensorFlow.



-----------
Motivation
-----------

There are different motivations for this open source project.

~~~~~~~~~~~~~~~~~~~~~
Why Deep Learning?
~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What's the point of this open source project?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There other similar repositories similar to this repository and are very
comprehensive and useful and to be honest they made me ponder if there is
a necessity for this repository! A great example is `awesome-tensorflow <https://github.com/jtoy/awesome-tensorflow>`_
repository which is a curated list of different TensorFlow resources.

**The point of this repository is that the resources are being targeted**. The organization
of the resources is such that the user can easily find the things he/she is looking for.
We divided the resources to a large number of categories that in the beginning one may
have a headache!!! However, if someone knows what is being located, it is very easy to find the most related resources.
Even if someone doesn't know what to look for, in the beginning, the general resources have
been provided.


------------------------------------
How to make the most of this effort
------------------------------------


============================
Programming with TensorFlow
============================

The references here, deal with the details of programming and writing TensorFlow code.

--------------------------------
Reading data and input pipeline
--------------------------------

The first part is always how to prepare data and how to provide the pipeline to feed it to TensorFlow.
Usually providing the input pipeline can be complicated, even more than the structure design!

* `Dataset API for TensorFlow Input Pipelines`_: A TensorFlow official documentation on *Using the Dataset API for TensorFlow Input Pipelines*
.. _Dataset API for TensorFlow Input Pipelines: https://github.com/tensorflow/tensorflow/tree/v1.2.0-rc1/tensorflow/contrib/data

----------
Variables
----------

Variables are supposed to hold the parameters and supersede by new values as the parameters are updated.
Variables must be clearly set and initialized.


~~~~~~~~~~~~~~~~~~~~~~~~
Creation, Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

* `Variables Creation and Initialization`_: An official documentation on setting up variables

.. _Variables Creation and Initialization: https://www.tensorflow.org/programmers_guide/variables

~~~~~~~~~~~~~~~~~~~~~~
Saving and restoring
~~~~~~~~~~~~~~~~~~~~~~

* `Saving and Loading Variables`_: The official documentation on saving and restoring variables

.. _save and restore Tensorflow models: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

~~~~~~~~~~~~~~~~~
Sharing Variables
~~~~~~~~~~~~~~~~~

* `Sharing Variables`_: The official documentation on how to share variables

.. _Sharing Variables: https://www.tensorflow.org/programmers_guide/variable_scope


--------------------
TensorFlow Utilities
--------------------

Different utilities empower TensorFlow for faster computation in a more monitored manner.


============
Contributing
============

*For typos, please do not create a pull request. Instead, declare them in issues or email the repository owner*. Please note we have a code of conduct, please follow it in all your interactions with the project.

--------------------
Pull Request Process
--------------------

Please consider the following criterions in order to help us in a better way:

1. The pull request is mainly expected to be a link suggestion.
2. Please make sure your suggested resources are not obsolete or broken.
3. Ensure any install or build dependencies are removed before the end of the layer when doing a
   build and creating a pull request.
4. Add comments with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
5. You may merge the Pull Request in once you have the sign-off of at least one other developer, or if you
   do not have permission to do that, you may request the owner to merge it for you if you believe all checks are passed.

----------
Final Note
----------

We are looking forward to your kind feedback. Please help us to improve this open source project and make our work better.
For contribution, please create a pull request and we will investigate it promptly. Once again, we appreciate
your kind feedback and support.
