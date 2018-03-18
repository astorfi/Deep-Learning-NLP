
###################################################
Deep Learning for Natural Language Processing
###################################################
.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/astorfi/Deep-Learning-NLP/pulls
.. image:: https://badges.frapsoft.com/os/v2/open-source.png?v=103
    :target: https://github.com/ellerbrock/open-source-badge/
.. image:: https://img.shields.io/github/contributors/cdnjs/cdnjs.svg
    :target: https://github.com/astorfi/Deep-Learning-NLP/graphs/contributors
.. image:: https://img.shields.io/dub/l/vibe-d.svg
    :target: https://github.com/astorfi/Deep-Learning-NLP/blob/master/LICENSE



*****************
Table of Contents
*****************
.. contents::
  :local:
  :depth: 3

============
Introduction
============

The purpose of this project is to introduce a shortcut to developers and researcher
for finding useful resources about Deep Learning for Natural Language Processing.

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
a necessity for this repository!

**The point of this repository is that the resources are being targeted**. The organization
of the resources is such that the user can easily find the things he/she is looking for.
We divided the resources to a large number of categories that in the beginning one may
have a headache!!! However, if someone knows what is being located, it is very easy to find the most related resources.
Even if someone doesn't know what to look for, in the beginning, the general resources have
been provided.


------------------------------------
How to make the most of this effort
------------------------------------

=======
Papers
=======

This chapter is associated with the papers published in NLP using deep learning.

-----------------------
Data Representation
-----------------------

~~~~~~~~~~~~~~~~~~~~~~~
One-hot representation
~~~~~~~~~~~~~~~~~~~~~~~

.. ################################################################################

.. For continuous lines, the lines must be start from the same locations.
* **Character-level convolutional networks for text classification** :
  Promising results by the use of one-hot encoding possibly due to their character-level information.
  [`Paper link <http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica>`_ ,
  `Torch implementation <https://github.com/zhangxiangxiao/Crepe>`_ ,
  `TensorFlow implementation <https://github.com/mhjabreel/CharCNN>`_ ,
  `Pytorch implementation <https://github.com/srviest/char-cnn-pytorch>`_]

.. @inproceedings{zhang2015character,
..   title={Character-level convolutional networks for text classification},
..   author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
..   booktitle={Advances in neural information processing systems},
..   pages={649--657},
..   year={2015}
.. }

.. ################################################################################


.. ################################################################################

.. For continuous lines, the lines must be start from the same locations.
* **Effective Use of Word Order for Text Categorization with Convolutional Neural Networks** :
  Exploiting the 1D structure (namely, word order) of text data for prediction.
  [`Paper link <https://arxiv.org/abs/1412.1058>`_ ,
  `Code implementation <https://github.com/riejohnson/ConText>`_]

.. @article{johnson2014effective,
..   title={Effective use of word order for text categorization with convolutional neural networks},
..   author={Johnson, Rie and Zhang, Tong},
..   journal={arXiv preprint arXiv:1412.1058},
..   year={2014}
.. }

.. ################################################################################


.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Neural Responding Machine for Short-Text Conversation** :
  Neural Responding Machine has been proposed to generate content-wise appropriate responses to input text.
  [`Paper link <https://arxiv.org/abs/1503.02364>`_ ,
  `Paper summary <https://isaacchanghau.github.io/2017/07/19/Neural-Responding-Machine-for-Short-Text-Conversation/>`_]

.. Please add bibtex here
.. @article{shang2015neural,
..   title={Neural responding machine for short-text conversation},
..   author={Shang, Lifeng and Lu, Zhengdong and Li, Hang},
..   journal={arXiv preprint arXiv:1503.02364},
..   year={2015}
.. }

.. ################################################################################

-----------------------
Applications
-----------------------

~~~~~~~~~~~~~~~~~~~~~~
Text classification
~~~~~~~~~~~~~~~~~~~~~~

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Convolutional Neural Networks for Sentence Classification** :
  By training the model on top of the pretrained word-vectors through finetuning, considerable improvement has been reported for learning task-specific vectors.
  [`Paper link <https://arxiv.org/abs/1408.5882>`_ ,
  `Code implementation 1 <https://github.com/yoonkim/CNN_sentence>`_,
  `Code implementation 2 <https://github.com/abhaikollara/CNN-Sentence-Classification>`_,
  `Code implementation 3 <https://github.com/Shawn1993/cnn-text-classification-pytorch>`_,
  `Code implementation 4 <https://github.com/mangate/ConvNetSent>`_]
  
  .. image:: _img/mainpage/progress-overall-100.png


  .. @article{kim2014convolutional,
  ..   title={Convolutional neural networks for sentence classification},
  ..   author={Kim, Yoon},
  ..   journal={arXiv preprint arXiv:1408.5882},
  ..   year={2014}
  .. }

.. ################################################################################



.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **A Convolutional Neural Network for Modelling Sentences** :
  Dynamic Convolutional Neural Network (DCNN) architecture, which technically is the CNN with a dynamic
  k-max pooling method, has been proposed for capturing the semantic modeling of the sentences.
  [`Paper link <https://arxiv.org/abs/1404.2188>`_ ,
  `Code implementation <https://github.com/FredericGodin/DynamicCNN>`_]

  .. image:: _img/mainpage/progress-overall-40.png

  .. @article{kalchbrenner2014convolutional,
  ..   title={A convolutional neural network for modelling sentences},
  ..   author={Kalchbrenner, Nal and Grefenstette, Edward and Blunsom, Phil},
  ..   journal={arXiv preprint arXiv:1404.2188},
  ..   year={2014}
  .. }

.. ################################################################################








============
Contributing
============

*For typos, unless significant changes, please do not create a pull request. Instead, declare them in issues or email the repository owner*. Please note we have a code of conduct, please follow it in all your interactions with the project.

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
