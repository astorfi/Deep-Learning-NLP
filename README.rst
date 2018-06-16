
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



##################
Table of Contents
##################
.. contents::
  :local:
  :depth: 4

***************
Introduction
***************

The purpose of this project is to introduce a shortcut to developers and researcher
for finding useful resources about Deep Learning for Natural Language Processing.

============
Motivation
============

There are different motivations for this open source project.

--------------------
Why Deep Learning?
--------------------

------------------------------------------------------------
What's the point of this open source project?
------------------------------------------------------------

There other similar repositories similar to this repository and are very
comprehensive and useful and to be honest they made me ponder if there is
a necessity for this repository!

**The point of this repository is that the resources are being targeted**. The organization
of the resources is such that the user can easily find the things he/she is looking for.
We divided the resources to a large number of categories that in the beginning one may
have a headache!!! However, if someone knows what is being located, it is very easy to find the most related resources.
Even if someone doesn't know what to look for, in the beginning, the general resources have
been provided.


================================================
How to make the most of this effort
================================================

************
Papers
************

This chapter is associated with the papers published in NLP using deep learning.

====================
Data Representation
====================

-----------------------
One-hot representation
-----------------------

.. ################################################################################

.. For continuous lines, the lines must be start from the same locations.
* **Character-level convolutional networks for text classification** :
  Promising results by the use of one-hot encoding possibly due to their character-level information.
  [`Paper link <http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica>`_ ,
  `Torch implementation <https://github.com/zhangxiangxiao/Crepe>`_ ,
  `TensorFlow implementation <https://github.com/mhjabreel/CharCNN>`_ ,
  `Pytorch implementation <https://github.com/srviest/char-cnn-pytorch>`_]

    .. image:: _img/mainpage/progress-overall-80.png

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

  .. image:: _img/mainpage/progress-overall-60.png

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

  .. image:: _img/mainpage/progress-overall-60.png

.. Please add bibtex here
.. @article{shang2015neural,
..   title={Neural responding machine for short-text conversation},
..   author={Shang, Lifeng and Lu, Zhengdong and Li, Hang},
..   journal={arXiv preprint arXiv:1503.02364},
..   year={2015}
.. }

.. ################################################################################


------------------------------
Continuous Bag of Words (CBOW)
------------------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Distributed Representations of Words and Phrases and their Compositionality** :
  Not necessarily about CBOWs but the techniques represented in this paper
  can be used for training the continuous bag-of-words model.
  [`Paper link <http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases>`_ ,
  `Code implementation 1 <https://code.google.com/archive/p/word2vec/>`_,
  `Code implementation 2 <https://github.com/deborausujono/word2vecpy>`_]


  .. image:: _img/mainpage/progress-overall-100.png

  .. @inproceedings{mikolov2013distributed,
  ..   title={Distributed representations of words and phrases and their compositionality},
  ..   author={Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg S and Dean, Jeff},
  ..   booktitle={Advances in neural information processing systems},
  ..   pages={3111--3119},
  ..   year={2013}
  .. }

.. ################################################################################


---------------------
Word-Level Embedding
---------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Efficient Estimation of Word Representations in Vector Space** :
  Two novel model architectures for computing continuous vector representations of words.
  [`Paper link <https://arxiv.org/abs/1301.3781>`_ ,
  `Official code implementation <https://code.google.com/archive/p/word2vec/>`_]

  .. image:: _img/mainpage/progress-overall-100.png

  .. @article{mikolov2013efficient,
  ..   title={Efficient estimation of word representations in vector space},
  ..   author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  ..   journal={arXiv preprint arXiv:1301.3781},
  ..   year={2013}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **GloVe: Global Vectors for Word Representation** :
  Combines the advantages of the two major models of global matrix
  factorization and local context window methods and efficiently leverages
  the statistical information of the content.
  [`Paper link <http://www.aclweb.org/anthology/D14-1162>`_ ,
  `Official code implementation <https://github.com/stanfordnlp/GloVe>`_]

  .. image:: _img/mainpage/progress-overall-100.png

  .. @inproceedings{pennington2014glove,
  ..   title={Glove: Global vectors for word representation},
  ..   author={Pennington, Jeffrey and Socher, Richard and Manning, Christopher},
  ..   booktitle={Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)},
  ..   pages={1532--1543},
  ..   year={2014}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Skip-Thought Vectors** :
  Skip-thought model applies word2vec at the sentence-level.
  [`http://papers.nips.cc/paper/5950-skip-thought-vectors>`_ ,
  `Code implementation <https://github.com/ryankiros/skip-thoughts>`_,
  `TensorFlow implementation <https://github.com/tensorflow/models/tree/master/research/skip_thoughts>`_]

  .. image:: _img/mainpage/progress-overall-100.png

  .. @inproceedings{kiros2015skip,
  ..   title={Skip-thought vectors},
  ..   author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan R and Zemel, Richard and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
  ..   booktitle={Advances in neural information processing systems},
  ..   pages={3294--3302},
  ..   year={2015}
  .. }

.. ################################################################################

-------------------------
Character-Level Embedding
-------------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Learning Character-level Representations for Part-of-Speech Tagging** :
  CNNs have successfully been utilized for learning character-level embedding.
  [`Paper link <http://proceedings.mlr.press/v32/santos14.pdf>`_ ]

  .. image:: _img/mainpage/progress-overall-60.png

  .. @inproceedings{santos2014learning,
  ..   title={Learning character-level representations for part-of-speech tagging},
  ..   author={Santos, Cicero D and Zadrozny, Bianca},
  ..   booktitle={Proceedings of the 31st International Conference on Machine Learning (ICML-14)},
  ..   pages={1818--1826},
  ..   year={2014}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Deep Convolutional Neural Networks forSentiment Analysis of Short Texts** :
  A new deep convolutional neural network has been proposed for exploiting
  the character- to sentence-level information for sentiment analysis application on short texts.
  [`Paper link <http://www.aclweb.org/anthology/C14-1008>`_ ]

  .. image:: _img/mainpage/progress-overall-80.png

  .. @inproceedings{dos2014deep,
  ..   title={Deep convolutional neural networks for sentiment analysis of short texts},
  ..   author={dos Santos, Cicero and Gatti, Maira},
  ..   booktitle={Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers},
  ..   pages={69--78},
  ..   year={2014}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation** :
  The usage of two LSTMs operate over the char-
  acters for generating the word embedding
  [`Paper link <https://arxiv.org/abs/1508.02096>`_ ]

  .. image:: _img/mainpage/progress-overall-60.png

  .. @article{ling2015finding,
  ..   title={Finding function in form: Compositional character models for open vocabulary word representation},
  ..   author={Ling, Wang and Lu{\'\i}s, Tiago and Marujo, Lu{\'\i}s and Astudillo, Ram{\'o}n Fernandez and Amir, Silvio and Dyer, Chris and Black, Alan W and Trancoso, Isabel},
  ..   journal={arXiv preprint arXiv:1508.02096},
  ..   year={2015}
  .. }

.. ################################################################################

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Improved Transition-Based Parsing by Modeling Characters instead of Words with LSTMs** :
  The effectiveness of modeling characters for dependency parsing.
  [`Paper link <https://arxiv.org/abs/1508.00657>`_ ]

  .. image:: _img/mainpage/progress-overall-40.png

  .. @article{ballesteros2015improved,
  ..   title={Improved transition-based parsing by modeling characters instead of words with lstms},
  ..   author={Ballesteros, Miguel and Dyer, Chris and Smith, Noah A},
  ..   journal={arXiv preprint arXiv:1508.00657},
  ..   year={2015}
  .. }

.. ################################################################################





====================
Applications
====================

--------------------
Text classification
--------------------

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

  .. image:: _img/mainpage/progress-overall-80.png

  .. @article{kalchbrenner2014convolutional,
  ..   title={A convolutional neural network for modelling sentences},
  ..   author={Kalchbrenner, Nal and Grefenstette, Edward and Blunsom, Phil},
  ..   journal={arXiv preprint arXiv:1404.2188},
  ..   year={2014}
  .. }

.. ################################################################################



.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Very Deep Convolutional Networks for Text Classification** :
  The Very Deep Convolutional Neural
  Networks (VDCNNs) has been presented and employed at
  character-level with the demonstration of the effectiveness of
  the network depth on classification tasks
  [`Paper link <http://www.aclweb.org/anthology/E17-1104>`_ ]

  .. image:: _img/mainpage/progress-overall-20.png

  .. @inproceedings{conneau2017very,
  ..   title={Very deep convolutional networks for text classification},
  ..   author={Conneau, Alexis and Schwenk, Holger and Barrault, Lo{\"\i}c and Lecun, Yann},
  ..   booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers},
  ..   volume={1},
  ..   pages={1107--1116},
  ..   year={2017}
  .. }

.. ################################################################################


.. ################################################################################

* **Character-level convolutional networks for text classification** :
  The character-level
  representation using CNNs investigated which argues
  the power of CNNs as well as character-level representation for
  language-agnostic text classification.
  [`Paper link <http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica>`_ ,
  `Torch implementation <https://github.com/zhangxiangxiao/Crepe>`_ ,
  `TensorFlow implementation <https://github.com/mhjabreel/CharCNN>`_ ,
  `Pytorch implementation <https://github.com/srviest/char-cnn-pytorch>`_]

  .. image:: _img/mainpage/progress-overall-80.png

  .. @inproceedings{zhang2015character,
  ..   title={Character-level convolutional networks for text classification},
  ..   author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  ..   booktitle={Advances in neural information processing systems},
  ..   pages={649--657},
  ..   year={2015}
  .. }

.. ################################################################################


.. ################################################################################

* **Multichannel Variable-Size Convolution for Sentence Classification** :
  Multichannel Variable Size Convolutional Neural Network (MV-CNN) architecture
  Combines different version of word-embeddings in addition to
  employing variable-size convolutional filters and is proposed
  in this paper for sentence classification.
  [`Paper link <https://arxiv.org/abs/1603.04513>`_]

  .. image:: _img/mainpage/progress-overall-20.png

  .. @article{yin2016multichannel,
  ..   title={Multichannel variable-size convolution for sentence classification},
  ..   author={Yin, Wenpeng and Sch{\"u}tze, Hinrich},
  ..   journal={arXiv preprint arXiv:1603.04513},
  ..   year={2016}
  .. }

.. ################################################################################


.. ################################################################################

* **A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification** :
  A practical sensitivity analysis of CNNs for exploring the effect
  of architecture on the performance, has been investigated in this paper.
  [`Paper link <https://arxiv.org/abs/1510.03820>`_]

  .. image:: _img/mainpage/progress-overall-60.png

  .. @article{zhang2015sensitivity,
  ..   title={A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification},
  ..   author={Zhang, Ye and Wallace, Byron},
  ..   journal={arXiv preprint arXiv:1510.03820},
  ..   year={2015}
  .. }

.. ################################################################################


* **Generative and Discriminative Text Classification with Recurrent Neural Networks** :
  RNN-based discriminative and generative models have been investigated for
  text classification and their robustness to the data distribution shifts has been
  claimed as well.
  [`Paper link <https://arxiv.org/abs/1703.01898>`_]

  .. image:: _img/mainpage/progress-overall-20.png

  .. @article{yogatama2017generative,
  ..   title={Generative and discriminative text classification with recurrent neural networks},
  ..   author={Yogatama, Dani and Dyer, Chris and Ling, Wang and Blunsom, Phil},
  ..   journal={arXiv preprint arXiv:1703.01898},
  ..   year={2017}
  .. }

.. ################################################################################


.. ################################################################################


* **Deep sentence embedding using long short-term memory networks: Analysis and application to information retrieval** :
  An LSTM-RNN architecture has been utilized
  for sentence embedding with special superiority in
  a defined web search task.
  [`Paper link <https://dl.acm.org/citation.cfm?id=2992457>`_]

  .. image:: _img/mainpage/progress-overall-60.png

  .. .. image:: _img/mainpage/progress-overall-20.png
  ..
  .. @article{palangi2016deep,
  ..   title={Deep sentence embedding using long short-term memory networks: Analysis and application to information retrieval},
  ..   author={Palangi, Hamid and Deng, Li and Shen, Yelong and Gao, Jianfeng and He, Xiaodong and Chen, Jianshu and Song, Xinying and Ward, Rabab},
  ..   journal={IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP)},
  ..   volume={24},
  ..   number={4},
  ..   pages={694--707},
  ..   year={2016},
  ..   publisher={IEEE Press}
  .. }

.. ################################################################################


* **Hierarchical attention networks for document classification** :
  Hierarchical
  Attention Network (HAN) has been presented and utilized to
  capture the hierarchical structure of the text by two word-
  level and sentence-level attention mechanism.
  [`Paper link <http://www.aclweb.org/anthology/N16-1174>`_ ,
  `Code implementation 1 <https://github.com/richliao/textClassifier>`_ ,
  `Code implementation 2 <https://github.com/ematvey/hierarchical-attention-networks>`_ ,
  `Code implementation 3 <https://github.com/EdGENetworks/attention-networks-for-classification>`_,
  `Summary 1 <https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/>`_,
  `Summary 2 <https://medium.com/@sharaf/a-paper-a-day-25-hierarchical-attention-networks-for-document-classification-dd76ba88f176>`_]

  .. image:: _img/mainpage/progress-overall-80.png

  .. @inproceedings{yang2016hierarchical,
  ..   title={Hierarchical attention networks for document classification},
  ..   author={Yang, Zichao and Yang, Diyi and Dyer, Chris and He, Xiaodong and Smola, Alex and Hovy, Eduard},
  ..   booktitle={Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  ..   pages={1480--1489},
  ..   year={2016}
  .. }

.. ################################################################################


.. ################################################################################


* **Recurrent Convolutional Neural Networks for Text Classification** :
  The combination of both RNNs and CNNs is used for text classification which technically
  is a recurrent architecture in addition to max-pooling with
  an effective word representation method and demonstrates
  superiority compared to simple windows-based neural network
  approaches.
  [`Paper link <http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552>`_ ,
  `Code implementation 1 <https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier>`_ ,
  `Code implementation 2 <https://github.com/knok/rcnn-text-classification>`_ ,
  `Summary <https://medium.com/paper-club/recurrent-convolutional-neural-networks-for-text-classification-107020765e52>`_]

  .. image:: _img/mainpage/progress-overall-60.png

  .. @inproceedings{lai2015recurrent,
  ..   title={Recurrent Convolutional Neural Networks for Text Classification.},
  ..   author={Lai, Siwei and Xu, Liheng and Liu, Kang and Zhao, Jun},
  ..   booktitle={AAAI},
  ..   volume={333},
  ..   pages={2267--2273},
  ..   year={2015}
  .. }

.. ################################################################################

* **A C-LSTM Neural Network for Text Classification** :
  A unified architecture proposed for sentence and document modeling for classification.
  [`Paper link <https://arxiv.org/abs/1511.08630>`_ ]

  .. image:: _img/mainpage/progress-overall-20.png

  .. @article{zhou2015c,
  ..   title={A C-LSTM neural network for text classification},
  ..   author={Zhou, Chunting and Sun, Chonglin and Liu, Zhiyuan and Lau, Francis},
  ..   journal={arXiv preprint arXiv:1511.08630},
  ..   year={2015}
  .. }

.. ################################################################################

--------------------
Sentiment Analysis
--------------------

.. ################################################################################
.. For continuous lines, the lines must be start from the same locations.
* **Domain adaptation for large-scale sentiment classification: A deep learning approach** :
  A deep learning approach which learns to extract a meaningful representation for each online review.
  [`Paper link <http://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/ICML2011_sentiment.pdf>`_]

  .. image:: _img/mainpage/progress-overall-80.png


  .. @inproceedings{glorot2011domain,
  ..   title={Domain adaptation for large-scale sentiment classification: A deep learning approach},
  ..   author={Glorot, Xavier and Bordes, Antoine and Bengio, Yoshua},
  ..   booktitle={Proceedings of the 28th international conference on machine learning (ICML-11)},
  ..   pages={513--520},
  ..   year={2011}
  .. }

* **Sentiment analysis: Capturing favorability using natural language processing** :
  A sentiment analysis approach to extract sentiments associated with polarities of positive or negative for specific subjects from a document.
  [`Paper link <https://dl.acm.org/citation.cfm?id=945658>`_]

  .. image:: _img/mainpage/progress-overall-80.png


  .. @inproceedings{nasukawa2003sentiment,
  ..   title={Sentiment analysis: Capturing favorability using natural language processing},
  ..   author={Nasukawa, Tetsuya and Yi, Jeonghee},
  ..   booktitle={Proceedings of the 2nd international conference on Knowledge capture},
  ..   pages={70--77},
  ..   year={2003},
  ..   organization={ACM}
  .. }


* **Document-level sentiment classification: An empirical comparison between SVM and ANN** :
  A comparison study. [`Paper link <https://dl.acm.org/citation.cfm?id=945658>`_]

  .. image:: _img/mainpage/progress-overall-60.png


  .. @article{moraes2013document,
  ..   title={Document-level sentiment classification: An empirical comparison between SVM and ANN},
  ..   author={Moraes, Rodrigo and Valiati, Jo{\~a}O Francisco and Neto, Wilson P Gavi{\~a}O},
  ..   journal={Expert Systems with Applications},
  ..   volume={40},
  ..   number={2},
  ..   pages={621--633},
  ..   year={2013},
  ..   publisher={Elsevier}
  .. }

* **Learning semantic representations of users and products for document level sentiment classification** :
  [`Paper <http://www.aclweb.org/anthology/P15-1098>`_]

  .. image:: _img/mainpage/progress-overall-40.png


* **Document modeling with gated recurrent neural network for sentiment classification** :
  [`Paper <http://www.aclweb.org/anthology/D15-1167>`_,
  `Implementation <https://github.com/NUSTM/DLSC>`_]

  .. image:: _img/mainpage/progress-overall-60.png


* **Semi-supervised recursive autoencoders for predicting sentiment distributions** :
  [`Paper <https://dl.acm.org/citation.cfm?id=2145450>`_]

  .. image:: _img/mainpage/progress-overall-80.png


* **A convolutional neural network for modelling sentences** :
  [`Paper <https://arxiv.org/abs/1404.2188>`_]

  .. image:: _img/mainpage/progress-overall-80.png


* **Recursive deep models for semantic compositionality over a sentiment treebank** :
  [`Paper <http://www.aclweb.org/anthology/D13-1170>`_]

  .. image:: _img/mainpage/progress-overall-60.png


* **Adaptive recursive neural network for target-dependent twitter sentiment classification** :
  [`Paper <http://www.aclweb.org/anthology/P14-2009>`_]

  .. image:: _img/mainpage/progress-overall-20.png

* **Aspect extraction for opinion mining with a deep convolutional neural network** :
  [`Paper <https://www.sciencedirect.com/science/article/pii/S0950705116301721>`_]

  .. image:: _img/mainpage/progress-overall-20.png






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
