# Machine Learning Systems

course website: https://pooyanjamshidi.github.io/mls/

![](ml-systems-cover.png)

## Overview 

When we talk about Artificial Intelligence (AI) in general and Machine Learning (ML) in particular, we typically refer to a technique or an algorithm that gives the computer systems the ability to learn and to reason with data. However, there is a lot more to AI/ML than just implementing an algorithm or a technique. In this course, we will learn the fundamental differences between AI/ML as a technique versus AI/ML as a system in production. By the end of this class, I hope:
You will be able to apply state of the art ML algorithms, in whatever problem you are interested in, at scale and learn how to deal with unique challenges that only may happen when building real-world production-ready AI/ML systems.
You will be able to do AI at the Edge; there will be projects for an end-to-end, cloud-to-edge, hardware + software infrastructure for facilitating the deployment of AI-based solutions using Edge TPU/NVIDIA Jetson Nano and TensorFlow Lite.
I also hope I can convey my own excitement about AI/ML systems to you.
You are well qualified for doing research in AI/ML systems.

## Lectures

- Lecture 0: Course Introduction: Machine Learning Systems
  - tl;dr: This lecture reviews course requirement, learning goals, policies, and expectations.
- Lecture 1: Introduction to Machine Learning Systems (Uber Case Study)
  - tl;dr: This lecture reviews challenges of building a real-world ML system that scales.
  - Blogs
    - [COTA: Improving Uber Customer Care with NLP & Machine Learning](https://eng.uber.com/cota/)
    - [Scaling Uber’s Customer Support Ticket Assistant (COTA) System with Deep Learning](https://eng.uber.com/cota-v2/)
  - Papers
    - [COTA: Improving the Speed and Accuracy of Customer Support through Ranking and Deep Networks](https://arxiv.org/pdf/1807.01337.pdf)
  - Videos
    - [Using NLP & Machine Learning to improve customer care at Uber, Huaixiu Zheng,20180418](https://www.youtube.com/watch?v=_l5wbgoLYTo)
- Lecture 2: Machine Learning Systems: Challenges and Solutions
  - tl;dr: This lecture reviews reactive strategies to incorporate ML-based components into a larger system.
  - Books
    - [Machine Learning Systems: Designs that scale](https://www.manning.com/books/machine-learning-systems)
  - Videos
    - [Reactive Machine Learning On and Beyond the JVM by Jeff Smith](https://www.youtube.com/watch?v=akPLphTykwI)
- Lecture 3: Optimization and Neural Nets
  - tl;dr: This lecture builds the foundation of optimization and deep learning.
- Lecture 4: Learning Theory
  - tl;dr: This lecture reviews basic concepts related to statistical learning theory (e.g., hypothesis space).
- Lecture 5: Deep Convolutional Neural Networks
  - tl;dr: This lecture builds the foundation of deep learning systems.
- Lecture 6: Deep Learning System Stack
  - tl;dr: This lecture reviews the full-stack deep learning system development.
- Lecture 7: Backpropagation and Automatic Differentiation
  - tl;dr: This lecture reviews backprop and automatic differentiation.
- Lecture 8: High-Performance Hardware for Deep Learning
  - tl;dr: This lecture reviews hardware backends for deep learning.
- Lecture 9: Optimization and Performance Understanding of ML Systems
  - tl;dr: This lecture discusses performance optimization of machine learning systems.
- Lecture 10: Compressing Deep Neural Networks: Pruning and Quantization
  - tl;dr: This lecture discusses DNN model compression techniques.
- Lecture 11: Machine Learning Platforms
  - tl;dr: This lecture reviews a platform that facilitates building an ML pipeline in production at scale.
  - Blogs
    - [Meet Michelangelo: Uber’s Machine Learning Platform](https://eng.uber.com/michelangelo-machine-learning-platform/)
  - Videos
    - [UBER : Big Data Infrastructure and Machine Learning Platform](https://www.youtube.com/watch?v=y3O94MnO_IU)
    - [Michelangelo: Uber's machine learning platform - Achal Shah](https://www.youtube.com/watch?v=hGy1cM7_koM)
    - [Michelangelo Palette: A Feature Engineering Platform at Uber](https://www.infoq.com/presentations/michelangelo-palette-uber/)
- Lecture 12: Scalable Machine Learning
  - tl;dr: This lecture introduces variations of gradient descent and ideas how to it scale up using parallel computing.
- Lecture 13: Distributed Machine Learning
  - tl;dr: This lecture introduces how to scale up deployment (over multiple nodes) to speed up training and inference.
  - Blogs
    - [Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow](https://eng.uber.com/horovod/)
  - Code
    - [Horovod](https://github.com/horovod/horovod)
- Lecture 14: Recurrent Neural Networks
  - tl;dr: This lecture studies RNNs and LSTM architectures for predicting rare events.
- Lecture 15: Intrinsic Dimension
  - tl;dr: This lecture introduces the concept of instrinsic dimension and its implications for model compression.

## Projects

- Project 1: Design Space Exploration of Deep Neural Networks
  - tl;dr: How the choice of configuration options (e.g., CPU frequency) affect inference time and energy?
- Project 2: Design Space Exploration of Distributed ML
  - tl;dr: How the choice of configuration parameters in distributed ML affect training time?
- Project 3: Design Space Exploration of Model Serving
  - tl;dr: How you can decrease latency of model serving by changing configurations such as caching?
- Project 4: Accelerating Deep Reinforcement Learning
  - tl;dr: The aim of this project is to utilize computer system capability to accelerate training of Deep RL agents.

## References

- Technical Company Engineering Blogs
  - [Uber](https://eng.uber.com/)
  - [Airbnb](https://airbnb.io/)
  - [Spotify](https://engineering.atspotify.com/)
  - [Netflix](https://netflixtechblog.com/)
  - [Coursera](https://medium.com/coursera-engineering)
  - [Facebook](https://engineering.fb.com/)
  - [Twitter](https://blog.twitter.com/engineering/en_us.html)
