---
title: Generative Adversarial Nets
date: 2020-12-14 14:25:08
categories: Computer Vision
tags: GAN
---

오늘 다룰 논문은 생성 모델에서의 한 획을 그은 **Generative Adversarial Nets**(**GAN**)이다. 이미 많은 글에서 **GAN**에 대해 쉽게 다루었기 때문에, 오늘은 좀 더 학문적으로 **GAN**을 이해해보는 시간을 가져보자.
<!--more-->

****

Generative Models (생성모델)
==
먼저 생성모델이란 무엇일까? **생성 모델**이란 우리가 알지 못 하는 확률 분포를 가진 대상 분포에서 추출(sampling)한 것과 같이 데이터를 만들어내는 모델을 말한다. 여기서 핵심은 우리가 **모르는 분포**이다. 우선 우리가 알고 있는 분포를 가지고 sampling 하는 방법을 알아보자.

Universality of Uniform
==
컴퓨터에서 난수를 생성하면 문득 그런 생각이 들 때가 있다. **Uniform Distribution**을 따르는 확률변수는 기본적인 난수 생성기에 의해서 추출할 수 있다. 하지만, **Normal Distribution**과 같은 다른 특정 분포를 따르는 확률변수는 어떻게 생성을 해내는 걸까? 이를 설명할 수 있는 이론이 바로 **Universality of Uniform**이다.

{% asset_img Universality_of_Uniform.png [hello?] %}

> **Theorem** (Universality of the Uniform). Let *F* : \mathbb{R}