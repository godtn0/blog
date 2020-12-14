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

> **Theorem** (Universality of the Uniform) 
> Let $F : \mathbb{R} \to (0,1)$ be a continuous function and strictly increasing on its support. Then $F$ is a valid CDF and the inverse function $F^{-1}:(0,1) \to \mathbb{R}$ exists. For such $F$<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;Let $U$~$Unif(0,1)$ and $X=F^{-1}(U)$. Then $X$ is a r.v. with CDF $F$. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;Let $X$ be an r.v. with CDF $F$. Then $F(X)$~$Unif(0,1)$. 

예시를 통해서 위 정리를 통해 어떻게 원하는 분포에서 확률 변수를 sample 할 수 있는지 알아보자. 아래 그래프는 **Normal Distribution**의 **CDF**이다.
{% asset_img gaussian_cdf.png hello%}
**Normal Distribution**의 **CDF**는 $F(X) = \frac{1}{2}(1 + erf\frac{x-\mu}{\sigma\sqrt{2}})$이다. 여기서 우리가 $F^{-1}(X)$를 구해낸다면, 우리는 **Uniform Distribution**으로부터 sample한 확률 변수 U를 이용하여
$X = F^{-1}(U)$를 통해 **Normal Distribution**으로부터 sample한 $X$를 얻어낼 수 있다.

여기서 하고자 하는 말은 어떤 확률분포를 가지는 함수라고 할 지라도, 그 확률분포의 **CDF**의 역함수 $F^{-1}(X)$를 구해낼 수 있다면 해당 분포로부터 변수를 sample할 수 있다는 것이다.
결국, 생성모델에서 하고자 하는것은 어떤 데이터의 집합이 어떤 분포로부터 왔는지는 모르지만 해당 분포의 **CDF**의 역함수 $F^{-1}(X)$를 추적하여 해당 분포에서 새로운 데이터를 sample하고싶은것이다.


Adversarial Loss
==
그래서 우리는 **Generative Adversarial Nets** 논문에서 어떤식으로 **CDF**의 역함수 $F^{-1}(X)$를 추적하는지를 살펴볼것이다. 그 중심에 바로 **Adversarial Loss**가 존재한다.