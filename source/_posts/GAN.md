---
title: "[GAN] Generative Adversarial Nets"
date: 2020-12-14 14:25:08
categories: Computer Vision
tags: 
  - GAN
  - Computer Vision
---

오늘 다룰 논문은 생성 모델에서의 한 획을 그은 ***Generative Adversarial Nets*** **(GAN)**이다. 이미 많은 글에서 **GAN**에 대해 쉽게 다루었기 때문에, 오늘은 좀 더 학문적으로 **GAN**을 이해해보는 시간을 가져보자.
<!--more-->

****

Generative Models (생성모델)
==
먼저 생성모델이란 무엇일까? **생성 모델**이란 우리가 알지 못 하는 확률 분포를 가진 대상 분포에서 추출(sampling)한 것과 같이 데이터를 만들어내는 모델을 말한다. 여기서 핵심은 우리가 **모르는 분포**이다. 예를 들면, 사람 얼굴 이미지를 생성해내고 싶다고 생각해보자. 
{% asset_img GAN_person_and_all.png hello%}
사람 얼굴 이미지는 각 픽셀들간의 유기적인 결합과 연결을 통해 사람 이미지로 인식된다. 사람 얼굴 이미지의 전체 분포가 존재한다고 가정했을 때, 우리는 정확히 그 분포가 어떤 확률 분포 함수를 가지고 sample되는지를 알 수 없다. 결국 **생성 모델**에서 다가가고자 하는 단계는 사람 얼굴의 분포와 같이 알려지지 않은 분포에서도 자연스러운 변수를 sample할 수 있는 것이다. 우선 우리가 알고 있는 분포를 가지고 sampling 하는 방법을 알아보자.

Universality of Uniform
==
컴퓨터에서 난수를 생성하면 문득 그런 생각이 들 때가 있다. **Uniform Distribution**을 따르는 확률변수는 기본적인 난수 생성기에 의해서 추출할 수 있다. 하지만, **Normal Distribution**과 같은 다른 특정 분포를 따르는 확률변수는 어떻게 생성을 해내는 걸까? 이를 설명할 수 있는 이론이 바로 **Universality of Uniform**이다.

> **Theorem** (Universality of the Uniform) 
> Let $F : \mathbb{R} \to (0,1)$ be a continuous function and strictly increasing on its support. Then $F$ is a valid CDF and the inverse function $F^{-1}:(0,1) \to \mathbb{R}$ exists. For such $F$<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;Let $U$~$Unif(0,1)$ and $X=F^{-1}(U)$. Then $X$ is a r.v. with CDF $F$. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;Let $X$ be an r.v. with CDF $F$. Then $F(X)$~$Unif(0,1)$. 

예시를 통해서 위 정리를 통해 어떻게 원하는 분포에서 확률 변수를 sample 할 수 있는지 알아보자. 아래 그래프는 **Normal Distribution**의 **Cumulative Density Function(CDF)**이다.
{% asset_img gaussian_cdf.png hello%}
**Normal Distribution**의 **CDF**는 $F(X) = \frac{1}{2}(1 + erf\frac{x-\mu}{\sigma\sqrt{2}})$이다. 여기서 우리가 $F^{-1}(X)$를 구해낸다면, 우리는 **Uniform Distribution**으로부터 sample한 확률 변수 U를 이용하여
$X = F^{-1}(U)$를 통해 **Normal Distribution**으로부터 sample한 $X$를 얻어낼 수 있다. 

여기서 하고자 하는 말은 어떤 확률분포를 가지는 변수라고 할 지라도, 그 확률분포의 **CDF**의 역함수 $F^{-1}(X)$를 구해낼 수 있다면 해당 분포로부터 변수를 sample할 수 있다는 것이다.
결국, 생성모델에서 하고자 하는것은 어떤 데이터의 집합이 어떤 분포로부터 왔는지는 모르지만 해당 분포의 **CDF**의 역함수 $F^{-1}(X)$를 추적하여 해당 분포에서 새로운 데이터를 sample하고싶은것이다.

GAN
==
문제는 위에서는 정확히 우리가 어떤 분포를 통해 생성하고싶은지 알고 있었다. 예를 들면, **Normal Distribution**생성 모델을 만들고 싶다면 해당 확률 분포의 **CDF**의 역함수를 계산하여 위 정리를 통해 확률 변수 $F^{-1}(U)=X \sim N(\mu,\sigma^{2})$을 생성할 수 있다. 하지만 대다수의 경우에서는 우리가 정확히 어떤 확률분포를 따르는지 모르는 상태에서 생성하기를 원한다. 이 과정에서 **CDF**의 역함수를  **Generator(G)**가 추적하고, **Discriminator(D)**가 잘 추적했는지 알려주며 만들어낸다. 
아래 그림은 그 예시를 보여준다. ~~(Normal Distribution처럼 생기긴 했지만 참자.)~~
{% asset_img GAN_process.png hello%}
위 그림에서 $z$ 는 **Uniform Distribution R.V.**이며, $x$ 는 추적하고자 하는 확률 변수이다.
**점선** 그래프는 그 확률분표를 정확히 모르지만, 그 분포로 부터 sample된 데이터들의 집합이다. 우리가 가지고 있는 사람 얼굴 이미지에 해당한다고 생각할 수 있다. **초록선** 그래프는 우리의 **Generator**가 추적해낸 **CDF**의 역함수이며 해당 함수를 거쳐 나온 $x$의 분포라고 할 수 있다. **파란선** 그래프는 **Discriminator**가 실제 분포에서 나온 값인지 **G**를 통해서 생성된 값인지 구분하는 함수이다.

**(a)**에서 **D**로 가면서 학습이 진행되었을 때의 **G**와 **D**의 형태를 보여준다. 초기에는 **G**로 생성된 값이 실제 분포에서 sample된 $x$와 거의 맞지 않지만, 학습이 진행될수록 실제 데이터 분포와 가까워지는 것을 볼 수 있다.

핵심은, **G**는 우리가 원하는 확률 분포 **CDF**의 역함수를 잘 추정해고자 학습될 것이며 **D**는 생성된 데이터와 실제 데이터를 잘 구분해내도록 학습될 것이다.

Adversarial Loss
==
그래서  ***Generative Adversarial Nets*** 논문에서 구체적으로 어떤 방식을 통해 **CDF**의 역함수 $F^{-1}(X)$를 추적하는지를 살펴보자. 그 중심에 바로 **Adversarial Loss**가 존재한다.

$$\begin{equation} \min_{G}\max_{D} V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}[log D(x)] + \mathbb{E}_{z\sim p_{z}(z)}[log( 1-D(G(z)) )] \end{equation}$$

위 식은 ***Generative Adversarial Nets***에서 설정한 objective function이다.
해당 식의 구조를 잠깐 살펴보자. **G**는 **Generator**를 뜻하며 **D**는 **Discriminator**를 뜻한다.
결국 우리는 두 가지의 function을 neural network로 optimize하는데, **G**는 우리가 원하는 분포의 **CDF**의 역함수 $F^{-1}(X)$를, **D**는 입력 변수가 실제 분포에서 sample됐는지 **G**에 의해 생성된 값인지를 구분하도록 한다. 
그렇다면 위의 **Adversarial Loss**를 optimize하면 정말 우리가 원하는 $G = F^{-1}$가 될 수 있는지 수학적으로 분석해보자. 

Optimal Solution of Adversarial Loss
==

우선 $V(G,D)$ 를 기대값 정의에 의해서 다음과 같이 적분식으로 고쳐 쓸 수 있다.

$$\begin{equation} V(G,D) = \displaystyle\int_{x}p_{data}(x)log(D(x))dx+\displaystyle\int_{z}p_{z}(z)log(1-D(G(z)))dz \end{equation}$$

여기서 $G$는 어떤 **CDF**의 역함수라고 할 수 있다. 이때, 이 **CDF**를 $F$라 두면 
$$\begin{equation} G(z) = F^{-1}(z) = x\end{equation}$$ 으로 쓸 수 있다. 식 (3)을 좀 정리해보면,

$$F^{-1}(z) = x$$
$$\begin{equation}z = F(x)\end{equation}$$
식 (4)를 양변 미분하면,
$$\begin{equation}dz = F^{'}(x)dx = p_{g}(x)dx\end{equation}$$

식 (5)를 이용하여 식 (2)에서의 오른쪽 적분식에 $z$를 $x$로 치환하면, 우리가 원하는 식 (6)을 다음과 같이 얻어낼 수 있다.
$$\begin{equation} V(G,D) = \displaystyle\int_{x}p_{data}(x)log(D(x)) + p_{g}(x)log(1-D(x))dx \end{equation}$$

이제 각각 $D$ 와 $G$ 에 대해서 식 (6)을 순차적으로 $maximize$, $minimize$ 했을 때 $G$ 에 대한 $x$ 의 분포가 $p_{g}(x) = p_{data}(x)$를 만족함을 보이자. 

우선 $D$ 에 대해서 $maximize$ 해보자. $y\to a\,log(y) + b\,log(1-y)$ 의 형태를 가진 모든 실수 함수는 $[0,1]$에서 $y=\frac{a}{a+b}$ 일 때 최대값을 갖는다. 따라서 $V(G,D)$가 최대값을 갖는 $D$ 는 다음과 같다.
$$\begin{equation} D^{*}(x) = \displaystyle\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)} \end{equation}$$

이제 마저 $G$ 에 대해서 $minimize$ 해보자. 우리는 $\displaystyle\max_{D}V(G, D)$를 만족하는 $D$를 구했기 때문에 $\displaystyle\\min_{G} \max_{D} V(G,D)$에 해당 $D$ 를 대입하므로써 $G$ 에 대한 식 $C(G) = \displaystyle\\max_{D} V(G,D)$를 표현할 수 있다. 즉, 아래 식에 대해서 $minimize$하는 것으로 문제를 바꿀 수 있다.
$$\displaystyle C(G) = \max_{D}V(G,D)$$
$$ =\mathbb{E}_{x \sim p_{data}}[logD^{*}_{G}(x)] + \mathbb{E}_{z \sim p_{z}}[log(1 - D^{*}_{G}(G(z)))]$$
$$ = \mathbb{E}_{x \sim p_{data}}[logD^{*}_{G}(x)] + \mathbb{E}_{x \sim p_{g}}[log(1 - D^{*}_{G}(x))]$$
$$\begin{equation} = \mathbb{E}_{x \sim p_{data}}[log\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}] +\mathbb{E}_{x \sim p_{g}}[log\frac{p_{g}(x)}{p_{data}(x)+p_{g}(x)}] \end{equation}$$

여기서 우리는 $KL-Divergence$ 와 $JS-Divergence$ 식을 가져와서 위 식 (8)을 고칠 수 있다. 우선 $KL-Divergence$ 식과 $JS-Divergence$ 식은 다음과 같다.
$$ D_{KL}(p||q) = \mathbb{E}_{x \sim p_{x}}[log \frac{p(x)}{q(x)}] $$
$$ \begin{equation} JSD(p||q) = \frac{1}{2}D_{KL}(p||\frac{p+q}{2}) + \frac{1}{2}D_{KL}(q||\frac{p+q}{2}) \end{equation}$$

식 (8)에서 우변에 $-log(4)$ 를 더해주고 빼주면,
$$\begin{equation} C(G) = -log(4) + D_{KL}(p_{data}||\frac{p_{data}+p_{g}}{2}) + D_{KL}(p_{g}||\frac{p_{data}+p_{g}}{2}) \end{equation}$$

으로 정리할 수 있다. 여기서 $D_{KL}(p_{data}||\frac{p_{data}+p_{g}}{2}) + D_{KL}(p_{g}||\frac{p_{data}+p_{g}}{2}) = 2\,JSD(p_{data}||p_{g})$ 인데, $JSD$ 는 $p_{data} = p_{g}$ 일 때 최소값 $0$ 을 갖는다. 따라서, 
$$ \begin{equation}\displaystyle\min_{G} C(G) = -log(4),\;when\;p_{data} = p_{g} \end{equation}$$

따라서, **Adversarial Loss** 인 식 (1)을 optimize 했을 때 **Generator**가 추적하는 분포인 $p_{g}$ 는 우리가 알고자하는 대상 분포인 $p_{data}$ 와 같아진다는 결론을 얻어낼 수 있다.

Conclusion
==
힘들었다...