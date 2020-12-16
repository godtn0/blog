---
title: "[NeuralODE] Neural Ordinary Differential Equations"
date: 2020-12-16 14:55:27
categories: Deep Learning
tags:
  - Deep Learning
---

오늘 다룰 논문은 ***2019 NeurIPS*** 에서 뜨거운 관심을 받았던 ***Neural Ordinary Differential Equations*** 이다. 이미 논문 제목에서 나와있듯이 Ordinary Differential Equations, 상미분방정식에 대해서 다룬다. 기존에 layer 단위로 쌓아서 성능을 높혀왔던 Neural Network 구조의 틀을 부수는 듯한 굉장히 신선한 접근을 다룬다. 많은 사람들이 해당 논문의 수식에 의해서 중간에 포기를 하는 경우가 많다. 그래서 최대한 많은 사람들이 이 논문을 완벽히 이해할 수 있도록 노력해서 정리해보겠다.
<!--more-->

****

Neural Network
==
아주 간단하게 **Neural Network**에 대한 나의 주관적인 생각을 언급하고 넘어가도록 하겠다. ***(매우 중요)***
**Neural Network**는 input과 output의 관계 함수이다. Neural Network에는 $Wx$ 와 같은 선형적인 연산과, $relu(x)$와 같은 비선형적인 연산을 매우 많이 포함하고있다. 즉, 우리는 Neural Network의 parameter를 잘 조정함으로써 원하는 함수를, 그 함수가 어떠한 형태일지라도 근사해낼 수 있는 것이다. (이 말이 모든 문제를 neural network로 해결할 수 있다는 말은 아니다. 합수의 입력과 출력의 관계가 추상적일지라도 명백히 존재한다면 neural network로 그 함수를 근사해낼 수 있다는 말이다.)

즉, Neural Network는 우리가 어떤 입력과 출력의 관계에 대해서 가정을 할 경우 그 가정에 가장 합당한 함수를 근사해내는 것이라고 생각할 수 있다. 예를 들어, "어떤 사진을 입력으로 넣었을 때 감자인지 고구마인지 구분할 수 있을거야"와 같은 가정 말이다.

Residual Network
==
**Neural ODE**의 이해를 돕기 위해서 **resnet** 의 구조를 살펴볼 필요가 있다. **resnet**은 **ImageNet Challenge** 에서 처음 등장했으며, 기존에 layer가 많아질수록 네트워크 하부에 전달되는 gradient가 작아지는 문제로 인해 더이상 깊은 층을 쌓기 힘들었던 **Deep Neural Networks**의 한계를 깨버린 아주 괄목할 만한 성과를 이뤄낸 구조이다. 그 방법은 매우 심플했다. hidden layer의 activation을 다음 hidden layer의 input으로 직접 주입하는 것이 아닌, **변화량**으로 고려한것이다. 아래 그림을 보도록 하자.
{% asset_img deep_neural_network.png 500 "그림 그리는게 젤 힘듦..." %}
위 그림은 기존의 Deep Neural Network의 구조이다. 단순히 hidden layer $W_{t}$ 의 output(activation)을 hidden layer $W_{t+1}$의 input으로 연결하는 구조이다. 여기다가 **resnet**에서는 아래의 Residual Unit 구조를 적용한다.

{% asset_img residual_unit.png  300 "그림 그리는게 젤 힘듦..." %}
residual unit을 적용한 **resnet**의 구조는 아래 그림과 같다.
{% asset_img resnet.png 500 "그림 그리는게 젤 힘듦..." %}

다시 말해, hidden layer $W_{t}$ 의 output(activation)을 $h_{t}$ 에서 $h_{t+1}$로 가는 변화량으로 보겠다는 것이다. 이를 수식으로 살짝 표현하면 아래와 같이 쓸 수 있다.

$$\begin{equation} h_{t+1} = h_{t} + f(h_{t}, \theta_{t}) \end{equation}$$

여기서 위 **resnet**의 Depth가 매우 깊어진다면 어떨까? 그 식은 $h$ 를 $t$에 대해서 미분한 식과 매우 유사할 것이다. 왜 그런 것인지 아래 식을 통해 납득해보자.

$$\begin{split}
h_{t+1} - h_{t} = f(h_{t}, \theta_{t})\\\\
\frac{h_{t+1} - h_{t}}{(t+1) - t} = f(h_{t}, \theta_{t})\\\\
\end{split}$$ 

여기서 layer가 매우 많아진다는 것은 $h_{t+1}$ 과 $h_{t}$ 사이의 간격이 매우 좁아진다는 것을 의미한다.
layer마다의 간격이 매우 좁아지면 아래 그림과 같이 input과 output을 연결하는 hidden layer들이 무수히 많아진다는 것을 뜻하고, 이는 hidden layer의 output(activation)이 **순간변화율에** 가까워진다고 생각할 수 있다.

{% asset_img resnet_many.png 400 "그림 그리는게 젤 힘듦..." %}

$$\begin{equation} \frac{dh(t)}{dt} = f(h(t), t, \theta) \end{equation}$$
우리는 이를 위 식(2)와 같이 미분방정식의 형식으로 표현할 수 있다. (사실 위 식 (1)에서 $t+1 \to t$로 보내준 것이다.) 다시 말해, **resnet**은 우리가 알고 싶은 input과 output의 관계 식에 대한 **미분방정식**을 **discrete**힌 형태로(이산적인) 근사한 함수라고 할 수 있다.

여기서 한번 정리하고 넘어가자.
우리는 입력과 출력의 관계인 함수를 근사하고자 **Neural Network**를 이용한다. 그 중에서도 **ResNet**의 구조는 해당 함수를 근사하기 위해 입력과 출력의 관계 함수에 대한 미분방정식을 이산적으로 근사한 형태로 해석할 수 있다. 아직 이해가 가지 않더라도 걱정하지 말자. 

Ordinary Differential Equations
==
위 식(2)와 같은 미분방정식은 특별하게 **Ordinary Differential Equation(ODE), 상미분방정식** 이라고 부른다. 함수를 결정하는 독립적인 변수가 하나인 경우에 이와 같이 부르며, 식 (2)에서는 독립 변수가 $t$밖에 존재하지 않으므로 **ODE**라고 부르는 것이다. 

실제로 많은 자연 현상이 미분방정식의 형태로 표현되는데, 대표적으로 슈뢰딩거의 파동방정식이 그러하다.
하지만 복잡한 형식의 미분방정식은 풀기 어렵고, 미분방정식의 해가 되는 함수의 근사 함수라도 얻고자 하는 시도가 활발하다.
대표적인 방법으로 **Euler Discretization Method**가 있다. (우리 학과는 미적분학이 필수 교과로 포함되어 있지 않기 떄문에 설명하고 넘어가겠다.) 임의로 정한 $\bigtriangleup x$ 에 대해서 initial value로부터 $y(x_{t+1}) = y(x_{t}) + \bigtriangleup x \cdot \frac{dy}{dx}(x_{t})$ 의 방식으로 근사해 나가는 것이다.

아래의 미분방정식을 풀어보자.
$$\displaystyle \frac{dy}{dx} = y\, , \,\,\,\, y(0)=1 $$
사실 위 미분방정식은 해는 $y=e^x$ 라는 것을 우리는 알고있다.
하지만, 여기서는 위와 같은 미분방정식을 대표적인 이산적 근사방법, **Euler Discretization Method**로 어떻게 풀어내는지 보도록 하자.
우리는 아래와 같은 표를 하나 만들어 낼 수 있다. 우리는 함수를 근사하는 표를 하나 만들것이다. 이 표는 $x$ 와 근사된 $y$ 그리고 해당 $(x,y)$ 에서의 $\frac{dy}{dx}$ 로 채워나갈것이다.
{% asset_img ex_table1.jpg 400 "그림 그리는게 젤 힘듦..." %}
우선 $x=0$ 에서의 initial value로 $y=1$ 이 주어졌으므로 해당 값을 채운다. 그리고 우리는 $\bigtriangleup x=1$ 단위로 $x$를 옮겨가며 $y$ 와 $\frac{dy}{dx}$를 채워나갈것이다. 즉, $y(1) = y(0) + 1 \cdot \frac{dy}{dx}(0)$ 으로 근사를 한다.
{% asset_img ex_table2.jpg 400 "그림 그리는게 젤 힘듦..." %}
{% asset_img ex_table3.jpg 400 "그림 그리는게 젤 힘듦..." %}
{% asset_img ex_table4.jpg 400 "그림 그리는게 젤 힘듦..." %}
휴... 다 채웠다. 이제 위 표를 바탕으로 그래프를 그려 실제 함수인 $y=e^x$ 과 얼마나 가까운지 확인해보자.
{% asset_img ex_graph.jpg 400 "그림 그리는게 젤 힘듦..." %}
보다시피, initial value와 멀어질 수록 함수의 근사치에 대한 오차는 증가한다.

그래서 이 얘기를 왜 했을까?
우리의 **ResNet** 의 형태를 다시 보자. 
$$h_{t+1} = h_{t} + f(h_{t}, \theta_{t})$$
이 식의 형태가 위에서 본 **Euler Discretization Method**와 매우 유사하지 않나? 결국, 하고자 했던 얘기는 **ResNet**은 어떠한 미분방정식의 함수 해를 찾기 위해 **Euler Discretization Method**와 매우 유사한 방법으로써 함수의 근사를 찾는다고 해석할 수 있다. (위에서 했던 얘기를 부연 설명한 것이다.)

Neural ODE
==
드디어 우리는 **Neural ODE**에서 하고자 하는 것을 이해할 수 있게 되었다. 
**Neural ODE**는 우리가 사용하는 Neural Network는 ODE의 근사를 얻어내는 형태이므로, 굳이 layer를 반복하지 않고 함수를 찾아내는 방법을 제시한다.