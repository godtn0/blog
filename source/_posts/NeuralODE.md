---
title: "[NeuralODE] Neural Ordinary Differential Equations"
date: 2020-12-16 14:55:27
categories: Deep Learning
tags:
  - Deep Learning
---

오늘 다룰 논문은 ***NeurIPS 2018*** 에서 뜨거운 관심을 받았던 ***Neural Ordinary Differential Equations*** 이다. 이미 논문 제목에서 나와있듯이 Ordinary Differential Equations, 상미분방정식에 대해서 다룬다. 기존에 layer 단위로 쌓아서 성능을 높혀왔던 Neural Network 구조의 틀을 부수는 듯한 굉장히 신선한 접근을 다룬다. 많은 사람들이 해당 논문의 수식에 의해서 중간에 포기를 하는 경우가 많다. 그래서 최대한 많은 사람들이 이 논문을 완벽히 이해할 수 있도록 노력해서 정리해보겠다.
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
우리는 이를 위 식(2)와 같이 미분방정식의 형식으로 표현할 수 있다. (사실 위 식 (1)에서 $t+1 \to t$로 보내준 것이다.) 다시 말해, **resnet**은 우리가 알고 싶은 input과 output의 관계 식에 대한 **미분방정식**을 **discrete**한(이산적인) 형태로 근사한 함수라고 할 수 있다.

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
우리가 사용하는 Neural Network는 **Ordinary Differential Equations**의 근사 함수 해를 얻어내는 방법으로 볼 수 있었다. 구체적으로, 각 layer 마다의 output을 $\bigtriangleup x=1$마다의 **Euler Discretization Method**를 통한 함수값의 근사치라고 해석 할 수 있다. **Neural ODE**에서는 결국 initial valude $h(0)$로부터 떨어진 some time $T$ 에서의 함수값을 한번에 계산해내고 싶은 것이다. 우리에게 주어진 아래의 미분방정식을 가지고,
$$\begin{equation} \frac{dh(t)}{dt} = f(h(t), t, \theta), \;\;h(t_{0}) \end{equation}$$
다음과 같이 $h(t_{1})$의 값을 계산할 수 있다.
$$\begin{equation} h(t_{1}) = h(t_{0}) + \int_{t_{0}}^{t_{1}}f(h(t),t,\theta)dt \end{equation}$$
아래 그림을 보면 좀 더 이해가 갈 것이다.
{% asset_img ode_resnet.jpg 400 "이 그림의 의도를 이해해줘 제발!!" %}
그렇다면, **ResNet**과 같이 $\bigtriangleup x$ 의 크기를 $1$ 로 제한을 둘 필요가 있을까?
그리고 모든 layer들이 동일한 함수 $f$ 에 대해서 근사하는데, 각 layer들의 parameter를 서로 달리 할 필요가 있을까?
이제 좀 감이 오는가?

ODE solver
==
**ODE solver**는 말 그대로 **ODE**를 해결하는 방법론을 말한다. 예를 들면, 위에서 봤던 **Euler Discretization Method**와 같은 method를 이용해서 **ODE**문제를 해결할 경우 우리는 **Euler Discretization Method ODE sovler**를 쓴다고 할 수 있다. 이 논문에서는 **ODE solver**가 핵심적인 역할을 한다. 이 **ODE solver**를 가지고 기존의 우리의 $loss\;function$ 을 표현해보자. 식을 보면 **ODE solver**의 역할이 더 와닿을 것이다.
$$\begin{equation} 
L(\mathbf{z}(t_{1})) = L(\mathbf{z}(t_{0}) + \int_{t_{0}}^{t_{1}}f(\mathbf{z}(t),t,\theta)dt) = L(ODESolver(\mathbf{z}(t_{0}), f, t_{0}, t_{1}, \theta))
\end{equation}$$
당황하지말고 천천히 위 식을 이해해보자. 우선 $\mathbf{z}(t)$는 위에서 얘기한 $t$ 번째의 time step t에서의 state, hidden layer의 activation이라고 생각하면 된다. 우리는 최종 $scalar\; loss$를 가지고 각 hidden layer activation에 대한 $loss$ 함수를 생각할 수 있다. 이를 표현한 것이 $L(\mathbf{z}(t_{1}))$ 이다. 우리는 실제로 위의 정적분 식을 계산 할 수 없다. 정적분을 계산하기 위해선 적어도 $f$ 에 대한 부정적분을 알아야 하지만, 만약 부정적분 식을 알고 있다면 Neural Network를 통한 함수의 근사조차 필요 없어질 것이다. 따라서 이를 근사하기 위해서 위에서 보았던 **Euler Discretization Method**와 같은 **ODE solver**를 이용하는 것이다. 

Reverse Mode Automatic Differentiation of ODE Solution
==
자 그럼 실제로 위 ODE solver를 가지고 계산한 Loss를 어떻게 optimize 할 것인지, backprop 할 것인지를 알아보자. ***(수식 주의)*** 제목은 사실 별 건 아니고, "그래서 backprop 어케 하누!"를 말하는 것이다.
{% asset_img ode_resnet_loss.jpg 450 "이 그림의 의도를 이해해줘 제발!!" %}
위 그림은 **ResNet**을 가지고 함수를 근사할 경우에 각 **hidden state**마다 **Loss**와 연결되는 듯한 그림이다. 위 그림에서도 보다시피, **ResNet**에서는 time step $t$ 에 대해서 이산적이고 불연속적으로 state가 전파되므로 우리는 $chain\; rule$ 을 통해 모든 hidden state에 대한 gradient를 계산할 수 있었다. 반면, 우리의 **ODE Solver**를 통한 근사는 time step $t$ 에 대해서 연속적이므로 $chain\; rule$ 을 적용하는데 어려움을 겪게 된다. 여기서 **adjoint sensitivity method**를 이용한다.

먼저, hidden state에서의 $loss$ 에 대한 미분을 결정하자. 이를 우리는 $adjoint \; a(t)$ 라고 하자.
$$\begin{equation}
adjoint, \;\;\; a(t) = {\partial L\over\partial \mathbf{z}(t)} 
\end{equation}$$
우리의 $z(t)$ 는 아래 미분방정식에 대한 해이다.
$$\frac{d\mathbf{z}(t)}{d(t)} = f(\mathbf{z}(t), t, \theta)$$
기존의 Neural Nets 에서는 hidden state $h_{t}$ 에 대한 $L$의 미분을 아래와 같은 $chain \; rule$ 을 적용한 식으로 표현할 수 있었다.
$$\frac{dL}{dh_{t}} = \frac{dL}{dh_{t+1}} \cdot \frac{dh_{t+1}}{dh_{t}}$$
$ODE Solver$ 에서 계산된 hidden state는 위와 같이 discrete 하지 않고 연속적이다. 위와 마찬가지로 아주 작은 time step $\varepsilon$ 에 대해서 time step $t+\varepsilon$에서의 미분을 다음과 같이 표현할 수 있다.
$$\begin{equation}
\mathbf{z}(t+\varepsilon) = \int_{t}^{t+\varepsilon}f(\mathbf{z}, t, \theta) dt + \mathbf{z}(t) = T_{\varepsilon}(\mathbf{z}(t, t)
\end{equation}$$
$$\begin{equation}\frac{dL}{d\mathbf{z}(t)} = \frac{dL}{d\mathbf{z}(t+\varepsilon)} \cdot \frac{d\mathbf{z}(t+\varepsilon)}{d\mathbf{z}(t)} \;\;or\;\;
a(t) = a(t+\varepsilon) \cdot {\partial{T_{\varepsilon}(\mathbf{z}(t), t)} \over \partial{\mathbf{z}(t)}}
\end{equation}$$
여기서 미분의 정의에 의해, 
$$\begin{aligned}
\frac{da(t)}{dt} &= \lim_{\varepsilon \to 0+} \frac{a(t+\varepsilon) - a(t)}{\varepsilon} \\
&= \lim_{\varepsilon \to 0+} \frac{a(t+\varepsilon) - a(t+\varepsilon) \cdot {\partial \over \partial{\mathbf{z}(t)}}T_{\varepsilon} (\mathbf{z}(t), t)}{\varepsilon} \\
&= \lim_{\varepsilon \to 0+} \frac{a(t+\varepsilon) - a(t+\varepsilon) \cdot {\partial \over \partial{\mathbf{z}(t)}}(\mathbf{z}(t) + \varepsilon \cdot f(\mathbf{z}(t), t, \theta) + O(\varepsilon^{2})) }{\varepsilon} \\
&= \lim_{\varepsilon \to 0+} \frac{a(t+\varepsilon) - a(t+\varepsilon) \cdot (I + \varepsilon \cdot {\partial{f(\mathbf{z}(t),t,\theta)} \over \partial{\mathbf{z}(t)}} + O(\varepsilon^{2})) }{\varepsilon} \\
&= \lim_{\varepsilon \to 0+} \frac{ -\varepsilon \cdot a(t+\varepsilon) \cdot {\partial{f(\mathbf{z}(t),t,\theta)} \over \partial{\mathbf{z}(t)}} + O(\varepsilon^{2}) }{\varepsilon} \\
&= \lim_{\varepsilon \to 0+} - a(t+\varepsilon) \cdot {\partial{f(\mathbf{z}(t),t,\theta)} \over \partial{\mathbf{z}(t)}} + O(\varepsilon^{2})\\
&= - a(t) \cdot {\partial{f(\mathbf{z}(t),t,\theta)} \over \partial{\mathbf{z}(t)}}
\end{aligned}$$
따라서, 우리는 다음과 같이 $a(t)$ 에 대한 새로운 **ODE**를 얻어낼 수 있다.
$$\begin{equation} 
\frac{da(t)}{dt} = -a(t) \cdot {\partial{f(\mathbf{z}(t)}, t, \theta) \over \partial{\mathbf{z}(t)}} 
\end{equation}$$
우리가 이걸 왜 구했냐? 우리는 hidden state에 대한 $loss$의 미분을 구하고싶다. 왜냐하면 backprop을 해야하니깐. 그런데 $a(t) = {\partial L\over\partial \mathbf{z}(t)}$ 이었다. 이 말은 $a(t)$ 에 대한 미분방정식을 풀어내면, gradient에 대한 함수 해를 얻어낼 수 있다는 것이다. $\mathbf{z}(t)$ 를 구하기 위해서 $ODE Solver$ 를 썼던 것과 마찬가지로 $a(t) = {\partial L\over\partial \mathbf{z}(t)}$ 라는 미분방정식의 해를 구하기 위해서 또 다른 $ODE Solver$ 를 쓸 수 있다는 말이다. 
$$\begin{equation} 
a(t_{0}) = a(t_{1}) - \int_{t_{1}}^{t_{0}} a(t) \cdot {\partial{f(\mathbf{z}(t), t, \theta)} \over \partial{\mathbf{z}(t)}} dt
= ODESolver(a(t_{1}), deriv, t_{1}, t_{0}, \theta)
\end{equation}$$

(진짜 어마무시하지 않음?)

결국 우리가 계산해야하는 것은 hidden state에 대한 gradient를 이용해서 network의 parameter $\theta$ 에 대한 gradient이다. 이제 식 (9)를 가지고 최종적으로 $\frac{dL}{d\theta}$ 를 구해보자. 

