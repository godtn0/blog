---
title: "[CycleGAN] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
date: 2020-12-15 14:40:35
categories: Computer Vision
tags:
  - GAN
  - Computer Vision
---

오늘 다룰 논문은 **Image to Image Translation(I2I)**에서 **Reconstruction Loss**의 개념을 추가하여 아주 높은 성능을 달성했던 **CycleGAN** 이다. 최근까지 **Reconstruction Loss** 는 사용되고 있으며, 앞으로도 **이미지 번역(I2I Translation), 이미지 합성(Image Synthesis)** 과 같은 분야에서 계속 사용될 것으로 보여지기 때문에 그 시초격의 논문인 ***Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*** 을 리뷰해보고자 한다. 앞서 보았던 {% post_link GAN [GAN] %} 과 같이 최대한 수식적으로 접근하도록 노력해보겠다.
<!--more-->

****

Image to Image Translation
==
우선 **Image to Image Translation** 이란 무엇인가? 
말 그대로 어떤 이미지를 다른 형식의 이미지로 번역/변환 하는 것이다.
예를 들어, 말의 이미지를 얼룩말의 이미지로 바꾸거나, 풍경 사진을 고흐의 화풍으로 변환하는 것을 말한다.
{% asset_img horse_zebra.png "말과 얼룩말 싸우면 누가 이길까..." %}
즉, 이미지에서의 핵심이 되는 **feature**(말의 모양, 풍경 등)들은 그대로 둔 채, 이미지의 색감이나 텍스쳐와 같은 성질만을 변화시키는 것을 목적으로 한다.

이를 좀 더 수학적으로 생각해보자. 어떤 두 **domain** $X$, $Y$ 가 존재할 때, $X$ 에서 sample된 확률 변수 $x$를 가지고 