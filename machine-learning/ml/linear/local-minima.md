# Local Minima Problem

{% hint style="info" %}
여기에 사진을 추가할 예정
{% endhint %}

위와 같은 L\(손실\)함수가 존재했을 때, 어떤 문제가 발생할 수 있을까?

이같은 경우는 지역마다 최소값\(**local minima**\)이 여러 개 있기 때문에 기존에 Gradient descent를 사용해서는 궁국적으로 우리가 찾는 **global minima**를 찾을 수 없게 된다.

그렇다면 우리는 어떤 방법으로 이 문제를 **해결**할 수 있을까?

음... 학습을 시킬 때 샘플을 _**랜덤**_한 걸로 지정해보면 어떨까

이렇게 학습시키는 방법을 **SGD** 라고 한다. 자세한 이야기는 뒤에서 하겠다.



