# Keywords

![image](./1.png)

## Semantic Segmantation
Semantic Segmantation은 이미지의 각 픽셀을 사전 정의된 클래스 중 하나로 분류하는 작업이다.
같은 class인 서로 다른 object는 하나로 취급된다.
'배경', '도로' 등의 'stuff'는 취급하지 않으며, '사람' 등의 'things'만 취급한다.

## Instance Segmantation
Instance Segmantation은 각 객체를 개별적으로 식별하고, 객체의 경계를 구분하여 각 객체에 대한 binary mask와 class label을 예측하는 작업이다.
같은 class인 서로 다른 object는 각각의 object로 취급된다.
'배경', '도로' 등의 'stuff'는 취급하지 않으며, '사람' 등의 'things'만 취급한다.

## Panoptic Segmantation
Panoptic Segmantation은 Semantic Segmantation과 Instance Segmantation을 통합한 방식으로, 이미지의 모든 픽셀을 분류함과 동시에 각 객체를 개별적으로 식별하는 작업이다.
이 방법은 `stuff`와 `things` 로 카테고리를 분류한다. 
`stuff`는 **형태가 일정하지 않은 물체**(하늘, 도로 등)를 의미하며, 
`things`는 **독립적인 인스턴스를 가진 물체**(사람 등)를 의미한다.
