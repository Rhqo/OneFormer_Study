# Evaluation Metrics

## PQ

$$
PQ = \frac{\sum_{(p,g) \in TP}IoU(p, g)}{|TP|+\frac{1}{2}|FP|+\frac{1}{2}|FN|} \\ = \frac{\sum_{(p,g) \in TP}IoU(p, g)}{|TP|} \times \frac{|TP|}{|TP|+\frac{1}{2}|FP|+\frac{1}{2}|FN|}\\ = SQ \times RQ
$$

SQ (Segmentation Quality)

RQ (Recognition Quality)

## AP (Average Precision)

하나의 class에 대해 얼마나 잘, 정확히 감지했는지

precision, recall을 누적하여 커브를 그리고 그 안의 영역

certain한 결과부터 uncertain한 결과 순서로 나열하여 누적

모든 클래스에 대해 평균을 내면 mAP

## IoU

$$
IoU = \frac{Prediction \cap Ground Truth}{Prediction\cup Ground Truth} = \frac{Area of OverLap}{Area of Union}
$$

모든 클래스에 대해 평균을 내면 mIoU
