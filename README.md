# MelodicSeasons
Melon 차트에서 수집한 음악 데이터를 바탕으로, 가사의 감정과 계절(기온) 간의 상관관계를 분석한 프로젝트입니다.

## Methods
국내 음원 사이트인 Melon사의 지난 25년간 인기 음악 2,468곡의 데이터를 이용하여 분석을 진행하였습니다. [기상청](https://data.kma.go.kr/stcs/grnd/grndTaList.do)으로부터 음악 발매월의 월 평균 기온을 수집하였으며, [감정 분류 모델](https://huggingface.co/daniel604/koelectra-base-v3-finetuned-emotion)을 생성하여 가사의 감정을 취득하였습니다.

![Image](https://github.com/user-attachments/assets/9529a8ed-2e8c-4434-8dec-e2d5d458c8c0)

#### Fine tuned emotion classification model

[감정 분류 모델](https://huggingface.co/daniel604/koelectra-base-v3-finetuned-emotion)은 [KoELECTRA-Base-v3](https://github.com/monologg/KoELECTRA)를 AI hub의 [감정 분류를 위한 대화 음성 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&dataSetSn=263&aihubDataSe=extrldata), [한국어 감정 정보가 포함된 단발성 대화 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=270)을 이용하여 감정 분류를 위해 파인 튜닝하였습니다. 모델을 학습시키는 과정의 Figure는 [이곳](https://github.com/daniel-604/MelodicSeasons/tree/main/figure/model_train)에서 보실 수 있습니다.

모델은 [huggingface에 게시](https://huggingface.co/daniel604/koelectra-base-v3-finetuned-emotion)되어 있으며 아래의 코드로 사용해보실 수 있습니다.

```python
import pandas as pd
from transformers import pipeline, ElectraTokenizer

model_id = "daniel604/koelectra-base-v3-finetuned-emotion"
model_ckpt = "monologg/koelectra-base-v3-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(model_ckpt)
classifier = pipeline("text-classification", model=model_id, tokenizer=tokenizer)

labels = ['fear', 'surprise', 'angry', 'sadness', 'neutral', 'happiness', 'disgust']

def predict(text):
    preds = classifier(text, top_k=None)

    if isinstance(text, list):
        df_list = [pd.DataFrame(p) for p in preds]
        avg_df = pd.concat(df_list).groupby('label', as_index=False).mean()
        return 100 * avg_df['score'].values

    else:
        preds_df = pd.DataFrame(preds[0])
        return 100 * preds_df['score'].values
```

## Results
<p align="center">
  <img src="https://raw.githubusercontent.com/daniel-604/MelodicSeasons/refs/heads/main/figure/result_analysis/emotion_temp_heatmap.png" width="45%">
  <img src="https://raw.githubusercontent.com/daniel-604/MelodicSeasons/refs/heads/main/figure/result_analysis/emotion_season_heatmap.png" width="45%" style="margin-right: 10px;">
</p>

#### Pearson and Spearman Correlation Coefficient

| Emotion | Pearson Correlation | p-value | Spearman Correlation | p-value |
|:---------:|:---------------------:|:---------------:|:-----------------------:|:---------------:|
| Happy | 0.08 | 3.48 × 10⁻⁵ | 0.09 | 6.88 × 10⁻⁶ |
| Sad | -0.11 | 1.07 × 10⁻⁸ | -0.12 | 1.04 × 10⁻⁹ |

이외의 추가적인 차트는 [이곳](https://github.com/daniel-604/MelodicSeasons/tree/main/figure/result_analysis)에서 보실 수 있습니다.

## References
KoELECTRA(https://github.com/monologg/KoELECTRA)

Natural Language Processing with Transformers, Revised Edition(https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)

Melon chart(https://www.melon.com/chart/age/index.htm)

기상청 기상자료개방포털(https://data.kma.go.kr/stcs/grnd/grndTaList.do)