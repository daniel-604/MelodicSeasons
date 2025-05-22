import pandas as pd
from transformers import pipeline, ElectraTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
from soynlp.normalizer import repeat_normalize

# plot korean text
matplotlib.rc('font', family='Nanum Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# load model
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

pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x): 
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x


df = pd.read_csv('data/song_data_with_temp.csv')
score_columns = [f'{label}' for label in labels]

cnt = 0
def get_scores(text):
    global cnt
    sentences = text.split('<EOS>')
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [clean(s) for s in sentences]
    scores = predict(sentences)
    cnt += 1
    print(cnt)
    return pd.Series(scores, index=score_columns)

df[score_columns] = df['lyric'].apply(get_scores)
df.to_csv('data/song_data_with_emotion.csv', index=False)