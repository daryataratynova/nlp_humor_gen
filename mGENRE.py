import pickle
from genre.trie import Trie, MarisaTrie
import torch
from genre.fairseq_model import mGENRE
import pandas as pd
from tqdm import tqdm
import re

data = pd.read_excel('./Our Dataset MY_VERSION.xlsx')
first_column = data.iloc[:, 0].values

# def extract_first_three_sentences(text):
#     pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
#     sentences = re.split(pattern, text)
#     return ' '.join(sentences[:3])

# combined_strings = []

# for index, row in data.iterrows():
#     title = row['title']
#     content = row['content']

#     title_without_nyt = title.replace("The New York Times", "")

#     content_without_nyt = content.replace("New York Times", "")

#     first_three_sentences = extract_first_three_sentences(content_without_nyt)
#     # combined_string = f"{title_without_nyt} {first_three_sentences}"
#     # combined_string = f"{title_without_nyt}"
#     combined_string = f"{first_three_sentences}"
#     combined_strings.append(combined_string)
# print('lead only')


with open("lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
    lang_title2wikidataID = pickle.load(f)

with open("titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_mGENRE = mGENRE.from_pretrained("fairseq_multilingual_entity_disambiguation").eval()
model_mGENRE.to(device)
print("mGENRE loaded")

#prediction procedure is taken from here https://github.com/facebookresearch/GENRE/tree/main/examples_mgenre

def pred(sentences, model):
    
    prediction = model.sample(
        sentences,
        prefix_allowed_tokens_fn=lambda batch_id, sent: [
            e for e in trie.get(sent.tolist())
            if e < len(model.task.target_dictionary)
        ],
        text_to_id=lambda x: max(lang_title2wikidataID[tuple(reversed(x.split(" >> ")))], key=lambda y: int(y[1:])),
        marginalize=True,
    )
    return prediction

results = []
wiki_ids = []
texts = []
for sen in first_column:
    sentence = [sen]
    prediction = pred(sentence, model_mGENRE)
    results.append(prediction)
    
    id_list = list(map(lambda x: x['id'], prediction[0]))
    text = list(map(lambda x: x['texts'], prediction[0]))
    wiki_ids.append(id_list)
    texts.append(text)

data = {'Column': wiki_ids, 'Texts': texts}
df = pd.DataFrame(data)
excel_file_path = './mGENRE_headlines-2.xlsx'
df.to_excel(excel_file_path, index=False)
print("Excel file saved successfully.")