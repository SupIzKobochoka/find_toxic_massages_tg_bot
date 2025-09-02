import joblib
import torch
from transformers import AutoTokenizer, AutoModel


model_rf = joblib.load('tg_bot/main_model/model.pkl')


tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embed(text: list[str]):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask']).numpy()

def predict_toxic_proba(text):
    embed = get_embed(text)
    return model_rf.predict_proba(embed)[:,1]


if __name__ == '__main__':
    print(predict_toxic_proba(['ты конечно не прав в данном случае, такая ситуация явно неоднозначна',
                               "ты блять дурак чтоли? Ты не понимаешь что ли что эти мудаки ебаные во всем виноваты?"]))

