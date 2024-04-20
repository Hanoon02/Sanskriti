import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import json
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam

# Emotion categories
emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
label_encoder = LabelEncoder()
label_encoder.fit(emotions)

class EmotionModel1(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.emotion_classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.flip_classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        emotion_output = self.emotion_classifier(output.pooler_output)
        flip_output = self.flip_classifier(output.pooler_output)
        return emotion_output, flip_output

class EmotionModel2(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=30522, embedding_dim=768)
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.emotion_classifier = nn.Linear(512, num_labels)
        self.flip_classifier = nn.Linear(512, 2)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        gru_output, _ = self.gru(embeddings)
        last_hidden_state = gru_output[:, -1, :]
        last_hidden_state = self.dropout(last_hidden_state)
        emotion_output = self.emotion_classifier(last_hidden_state)
        flip_output = self.flip_classifier(last_hidden_state)
        return emotion_output, flip_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = len(emotions)

def load_model(model_path, model_class, num_labels, device):
    model = model_class(num_labels)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

model1 = load_model('A4_43//TASK 1//Bert_base_uncased.pth', EmotionModel1, num_labels, device)
model2 = load_model('A4_43//TASK 1//GRU.pth', EmotionModel2, num_labels, device)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def prepare_data(tokenizer, text, max_length=128):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']

def predict_emotions(models, tokenizer, texts, label_encoder, device):
    results = []
    for text in texts:
        data = prepare_data(tokenizer, text)
        input_ids, attention_mask = data[0].to(device), data[1].to(device)
        outputs = []
        for model in models:
            with torch.no_grad():
                if isinstance(model, EmotionModel2):
                    output = model(input_ids)
                else:
                    output = model(input_ids, attention_mask)
            emotion_logits = output[0]
            emotion_preds = torch.argmax(emotion_logits, dim=1)
            decoded_emotion = label_encoder.inverse_transform([emotion_preds.item()])[0]
            outputs.append(decoded_emotion)
        results.append((text, outputs))
    return results

if __name__ == "__main__":
    print("Welcome to the Emotion Detection System!")
    print("Type a sentence to analyze or 'exit' to quit.")

    models = [model1, model2]

    while True:
        input_text = input("Enter your sentence: ")
        if input_text.lower() == 'exit':
            print("Exiting the system.")
            break
        
        texts = [input_text]
        predictions = predict_emotions(models, tokenizer, texts, label_encoder, device)

        for _, emotion_preds in predictions:
            print(f"Sentence: {input_text}")
            print(f"Predicted emotion by Model 1: {emotion_preds[0]}")
            print(f"Predicted emotion by Model 2: {emotion_preds[1]}")
            print()

    print("Thank you for using the Emotion Detection System!")
