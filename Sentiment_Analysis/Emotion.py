import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModel
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os
from torch.optim import Adam

def load_data(filepath):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            sequence_data = []
            for entry in data:
                emotions = entry['emotions']
                flips = [0] + [1 if emotions[i] != emotions[i-1] else 0 for i in range(1, len(emotions))]
                sequence_data.extend([{'utterance': u, 'emotion': e, 'flip': f} for u, e, f in zip(entry['utterances'], emotions, flips)])
            return pd.DataFrame(sequence_data)
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        return pd.DataFrame()

class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.emotion_encoder = LabelEncoder()
        self.emotions = self.emotion_encoder.fit_transform(self.data['emotion'])
        self.flips = torch.tensor(self.data['flip'].values, dtype=torch.long)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        encoded = self.tokenizer.encode_plus(
            item['utterance'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'labels': torch.tensor(self.emotions[idx], dtype=torch.long),
            'flips': self.flips[idx]
        }

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

train_df = load_data('train.json')
val_df = load_data('val.json')

tokenizer1 = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer2 = AutoTokenizer.from_pretrained('bert-base-uncased')  

train_dataset1 = EmotionDataset(train_df, tokenizer1)
val_dataset1 = EmotionDataset(val_df, tokenizer1)
train_dataset2 = EmotionDataset(train_df, tokenizer2)
val_dataset2 = EmotionDataset(val_df, tokenizer2)

train_loader1 = DataLoader(train_dataset1, batch_size=16, shuffle=True)
val_loader1 = DataLoader(val_dataset1, batch_size=16, shuffle=False)
train_loader2 = DataLoader(train_dataset2, batch_size=16, shuffle=True)
val_loader2 = DataLoader(val_dataset2, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = EmotionModel1(len(np.unique(train_df['emotion']))).to(device)
model2 = EmotionModel2(len(np.unique(train_df['emotion']))).to(device)

optimizer1 = Adam(model1.parameters(), lr=1e-5)
optimizer2 = Adam(model2.parameters(), lr=1e-5)

criterion = nn.CrossEntropyLoss()

def train_and_validate_model(model, optimizer, train_loader, val_loader, epochs=4, model_name='model'):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            flips = batch['flips'].to(device)

            if isinstance(model, EmotionModel2):
                emotion_preds, flip_preds = model(input_ids)  
            else:
                emotion_preds, flip_preds = model(input_ids, attention_mask)

            emotion_loss = criterion(emotion_preds, labels)
            flip_loss = criterion(flip_preds, flips)
            loss = emotion_loss + flip_loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss, metrics = validate_model(model, val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Avg Training Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_val_loss:.4f}")
        print_metrics(metrics, model_name, epoch)

    plot_losses(train_losses, val_losses, model_name)

def validate_model(model, val_loader):
    model.eval()
    total_val_loss = 0
    total_emotion_true, total_emotion_pred, total_flip_true, total_flip_pred = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            flips = batch['flips'].to(device)

            if isinstance(model, EmotionModel2):
                emotion_preds, flip_preds = model(input_ids)  
            else:
                attention_mask = batch['attention_mask'].to(device)
                emotion_preds, flip_preds = model(input_ids, attention_mask)

            emotion_loss = criterion(emotion_preds, labels)
            flip_loss = criterion(flip_preds, flips)
            loss = emotion_loss + flip_loss
            total_val_loss += loss.item()

            total_emotion_true.extend(labels.cpu().numpy())
            total_emotion_pred.extend(emotion_preds.argmax(dim=1).cpu().numpy())
            total_flip_true.extend(flips.cpu().numpy())
            total_flip_pred.extend(flip_preds.argmax(dim=1).cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss, (total_emotion_true, total_emotion_pred, total_flip_true, total_flip_pred)

def print_metrics(metrics, model_name, epoch):
    emotion_true, emotion_pred, flip_true, flip_pred = metrics
    emotion_acc = accuracy_score(emotion_true, emotion_pred)
    emotion_precision = precision_score(emotion_true, emotion_pred, average='macro', zero_division=0)
    emotion_recall = recall_score(emotion_true, emotion_pred, average='macro', zero_division=0)
    emotion_f1 = f1_score(emotion_true, emotion_pred, average='macro', zero_division=0)
    flip_acc = accuracy_score(flip_true, flip_pred)
    flip_precision = precision_score(flip_true, flip_pred, average='binary', zero_division=0)
    flip_recall = recall_score(flip_true, flip_pred, average='binary', zero_division=0)
    flip_f1 = f1_score(flip_true, flip_pred, average='binary', zero_division=0)

    print(f"Emotion Accuracy: {emotion_acc:.4f}, Precision: {emotion_precision:.4f}, Recall: {emotion_recall:.4f}, F1: {emotion_f1:.4f}")
    print(f"Flip Accuracy: {flip_acc:.4f}, Precision: {flip_precision:.4f}, Recall: {flip_recall:.4f}, F1: {flip_f1:.4f}")

def plot_losses(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Training and Validation Loss for {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plot_path = "plots"  
    os.makedirs(plot_path, exist_ok=True) 
    plot_filename = os.path.join(plot_path, f"{model_name.replace(' ', '_')}_loss_plot.png")
    plt.savefig(plot_filename)  
    plt.close()  
    print(f"Plot saved as {plot_filename}")

def save_model(model, optimizer, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename)

torch.cuda.empty_cache()
print("Evaluating Model-1")
train_and_validate_model(model1, optimizer1, train_loader1, val_loader1, epochs=4, model_name='Model 1')
save_model(model1, optimizer1, 'Bert_base_uncased.pth') # Saving Model 1 checkpoint

torch.cuda.empty_cache()
print("Evaluating Model-2")
train_and_validate_model(model2, optimizer2, train_loader2, val_loader2, epochs=4, model_name='Model 2')
save_model(model2, optimizer2, 'GRU.pth') # Saving Model 2 checkpoint
