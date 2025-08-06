import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_label(data):
    unique_labels = sorted(data['类别'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    data['label'] = data['类别'].map(label_to_id)

    return label_to_id, id_to_label, data

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertForSequenceClassification(nn.Module):
    def __init__(self, bert_model_name, num_labels=10, dropout=0.1):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size * 5, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        last_hidden_state = outputs[0]
        pooled_output = outputs[1]
        hidden_states = outputs[2]

        cls_list = []
        for i in range(-1, -6, -1):
            cls_output = hidden_states[i][:, 0, :]
            cls_list.append(cls_output)

        last_hidden = torch.cat(cls_list, dim=1)
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)

        return logits

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def train_model_with_FGM(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs,save_path):

    best_macro_f1 = 0.0
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        fgm = FGM(model)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward(retain_graph=True)

            fgm.attack()
            logits_adv = model(input_ids, attention_mask, token_type_ids)
            loss_adv = criterion(logits_adv, labels)
            loss_adv.backward(retain_graph=True)
            fgm.restore()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_dataloader)

        val_loss, val_macro_f1 = evaluate_model(model, val_dataloader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Val Macro F1: {val_macro_f1:.4f}")
        print("-" * 50)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path} with Val Macro F1: {val_macro_f1:.4f}")

    print(f"Training completed. Best Macro F1: {best_macro_f1:.4f}")

def evaluate_model(model, dataloader, criterion, device):
    """
    评估模型，返回损失、准确率和 Macro F1-Score
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, macro_f1

def softmax_with_temperature(logits, temperature=1.0):
    return np.exp((logits - np.max(logits)) / temperature) / np.sum(np.exp((logits - np.max(logits)) / temperature))

if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已释放 PyTorch CUDA 缓存")

    seed_everything(seed=2025)
    data = pd.read_csv('/workspace/敏感词分类/dataset/train_all.csv')
    label_to_id,id_to_label, data = get_label(data)

    MODEL_NAME = '/workspace/models/chinese-bert-wwm'
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.1
    NUM_EPOCHS = 10
    DROPOUT = 0.2
    NUM_LABELS = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification(bert_model_name=MODEL_NAME,num_labels=NUM_LABELS,dropout=DROPOUT).to(device)

    train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['label'])

    train_data_text = train_data['文本'].to_list()
    train_data_label = train_data['label'].to_list()

    val_data_text = val_data['文本'].to_list()
    val_data_label = val_data['label'].to_list()

    train_dataset = TextClassificationDataset(train_data_text, train_data_label, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(val_data_text, val_data_label, tokenizer, MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # CrossEntropyLoss
    # criterion = nn.CrossEntropyLoss()

    # class_weights CrossEntropyLoss
    class_weights = torch.tensor(100 * softmax_with_temperature(compute_class_weight('balanced', classes=np.array(list(id_to_label.keys())), y=train_data_label), temperature=25.0), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    save_path = 'model_path/best_model_with_FGM_macro_f1.pth'
    train_model_with_FGM(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS, save_path=save_path)
