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
from datetime import datetime
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

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("已释放 PyTorch CUDA 缓存")

seed_everything(seed=2025)

def get_label(data):
    unique_labels = sorted(data['类别'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    data['label'] = data['类别'].map(label_to_id)

    return label_to_id, id_to_label, data

def truncate_with_head_tail(text, tokenizer, max_length=512, head_length=128, tail_length=382):
    """
    对文本进行头尾截断
    :param text: 原始文本
    :param tokenizer: BERT tokenizer
    :param max_length: 最大长度（通常512）
    :param head_length: 保留前面多少个token
    :param tail_length: 保留后面多少个token
    :return: 截断后的 input_ids, attention_mask, token_type_ids
    """
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False
    )
    input_ids = encoded['input_ids']

    usable_length = max_length - 2

    if len(input_ids) <= usable_length:
        final_input_ids = input_ids
    else:

        head_ids = input_ids[:head_length]
        tail_ids = input_ids[-tail_length:] if tail_length > 0 else []
        final_input_ids = head_ids + tail_ids

    final_input_ids = [tokenizer.cls_token_id] + final_input_ids + [tokenizer.sep_token_id]

    attention_mask = [1] * len(final_input_ids)

    padding_length = max_length - len(final_input_ids)
    if padding_length > 0:
        final_input_ids = final_input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length

    token_type_ids = [0] * max_length

    return {
        'input_ids': torch.tensor(final_input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'token_type_ids': torch.tensor(token_type_ids)
    }

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, head_len=128, tail_len=382):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_len = head_len
        self.tail_len = tail_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = truncate_with_head_tail(
            text, self.tokenizer,
            max_length=self.max_length,
            head_length=self.head_len,
            tail_length=self.tail_len
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids'],
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

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.emb_backup:
                param.data = self.emb_backup[name]
        self.emb_backup.clear()

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:   # 关键判空
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = self.grad_backup[name]

def train_model_with_PGD(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs, save_dir):
    """
    训练模型并保存每个 epoch 的检查点，文件名包含验证集性能
    Args:
        save_dir: 保存模型的目录路径（会自动创建）
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    best_macro_f1 = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 防止不同训练任务文件名冲突

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        pgd = PGD(model)
        K = 3

        # 用于计算训练集指标
        all_train_labels = []
        all_train_preds = []

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward(retain_graph=True)

            # 对抗攻击
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t==0))
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                logits_adv = model(input_ids, attention_mask, token_type_ids)
                loss_adv = criterion(logits_adv, labels)
                loss_adv.backward(retain_graph=True)
            pgd.restore()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            all_train_preds.extend(preds)
            all_train_labels.extend(labels_np)

        # 计算训练集 Macro F1
        train_macro_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        avg_train_loss = total_loss / len(train_dataloader)

        # 验证模型
        val_loss, val_macro_f1, classification_report = evaluate_model(model, val_dataloader, criterion, device)

        # 输出训练和验证指标
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Macro F1: {train_macro_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report)
        print("-" * 60)

        val_f1_str = f"{val_macro_f1:.3f}".replace('.', '_')
        # 保存最佳模型（可选：也可单独保存最佳模型）
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            if best_macro_f1 > 0.65:
                best_model_path = os.path.join(save_dir, f"best_model_PGD_valF1_{val_f1_str}_{timestamp}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved: {best_model_path}")

    print(f"Training completed. Best Val Macro F1: {best_macro_f1:.4f}")
    return best_model_path

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
    classification_reports = classification_report(all_labels, np.array(all_preds))
    
    return avg_loss, macro_f1, classification_reports

def softmax_with_temperature(logits, temperature=1.0):
    return np.exp((logits - np.max(logits)) / temperature) / np.sum(np.exp((logits - np.max(logits)) / temperature))


if __name__ == "__main__":

    device = torch.device('cuda:0')

    MODEL_NAME = '/root/lanyun-fs/models/chinese-roberta-wwm-ext'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 5e-3
    NUM_EPOCHS = 10
    DROPOUT = 0.2
    NUM_LABELS = 10

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification(bert_model_name=MODEL_NAME,num_labels=NUM_LABELS,dropout=DROPOUT).to(device)
    data = pd.read_csv('dataset/train_all.csv')
    label_to_id,id_to_label, data = get_label(data)
    
    train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['label'])
    
    train_data_text = train_data['文本'].to_list()
    train_data_label = train_data['label'].to_list()
    
    val_data_text = val_data['文本'].to_list()
    val_data_label = val_data['label'].to_list()
    
    train_dataset = TextClassificationDataset(train_data_text, train_data_label, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(val_data_text, val_data_label, tokenizer, MAX_LENGTH)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    class_weights = torch.tensor(softmax_with_temperature(compute_class_weight('balanced', classes=np.array(list(id_to_label.keys())), y=train_data_label), temperature=1), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    best_model_path = train_model_with_PGD(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS, save_dir="./model_path")
