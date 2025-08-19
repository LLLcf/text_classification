import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BertTokenizer, BertModel, get_constant_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from datetime import datetime
import os
from collections import defaultdict


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

class RDropLoss(nn.Module):
    '''R-Drop的Loss实现，官方项目：https://github.com/dropreg/R-Drop
    核心修改：仅接受两组独立预测结果+标签，计算双向KL散度约束
    '''
    def __init__(self, weights, alpha=4):
        super().__init__()
        self.alpha = alpha
        self.loss_sup = nn.CrossEntropyLoss(weight=weights)  # 带类别权重的交叉熵
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, y_pred1, y_pred2, y_true):
        '''
        Args:
            y_pred1: 同一输入在dropout下的第一次预测logits
            y_pred2: 同一输入在dropout下的第二次预测logits
            y_true: 真实标签
        '''
        # 监督损失：两次预测的交叉熵平均值
        ce_loss1 = self.loss_sup(y_pred1, y_true)
        ce_loss2 = self.loss_sup(y_pred2, y_true)
        loss_sup = (ce_loss1 + ce_loss2) / 2
        
        # 确保两组预测结果形状一致
        assert y_pred1.shape == y_pred2.shape, "两次预测结果形状必须一致"
        
        # 计算双向KL散度损失
        loss_rdrop1 = self.loss_rdrop(
            F.log_softmax(y_pred1, dim=-1), 
            F.softmax(y_pred2, dim=-1)
        ).sum(dim=-1)  # 对类别维度求和
        loss_rdrop2 = self.loss_rdrop(
            F.log_softmax(y_pred2, dim=-1), 
            F.softmax(y_pred1, dim=-1)
        ).sum(dim=-1)
        
        # 平均KL损失
        loss_kl = (loss_rdrop1 + loss_rdrop2).mean() / 2
        
        # 总损失 = 监督损失 + alpha * KL损失
        total_loss = loss_sup + self.alpha * loss_kl
        
        return total_loss

def get_label(data):
    """获取标签映射关系并添加数值标签列"""
    unique_labels = sorted(data['类别'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    data['label'] = data['类别'].map(label_to_id)
    return label_to_id, id_to_label, data

def truncate_with_head_tail(text, tokenizer, max_length=512, head_length=128, tail_length=382):
    """文本头尾截断处理"""
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False
    )
    input_ids = encoded['input_ids']

    usable_length = max_length - 2  # 预留CLS和SEP

    if len(input_ids) <= usable_length:
        final_input_ids = input_ids
    else:
        head_ids = input_ids[:head_length]
        tail_ids = input_ids[-tail_length:] if tail_length > 0 else []
        final_input_ids = head_ids + tail_ids

    # 添加特殊符号
    final_input_ids = [tokenizer.cls_token_id] + final_input_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(final_input_ids)

    # 补齐长度
    padding_length = max_length - len(final_input_ids)
    if padding_length > 0:
        final_input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

    token_type_ids = [0] * max_length

    return {
        'input_ids': torch.tensor(final_input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'token_type_ids': torch.tensor(token_type_ids)
    }

class TextClassificationDataset(Dataset):
    """文本分类数据集"""
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
    """BERT分类模型（使用最后5层CLS拼接）"""
    def __init__(self, bert_model_name, num_labels=10, dropout=0.1):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size * 5, num_labels)  # 5层CLS拼接

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        hidden_states = outputs[2]  # 所有隐藏层输出

        # 取最后5层的CLS token
        cls_list = []
        for i in range(-1, -6, -1):  # 倒数第1到第5层
            cls_output = hidden_states[i][:, 0, :]
            cls_list.append(cls_output)

        # 拼接后过dropout和分类头
        last_hidden = torch.cat(cls_list, dim=1)
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)

        return logits

class FGM():
    """Fast Gradient Method对抗训练"""
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

def train_model_with_FGM(fold, MODEL_NAME, model, train_dataloader, val_dataloader, 
                        criterion, optimizer, scheduler, device, num_epochs, save_dir):
    """训练单折模型并保存最优模型"""
    os.makedirs(save_dir, exist_ok=True)
    best_macro_f1 = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = None  # 记录当前折最优模型路径

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Fold {fold+1} | Training Epoch {epoch + 1}")

        fgm = FGM(model)
        all_train_labels = []
        all_train_preds = []

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            # R-Drop核心：同一输入两次forward（启用dropout）
            logits1 = model(input_ids, attention_mask, token_type_ids)
            logits2 = model(input_ids, attention_mask, token_type_ids)
            
            # 计算基础损失
            loss = criterion(logits1, logits2, labels)
            loss.backward(retain_graph=True)

            # 对抗攻击：基于第一次预测的梯度进行攻击
            fgm.attack()
            logits_adv1 = model(input_ids, attention_mask, token_type_ids)
            logits_adv2 = model(input_ids, attention_mask, token_type_ids)
            loss_adv = criterion(logits_adv1, logits_adv2, labels)
            loss_adv.backward()
            fgm.restore()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            # 用第一次预测结果计算训练指标
            preds = torch.argmax(logits1, dim=-1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels_np)

        # 训练集指标
        train_macro_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        avg_train_loss = total_loss / len(train_dataloader)

        # 验证集指标
        val_loss, val_macro_f1, classification_report = evaluate_model(model, val_dataloader, criterion, device)

        # 输出日志
        print(f"\nFold {fold+1} | Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Macro F1: {train_macro_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f}")
        print("Classification Report:")
        print(classification_report)
        print("-" * 60)
        
        val_f1_str = f"{val_macro_f1:.4f}".replace('.', '_')
        
        # 保存当前折最优模型（移除硬编码阈值，只要更优就保存）
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            if best_macro_f1 > 0.68:
                # 文件名格式：fold_{折数}_模型名_损失类型_valF1_{分数}_时间戳.pth
                loss_type = "rdrop"
                best_model_path = os.path.join(save_dir, 
                                              f"fold_{fold+1}_{MODEL_NAME}_{loss_type}_valF1_{val_f1_str}_{timestamp}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"Fold {fold+1} best model saved: {best_model_path}\n")

    return best_macro_f1, best_model_path

def evaluate_model(model, dataloader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # 评估时仅需一次forward（无dropout）
            logits = model(input_ids, attention_mask, token_type_ids)
            # 为计算损失，用两次forward（但评估损失仅作参考）
            logits2 = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, logits2, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    classification_reports = classification_report(all_labels, np.array(all_preds))
    
    return avg_loss, macro_f1, classification_reports

def softmax_with_temperature(logits, temperature=1.0):
    """带温度系数的softmax（用于类权重调整）"""
    return np.exp((logits - np.max(logits)) / temperature) / np.sum(np.exp((logits - np.max(logits)) / temperature))

if __name__ == "__main__":
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    MODEL_PATH = '/root/lanyun-fs/models/chinese-roberta-wwm-ext'
    MODEL_NAME = os.path.basename(MODEL_PATH)
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    DROPOUT = 0.2
    N_SPLITS = 5
    SAVE_DIR = "./model_path/rdrop"
    USE_FOCAL_LOSS = False  # 定义缺失变量，当前使用RDropLoss

    data = pd.read_csv('dataset/train_all.csv')
    label_to_id, id_to_label, data = get_label(data)
    NUM_LABELS = len(label_to_id)  # 动态获取类别数
    print(f"类别数: {NUM_LABELS}, 标签映射: {label_to_id}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2025)
    texts = data['文本'].values
    labels = data['label'].values

    # 记录各折性能
    fold_performances = defaultdict(list)
    fold_model_paths = []

    # 循环每一折
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n{'='*20} 开始第 {fold+1}/{N_SPLITS} 折训练 {'='*20}")

        train_texts = texts[train_idx]
        train_labels = labels[train_idx]
        val_texts = texts[val_idx]
        val_labels = labels[val_idx]

        print(f"第 {fold+1} 折数据量 - 训练集: {len(train_texts)}, 验证集: {len(val_texts)}")

        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification(
            bert_model_name=MODEL_PATH,
            num_labels=NUM_LABELS,
            dropout=DROPOUT
        ).to(device)

        train_dataset = TextClassificationDataset(
            train_texts, train_labels, tokenizer, MAX_LENGTH
        )
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, tokenizer, MAX_LENGTH
        )

        # 训练集和验证集使用相同batch_size，提高效率
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 计算类别权重
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.array(list(id_to_label.keys())), 
            y=train_labels
        )
        class_weights = torch.tensor(
            100 * softmax_with_temperature(class_weights, temperature=25), 
            dtype=torch.float32
        ).to(device)

        # 初始化R-Drop损失
        criterion = RDropLoss(weights=class_weights, alpha=4)

        # 优化器和学习率调度器
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        total_steps = len(train_dataloader) * NUM_EPOCHS
        scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps)
        )

        # 训练模型
        best_f1, best_model_path = train_model_with_FGM(
            fold=fold,
            MODEL_NAME=MODEL_NAME,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=NUM_EPOCHS,
            save_dir=SAVE_DIR
        )

        fold_performances['fold'].append(fold+1)
        fold_performances['best_val_macro_f1'].append(best_f1)
        fold_model_paths.append(best_model_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("五折交叉验证性能汇总")
    print(f"使用损失函数: {'R-Drop Loss'}")  # 明确当前使用的损失函数
    print("="*50)
    for i in range(N_SPLITS):
        print(f"第 {i+1} 折 - 最优验证集Macro F1: {fold_performances['best_val_macro_f1'][i]:.4f}")
        print(f"模型路径: {fold_model_paths[i]}")
    print("-"*50)
    avg_f1 = np.mean(fold_performances['best_val_macro_f1'])
    print(f"平均验证集Macro F1: {avg_f1:.4f} ± {np.std(fold_performances['best_val_macro_f1']):.4f}")
    print("="*50)