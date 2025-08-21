import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import BertTokenizer, BertModel
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

# 早停机制类
class EarlyStopping:
    """早停机制实现"""
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_score, model, model_path):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_score, model, model_path):
        if self.verbose:
            print(f'Validation score improved ({self.val_loss_min:.6f} --> {val_score:.6f}).  Saving model ...')
        torch.save(model.module.state_dict(), model_path)
        self.val_loss_min = val_score

# EMA指数移动平均类
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

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
        
# # 注意力池化层
# class AttentionPooling(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(in_dim, in_dim),
#             nn.LayerNorm(in_dim),
#             nn.GELU(),
#             nn.Linear(in_dim, 1),
#         )

#     def forward(self, last_hidden_state, attention_mask):
#         w = self.attention(last_hidden_state).float()
#         w[attention_mask == 0] = float('-inf')
#         w = torch.softmax(w, dim=1)
#         attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
#         return attention_embeddings

# # BERT分类模型（使用最后4层输出+AttentionPooling）
# class BertForSequenceClassification(nn.Module):
#     def __init__(self, bert_model_name, num_labels=10, dropout=0.1):
#         super(BertForSequenceClassification, self).__init__()
#         self.num_labels = num_labels
#         self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
#         self.dropout = nn.Dropout(dropout)
#         self.hidden_size = self.bert.config.hidden_size
        
#         self.attention_pool = AttentionPooling(self.hidden_size)
#         self.classifier = nn.Linear(self.hidden_size * 4, num_labels)

#     def forward(self, input_ids, attention_mask, token_type_ids):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=False
#         )

#         hidden_states = outputs[2]

#         layer_embeddings = []
#         for i in range(-1, -5, -1):
#             layer_output = hidden_states[i]

#             pooled_output = self.attention_pool(layer_output, attention_mask.unsqueeze(-1))
#             layer_embeddings.append(pooled_output)
        
#         combined_output = torch.cat(layer_embeddings, dim=1)
#         combined_output = self.dropout(combined_output)
#         logits = self.classifier(combined_output)

#         return logits

# FGM对抗训练类
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

# R-Drop损失函数
class RDropLoss(nn.Module):
    '''R-Drop的Loss实现'''
    def __init__(self, class_weights, label_smoothing=0.1,alpha=4):
        super().__init__()
        self.alpha = alpha
        self.loss_sup = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, y_pred1, y_pred2, y_true):
        ce_loss1 = self.loss_sup(y_pred1, y_true)
        ce_loss2 = self.loss_sup(y_pred2, y_true)
        loss_sup = (ce_loss1 + ce_loss2) / 2
        
        assert y_pred1.shape == y_pred2.shape

        loss_rdrop1 = self.loss_rdrop(
            F.log_softmax(y_pred1, dim=-1), 
            F.softmax(y_pred2, dim=-1)
        ).sum(dim=-1)
        loss_rdrop2 = self.loss_rdrop(
            F.log_softmax(y_pred2, dim=-1), 
            F.softmax(y_pred1, dim=-1)
        ).sum(dim=-1)

        loss_kl = (loss_rdrop1 + loss_rdrop2).mean() / 2
        
        total_loss = loss_sup + self.alpha * loss_kl
        
        return total_loss

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

def evaluate_model(model, dataloader, criterion, device):
    """评估模型"""
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

            logits1 = model(input_ids, attention_mask, token_type_ids)
            logits2 = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits1, logits2, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits1, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    classification_reports = classification_report(all_labels, np.array(all_preds))
    
    return avg_loss, macro_f1, classification_reports

def softmax_with_temperature(logits, temperature=1.0):
    """带温度系数的softmax"""
    return np.exp((logits - np.max(logits)) / temperature) / np.sum(np.exp((logits - np.max(logits)) / temperature))

# 训练函数
def train_model_with_FGM_EMA(fold, MODEL_NAME, model, train_dataloader, val_dataloader, 
                           criterion, optimizer, scheduler, device, num_epochs, save_dir, 
                           patience=3, ema_decay=0.999, ema_start_epoch=4):
    """训练模型"""
    
    os.makedirs(save_dir, exist_ok=True)
    best_macro_f1 = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(save_dir, 
                                 f"fold_{fold+1}_{MODEL_NAME}_rdrop_ema_cosine_best_{timestamp}.pth")
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.001)
    ema = EMA(model, ema_decay)
    ema_enabled = False

    for epoch in range(num_epochs):
        if epoch >= ema_start_epoch and not ema_enabled:
            print(f"\n===== 从第 {epoch+1} 轮开始启用EMA =====")
            ema.register()
            ema_enabled = True
        
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Fold {fold+1} | Training Epoch {epoch + 1}")

        fgm = FGM(model)
        all_train_labels = []
        all_train_preds = []

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
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
            
            scheduler.step(epoch + batch_idx / len(train_dataloader))
            
            # 只有EMA启用后才更新
            if ema_enabled:
                ema.update()

            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': loss.item(), 'lr': f'{current_lr:.6f}'})

            preds = torch.argmax(logits1, dim=-1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels_np)

        train_macro_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        avg_train_loss = total_loss / len(train_dataloader)

        if ema_enabled:
            ema.apply_shadow()
            val_loss, val_macro_f1, classification_report = evaluate_model(model, val_dataloader, criterion, device)
            ema.restore()
        else:
            val_loss, val_macro_f1, classification_report = evaluate_model(model, val_dataloader, criterion, device)

        print(f"\nFold {fold+1} | Epoch {epoch+1}/{num_epochs}")
        print(f"  EMA状态: {'已启用' if ema_enabled else '未启用'}")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Macro F1: {train_macro_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f}")
        print("Classification Report:")
        print(classification_report)
        print("-" * 60)

        if val_macro_f1 > best_macro_f1 and val_macro_f1 >= 0.68:
            best_macro_f1 = val_macro_f1
            best_model_path = os.path.join(save_dir, 
                                          f"fold_{fold+1}_{MODEL_NAME}_rdrop_ema_valF1_{val_macro_f1:.5f}_{timestamp}.pth")
            if ema_enabled:
                ema.apply_shadow()
                torch.save(model.module.state_dict(), best_model_path)
                ema.restore()
            else:
                torch.save(model.module.state_dict(), best_model_path)
            print(f"模型保存成功！当前验证集Macro F1: {val_macro_f1:.4f}")

        elif val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            print(f"当前验证集Macro F1: {val_macro_f1:.4f} (未达0.68阈值，不保存模型)")

        # 早停检查
        early_stopping(val_macro_f1, model, best_model_path)
        
        if early_stopping.early_stop:
            print(f"\nFold {fold+1} 早停触发! 在第 {epoch+1} 轮停止训练")
            break

    return best_macro_f1, best_model_path

# 主程序
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已释放 PyTorch CUDA 缓存")
    
    seed_everything(seed=42)
    # ===================== 训练配置 =====================
    device_ids = [0, 1]  # 多卡配置
    if torch.cuda.is_available() and len(device_ids) > torch.cuda.device_count():
        raise ValueError(f"指定了{len(device_ids)}个GPU，但实际可用GPU数量为{torch.cuda.device_count()}")
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}，多卡设备列表: {device_ids}")

    # 模型与数据配置
    MODEL_PATH = '/root/lanyun-fs/models/chinese-bert-wwm'  # BERT模型路径
    MODEL_NAME = os.path.basename(MODEL_PATH)
    MAX_LENGTH = 512  # 文本最大长度
    BATCH_SIZE = 32   # 批次大小
    LEARNING_RATE = 2e-5  # 初始学习率
    WEIGHT_DECAY = 0.01   # 权重衰减
    NUM_EPOCHS = 10       # 最大训练轮数
    DROPOUT = 0.2         # Dropout比例
    N_SPLITS = 5          # 交叉验证折数
    SAVE_DIR = "./model_path/bert"  # 模型保存目录

    # 策略配置
    EARLY_STOPPING_PATIENCE = 3  # 早停耐心值
    EMA_DECAY = 0.999            # EMA衰减系数
    EMA_START_EPOCH = 11          # EMA开始轮数
    R_DROP_ALPHA = 4             # R-Drop的alpha参数
    
    # 余弦退火调度器参数
    T_0 = 500       # 初始周期长度
    T_MULT = 5       # 周期乘数
    ETA_MIN = 1e-6   # 最小学习率

    # 加载数据
    data = pd.read_csv('dataset/train_all.csv')
    label_to_id, id_to_label, data = get_label(data)
    NUM_LABELS = len(label_to_id)
    print(f"类别数: {NUM_LABELS}, 标签映射: {label_to_id}")

    # 交叉验证设置
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
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

        # 初始化tokenizer和模型
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification(
            bert_model_name=MODEL_PATH,
            num_labels=NUM_LABELS,
            dropout=DROPOUT
        ).to(device)
        
        model = nn.DataParallel(model, device_ids=device_ids)

        # 数据集和数据加载器
        train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
        val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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

        # 初始化损失函数、优化器和调度器
        criterion = RDropLoss(class_weights=class_weights, alpha=R_DROP_ALPHA)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )

        # 余弦退火带重启调度器
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_MULT,
            eta_min=ETA_MIN,
            last_epoch=-1
        )

        # 训练模型
        best_f1, best_model_path = train_model_with_FGM_EMA(
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
            save_dir=SAVE_DIR,
            patience=EARLY_STOPPING_PATIENCE,
            ema_decay=EMA_DECAY,
            ema_start_epoch=EMA_START_EPOCH
        )

        fold_performances['fold'].append(fold+1)
        fold_performances['best_val_macro_f1'].append(best_f1)
        fold_model_paths.append(best_model_path)

        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 输出交叉验证结果汇总
    print("\n" + "="*50)
    print("五折交叉验证性能汇总")
    print(f"使用策略: R-Drop Loss + FGM对抗训练 + EMA({EMA_DECAY}) + 余弦退火调度")
    print(f"早停参数: 耐心值 = {EARLY_STOPPING_PATIENCE}")
    print(f"EMA启动轮数: {EMA_START_EPOCH}")
    print("="*50)
    for i in range(N_SPLITS):
        print(f"第 {i+1} 折 - 最优验证集Macro F1: {fold_performances['best_val_macro_f1'][i]:.4f}")
        print(f"模型路径: {fold_model_paths[i]}")
    print("-"*50)
    avg_f1 = np.mean(fold_performances['best_val_macro_f1'])
    print(f"平均验证集Macro F1: {avg_f1:.4f} ± {np.std(fold_performances['best_val_macro_f1']):.4f}")
    print("="*50)
