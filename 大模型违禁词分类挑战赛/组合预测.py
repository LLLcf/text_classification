import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os
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

class FocalLoss(nn.Module):
    """多类别Focal Loss实现"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C] 模型输出的logits
        target: [N, ] 真实标签
        """
        # 计算log_softmax
        log_pt = torch.log_softmax(input, dim=1)
        # 计算softmax概率
        pt = torch.exp(log_pt)
        # 应用聚焦因子
        log_pt = (1 - pt) ** self.gamma * log_pt
        # 计算带权重的负对数似然损失
        loss = F.nll_loss(
            log_pt, 
            target, 
            weight=self.weight, 
            reduction=self.reduction, 
            ignore_index=self.ignore_index
        )
        return loss

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
        
# 复用训练时的工具函数和模型类
# ------------------------------
# 1. 复用训练时的文本截断函数
def truncate_with_head_tail(text, tokenizer, max_length=512, head_length=128, tail_length=382):
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
    
    final_input_ids = [tokenizer.cls_token_id] + final_input_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(final_input_ids)
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

# 2. 复用训练时的模型类
class BertForSequenceClassification(nn.Module):
    def __init__(self, bert_model_name, num_labels=10, dropout=0.1):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size * 5, num_labels)  # 最后5层CLS拼接

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        hidden_states = outputs[2]  # 所有隐藏层输出
        
        # 取最后5层的CLS token
        cls_list = [hidden_states[i][:, 0, :] for i in range(-1, -6, -1)]
        last_hidden = torch.cat(cls_list, dim=1)
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        return logits

# 3. 复用标签映射函数
def get_label(data):
    unique_labels = sorted(data['类别'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label

# 4. 测试数据集类（无标签）
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, head_len=128, tail_len=382):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_len = head_len
        self.tail_len = tail_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = truncate_with_head_tail(
            text, self.tokenizer,
            max_length=self.max_length,
            head_length=self.head_len,
            tail_length=self.tail_len
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids']
        }

# 集成预测核心函数
# ------------------------------
def predict_ensemble(test_dataloader, models, device):
    """获取所有模型对测试集的预测类别和概率"""
    all_preds = []  # 存储每个样本的所有模型预测类别 [样本数, 模型数]
    all_probs = []  # 存储每个样本的所有模型预测概率 [样本数, 模型数]
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="集成预测中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            sample_preds = []  # 当前样本在各模型上的预测类别
            sample_probs = []  # 当前样本在各模型上的预测概率
            
            for model in models:
                logits = model(input_ids, attention_mask, token_type_ids)
                probs = torch.softmax(logits, dim=1)  # 转换为概率
                pred_id = torch.argmax(probs, dim=1).item()  # 预测类别ID
                pred_prob = probs[0, pred_id].item()  # 预测类别的概率
                
                sample_preds.append(pred_id)
                sample_probs.append(pred_prob)
            
            all_preds.append(sample_preds)
            all_probs.append(sample_probs)
    
    return all_preds, all_probs

def vote_preds_with_probs(all_preds, all_probs, id_to_label):
    """基于投票+概率加权解决平局的最终预测函数"""
    final_preds = []
    
    for preds, probs in zip(all_preds, all_probs):
        # 统计每个类别的出现次数和对应概率
        label_stats = defaultdict(lambda: {'count': 0, 'probs': []})
        for pred_id, prob in zip(preds, probs):
            label_stats[pred_id]['count'] += 1
            label_stats[pred_id]['probs'].append(prob)
        
        # 找到最高票数
        max_count = max(stats['count'] for stats in label_stats.values())
        # 筛选出获得最高票数的候选类别
        candidates = [label_id for label_id, stats in label_stats.items() 
                     if stats['count'] == max_count]
        
        # 确定最终预测（处理平局）
        if len(candidates) == 1:
            final_id = candidates[0]
        else:
            # 平局时选择平均概率最高的类别
            avg_probs = {label_id: np.mean(label_stats[label_id]['probs']) 
                        for label_id in candidates}
            final_id = max(avg_probs, key=avg_probs.get)
        
        # 转换为原始标签名称
        final_preds.append(id_to_label[final_id])
    
    return final_preds

# 主预测流程
# ------------------------------
if __name__ == "__main__":
    # 1. 配置参数（需与训练时保持一致）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    MODEL_PATH = '/root/lanyun-fs/models/chinese-roberta-wwm-ext'  # 预训练模型路径
    fold_model_paths = [
        "./model_path/rdrop_multi_gpu/fold_1_chinese-roberta-wwm-ext_rdrop_valF1_0_7577_20250818_154517.pth",
        "./model_path/rdrop_multi_gpu/fold_2_chinese-roberta-wwm-ext_rdrop_valF1_0_7351_20250818_193058.pth",
        "./model_path/rdrop_multi_gpu/fold_3_chinese-roberta-wwm-ext_rdrop_valF1_0_7240_20250818_231603.pth",
        "./model_path/rdrop_multi_gpu/fold_4_chinese-roberta-wwm-ext_rdrop_valF1_0_6883_20250819_031413.pth",
        "./model_path/rdrop_multi_gpu/fold_5_chinese-roberta-wwm-ext_rdrop_valF1_0_7379_20250819_071253.pth",
        "./model_path/focal_loss/fold_1_chinese-roberta-wwm-ext_focal_valF1_0_7328_20250818_000623.pth",
        './model_path/focal_loss/fold_2_chinese-roberta-wwm-ext_focal_valF1_0_7226_20250818_024536.pth',
        "./model_path/focal_loss/fold_3_chinese-roberta-wwm-ext_focal_valF1_0_7005_20250818_053611.pth",
        "./model_path/focal_loss/fold_4_chinese-roberta-wwm-ext_focal_valF1_0_6913_20250818_082419.pth",
        "./model_path/focal_loss/fold_5_chinese-roberta-wwm-ext_focal_valF1_0_7282_20250818_110457.pth",
        "./model_path/rdrop_multi_gpu/fold_1_chinese-roberta-wwm-ext_rdrop_valF1_0_7442_20250818_154517.pth",
        "./model_path/rdrop_multi_gpu/fold_2_chinese-roberta-wwm-ext_rdrop_valF1_0_7255_20250818_193058.pth",
        "./model_path/rdrop_multi_gpu/fold_5_chinese-roberta-wwm-ext_rdrop_valF1_0_7125_20250819_071253.pth",
    ]

    # fold_model_paths = [
    #     "./model_path/rdrop_multi_gpu/fold_4_chinese-roberta-wwm-ext_rdrop_valF1_0_6883_20250819_031413.pth",
    # ]
    
    # 数据配置
    TEST_FILE = 'dataset/test_text.csv'  # 测试数据路径
    OUTPUT_FILE = 'test_predictions1.csv'  # 预测结果保存路径
    MAX_LENGTH = 512  # 与训练时一致
    DROPOUT = 0.2  # 与训练时一致
    
    # 2. 加载标签映射（需从训练数据中获取类别信息）
    train_data = pd.read_csv('dataset/train_all.csv')  # 训练数据路径
    label_to_id, id_to_label = get_label(train_data)
    NUM_LABELS = len(label_to_id)
    print(f"类别映射: {id_to_label}")
    
    # 3. 加载测试数据
    test_data = pd.read_csv(TEST_FILE)
    test_texts = test_data['文本'].values  # 假设测试数据文本列名为“文本”
    print(f"测试样本数量: {len(test_texts)}")
    
    # 4. 加载所有折的模型
    models = []
    for path in fold_model_paths:
        if not path or not os.path.exists(path):
            print(f"模型路径不存在: {path}，已跳过")
            continue
        
        # 初始化模型并加载权重
        model = BertForSequenceClassification(
            bert_model_name=MODEL_PATH,
            num_labels=NUM_LABELS,
            dropout=DROPOUT
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()  # 设置为评估模式
        models.append(model)
    
    print(f"成功加载 {len(models)} 个模型用于集成预测")
    if len(models) == 0:
        raise ValueError("未加载到任何模型，请检查模型路径")
    
    # 5. 准备测试数据加载器
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    test_dataset = TestDataset(
        texts=test_texts,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 单样本处理确保顺序
    
    # 6. 集成预测
    all_preds, all_probs = predict_ensemble(test_dataloader, models, device)
    
    # 7. 投票确定最终预测结果
    final_preds = vote_preds_with_probs(all_preds, all_probs, id_to_label)
    
    # 8. 保存预测结果
    test_data['预测类别'] = final_preds
    test_data.to_csv(OUTPUT_FILE, index=False)
    print(f"预测结果已保存至: {OUTPUT_FILE}")

    submit = pd.read_csv('dataset/example.csv')
    submit['类别'] = final_preds
    submit.to_csv('dataset/submit.csv')

    # 打印部分预测结果示例
    print("\n预测结果示例:")
    print(test_data[['文本', '预测类别']].head(5))