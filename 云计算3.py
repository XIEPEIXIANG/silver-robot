# -*- coding: utf-8 -*-
"""极速复现论文《COGN: 对话情感因果推理》核心机制的验证代码
功能：在1个典型对话上验证说话人感知因果图+隐变量建模的有效性
时间：约30分钟（含可视化）
"""
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score


# ===================== 1. 数据准备 =====================
def prepare_demo_data():
    """构造一个典型冲突对话样本（模拟论文数据格式）"""
    return {
        "text": [
            "You never listen to me!",  # 说话人A的愤怒句（情感触发点）
            "I was busy with work.",  # 说话人B的解释（非情感句）
            "That's just an excuse!",  # 说话人A的升级愤怒（被前序语句触发）
            "Fine, I won't bother you then."  # 说话人A的失望（情感变化结果）
        ],
        "speaker": ["A", "B", "A", "A"],  # 说话人标签
        "true_cause": [(2, 0), (3, 2)]  # 真实因果边：(结果句, 原因句)
    }


# ===================== 2. 特征工程 =====================
def extract_handcraft_features(texts):
    """手工特征提取（替代BERT加速实验）：
    - 愤怒词计数: never, excuse, bother等关键词
    - 疑问词计数: ?等
    """
    anger_words = ["never", "excuse", "bother", "listen"]
    features = []
    for text in texts:
        # 特征1: 文本中的愤怒词数量（简单情感强度代理）
        anger_count = sum(text.lower().count(word) for word in anger_words)
        # 特征2: 是否包含问号（对话行为特征）
        is_question = 1 if "?" in text else 0
        features.append([anger_count, is_question])
    return torch.tensor(features, dtype=torch.float)


# ===================== 3. 基线模型 =====================
def baseline_model(feats):
    """基线方法：全连接GAT无任何约束"""
    # 构建全连接边（排除自环）
    n = len(feats)
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j]).t()

    # 计算边权重（简单点积相似度）
    sim_matrix = torch.mm(feats, feats.t())
    edge_weights = sim_matrix[edge_index[0], edge_index[1]]

    # 预测因果边（权重>0.5的边）
    pred_edges = edge_index[:, (edge_weights > 0.5).nonzero().squeeze()]
    return pred_edges.t().tolist()


# ===================== 4. 论文方法 =====================
def paper_method(feats, speakers):
    """论文核心方法：说话人掩码 + 隐变量时序约束"""
    # 构建全连接边
    n = len(feats)
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j]).t()

    # 说话人掩码（同说话人=1，不同=0）
    speaker_ids = torch.tensor([0 if s == "A" else 1 for s in speakers])  # A=0, B=1
    same_speaker_mask = (speaker_ids[edge_index[0]] == speaker_ids[edge_index[1]]).float()

    # 隐变量建模（用上一句特征作为当前句的隐变量）
    z = torch.cat([feats[0:1], feats[:-1]])  # 第一句重复，其余用前一句

    # 计算因果权重 = (当前句·隐变量) * 说话人掩码 * 时序掩码(i>j)
    causal_weights = (feats[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    causal_weights *= same_speaker_mask
    causal_weights *= (edge_index[0] > edge_index[1]).float()  # 只允许时间向后的因果

    # 预测因果边（权重>0.3的边，更高阈值减少噪声）
    pred_edges = edge_index[:, (causal_weights > 0.3).nonzero().squeeze()]
    return pred_edges.t().tolist()


# ===================== 5. 评估与可视化 =====================
def visualize_results(true_edges, pred_baseline, pred_paper):
    """绘制因果图对比"""
    plt.figure(figsize=(12, 5))

    # 真实因果图
    plt.subplot(131)
    G_true = nx.DiGraph()
    G_true.add_edges_from(true_edges)
    nx.draw(G_true, with_labels=True, node_color='lightgreen',
            edge_color='green', width=2, arrowsize=20)
    plt.title("Ground Truth")

    # Baseline预测
    plt.subplot(132)
    G_base = nx.DiGraph()
    G_base.add_edges_from(pred_baseline)
    nx.draw(G_base, with_labels=True, node_color='lightblue',
            edge_color='red', width=2, arrowsize=20)
    plt.title("Baseline (No Constraints)")

    # 论文方法预测
    plt.subplot(133)
    G_paper = nx.DiGraph()
    G_paper.add_edges_from(pred_paper)
    nx.draw(G_paper, with_labels=True, node_color='pink',
            edge_color='blue', width=2, arrowsize=20)
    plt.title("Paper Method (Speaker+Time)")

    plt.tight_layout()
    plt.show()


def calculate_precision(true_edges, pred_edges, total_possible_edges):
    """计算精确率（预测正确的边占比）"""
    true_set = set(map(tuple, true_edges))
    pred_set = set(map(tuple, pred_edges))
    return len(true_set & pred_set) / len(pred_set) if len(pred_set) > 0 else 0


# ===================== 主执行流程 =====================
if __name__ == "__main__":
    # 1. 加载数据
    data = prepare_demo_data()
    print("测试对话:", data["text"])

    # 2. 特征提取
    features = extract_handcraft_features(data["text"])
    print("特征矩阵:\n", features)

    # 3. 运行基线模型
    baseline_edges = baseline_model(features)
    print("\nBaseline预测边:", baseline_edges)

    # 4. 运行论文方法
    paper_edges = paper_method(features, data["speaker"])
    print("论文方法预测边:", paper_edges)

    # 5. 评估结果
    n = len(data["text"])
    total_possible = n * (n - 1)  # 全连接边数量
    prec_base = calculate_precision(data["true_cause"], baseline_edges, total_possible)
    prec_paper = calculate_precision(data["true_cause"], paper_edges, total_possible)
    print(f"\n精确率对比: Baseline={prec_base:.1%}, Paper={prec_paper:.1%}")

    # 6. 可视化
    visualize_results(data["true_cause"], baseline_edges, paper_edges)