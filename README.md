# silver-robot
云计算大作业
# -*- coding: utf-8 -*-
"""极速复现论文《COGN: 对话情感因果推理》核心机制的验证代码
功能：在1个典型对话上验证说话人感知因果图+隐变量建模的有效性
时间：约30分钟（含可视化）
"""
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# ===================== 一. 云计算3 =====================
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

    # ===================== 二. 云计算1 =====================
# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import ollama  # Ollama客户端库，用于与本地大模型交互
import subprocess  # 用于启动外部进程
from tqdm import tqdm  # 用于显示进度条
from time import sleep  # 用于添加延迟
from typing import List, Dict, Tuple  # 类型提示


def ensure_ollama_running():
    """
    确保Ollama服务正在运行
    如果服务未运行，则自动启动deepseek-r1:1.5b模型
    """
    try:
        # 尝试列出模型，验证服务是否可用
        ollama.list()
        print("Ollama服务已就绪")
    except:
        print("正在启动Ollama服务...")
        # 在新控制台窗口中启动模型服务
        # creationflags=subprocess.CREATE_NEW_CONSOLE仅适用于Windows
        subprocess.Popen(["ollama", "run", "deepseek-r1:1.5b"],
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
        # 等待10秒让服务初始化完成
        sleep(10)


def load_news_data(file_path: str, sample_size: int = 50) -> List[Dict]:
    """
    从文本文件加载新闻数据

    参数:
        file_path: 数据文件路径
        sample_size: 要读取的数据条数（默认为50）

    返回:
        包含新闻数据的字典列表
    """
    try:
        # 读取制表符分隔的文本文件，限制行数
        # 替换路径中的反斜杠以适应不同操作系统
        df = pd.read_csv(file_path.replace('\\', '/'),
                         sep='\t',
                         encoding='utf-8',
                         nrows=sample_size)

        # 将数据转换为字典列表格式
        news_data = []
        for _, row in df.iterrows():
            news_item = {
                "post_id": row['post_id'],
                "post_text": row['post_text'],
                "user_id": row['user_id'],
                "username": row['username'],
                "image_id": row['image_id'],
                "timestamp": row['timestamp'],
                # 将数值标签转换为文字标签
                "label": "真新闻" if row['label'] == 1 else "假新闻"
            }
            news_data.append(news_item)

        print(f"成功加载前 {len(news_data)} 条新闻数据")
        return news_data

    except Exception as e:
        print(f"加载数据文件出错: {e}")
        return []


def analyze_sentiment(text: str, max_retries: int = 3) -> str:
    """
    分析文本情感倾向

    参数:
        text: 待分析文本
        max_retries: 最大重试次数

    返回:
        情感倾向 ("正面", "中性"或"负面")
    """
    # 构造情感分析prompt
    prompt = f"""请分析以下文本的情感倾向。只回答"正面"、"中性"或"负面"，不要解释。

文本内容：{text}"""

    for attempt in range(max_retries):
        try:
            # 调用模型生成响应
            response = ollama.generate(
                model='deepseek-r1:1.5b',
                prompt=prompt,
                options={'temperature': 0.2}  # 降低输出随机性
            )
            response_text = response['response'].strip()

            # 解析模型输出
            if "正面" in response_text:
                return "正面"
            elif "负面" in response_text:
                return "负面"
            return "中性"

        except Exception as e:
            print(f"情感分析尝试 {attempt + 1} 失败: {e}")
            if attempt < max_retries - 1:
                sleep(2)  # 等待后重试
            continue

    print(f"情感分析失败，已达到最大重试次数 {max_retries}")
    return "中性"  # 默认返回中性


def detect_fake_news_basic(news_list: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    基础新闻真实性检测

    参数:
        news_list: 新闻数据列表

    返回:
        (准确率, 结果列表)
    """
    correct = 0
    results = []

    # 基础检测prompt模板
    prompt_template = """请判断以下新闻的真实性。只回答"真新闻"或"假新闻"，不要解释。

新闻内容：{news_text}"""

    for news in tqdm(news_list, desc="基础真实性检测"):
        try:
            # 格式化prompt
            prompt = prompt_template.format(news_text=news["post_text"])

            # 调用模型
            response = ollama.generate(
                model='deepseek-r1:1.5b',
                prompt=prompt,
                options={'temperature': 0.2}
            )
            response_text = response['response'].strip()

            # 解析预测结果
            prediction = "真新闻" if "真新闻" in response_text else "假新闻"
            is_correct = (prediction == news["label"])
            correct += int(is_correct)

            # 保存结果
            result = news.copy()
            result.update({
                "prediction": prediction,
                "correct": is_correct,
                "method": "basic"
            })
            results.append(result)

        except Exception as e:
            print(f"处理新闻 {news['post_id']} 出错: {e}")
            continue

    # 计算准确率
    accuracy = correct / len(news_list) if news_list else 0
    return accuracy, results


def detect_fake_news_with_sentiment(news_list: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    结合情感分析的新闻真实性检测

    参数:
        news_list: 新闻数据列表

    返回:
        (准确率, 结果列表)
    """
    correct = 0
    results = []

    # 结合情感分析的prompt模板
    prompt_template = """请根据新闻内容和情感倾向判断真实性。只回答"真新闻"或"假新闻"，不要解释。

新闻内容：{news_text}
情感分析：{sentiment}"""

    for news in tqdm(news_list, desc="结合情感的真实性检测"):
        try:
            # 先进行情感分析
            sentiment = analyze_sentiment(news["post_text"])

            # 构造prompt
            prompt = prompt_template.format(
                news_text=news["post_text"],
                sentiment=sentiment
            )

            # 调用模型
            response = ollama.generate(
                model='deepseek-r1:1.5b',
                prompt=prompt,
                options={'temperature': 0.2}
            )
            response_text = response['response'].strip()

            # 解析预测结果
            prediction = "真新闻" if "真新闻" in response_text else "假新闻"
            is_correct = (prediction == news["label"])
            correct += int(is_correct)

            # 保存结果
            result = news.copy()
            result.update({
                "sentiment": sentiment,
                "prediction": prediction,
                "correct": is_correct,
                "method": "with_sentiment"
            })
            results.append(result)

        except Exception as e:
            print(f"处理新闻 {news['post_id']} 出错: {e}")
            continue

    # 计算准确率
    accuracy = correct / len(news_list) if news_list else 0
    return accuracy, results


def save_results_to_csv(results: List[Dict], file_path: str):
    """
    将结果保存为CSV文件

    参数:
        results: 结果数据
        file_path: 保存路径
    """
    try:
        df = pd.DataFrame(results)
        # 选择要保存的列
        columns = ['post_id', 'post_text', 'label', 'prediction', 'correct']
        if 'sentiment' in df.columns:
            columns.insert(3, 'sentiment')
        # 保存为UTF-8编码的CSV（带BOM以兼容Excel）
        df[columns].to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"结果已保存到 {file_path}")
    except Exception as e:
        print(f"保存结果失败: {e}")


def evaluate_results(basic_results: List[Dict], sentiment_results: List[Dict]):
    """
    评估并比较两种方法的性能

    参数:
        basic_results: 基础方法结果
        sentiment_results: 结合情感分析的结果
    """
    # 计算基础方法指标
    basic_correct = sum(1 for r in basic_results if r['correct'])
    basic_accuracy = basic_correct / len(basic_results) if basic_results else 0

    # 计算结合情感分析的方法指标
    sentiment_correct = sum(1 for r in sentiment_results if r['correct'])
    sentiment_accuracy = sentiment_correct / len(sentiment_results) if sentiment_results else 0

    print("\n=== 评估结果 ===")
    print(f"基础方法准确率: {basic_accuracy:.2%} ({basic_correct}/{len(basic_results)})")
    print(f"结合情感分析方法准确率: {sentiment_accuracy:.2%} ({sentiment_correct}/{len(sentiment_results)})")
    print(f"准确率变化: {sentiment_accuracy - basic_accuracy:+.2%}")

    # 分析情感分布
    if sentiment_results and 'sentiment' in sentiment_results[0]:
        sentiment_dist = pd.DataFrame(sentiment_results)['sentiment'].value_counts(normalize=True)
        print("\n情感分布:")
        print(sentiment_dist)


def main():
    """主执行函数"""
    # 确保Ollama服务运行
    ensure_ollama_running()

    # 加载数据（只读取前50条）
    data_path = r"C:\Users\Administrator\Desktop\posts_groundtruth.txt"
    news_data = load_news_data(data_path, sample_size=50)

    if not news_data:
        print("无法加载数据，程序退出")
        return

    # 基础真实性检测
    print("\n=== 基础真实性检测 ===")
    basic_acc, basic_results = detect_fake_news_basic(news_data)
    print(f"基础版准确率: {basic_acc:.2%}")
    save_results_to_csv(basic_results, "basic_results.csv")

    # 结合情感分析的真实性检测
    print("\n=== 结合情感分析的真实性检测 ===")
    sentiment_acc, sentiment_results = detect_fake_news_with_sentiment(news_data)
    print(f"结合情感分析的准确率: {sentiment_acc:.2%}")
    save_results_to_csv(sentiment_results, "sentiment_results.csv")

    # 评估结果
    evaluate_results(basic_results, sentiment_results)


if __name__ == "__main__":
    main()
    
    # ===================== 三. 云计算2 =====================
# -*- coding: utf-8 -*-
"""
Twitter安全分析专业版（解决无效词问题）
"""

import re
import os
from nltk.tokenize.punkt import PunktLanguageVars, PunktTrainer, PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ===== 1. 专业分词器配置 =====
class ProfessionalPunkt(PunktLanguageVars):
    """专业级分词规则"""
    _re_word_start = r"[a-zA-Z]"
    _re_non_word_chars = r"(?:[?!)\";}\]\*:@\'\({$$$$])"
    _re_urls = r"https?://\S+"  # 专门处理URL


# 使用高质量语料训练
train_text = """
Scientific reports show climate change impacts. Technology companies announce new innovations.
Economic indicators demonstrate steady growth patterns. Academic research reveals breakthrough findings.
"""
trainer = PunktTrainer()
trainer.train(train_text)
tokenizer = PunktSentenceTokenizer(trainer.get_params(), lang_vars=ProfessionalPunkt())

# ===== 2. 敏感词检测系统 =====
sensitive_terms = {
    'terror', 'isis', 'attack', 'weapon', 'bomb',
    'arabianblood', 'antiterror', 'violence'
}


def is_sensitive(text):
    """增强型敏感词检测"""
    text_lower = text.lower()
    # 同时检测完整词和词根
    return any(
        term in text_lower or
        re.search(rf'\b{term[:4]}\w*', text_lower)
        for term in sensitive_terms
    )


# ===== 3. 高质量数据加载 =====
def load_quality_tweets(filepath, n=200):
    """加载并过滤高质量内容"""
    quality_tweets = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                text = line.split('\t')[1] if '\t' in line else line.strip()
                if (len(text) > 20 and
                        not is_sensitive(text) and
                        re.search(r'\b[a-zA-Z]{4,}\b', text)):  # 至少包含4字母单词
                    quality_tweets.append(text)
                    if len(quality_tweets) >= n:
                        break
            except:
                continue
    print(f"加载高质量推文: {len(quality_tweets)}条")
    return quality_tweets


tweets = load_quality_tweets(r"C:\Users\Administrator\Desktop\posts_groundtruth.txt")

# ===== 4. 专业预处理流水线 =====
lemmatizer = WordNetLemmatizer()
stopwords = set([
    'http', 'https', 'com', 'www', 'rt', 'co', 'amp',
    'nhttps', 'awn'  # 添加发现的无效词
])


def professional_clean(text):
    """工业级清洗流程"""
    try:
        # 移除URL和无效字符
        text = re.sub(r'https?://\S+|@\w+|[^a-zA-Z\s]', ' ', text)

        # 分句和分词
        sentences = tokenizer.tokenize(text.lower())
        words = []
        for sent in sentences:
            for word in sent.split():
                # 严格词汇过滤
                if (len(word) >= 4 and  # 至少4个字母
                        word.isalpha() and  # 纯字母
                        word not in stopwords and  # 不在停用词表
                        not re.search(r'(\w)\1{2}', word)):  # 排除连续重复字符(如'awww')

                    # 专业词形还原
                    lemma = lemmatizer.lemmatize(word)
                    if len(lemma) >= 3:  # 还原后仍有效
                        words.append(lemma)
        return words
    except Exception as e:
        print(f"清洗异常: {str(e)}")
        return []


processed_data = [professional_clean(tweet) for tweet in tweets]
processed_data = [doc for doc in processed_data if len(doc) >= 5]  # 至少5个有效词

print(f"\n有效专业文档: {len(processed_data)}")
if processed_data:
    print("专业处理示例:", processed_data[0][:5])

# ===== 5. 专业建模 =====
if len(processed_data) >= 10:  # 提高最小文档要求
    # 构建高质量词典
    dictionary = corpora.Dictionary(processed_data)
    dictionary.filter_extremes(no_below=3, no_above=0.8, keep_n=5000)  # 更严格过滤

    # 验证词典质量
    print(f"\n词典质量检查:")
    print(f"总词条: {len(dictionary)}")
    print("高频有效词:", sorted(
        [(dictionary[id], cnt) for id, cnt in dictionary.cfs.items()],
        key=lambda x: -x[1]
    )[:5])

    # 构建语料库
    corpus = [dictionary.doc2bow(doc) for doc in processed_data]

    # 专业LDA训练
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=min(5, len(processed_data) // 3),
        passes=30,
        random_state=42,
        alpha='auto',
        eta=0.1,
        iterations=300
    )


    # ===== 6. 专业可视化 =====
    def pro_visualization():
        """工业级可视化"""
        try:
            # 交互式主题图
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
            pyLDAvis.save_html(vis, 'pro_topics.html')

            # 专业词云
            for i in range(lda_model.num_topics):
                try:
                    terms = lda_model.show_topic(i, topn=15)
                    word_freq = {}
                    valid_count = 0

                    for id, freq in terms:
                        word = dictionary[id]
                        if (word.isalpha() and
                                len(word) >= 4 and
                                word not in stopwords):
                            word_freq[word] = freq
                            valid_count += 1

                    if valid_count >= 5:  # 至少5个有效词
                        plt.figure(figsize=(12, 6), dpi=120)
                        wordcloud = WordCloud(
                            width=1000,
                            height=600,
                            background_color='white',
                            colormap='tab20',
                            collocations=False
                        ).generate_from_frequencies(word_freq)

                        plt.imshow(wordcloud)
                        plt.axis("off")
                        plt.title(f"Pro Topic {i}", pad=20)
                        plt.savefig(f"pro_topic_{i}.png", bbox_inches='tight', quality=95)
                        plt.close()
                        print(f"生成主题{i}词云（{valid_count}个有效词）")
                    else:
                        print(f"主题{i}有效词不足（{valid_count}）")
                except Exception as e:
                    print(f"主题{i}可视化错误:", str(e))
        except Exception as e:
            print("可视化系统错误:", str(e))


    pro_visualization()

    # ===== 7. 专业分析报告 =====
    print("\n=== 专业主题分析 ===")
    for i in range(lda_model.num_topics):
        try:
            terms = lda_model.show_topic(i, topn=10)
            keywords = []
            for id, _ in terms:
                word = dictionary[id]
                if (word.isalpha() and
                        len(word) >= 4 and
                        word not in stopwords):
                    keywords.append(word)

            if keywords:
                print(f"\nTopic {i}:")
                print(", ".join(keywords[:5]))
            else:
                print(f"\nTopic {i}: 无高质量关键词")
        except Exception as e:
            print(f"\nTopic {i}分析错误:", str(e))
else:
    print("\n警告: 专业级数据不足，需更多输入")

print("\n=== 生成文件 ===")
if len(processed_data) >= 10:
    print("- pro_topics.html (专业主题图)")
    print("- pro_topic_*.png (专业词云)")
print("注: 已跳过所有敏感内容和低质量数据")
