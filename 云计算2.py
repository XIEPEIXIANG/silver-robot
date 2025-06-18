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