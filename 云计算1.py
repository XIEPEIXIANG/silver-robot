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