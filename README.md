# 📱 垃圾短信分类系统

## 📖 项目简介
本项目是一个针对**中文短信**的垃圾短信分类系统，基于机器学习方法实现自动识别。  
主要流程包含：文本清洗与分词 → 特征提取（BoW / TF-IDF）→ 模型训练（朴素贝叶斯、逻辑回归）→ 评估与可视化。  
系统可生成混淆矩阵、ROC 曲线、模型对比柱状图与词云，便于结果分析与报告展示。

---

## ⚙️ 环境要求
建议使用 Python 3.8 及以上，并安装以下依赖：

```bash
pip install numpy jieba matplotlib seaborn wordcloud scikit-learn
```

## 🚀 运行方法

准备数据（放在 data/ 目录）：
```bash
data/
├── ham_data.txt        # 正常短信，每行一条
├── spam_data.txt       # 垃圾短信，每行一条
└── stop_words.utf8     # 停用词表（每行一个词）
```

运行主程序（假设文件名为 spam_classifier.py）：
```bash
python spam_classifier.py
```

程序流程会自动执行：

读取与清洗数据（去空行、去除标点、分词、去停用词）

特征提取（BoW、TF-IDF）

划分训练/测试集（默认 test_size=0.3, random_state=42）

训练模型（MultinomialNB & LogisticRegression）

输出评估指标并保存可视化图像

运行结束后将在项目根目录生成（示例）图像文件：

BoW+NaiveBayes_混淆矩阵.png

![image](https://github.com/mire403/Spam-Message-Classification-System/blob/main/ham_spam/%E7%A4%BA%E4%BE%8B%E8%BE%93%E5%87%BA/BoW%2BNaiveBayes_%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png)

TFIDF+NaiveBayes_混淆矩阵.png

![image](https://github.com/mire403/Spam-Message-Classification-System/blob/main/ham_spam/%E7%A4%BA%E4%BE%8B%E8%BE%93%E5%87%BA/TFIDF%2BNaiveBayes_%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png)

TFIDF+LogisticRegression_混淆矩阵.png

![image](https://github.com/mire403/Spam-Message-Classification-System/blob/main/ham_spam/%E7%A4%BA%E4%BE%8B%E8%BE%93%E5%87%BA/TFIDF%2BLogisticRegression_%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png)

模型准确率对比柱状图.png

![image](https://github.com/mire403/Spam-Message-Classification-System/blob/main/ham_spam/%E7%A4%BA%E4%BE%8B%E8%BE%93%E5%87%BA/%E6%A8%A1%E5%9E%8B%E5%87%86%E7%A1%AE%E7%8E%87%E5%AF%B9%E6%AF%94%E6%9F%B1%E7%8A%B6%E5%9B%BE.png)

模型ROC曲线比较.png

![image](https://github.com/mire403/Spam-Message-Classification-System/blob/main/ham_spam/%E7%A4%BA%E4%BE%8B%E8%BE%93%E5%87%BA/%E6%A8%A1%E5%9E%8BROC%E6%9B%B2%E7%BA%BF%E6%AF%94%E8%BE%83.png)

正常短信高频词云.png

![image](https://github.com/mire403/Spam-Message-Classification-System/blob/main/ham_spam/%E7%A4%BA%E4%BE%8B%E8%BE%93%E5%87%BA/%E6%AD%A3%E5%B8%B8%E7%9F%AD%E4%BF%A1%E9%AB%98%E9%A2%91%E8%AF%8D%E4%BA%91.png)

垃圾短信高频词云.png

![image](https://github.com/mire403/Spam-Message-Classification-System/blob/main/ham_spam/%E7%A4%BA%E4%BE%8B%E8%BE%93%E5%87%BA/%E5%9E%83%E5%9C%BE%E7%9F%AD%E4%BF%A1%E9%AB%98%E9%A2%91%E8%AF%8D%E4%BA%91.png)



## 📊 效果展示（示例）

注：以下为示例格式和展示方式，最终数值依赖于你的数据集与预处理。

模型	特征提取方式	测试集准确率（示例）	说明
MultinomialNB	BoW	0.94	训练快速、资源占用低
MultinomialNB	TF-IDF	0.96	TF-IDF 能弱化高频停用词影响
LogisticRegression	TF-IDF	0.97	泛化性能更好（示例）

可视化输出包括：

混淆矩阵：展示真/假正例的分类情况

ROC 曲线与 AUC：衡量模型整体判别能力

训练/测试准确率柱状图：直观对比不同模型表现

词云：展示正常短信与垃圾短信的高频词差异

## 说明

若要调整模型或超参数，请修改 spam_classifier.py 中对应段落（如 CountVectorizer / TfidfVectorizer 参数、LogisticRegression(max_iter=...) 等）。

若数据量较大，建议在向量化和模型训练时使用更高效的稀疏存储或切换到基于深度学习的方法（如预训练语言模型）进行扩展。

## ⭐ Star Support

如果你觉得这个项目对你有帮助，请给仓库点一个 ⭐ Star！
你的鼓励是我继续优化此项目的最大动力 😊
