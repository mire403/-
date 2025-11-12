#%%
import numpy as np
import jieba
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# 设置中文字体和全局风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 负号正常显示
sns.set(style="whitegrid", font="SimHei")

#%%
def get_data():
    with open("./data/ham_data.txt", encoding="utf8") as ham_f, open("./data/spam_data.txt", encoding="utf8") as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()
        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()
        corpus = ham_data + spam_data
        labels = ham_label + spam_label
    return corpus, labels

#%%
def remove_empty_docs(corpus, labels):
    corpus_clean, labels_clean = [], []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            corpus_clean.append(doc)
            labels_clean.append(label)
    return corpus_clean, labels_clean

#%%
def tokenize_text(text):
    tokens = jieba.cut(text)
    return [t.strip() for t in tokens]

def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    clean_tokens = [pattern.sub('', token) for token in tokens if token.strip()]
    return ' '.join(clean_tokens)

with open("./data/stop_words.utf8", encoding="utf8") as f:
    stopwords = f.readlines()

def remove_stopwords(text):
    tokens = tokenize_text(text)
    clean_tokens = [t for t in tokens if t not in stopwords]
    return ''.join(clean_tokens)

def normalize_corpus(corpus):
    normalized = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized.append(text)
    return normalized

#%%
def bow_extractor(corpus):
    vectorizer = CountVectorizer(min_df=1)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def tfidf_extractor(corpus):
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

#%%
corpus, labels = get_data()
print("总的数据量:", len(labels))
corpus, labels = remove_empty_docs(corpus, labels)
train_corpus, test_corpus, train_labels, test_labels = train_test_split(corpus, labels, test_size=0.3, random_state=42)

#%%
norm_train_corpus = normalize_corpus(train_corpus)
norm_test_corpus = normalize_corpus(test_corpus)

bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
bow_test_features = bow_vectorizer.transform(norm_test_corpus)

tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

#%%
mnb_bow = MultinomialNB().fit(bow_train_features, train_labels)
mnb_tfidf = MultinomialNB().fit(tfidf_train_features, train_labels)
lr_tfidf = LogisticRegression(max_iter=1000).fit(tfidf_train_features, train_labels)

#%%
print("基于词袋模型的多项式朴素贝叶斯模型")
print("训练集得分：", mnb_bow.score(bow_train_features, train_labels))
print("测试集得分：", mnb_bow.score(bow_test_features, test_labels))

print("基于tfidf的多项式朴素贝叶斯模型")
print("训练集得分：", mnb_tfidf.score(tfidf_train_features, train_labels))
print("测试集得分：", mnb_tfidf.score(tfidf_test_features, test_labels))

print("基于tfidf的逻辑回归模型")
print("训练集得分：", lr_tfidf.score(tfidf_train_features, train_labels))
print("测试集得分：", lr_tfidf.score(tfidf_test_features, test_labels))

#%%
# ===================== 📊 学术可视化分析模块 =====================

def model_report(name, model, X_test, y_test):
    print(f"\n📘 模型分析报告：{name}")
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"精确率（Precision）: {precision:.4f}")
    print(f"召回率（Recall）: {recall:.4f}")
    print(f"F1 值: {f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title(f'{name} 混淆矩阵', fontsize=14, fontweight='bold')
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{name}_混淆矩阵.png', dpi=300)
    plt.close()

# 模型报告
model_report("BoW+NaiveBayes", mnb_bow, bow_test_features, test_labels)
model_report("TFIDF+NaiveBayes", mnb_tfidf, tfidf_test_features, test_labels)
model_report("TFIDF+LogisticRegression", lr_tfidf, tfidf_test_features, test_labels)

#%%
# 📈 各模型准确率对比
models = ['BoW+NB', 'TFIDF+NB', 'TFIDF+LR']
train_scores = [
    mnb_bow.score(bow_train_features, train_labels),
    mnb_tfidf.score(tfidf_train_features, train_labels),
    lr_tfidf.score(tfidf_train_features, train_labels)
]
test_scores = [
    mnb_bow.score(bow_test_features, test_labels),
    mnb_tfidf.score(tfidf_test_features, test_labels),
    lr_tfidf.score(tfidf_test_features, test_labels)
]

x = np.arange(len(models))
width = 0.35
plt.figure(figsize=(7,5))
plt.bar(x - width/2, train_scores, width, label='训练集', color='#4A90E2')
plt.bar(x + width/2, test_scores, width, label='测试集', color='#F5A623')
plt.xticks(x, models, fontsize=11)
plt.ylabel('准确率', fontsize=12)
plt.title('各模型训练/测试准确率对比', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('模型准确率对比柱状图.png', dpi=300)
plt.close()

#%%
# ROC曲线比较
plt.figure(figsize=(6,5))
for name, model, X in [
    ("BoW+NB", mnb_bow, bow_test_features),
    ("TFIDF+NB", mnb_tfidf, tfidf_test_features),
    ("TFIDF+LR", lr_tfidf, tfidf_test_features)
]:
    y_prob = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(test_labels, y_prob)
    auc_value = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc_value:.3f})')

plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel('假阳率 (FPR)', fontsize=12)
plt.ylabel('真阳率 (TPR)', fontsize=12)
plt.title('模型ROC曲线比较', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('模型ROC曲线比较.png', dpi=300)
plt.close()

#%%
# ☁️ 高频词云（区分色调）
def plot_wordcloud(corpus, title, filename, color):
    text = ' '.join(corpus)
    wc = WordCloud(
        font_path='simhei.ttf',
        background_color='white',
        width=800, height=600,
        colormap=color,
        max_words=200
    ).generate(text)
    plt.figure(figsize=(8,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

ham_texts = [norm_train_corpus[i] for i, label in enumerate(train_labels) if label == 1]
spam_texts = [norm_train_corpus[i] for i, label in enumerate(train_labels) if label == 0]

plot_wordcloud(ham_texts, "正常短信高频词云", "正常短信高频词云.png", "Blues")
plot_wordcloud(spam_texts, "垃圾短信高频词云", "垃圾短信高频词云.png", "Oranges")

print("\n✅ 可视化结果已生成，包含：")
print(" - 各模型混淆矩阵（蓝色方格）")
print(" - 模型准确率对比柱状图.png")
print(" - 模型ROC曲线比较.png")
print(" - 正常/垃圾短信高频词云.png")
print("所有图像已优化为论文/汇报风格。")
