# 代码核心功能说明
## 1.文本预处理与特征提取
### 文本预处理与分词过滤
**使用`jieba`对邮件文本分词，并过滤无效字符（如标点、数字）和长度为1的词汇。**
```python
def get_words(filename):
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)  # 过滤无效字符
            line = cut(line)  # jieba分词
            line = filter(lambda word: len(word) > 1, line)  # 过滤长度为1的词
            words.extend(line)
    return words
```
### 构建高频词库
**遍历151个邮件文件，统计所有词汇的出现频率，筛选出前100个高频词作为特征词库。**
```python
def get_top_words(top_num):
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    for filename in filename_list:
        all_words.append(get_words(filename))  # 遍历所有邮件生成词库
    freq = Counter(chain(*all_words))  # 统计词频
    return [i[0] for i in freq.most_common(top_num)]  # 返回前 top_num 个高频词
```
## 2.特征向量化
### 词频统计
**将每封邮件转换为一个100维向量，每个维度对应特征词库中某个词在邮件中的出现次数。**
```python
vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))  # 统计每个特征词的词频
    vector.append(word_map)
vector = np.array(vector)  # 转换为 NumPy 数组
```
### 标签标记
**为训练数据分配标签，标记垃圾邮件和普通邮件。前127封邮件标记为垃圾邮件（1），后24封标记为普通邮件（0）。**
## 3.模型训练
### 分类算法
**使用 `MultinomialNB`（多项式朴素贝叶斯）模型，基于词频向量和标签进行训练，学习垃圾邮件的词汇分布模式。**
```python
model = MultinomialNB()  # 初始化多项式朴素贝叶斯模型
model.fit(vector, labels)  # 使用词频向量和标签进行训练
```
## 4.新邮件分类
### 预测逻辑
**对新邮件进行同样的分词和过滤，生成其词频向量，输入训练好的模型进行分类，输出结果为“垃圾邮件”或“普通邮件”。**
```python
def predict(filename):
    words = get_words(filename)  # 预处理新邮件
    current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))  # 生成词频向量
    result = model.predict(current_vector.reshape(1, -1))  # 预测结果
    return '垃圾邮件' if result == 1 else '普通邮件'
```
# 高频词/TF-IDF两种特征模式及其切换方法
## 高频词特征模式
### 使用词袋模型
**使用`CountVectorizer`来提取文本中的词频特征。`CountVectorizer`会将文本转换为词频矩阵，其中每个词的频率表示其在文档中出现的次数。**
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["The quick brown fox jumps over the lazy dog.", "The quick brown fox is fast."]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())
```
## TF-IDF特征模式
### 使用TF-IDF模型
**使用`TfidfVectorizer`直接将文本转换为TF-IDF特征矩阵。`TfidfVectorizer`结合了词频（TF）和逆文档频率（IDF），能够更好地反映词在文档中的重要性。**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["The quick brown fox jumps over the lazy dog.", "The quick brown fox is fast."]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())
```
## 转换方法
### 从高频词切换到TF-IDF
**如果已经使用`CountVectorizer`提取了词频特征，可以通过`TfidfTransformer`将其转换为TF-IDF特征：**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus = ["The quick brown fox jumps over the lazy dog.", "The quick brown fox is fast."]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
print(tfidf.toarray())
```
### 从TF-IDF切换到高频词
**如果已经使用`TfidfVectorizer`提取了TF-IDF特征，可以通过`CountVectorizer`重新提取词频特征**
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["The quick brown fox jumps over the lazy dog.", "The quick brown fox is fast."]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
```