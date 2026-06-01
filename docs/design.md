# 系统设计说明

## 1. 任务定义

句子语义相似度计算是自然语言处理中的基础任务，目标是判断两个句子在语义层面是否表达相同或相近含义。它常用于问答匹配、信息检索、智能客服、重复问题检测和文本聚类。

本系统输入两个句子，输出一个 `[0, 1]` 区间的分数。分数越高，表示两个句子越相似。

## 2. 总体架构

系统分为五层：

1. 输入层：Web 表单、命令行参数、批量示例、候选句检索。
2. 预处理层：文本规范化、中英文混合分词、停用词过滤。
3. 表示层：词级 TF-IDF、字符 n-gram TF-IDF、同义词归一化词集合。
4. 决策层：规则多特征融合、否定/反义极性修正、监督学习融合和等级判断。
5. 评测层：小型标注数据集、LCQMC 公开数据集抽样、Accuracy、Precision、Recall、F1。

## 3. 核心算法

### 3.1 文本预处理

系统首先将英文转为小写，去除多余空格和标点。中文部分优先使用领域词典和同义词词典进行最长匹配，例如可以识别“自然语言处理”“语义相似度”等短语；无法匹配的中文片段使用单字切分，保证系统在没有第三方分词库时仍可运行。

### 3.2 语义归一化

系统维护 `data/synonyms.json` 作为轻量级知识资源，将同义或近义表达映射到统一形式。例如：

- `NLP`、`自然语言理解` 映射为 `自然语言处理`
- `喜欢`、`热爱`、`感兴趣` 映射为 `喜欢`
- `教师`、`讲师` 映射为 `老师`
- `餐厅`、`店`、`饭店` 映射为 `餐厅`

这样可以提高系统对改写句的识别能力。

### 3.3 TF-IDF 与余弦相似度

TF-IDF 用于衡量词项在句子中的重要程度。对于两个句子，系统分别构建稀疏向量，并使用余弦相似度计算向量夹角：

```text
cosine(A, B) = dot(A, B) / (||A|| * ||B||)
```

词级 TF-IDF 关注关键词语义重合，字符 n-gram TF-IDF 能捕捉局部片段相似性，对中文短句和轻微改写更鲁棒。

### 3.4 否定与反义极性处理

传统词面相似度难以区分“我喜欢这门课”和“我不喜欢这门课”。系统加入否定词检测机制，检测词包括：

```text
不、没、没有、不是、无法、不能、并非、讨厌、not、never、cannot、dislike
```

当两个句子本身相似度较高，但只有一个句子包含否定表达时，系统将最终分数乘以 `0.40`，从而降低语义相反句子的误判风险。

系统还加入了少量反义极性词组，例如“高/低”“快/慢”“好/差”“喜欢/讨厌”。当两个句子触发反义极性冲突时，继续乘以 `0.40` 作为惩罚。

### 3.5 多特征融合

最终基础融合分数为：

```text
fusion = 0.50 * word_tfidf + 0.15 * char_ngram_tfidf
       + 0.25 * normalized_jaccard + 0.10 * edit_similarity
```

融合后再根据否定和反义极性进行修正：

```text
score = fusion * negation_penalty * opposite_penalty
```

等级规则：

- `score >= 0.70`：高相似
- `0.42 <= score < 0.70`：中等相似
- `score < 0.42`：低相似

### 3.6 监督学习融合

为了避免人工固定权重过于主观，系统进一步加入监督学习融合模型。模型使用 LCQMC 标注数据训练 Logistic Regression 分类器，输入特征来自规则相似度模块：

```text
word_tfidf
char_ngram_tfidf
normalized_jaccard
edit_similarity
fusion_before_penalty
rule_score
negation_mismatch
opposite_mismatch
negation_penalty
opposite_penalty
```

模型形式为：

```text
p(y=1|x) = sigmoid(w·x + b)
```

其中 `x` 为特征向量，`w` 为从训练数据中学习得到的特征权重，`b` 为偏置项。该模型不依赖第三方机器学习库，由项目使用 Python 标准库实现。训练完成后，模型权重保存到 `data/model_weights.json`。

## 4. 方法对比实验

系统输出并评测四种方法的结果：

| 方法 | 含义 |
| --- | --- |
| 方法一：TF-IDF 词面基线 | 基础词面相似 |
| 方法二：同义词归一化 + Jaccard | 加入简单语义规则 |
| 方法三：多特征融合相似度 | 最终系统结果 |
| 方法四：监督学习融合模型 | 使用 LCQMC 标注数据自动学习特征权重 |

通过这种方式可以展示：只依赖词面相似度会受表达差异影响；加入同义词归一化后可以识别部分语义改写；多特征融合和极性修正能提供更稳定的最终判断；监督学习模型可以进一步减少人工权重设计的主观性。

## 5. 小型数据集评测

系统提供 `data/test_pairs.csv`，包含 40 组人工标注样例，每组包括：

```text
sentence_a,sentence_b,label
```

其中 `label=1` 表示相似，`label=0` 表示不相似。运行：

```bash
python evaluate.py
```

可以输出 Accuracy、Precision、Recall、F1 和混淆矩阵统计。当前默认阈值为 `0.30`，内置数据集上的当前结果为：

```text
Accuracy：0.8250
Precision：0.9375
Recall：0.7143
F1：0.8108
TP=15 FP=1 TN=18 FN=6
```

为了进一步使用公开数据集测试，项目提供 `scripts/download_lcqmc_sample.py`。该脚本会从 Hugging Face 的 `C-MTEB/LCQMC` 下载训练集样本，并转换为本项目的 CSV 格式：

```bash
python scripts/download_lcqmc_sample.py --sample-size 1000 --pool-size 5000
python evaluate.py --data data/lcqmc_sample.csv
```

如果当前网络无法访问 Hugging Face，可以手动下载 LCQMC 原始文件，再运行：

```bash
python scripts/convert_lcqmc_local.py 路径\train.tsv --sample-size 1000
python evaluate.py --data data/lcqmc_sample.csv
```

LCQMC 是中文问题匹配数据集，适合检验句子语义相似度系统在真实问句匹配任务上的效果。

## 6. 监督学习实验

使用 LCQMC 训练集抽样训练监督学习融合模型：

```bash
python train_model.py --data data/lcqmc_sample.csv
```

使用 LCQMC 测试集抽样评测监督学习模型：

```bash
python evaluate_ml.py --data data/lcqmc_test_sample.csv --threshold 0.16
```

对比规则融合和监督学习融合：

```bash
python compare_methods.py --data data/lcqmc_test_sample.csv
```

在 LCQMC 测试集抽样 1000 条样本上，当前结果为：

| 方法 | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 规则多特征融合 | 0.5520 | 0.5199 | 0.9482 | 0.6716 |
| 监督学习融合 | 0.6680 | 0.6119 | 0.8551 | 0.7133 |

实验结果说明，监督学习融合模型能够在保持较高召回率的同时提升准确率和 F1 值，说明从标注数据中学习特征权重比人工固定权重更适合公开数据集场景。

## 7. Top-K 相似句检索

系统还支持用户输入一个查询句，从候选句库中找出最相似的前 K 个句子：

```bash
python retrieve.py "我对NLP很感兴趣" -k 3
```

该功能可以对应问答匹配、智能客服、重复问题检测等实际应用场景。

## 8. 局限性

本系统为轻量级课程原型，虽然加入了监督学习融合，但特征仍主要来自 TF-IDF、词典规则和简单极性检测，不能真正理解复杂语义、指代、省略和长距离依赖。后续可以接入 Sentence-BERT、SimCSE、ERNIE 等预训练模型，将句子编码为深层语义向量，以提升泛化能力。
