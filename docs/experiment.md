# 实验说明与结果分析

本文档记录句子语义相似度计算系统的实验设置、复现命令、主要结果和错误案例分析，可作为课程报告实验部分的材料。

## 1. 实验目标

本实验希望验证两类方法在句子语义相似度任务上的效果：

1. 规则多特征融合：人工设定 TF-IDF、字符 n-gram、同义词 Jaccard、编辑相似度和极性惩罚的融合权重。
2. 监督学习融合：基于 LCQMC 标注数据训练 Logistic Regression，自动学习各特征权重。

## 2. 数据集

实验使用两类数据：

- 自建小型测试集：`data/test_pairs.csv`，共 40 条样例，用于验证系统功能。
- LCQMC 抽样数据：从 LCQMC 原始数据中抽样得到训练集、验证集和测试集，每个抽样文件 1000 条。

LCQMC 是中文问题匹配数据集，标签为二分类：

```text
label=1 表示两个句子语义相似
label=0 表示两个句子语义不相似
```

## 3. 评价指标

实验采用以下指标：

- Accuracy：准确率
- Precision：精确率
- Recall：召回率
- F1：精确率和召回率的调和平均

## 4. 复现命令

自建测试集评测：

```bash
python evaluate.py --data data/test_pairs.csv
```

规则多特征融合在 LCQMC 测试集抽样上的评测：

```bash
python evaluate.py --data data/lcqmc_test_sample.csv --threshold 0.40
```

训练监督学习融合模型：

```bash
python train_model.py --data data/lcqmc_sample.csv
```

监督学习融合模型评测：

```bash
python evaluate_ml.py --data data/lcqmc_test_sample.csv --threshold 0.16
```

方法对比：

```bash
python compare_methods.py --data data/lcqmc_test_sample.csv
```

错误案例导出：

```bash
python analyze_errors.py --data data/lcqmc_test_sample.csv --threshold 0.16
```

## 5. 实验结果

### 5.1 自建测试集

```text
Accuracy：0.8250
Precision：0.9375
Recall：0.7143
F1：0.8108
```

该结果说明系统在人工构造的典型样例上能够正确处理同义词归一化、否定极性和部分反义关系。

### 5.2 LCQMC 测试集抽样

在 LCQMC 测试集抽样 1000 条样本上，结果如下：

| 方法 | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 规则多特征融合 | 0.5520 | 0.5199 | 0.9482 | 0.6716 |
| 监督学习融合 | 0.6680 | 0.6119 | 0.8551 | 0.7133 |

从结果可以看出，规则多特征融合模型召回率较高，但容易将不相似句误判为相似，精确率较低。监督学习融合模型通过 LCQMC 标注数据自动学习特征权重，在 Accuracy、Precision 和 F1 上均有提升。

## 6. 特征权重分析

监督学习模型的主要特征权重如下：

| 特征 | 权重 |
| --- | ---: |
| normalized_jaccard | 2.3841 |
| word_tfidf | 1.7976 |
| fusion_before_penalty | 1.5385 |
| rule_score | 1.5093 |
| char_ngram_tfidf | 0.3767 |
| edit_similarity | -0.1338 |
| negation_mismatch | -0.3723 |
| opposite_penalty | -0.4850 |

可以看到，同义词归一化后的 Jaccard 相似度和词级 TF-IDF 对最终判断贡献较大，说明关键词语义重合仍是当前轻量模型的重要依据。否定和反义相关特征多为负向权重，说明极性冲突会降低相似判断概率。

## 7. 错误案例分析

运行 `analyze_errors.py` 后，会生成：

```text
docs/error_cases.csv
```

错误类型包括：

- FP：真实标签为不相似，但模型预测为相似。
- FN：真实标签为相似，但模型预测为不相似。

常见错误原因包括：

1. 词面高度重合但语义不同，例如两个句子包含大量相同词，但询问重点不同。
2. 表达差异较大但语义相同，例如同义表达超出当前词典覆盖范围。
3. 需要常识或上下文推理，例如简短问句中的省略、指代、语境差异。
4. LCQMC 中部分问句带有口语、错别字或网络表达，轻量分词和词典方法难以完全覆盖。

## 8. 实验结论

本系统从规则多特征融合升级到监督学习融合后，能够利用标注数据自动学习特征权重，在公开数据集抽样上取得更好的综合表现。实验同时表明，仅依赖 TF-IDF、词典和浅层规则仍难以完全解决深层语义匹配问题。后续可引入 Sentence-BERT、SimCSE 或中文预训练语言模型，将句子编码为深层语义向量，以进一步提升泛化能力。
