# 基于多特征融合的句子语义相似度计算系统

这是一个面向自然语言处理课程作业的可用原型系统，用于计算两个句子的语义相似度。系统提供两种核心模型：规则多特征融合模型与监督学习融合模型。系统支持 Web 界面、命令行调用、Top-K 相似句检索、公开数据集评测和单元测试，核心算法使用纯 Python 标准库实现，可以离线运行。

## 功能特点

- 输入两个中文或英文句子，输出 0 到 1 之间的语义相似度分数。
- 同时提供规则多特征融合模型和监督学习融合模型，便于进行方法对比实验。
- 输出三种方法的对比结果：TF-IDF 词面基线、同义词归一化规则、多特征融合结果。
- 展示词级 TF-IDF、字符 n-gram TF-IDF、归一化 Jaccard、编辑相似度、否定极性惩罚和反义极性惩罚。
- 加入否定词和反义词检测，缓解“喜欢/不喜欢”“高/低”这类词面相似但语义相反的问题。
- 支持从候选句库中检索 Top-K 相似句，可对应问答匹配、智能客服和重复问题检测场景。
- 提供 `data/test_pairs.csv` 小型标注数据集，并用 Accuracy、Precision、Recall、F1 进行评测。
- 支持使用 LCQMC 标注数据训练 Logistic Regression 监督学习融合模型，自动学习特征权重。

## 项目结构

```text
.
├── app.py                         # Web 原型系统入口
├── cli.py                         # 两句相似度命令行入口
├── evaluate.py                    # 小型数据集评测脚本
├── evaluate_ml.py                 # 监督学习融合模型评测脚本
├── train_model.py                 # 监督学习融合模型训练脚本
├── compare_methods.py             # 规则模型与监督模型对比脚本
├── analyze_errors.py              # 错误案例分析脚本
├── retrieve.py                    # Top-K 相似句检索入口
├── examples.py                    # 批量示例
├── requirements.txt
├── README.md
├── docs/
│   ├── design.md                  # 系统设计与算法说明
│   ├── experiment.md              # 实验结果与错误分析
│   └── error_cases.csv            # 导出的误判样例
├── data/
│   ├── synonyms.json              # 近义词词典
│   ├── test_pairs.csv             # 评测数据集
│   └── candidates.txt             # 检索候选句库
├── nlp_similarity/
│   ├── __init__.py
│   ├── similarity.py              # 相似度融合、极性处理、Top-K 排序
│   ├── ml_model.py                # Logistic Regression 监督学习融合模型
│   ├── tokenizer.py               # 中英文混合分词
│   └── vectorizer.py              # TF-IDF 与余弦相似度
└── tests/
    └── test_similarity.py
```

## 快速开始

本项目不需要安装第三方依赖。确认本机已经安装 Python 3.10 或更高版本。

启动 Web 系统：

```bash
python app.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

命令行计算：

```bash
python cli.py "我喜欢自然语言处理" "我对NLP很感兴趣"
```

Top-K 相似句检索：

```bash
python retrieve.py "我对NLP很感兴趣" -k 3
```

运行小型数据集评测：

```bash
python evaluate.py
```

默认阈值为 `0.30`，当前内置数据集评测结果为 Accuracy `0.8250`、Precision `0.9375`、Recall `0.7143`、F1 `0.8108`。

下载并转换 LCQMC 公开数据集抽样：

```bash
python scripts/download_lcqmc_sample.py --sample-size 1000 --pool-size 5000
```

生成文件：

```text
data/lcqmc_sample.csv
```

然后运行公开数据集评测：

```bash
python evaluate.py --data data/lcqmc_sample.csv
```

如果 Hugging Face 无法联网访问，也可以先用浏览器或其他下载工具手动下载 LCQMC 的 `train.tsv` / `train.csv` / `train.jsonl`，再转换：

```bash
python scripts/convert_lcqmc_local.py 路径\train.tsv --sample-size 1000
python evaluate.py --data data/lcqmc_sample.csv
```

训练监督学习融合模型：

```bash
python train_model.py --data data/lcqmc_sample.csv
```

评测监督学习融合模型：

```bash
python evaluate_ml.py --data data/lcqmc_test_sample.csv --threshold 0.16
```

对比规则融合和监督学习融合：

```bash
python compare_methods.py --data data/lcqmc_test_sample.csv
```

导出错误案例：

```bash
python analyze_errors.py --data data/lcqmc_test_sample.csv --threshold 0.16
```

运行单元测试：

```bash
python -m unittest
```

## 算法简介

系统将句子相似度拆成多个互补特征：

1. 词级 TF-IDF 余弦相似度：关注句子的关键词重合与重要性。
2. 字符 n-gram TF-IDF 余弦相似度：缓解中文分词不准、短句表达变化等问题。
3. 近义词归一化：将“喜欢/热爱/感兴趣”“NLP/自然语言处理”等词映射到统一语义标签。
4. Jaccard 相似度：衡量归一化词集合的重合程度。
5. 编辑相似度：处理短文本中的局部改写和轻微差异。
6. 否定与反义极性惩罚：当两个句子核心内容相近但极性相反时降低最终分数。

最终基础融合分数为：

```text
fusion = 0.50 * word_tfidf + 0.15 * char_ngram_tfidf
       + 0.25 * normalized_jaccard + 0.10 * edit_similarity
```

若存在明显否定极性不一致，并且两个句子原本较相似，则使用：

```text
score = fusion * 0.40
```

若存在反义极性冲突，例如“高/低”“快/慢”“喜欢/讨厌”，继续乘以 `0.40`。否则：

```text
score = fusion
```

## 方法对比

系统会显式输出并评测多种方法，便于在报告中做对比实验：

| 方法 | 含义 |
| --- | --- |
| 方法一：TF-IDF 词面基线 | 基础词面相似 |
| 方法二：同义词归一化 + Jaccard | 加入简单语义规则 |
| 方法三：多特征融合相似度 | 最终系统结果 |
| 方法四：监督学习融合模型 | 使用 LCQMC 数据自动学习特征权重 |

在 LCQMC 测试集抽样 1000 条样本上，当前对比结果为：

| 方法 | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 规则多特征融合 | 0.5520 | 0.5199 | 0.9482 | 0.6716 |
| 监督学习融合 | 0.6680 | 0.6119 | 0.8551 | 0.7133 |

更多实验设置、特征权重解释和错误案例分析见 `docs/experiment.md`。

## 可扩展方向

- 接入 Sentence-BERT、SimCSE 等预训练语义向量模型。
- 扩充领域词典和同义词资源。
- 使用 LCQMC、ATEC、STS-B 等公开数据集进行更大规模评测。
- 使用 Flask、FastAPI 或 Streamlit 改造成更完整的服务。
