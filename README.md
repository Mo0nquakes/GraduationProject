# injlab

**面向大模型提示注入的检测器及其对抗鲁棒性研究** —— 实验代码骨架。

一句话定位：这不是一个"防护产品"，而是一个**实验台**。
它要回答的问题是：*检测器在真实攻击者手里，还能剩下多少防护力？*

---

## 30 秒跑通

```bash
# 零依赖，只需 Python 3.12
python3 -m injlab demo
```

不用装任何包就能看到完整实验结果：

```
检测器: keyword_stub   （阈值在训练集上校准）
========================================================
  阈值 threshold      = 0.5000
  TPR @ FPR=1%        = 0.697   (实际 FPR 0.000)
  TPR @ FPR=10%       = 0.697
  AUC                 = 0.848
  Precision / Recall / F1 = 1.000 / 0.697 / 0.821
  [注意] 测试集只有 18 条良性样本，FPR=1% 的估计不可靠（分辨率约 5.6%）。

========================================================
攻击谱系（ASR = 恶意样本被判成良性的比例，越高越脆弱）
========================================================
  attack          family            ASR     TPR      F1
  -----------------------------------------------------
  identity        sanity          0.303   0.697   0.821
  zero_width      obfuscation     0.970   0.030   0.059
  fullwidth       obfuscation     1.000   0.000   0.000
  homoglyph       obfuscation     0.788   0.212   0.350
  base64_wrap     obfuscation     1.000   0.000   0.000
```

**这张表就是你开题报告的预演**：一个在干净测试集上看似能用的规则检测器，
被四种**不需要任何机器学习知识**的字符级混淆直接打穿。
这不是危言耸听，是 10 行 base64 编码的结果。

> identity 的 ASR = 0.303 = 1 − TPR，是链路自检：恒等变换下"逃逸率"必须等于漏检率。
> 如果这个等式不成立，说明评测代码有 bug，后面所有数字都别信。

---

## 常用命令

| 命令 | 作用 |
|---|---|
| `python -m injlab demo` | 跑完整实验（数据→划分→训练→攻击→指标→落库→落盘） |
| `python -m injlab demo --detector tfidf-word` | 换成 TF-IDF + 逻辑回归（需 scikit-learn） |
| `python -m injlab demo --hard-negatives` | **A6 实验**：混入 3 条难负样本，看规则基线如何崩 |
| `python -m injlab truncation` | **截断攻击**：同一篇文档，朴素截断漏检、分块滑窗检出 |
| `python -m injlab rag` | **端到端 ASR**：恶意文档污染检索库，看密钥是否泄露 |
| `python -m injlab er` | 由 `schema.sql` 生成 Mermaid ER 图 |
| `python -m injlab verify` | 校验检测日志的哈希链（防篡改审计） |
| `python -m injlab serve` | 启 FastAPI 服务，`/docs` 有交互式界面 |

三个"便宜但出彩"的实验都能一条命令复现：

```bash
$ python -m injlab truncation
  文档长度               = 4011 词（近似 token）
  朴素截断  512 词        得分 = 0.5000   -> 漏检
  分块滑窗 256/128      得分 = 1.0000   -> 检出   (命中块起始词 3840)

$ python -m injlab rag
  [无防御  ] 文档分数 [0.0, 0.0] -> 泄漏
  [有检测器] 文档分数 [1.0, 0.5] -> 已拦截
```

---

## 目录结构

```
injlab/
├── config.py          全局配置（dataclass + 可选 YAML 覆盖）
├── labels.py          三分类标签空间：benign / direct / indirect
├── metrics.py         手写 TPR@FPR、AUC、ASR —— 安全指标 sklearn 里没有
├── chunking.py        分块滑窗推理 + 截断攻击用例生成
├── runner.py          实验运行器：阈值只在训练集校准，产物三件套落盘
├── cli.py             命令行入口
├── data/
│   ├── toy.py         冒烟数据（模板生成，含难负样本开关）
│   └── split.py       去重 + 分层划分
├── db/
│   ├── schema.sql     8 张表的唯一真源
│   ├── store.py       SQLite 封装 + 哈希链审计
│   └── er.py          由 schema 生成 ER 图
├── models/
│   ├── base.py        检测器协议（新增模型不用改其他代码）
│   ├── stub.py        关键词基线（零依赖，也是论文的规则对照组）
│   └── tfidf_lr.py    TF-IDF + LR（word / char_wb 两个变体）
├── attacks/
│   └── obfuscation.py 零宽 / 全角 / 同形字 / base64 + identity 自检
├── rag/
│   └── mini_rag.py    最小 RAG 靶子，量端到端 ASR
└── api/
    └── main.py        FastAPI：/detect 返回逐块分数供界面高亮
docs/
├── 威胁模型.md         ← 论文第 3 章的骨架，答辩必问的都在这里
└── er_diagram.md      自动生成，勿手改
```

---

## 实验矩阵与代码的对应关系

| 实验 | 内容 | 代码现状 |
|---|---|---|
| A1 基线对比 | 规则 / TF-IDF+LR / DistilBERT / 现成 guardrail | 前两个已实现，后两个待接 |
| A2 跨集泛化 | 训练 WildGuardMix，测 ToxicChat / PINT | 待接真实语料 |
| A3 FGSM-AT 失效 | embedding 对抗训练后，文本改写 ASR 几乎不降 | 待实现（寒假） |
| A4 攻击谱系 | 编码 / 截断 / 语义改写 / 多轮 | 4 种编码攻击 + 截断已实现 |
| A5 红队回路 AT | LLM 生成攻击 → 增量训练 → 再攻 | 待实现 |
| A6 难负样本消融 | 有无 hard negative 的差距 | `--hard-negatives` 已可演示 |
| A7 端到端 ASR | RAG 靶子上的真实泄漏率 | `python -m injlab rag` 已可演示 |

---

## 现在是"骨架"，哪些是假的

诚实标注，避免答辩时被问倒：

| 组件 | 现状 | 换成真的需要做什么 |
|---|---|---|
| 数据 | 手写模板，120 条级 | 接 WildGuardMix / ToxicChat / XSTest / PINT，做近重复去重 |
| 检测器 | 关键词 + TF-IDF | 加 DistilBERT 微调（手写训练循环，别用 Trainer 黑盒） |
| token 计数 | 用空格分词近似 | 换 tokenizer 的 `offset_mapping`，才能把块分数映射回字符区间 |
| RAG 检索 | 哈希向量 | 换 sentence-transformers 或 BM25 |
| 靶子模型 | 确定性假模型 | 换本地小模型（Qwen2.5-1.5B 级别）或 API |
| 攻击 | 4 种字符级混淆 | 加 LLM 改写攻击（PAIR 类），这才是有威胁的那一类 |
| 数据库 | SQLite 单机 | 够用，不建议换 |

**下一步优先级**（按"性价比 / 风险"排）：

1. 接真实语料 + 近重复去重 —— 没有这一步，后面全是自娱自乐。
2. 加 DistilBERT 基线 —— 让"传统 ML vs 预训练模型"的对比站得住。
3. 加 LLM 改写攻击 —— 证明字符级防御不够，把论文从"字符游戏"拉到"语义对抗"。
4. 红队回路 —— 这是 C3 贡献点，也是最能体现"对抗"的地方。

---

## 设计上几个刻意的决定

- **阈值只在训练集校准**。用测试集挑阈值，TPR@FPR 会系统性虚高，
  而这是审稿/答辩最常问的一个点。
- **判定规则全系统统一为 `score > threshold`**。`>=` 和 `>` 混用会让 ASR 与 TPR
  出现 1 个样本的系统性偏差，identity 攻击的自检就是拿来抓这个的。
- **报 TPR@FPR 而不是 Accuracy**。99% 良性的数据上，全判良性也有 99% 准确率。
- **同时报负样本量 N**。18 条负样本根本估不出 1% 的 FPR，分辨率只有 5.6%。
  把 N 一并报出来，是自证严谨的最低成本方式。
- **ER 图由 schema 生成**。手画的图一定会和代码脱节。
- **审计链 30 行实现**。改任何一条历史日志，`verify` 立刻报错。

---

## 时间线提醒

现在（2026-10）到考研（2026-12）之间**不要开新战线**。
这个骨架的意义就是：把"能不能跑通"这件事提前解决掉，
让你 12 月停摆、1 月回来时，面对的是一堆能跑的命令，而不是一个空仓库。

真正的重活（真实语料 + DistilBERT + LLM 改写攻击 + 红队回路）留给寒假。
