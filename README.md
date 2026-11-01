# injlab

**面向大模型提示注入的检测器及其对抗鲁棒性研究** —— 实验代码骨架。

一句话定位：这不是一个"防护产品"，而是一个**实验台**。
它要回答的问题是：*检测器在真实攻击者手里，还能剩下多少防护力？*

---

## 30 秒跑通

```bash
# 一次性：建虚拟环境（系统自带的 python3 是 3.9.6，太老，别用）
bash scripts/setup_venv.sh

# 每次开新终端：
source .venv/bin/activate

# 跑（keyword 检测器零依赖，装了 sklearn 才能用 tfidf 两版）
python -m injlab demo
```

> VS Code 用户：打开项目后直接点右上角三角即可，仓库已带 `.vscode/settings.json`，
> 会把解释器和 Code Runner 都指到 `.venv/bin/python`。没生效的话点右下角解释器手动选 `.venv/bin/python`。

不用装任何包就能看到完整实验结果（数字随数据规模变化）：

```
检测器: keyword_stub   （阈值在训练集上校准）
========================================================
  阈值 threshold      = 0.5000
  TPR @ FPR=1%        = 0.778   (实际 FPR 0.000)
  TPR @ FPR=10%       = 0.778
  AUC                 = 0.889
  Precision / Recall / F1 = 1.000 / 0.778 / 0.875
  [注意] 测试集只有 9 条良性样本，FPR=1% 的估计不可靠（分辨率约 11.1%）。

========================================================
攻击谱系（ASR = 恶意样本被判成良性的比例，越高越脆弱）
========================================================
  attack          family            ASR     TPR      F1
  -----------------------------------------------------
  identity        sanity          0.222   0.778   0.875
  zero_width      obfuscation     0.889   0.111   0.200
  fullwidth       obfuscation     1.000   0.000   0.000
  homoglyph       obfuscation     0.788   0.212   0.350
  base64_wrap     obfuscation     1.000   0.000   0.000
```

**这张表就是你开题报告的预演**：一个在干净测试集上看似能用的规则检测器，
被四种**不需要任何机器学习知识**的字符级混淆直接打穿。
这不是危言耸听，是 10 行 base64 编码的结果。

> identity 的 ASR = 0.222 = 1 − TPR，是链路自检：恒等变换下"逃逸率"必须等于漏检率。
> 如果这个等式不成立，说明评测代码有 bug，后面所有数字都别信。

---

## 常用命令

| 命令 | 作用 |
|---|---|
| `python -m injlab demo` | 跑完整实验（数据→划分→训练→攻击→指标→落库→落盘） |
| `python -m injlab demo --detector tfidf-word` | 换成 TF-IDF + 逻辑回归（需 scikit-learn） |
| `python -m injlab demo --detector distilbert` | 换成 DistilBERT 微调（需 torch + transformers，A1 第三列） |
| `python -m injlab demo --hard-negatives` | **A6 实验**：混入 3 条难负样本，看规则基线如何崩 |
| `python -m injlab build-data` | **拉真实语料**：下载→清洗→去重→配平→切分→落盘→入库 |
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

## 数据集构建（build-data）

前面所有 demo 用的都是模板生成的玩具数据。`build-data` 才是真正接公开语料的入口：

```bash
# 默认：deepset/prompt-injections + 合成间接注入（离线缓存可重跑）
python -m injlab build-data --limit 250 --no-db

# 只看统计，不落盘
python -m injlab build-data --limit 250 --dry-run

# 用缓存 + 写数据库（dataset_version 会记下 manifest 的 sha256）
python -m injlab build-data --limit 250

# 显式点名多个源（下面这些源网络好的时候才成功，见"已知问题"）
python -m injlab build-data --sources deepset,pint,wildguardmix,toxicchat --limit 800
```

产物在 `data/processed/<时间戳>_<实际用到的源>/`：

| 文件 | 内容 |
|---|---|
| `train.jsonl` / `test.jsonl` | 每行 `text / label / source / split / is_hard_negative / group / meta` |
| `manifest.json` | 参数、去重统计、标签分布、来源分布、两个文件的 sha256 |

**管线做了什么**：

1. **清洗**（`prep.clean_text`）：Unicode NFKC + 剥零宽/格式字符 + 折叠空白。注意这里只清洗用于**建库和去重**，训练用的还是原文——否则等于提前替模型做了输入规范化防御，实验就证明不了防御有用。
2. **去重**（`prep.dedup`）：精确哈希 + MinHash 近似去重（8 band × 4 row，shingle=5，32 个哈希）。公开语料里同一条注入模板会以几十种微小变体出现，不去重会把测试集污染成训练集。
3. **分组切分**（`prep.group_split`）：同一个 `group` 的样本只会落在训练或测试的一侧。没有 group 就退化成按 `dedup_key` 切。
4. **入库**：`dataset_version` 记录 manifest 里的 sha256，`sample` 表里每条也存 `sha256`，配合唯一索引 `ux_sample_hash` 防止重复入库。

**一个必须写进论文的标注决定**：WildGuardMix 里 `prompt_harm_label == "harmful"` 的样本被标成 **benign + is_hard_negative=True**。本课题判的是"这段文本是否试图覆盖系统指令"，不是"内容是否有害"——一个暴力请求并没有覆盖任何指令，注入检测器对它报警就是误报。这条区分让"恶意内容"变成天然的难负样本，也正是 A6 实验要用的东西。

**已知问题（都会写进论文的威胁部分）**：

| 源 | 状况 |
|---|---|
| `deepset/prompt-injections` | 可用，662 行，只有 benign / direct 两类 |
| `synth`（合成间接注入） | 模板生成，只为补 `indirect_injection` 类别空缺，**论文必须单独报剔除后的指标** |
| `walledai/XSTest` | **gated**，需 `huggingface-cli login` 并同意条款；其他 XSTest 镜像经 hf-mirror 会卡住 |
| `lmsys/toxic-chat` | 首包很大，镜像偶发 `ReadTimeout` |
| `allenai/wildguardmix` | 体积大，仅在网络好时拉 |
| `Lakera/pint-benchmark` | HF 上不稳，失败时退到 GitHub 原始文件 |

`huggingface.co` 在本机直连超时，代码默认走 `HF_ENDPOINT=https://hf-mirror.com`。拉过一次的源会缓存在 `data/raw/<source>.jsonl`，之后加 `--limit` 不会再出网。

## 三条评审意见对应到哪些代码

有人指出这类课题最容易翻车的三个地方，下面是逐条对应的实现（不是口头承诺）：

| 评审意见 | 代码位置 | 怎么体现 |
|---|---|---|
| ① 攻击"绕过检测器"不等于"真的有效"，还得证明意图保留、以及被保护模型真的照做 | `injlab/eval/intent.py`、`injlab/runner.py` 的 `ASR*`、`python -m injlab rag` | `ASR` 报全量，`ASR*` 只在"规范化后指令仍可读"的攻击上统计；`rag` 命令打印 `检测器放行 ≠ 攻击成功` 的对照表 |
| ② 不能靠"全都拦截"刷好看的检测率 | `injlab/runner.py: overdefense_block()` | 分别报良性整体 FPR、良性难负样本 FPR、**含触发词的良性 FPR** 三档 |
| ③ 训练/测试不能混入同模板的近似改写，否则只是背样本 | `injlab/data/prep.py: group_split()` + `leakage_report()` | 同 group 不跨侧；切分后再做一次 MinHash 两两比对，结果写进 `manifest.json` 的 `leakage` 字段 |

三者的共同点：**默认往坏里报**。判不了意图的攻击算"未知"而不是"有效"；
一条有效攻击都没有时 `ASR*` 返回 `n/a` 而不是 0；发现近似重复直接打警告。

## DistilBERT 基线（A1 的第三列）

```bash
pip install torch transformers          # 约 1GB，装一次
python -m injlab demo --detector distilbert        # 朴素截断 256 token
python -m injlab demo --detector distilbert-chunk  # 分块滑窗取最大分
```

`injlab/models/distilbert.py` 用手写训练循环，没用 `Trainer`——因为要能直接拿 logits、
自己控制 padding。三个 macOS/MPS 的坑已在代码里处理：

1. MPS 不支持 float64：softmax 后显式 `.float()`。
2. `DataLoader(num_workers>0)` 在 macOS 会 spawn 子进程而崩：固定 `num_workers=0`。
3. `pin_memory` 在 MPS 上无效：固定 `False`。

`fpr_target` 对它没有作用（神经网络不按分位数定阈值），阈值统一由 `runner` 在训练集上校准——
这一点全校统一，否则各检测器的实验不可比。


## 前端（可视化实验台）

```bash
python -m injlab serve          # 然后打开 http://127.0.0.1:8000
```

五个页签：**检测台**（分数/阈值/逐片段高亮/句级扫描）、**一键剧本**（七步论证链，
内容每轮随机、顺序固定、seed 可重放）、**攻击谱系**（每种混淆的变换样子 + 一键试）、
**实验对比**（A1 图表 + 运行记录）、**审计链**（哈希链校验 + 库统计）。

几个刻意的设计：

- **完全离线**：CSS/JS 全内联进 `injlab/web/index.html`，不引任何 CDN；图表用
  matplotlib 预生成 PNG 存 `data/charts/`，前端只 `<img>`。断网可用。
- **内存**：主进程只常驻三个轻量检测器（关键词 / TF-IDF word / TF-IDF char，几十 MB）。
  guardrail 与 DistilBERT 一律起子进程、用完即退，主进程永不持有模型权重——
  这台机器只有 8GB，神经网络模型同进程共存会把系统压垮。
- **`?print=1`**：去掉按钮、页签、阴影，四个页面全部展开，专门给论文截图用。
- **`◐ 深色`**：右上角切换深浅主题，选择记在 localStorage。
- **剧本里的"③ 混淆绕过"用的攻击不是随机的**：从最近一次 A1 结果里取该检测器
  ASR* 最高的那一种（最坏情况分析），页面上也这么写。没有 A1 结果时退回全角。
- **⑦ 中文载荷（零样本）会故意失败**：训练语料是英文，中文注入目前漏检。
  演示里主动暴露这个缺口，比藏起来好。

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
| A2 跨集泛化 | 训练 deepset，测 PINT / ToxicChat | 数据源已接（`build-data --sources`），实验脚本待写 |
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
| 数据 | **已接真实语料**（`build-data`）：deepset 662 行可用；WildGuardMix/ToxicChat/XSTest/PINT 是可选源，网络与 gated 限制见上文 | indirect 类目前靠合成模板补，需找 BIPIA 之类真实间接注入语料替换 |
| 检测器 | 关键词 + TF-IDF | 加 DistilBERT 微调（手写训练循环，别用 Trainer 黑盒） |
| token 计数 | 用空格分词近似 | 换 tokenizer 的 `offset_mapping`，才能把块分数映射回字符区间 |
| RAG 检索 | 哈希向量 | 换 sentence-transformers 或 BM25 |
| 靶子模型 | 确定性假模型 | 换本地小模型（Qwen2.5-1.5B 级别）或 API |
| 攻击 | 4 种字符级混淆 | 加 LLM 改写攻击（PAIR 类），这才是有威胁的那一类 |
| 数据库 | SQLite 单机 | 够用，不建议换 |

**下一步优先级**（按"性价比 / 风险"排）：

1. ~~接真实语料 + 近重复去重~~ —— 已完成（`python -m injlab build-data`）。下一个缺口是**真实的间接注入语料**。
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
