"""真实语料源适配器。

设计原则（这几条决定了后面论文能不能站得住）：

1. **每个源只负责"吐出一批 Record"，不做切分、不去重、不归一化。**
   那些是 prep.py 的事。职责分开，是因为"数据从哪来"和"数据怎么处理"
   在论文里要分别论证。

2. **正类和负类必须来自不同的语料。** 如果一个数据集里既有注入又有良性、
   还来自同一批标注者，模型学到的是"标注者习惯"而不是"注入"。

3. **列名用候选列表探测。** HuggingFace 上的数据集经常改列名，
   写死 `row["prompt"]` 第二天就会崩，而崩在答辩前一天是最糟的。

4. **网络默认走 hf-mirror。** 这台机器上 huggingface.co 直连 8 秒超时，
   hf-mirror.com 两秒返回。这是环境事实，不是偏好。

标签口径（重要，写论文时必须照这个说）：

    direct_injection    用户或攻击者直接输入的指令覆盖/越狱
    indirect_injection  藏在"被检索的第三方内容"里的指令
    benign              既不是上面两种的普通文本

**一个反直觉但正确的决定**：WildGuardMix 里的 harmful prompt（"怎么制作
炸弹"这类）在我们这个任务里算 **benign + hard=True**。

理由是：本课题的检测目标不是"内容是否有害"，而是"这段文本是不是在
覆盖系统指令"。一个暴力请求并没有试图覆盖任何指令，注入检测器对它报警
就是误报。把这类样本当良性难负样本，检验的正是检测器能不能区分
"恶意内容"和"恶意指令"——这是整篇论文最容易被评审认可的一个设计。
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..labels import BENIGN, DIRECT, INDIRECT

DEFAULT_ENDPOINT = "https://hf-mirror.com"
RAW_CACHE = "data/raw"


@dataclass
class Record:
    """一条待入库的样本。"""

    text: str
    label: str
    source: str
    hard: bool = False          # 是否难负样本（良性但容易被误判）
    group: str = ""             # 分组键：同一会话/同一模板的样本不跨集合
    meta: dict[str, Any] = field(default_factory=dict)


def _pick(row: dict, names: Iterable[str], default: Any = None) -> Any:
    """按候选列名依次取第一个非空值。数据集改列名时不至于崩。"""
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return default


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return "\n".join(str(x) for x in v)
    return str(v)


def _hf_rows(dataset_id: str, config: str | None = None,
             split: str = "train", limit: int | None = None,
             endpoint: str | None = None) -> list[dict]:
    """从 HuggingFace（或其镜像）流式取若干行。必须在 import datasets 前设好 HF_ENDPOINT。"""
    endpoint = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # 镜像偶尔很慢，默认超时 10s 不够，拉到 60s
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "缺少依赖: datasets\n请执行: .venv/bin/pip install datasets huggingface_hub"
        ) from e

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset_id, **kwargs)

    rows: list[dict] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        rows.append(dict(row))
    return rows


def _hf_first(ids, config=None, split="train", limit=None, endpoint=None):
    """按顺序尝试多个数据集 id，返回第一个能用的。

    HuggingFace 上同一个公开数据集常有多个镜像，其中一部分是 gated
    （需要登录 + 同意条款）或已删除。写死一个 id 迟早会挂在 gated 上。
    """
    errors = []
    for did in ids:
        try:
            rows = _hf_rows(did, config=config, split=split,
                            limit=limit, endpoint=endpoint)
            if rows:
                print(f"    [hf] 使用 {did}")
                return rows, did
            errors.append(f"{did}: 空数据集")
        except Exception as e:
            msg = str(e)
            if "gated" in msg.lower():
                errors.append(f"{did}: gated（需要登录 HF 并同意条款）")
            else:
                errors.append(f"{did}: {type(e).__name__}")
    raise RuntimeError("所有候选都失败 -> " + " | ".join(errors))


def _cached_jsonl(source: str, loader: Callable[[], list[Record]],
                  limit: int | None = None) -> list[Record]:
    """把拉取结果缓存成本地 jsonl，二次运行不再联网。"""
    path = os.path.join(RAW_CACHE, f"{source}.jsonl")
    have: list[Record] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                have.append(Record(**{**d, "meta": d.get("meta") or {}}))
        # 缓存够用就直接切片返回；不够则重新拉（避免 limit 变大后拿到旧的小缓存）
        if not limit or len(have) >= limit:
            return have[:limit] if limit else have
    recs = loader()
    os.makedirs(RAW_CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    return recs


# --------------------------------------------------------------------------
# 各源适配器
# --------------------------------------------------------------------------

def load_fixture(limit: int | None = None, **_) -> list[Record]:
    """离线冒烟源：不联网、不依赖 datasets，保证 build-data 永远能跑通。"""
    from .toy import build

    rows = build(n_per_class=min(limit or 30, 40), include_hard=True)
    out = []
    for r in rows:
        out.append(Record(
            text=r["text"], label=r["label"], source="fixture",
            hard=bool(int(r.get("is_hard_negative", r.get("hard", 0)) or 0)),
            group=r.get("group", ""),
            meta={"synthetic": True},
        ))
    return out


def load_deepset(limit: int | None = None, endpoint: str | None = None, **_) -> list[Record]:
    """deepset/prompt-injections —— 规模小但标注干净的真实注入语料。

    字段: text, label(0=良性 / 1=注入)
    组键用 text 前 40 字，避免同一段的变体跨集合。
    """
    rows, _used = _hf_first(["deepset/prompt-injections"], split="train",
                            limit=limit, endpoint=endpoint)
    out = []
    for r in rows:
        text = _as_text(_pick(r, ["text", "prompt", "input"]))
        y = _pick(r, ["label", "is_injection", "injection"], 0)
        is_inj = int(y) == 1 if str(y).isdigit() else str(y).lower() in ("1", "true", "injection")
        out.append(Record(
            text=text,
            label=DIRECT if is_inj else BENIGN,
            source="deepset",
            hard=False,
            group=text[:40],
        ))
    return out


def load_toxicchat(limit: int | None = None, endpoint: str | None = None, **_) -> list[Record]:
    """lmsys/toxic-chat —— 真实用户对话。

    jailbreaking=1 → direct_injection（越狱就是指令覆盖）
    toxicity=1 但 jailbreaking=0 → benign + hard（有害内容但不是注入，检验区分能力）
    其余 → benign
    """
    try:
        rows = _hf_rows("lmsys/toxic-chat", config="toxicchat0124", split="train",
                        limit=limit, endpoint=endpoint)
    except Exception:
        rows, _used = _hf_first(["lmsys/toxic-chat"], split="train",
                                limit=limit, endpoint=endpoint)
    out = []
    for r in rows:
        text = _as_text(_pick(r, ["user_input", "text", "prompt"]))
        if not text.strip():
            continue
        tox = int(_pick(r, ["toxicity"], 0) or 0)
        jail = int(_pick(r, ["jailbreaking"], 0) or 0)
        conv = str(_pick(r, ["conversation_id", "conv_id"], ""))
        if jail:
            out.append(Record(text=text, label=DIRECT, source="toxicchat",
                              group=conv or text[:40], meta={"toxicity": tox}))
        else:
            out.append(Record(text=text, label=BENIGN, source="toxicchat",
                              hard=bool(tox), group=conv or text[:40],
                              meta={"toxicity": tox}))
    return out


def load_wildguardmix(limit: int | None = None, endpoint: str | None = None, **_) -> list[Record]:
    """allenai/wildguardmix —— 大规模安全语料，主要贡献良性池与硬负样本。

    prompt_harm_label: "unharmful" → benign
                        "harmful"  → benign + hard（有害内容但不是注入，见模块文档）
    注意：不要把它当注入正类。它是"内容有害性"标注，不是"指令覆盖"标注。
    """
    rows, _used = _hf_first(["allenai/wildguardmix"], config="wildguardtrain",
                            split="train", limit=limit, endpoint=endpoint)
    out = []
    for r in rows:
        text = _as_text(_pick(r, ["prompt", "text", "input"]))
        if not text.strip():
            continue
        lab = str(_pick(r, ["prompt_harm_label", "harm_label", "label"], "")).lower()
        harmful = lab.startswith("harmful") or lab in ("1", "true")
        out.append(Record(
            text=text,
            label=BENIGN,
            source="wildguardmix",
            hard=harmful,
            group=text[:40],
            meta={"prompt_harm_label": lab, "subcategory": _pick(r, ["subcategory"], "")},
        ))
    return out


def load_xstest(limit: int | None = None, endpoint: str | None = None, **_) -> list[Record]:
    """walledai/XSTest —— 专门构造的"看起来危险其实安全"的提示，天然难负样本。

    这是全项目最值钱的良性来源：它能直接证伪"靠关键词（kill/porn/bomb）
    做检测"的路子。
    """
    rows, _used = _hf_first(
        ["Paul/XSTest", "AlignmentResearch/XSTest", "purpcode/XSTest", "walledai/XSTest"],
        split="train", limit=limit, endpoint=endpoint,
    )
    out = []
    for r in rows:
        text = _as_text(_pick(r, ["prompt", "text"]))
        if not text.strip():
            continue
        lab = str(_pick(r, ["label", "safety"], "")).lower()
        out.append(Record(
            text=text,
            label=BENIGN,
            source="xstest",
            hard=(lab in ("safe", "0", "false", "")),
            group=text[:40],
            meta={"type": _pick(r, ["type", "category"], ""), "focus": _pick(r, ["focus"], "")},
        ))
    return out


def load_pint(limit: int | None = None, endpoint: str | None = None, **_) -> list[Record]:
    """Lakera PINT —— 业界实际在用的提示注入基准，覆盖多种注入家族。

    先试 HuggingFace 镜像，失败再退回 GitHub 原始文件。
    """
    rows: list[dict] = []
    try:
        rows, _used = _hf_first(
            ["Lakera/pint-benchmark", "lakera/pint-benchmark"],
            split="train", limit=limit, endpoint=endpoint,
        )
    except Exception:
        print("    [pint] HF 不可用，改用 GitHub 原始文件")
        rows = _pint_from_github(limit)

    out = []
    for r in rows:
        text = _as_text(_pick(r, ["text", "prompt", "input", "content"]))
        if not text.strip():
            continue
        # PINT 的 label 语义：1/True 表示"这段是注入"
        y = _pick(r, ["label", "is_injection", "injected"], 0)
        is_inj = (str(y).lower() in ("1", "true", "yes")) or y is True
        cat = str(_pick(r, ["category", "family", "attack_type"], "")).lower()
        indirect = any(k in cat for k in ("indirect", "rag", "document"))
        out.append(Record(
            text=text,
            label=(INDIRECT if indirect else DIRECT) if is_inj else BENIGN,
            source="pint",
            hard=(cat in ("benign", "safe", "clean")),
            group=text[:40],
            meta={"category": cat},
        ))
    return out


def _pint_from_github(limit: int | None) -> list[dict]:
    url = ("https://raw.githubusercontent.com/lakeraai/pint-benchmark/"
           "main/benchmark/data/example-dataset.yaml")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PINT 拉取失败（HF 与 GitHub 都不通）: {e}") from e

    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover
        return []
    data = yaml.safe_load(raw) or {}
    items = data if isinstance(data, list) else next(
        (v for v in data.values() if isinstance(v, list)), [])
    return items[:limit] if limit else items


def load_aegis(limit: int | None = None, endpoint: str | None = None, **_) -> list[Record]:
    """NVIDIA AEGIS 内容安全语料 —— **只当良性难负样本用**。

    这是 A2 里最有信息量的一个目标集：里面是真实的"有害内容请求"
    （写诈骗信、怎么伤害某人……），它们**没有任何一句在试图覆盖系统指令**。
    一个只学"看起来坏"的检测器会在这里误报一片，而正确的行为是一条都不报。

    所以它测的不是召回，是**误报**——正好对上评审的第 2 条意见。
    """
    rows, _used = _hf_first(
        ["nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
         "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"],
        split="train", limit=limit, endpoint=endpoint,
    )
    out = []
    for r in rows:
        text = _as_text(_pick(r, ["prompt", "text"]))
        if len(text.strip()) < 8:
            continue
        lab = str(_pick(r, ["prompt_label", "label"], "")).lower()
        out.append(Record(
            text=text,
            label=BENIGN,               # 全部良性：这是误报压力测试集
            source="aegis",
            hard=True,                  # 有害内容≠覆盖指令，正是难负样本
            group=text[:40],
            meta={"aegis_label": lab,
                  "harm_type": str(_pick(r, ["prompt_harm_type", "harm_type"], ""))},
        ))
    return out


def load_synth_indirect(limit: int | None = None, **_) -> list[Record]:
    """合成间接注入：补 indirect_injection 类的缺口。

    公开语料里"文档里藏指令"这类样本极少（BIPIA 有，但授权与格式都麻烦），
    而它恰恰是本课题的重点。这里用模板生成，**必须在论文里单独标注为合成数据**，
    并且报告"剔除合成数据后"的指标，否则等于自己给自己出题自己答。
    """
    from .toy import build

    rows = build(n_per_class=min(limit or 40, 40), include_hard=False)
    return [
        Record(text=r["text"], label=INDIRECT, source="synth",
               group=r["text"][:40], meta={"synthetic": True})
        for r in rows
        if r["label"] == INDIRECT and str(r["text"]).strip()
    ]


SOURCES: dict[str, tuple[Callable[..., list[Record]], str]] = {
    "fixture": (load_fixture, "离线合成数据，冒烟用，不联网"),
    "deepset": (load_deepset, "真实注入语料（662 条，二分类标注）"),
    "pint": (load_pint, "Lakera PINT 提示注入基准"),
    "toxicchat": (load_toxicchat, "真实用户对话，越狱标记 → 正类"),
    "wildguardmix": (load_wildguardmix, "大规模安全语料，贡献良性池与硬负样本"),
    "xstest": (load_xstest, "看起来危险其实安全 → 天然难负样本"),
    "aegis": (load_aegis, "内容安全语料，全良性 → 误报压力测试"),
    "synth": (load_synth_indirect, "合成间接注入，补 indirect 类缺口（论文需单独标注）"),
}

# 默认组合只放"实测能稳定下载"的源。
#
# 为什么不把 xstest / toxicchat / wildguardmix 放默认：
#   - walledai/XSTest 是 gated 数据集，必须先 `huggingface-cli login`
#   - 其他 XSTest 镜像经 hf-mirror 下载会卡住（实测 4 分钟取不到 50 行）
#   - toxicchat 首包很大，镜像偶发 ReadTimeout
# 它们都保留为可选源，网络好的时候用 --sources 显式加进来。
#
# synth 是模板生成的间接注入，用来补 indirect_injection 这个类别的空缺。
# 它进默认组合是为了让三分类一开始就能跑；论文里必须把合成数据单独的
# 指标和"剔除合成数据后"的指标分开报。
DEFAULT_SOURCES = ("deepset", "synth")


def load(names: Iterable[str], limit: int | None = None,
         endpoint: str | None = None, cache: bool = True) -> list[Record]:
    """把多个源拼起来。单个源失败不拖垮整体，但会打印原因。"""
    out: list[Record] = []
    for name in names:
        if name not in SOURCES:
            raise SystemExit(f"未知数据源: {name}（可选: {', '.join(SOURCES)}）")
        fn = SOURCES[name][0]
        try:
            if cache and name != "fixture":
                recs = _cached_jsonl(name, lambda: fn(limit=limit, endpoint=endpoint), limit=limit)
            else:
                recs = fn(limit=limit, endpoint=endpoint)
            print(f"  [source] {name:14} {len(recs):6d} 条")
            out.extend(recs)
        except Exception as e:
            print(f"  [source] {name:14} 失败: {type(e).__name__}: {str(e)[:120]}")
    return out
