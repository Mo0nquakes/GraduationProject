"""嵌入空间对抗训练（FGSM / PGD）——**只对神经网络检测器**，且只当对照用。

这个模块存在的唯一目的，是为论文里那个**证伪性实验**提供对照组：

    在嵌入空间做对抗训练，能不能降低真实文本改写攻击的攻击成功率？

我们的判断是不能（详见 docs/A3.md）。但"判断"必须用实验说话，所以要有这套实现，
而且要做得**规范**——按最有利的方式实现它（合适的 ε、多步 PGD、与干净样本混合），
免得结论被"你没调好参数"反驳。

## 为什么 sklearn 那族做不了

FGSM/PGD 要求对输入求梯度。词袋/TF-IDF 模型没有可微的输入表示：改一个词的
one-hot 是离散跳变，梯度不存在。所以"传统 ML vs 预训练模型的对抗鲁棒性对比"
这个提法在方法上就不成立——只能两边都用**文本空间**攻击来比，这正是 A4 做的事。

## 一个必须写在论文里的细节

嵌入扰动**不可解码回文本**。把一段嵌入加上 0.05 的符号扰动，没有任何分词器能把它
还原成一句人话。所以它防的攻击者在现实里不存在；它的作用更接近数据增强式的正则化。
这不是本实现的缺陷，是这个方法本身的性质。
"""
from __future__ import annotations

from typing import Any, Sequence

from .distilbert import DistilBertDetector


def _emb_layer(det):
    return det._model.get_input_embeddings()


def fgsm_embeddings(det, input_ids, attention_mask, labels, eps: float = 0.05):
    """一步 FGSM：沿梯度符号方向扰动词嵌入。返回扰动后的 inputs_embeds（已 detach）。"""
    torch = det._torch
    emb = _emb_layer(det)(input_ids)
    emb = emb.detach().requires_grad_(True)
    out = det._model(inputs_embeds=emb, attention_mask=attention_mask, labels=labels)
    grad = torch.autograd.grad(out.loss, emb)[0]
    return (emb + eps * grad.sign()).detach()


def pgd_embeddings(det, input_ids, attention_mask, labels, eps: float = 0.05,
                   alpha: float = 0.01, steps: int = 3):
    """多步 PGD（L∞ 球内投影）。比 FGSM 强，用来堵"你只试了最弱的攻击"这类质疑。"""
    torch = det._torch
    base = _emb_layer(det)(input_ids).detach()
    adv = base.clone()
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        out = det._model(inputs_embeds=adv, attention_mask=attention_mask, labels=labels)
        grad = torch.autograd.grad(out.loss, adv)[0]
        adv = adv + alpha * grad.sign()
        # 投影回 [base-eps, base+eps]
        adv = torch.min(torch.max(adv, base - eps), base + eps).detach()
    return adv


class AdversarialDistilBert(DistilBertDetector):
    """在 DistilBertDetector 基础上，训练时混入嵌入空间对抗样本。

    `adv_mode`：None / "fgsm" / "pgd"。`mix=0.5` 表示每个 batch 一半干净、一半对抗，
    这是 FreeLB/ALUM 一系常用的做法——纯对抗样本训会显著掉干净准确率，
    那样比出来的"鲁棒性提升"其实是"把模型训坏了"。
    """

    def __init__(self, *args: Any, adv_mode: str | None = "pgd", eps: float = 0.05,
                 alpha: float = 0.01, steps: int = 3, mix: float = 0.5, **kw: Any):
        # 对抗训练要做两遍前向（扰动要一次、训练要一次），8GB 统一内存扛不住默认尺寸
        kw.setdefault("batch_size", 4)
        kw.setdefault("max_len", 192)
        super().__init__(*args, **kw)
        self.adv_mode = adv_mode
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.mix = mix
        self.name = f"distilbert-at-{adv_mode or 'none'}"
        self._save_dir = f"data/models/distilbert-inj-at-{adv_mode or 'none'}"

    def fit(self, texts: Sequence[str], labels: Sequence[str],
            fpr_target: float = 0.01, **_: Any) -> "AdversarialDistilBert":
        if not self.adv_mode:
            return super().fit(texts, labels, fpr_target=fpr_target)

        self._lazy()
        # MPS 的 SDPA 不支持 train 模式下的 dropout（`NotImplementedError: ... does not
        # support dropout`），而对抗训练必须在 train 模式多做一次带梯度的前向 —— 重载成 eager。
        from transformers import AutoModelForSequenceClassification

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, attn_implementation="eager").to(self.device_name)
        torch = self._torch
        import random as _random

        _random.seed(self.seed)
        torch.manual_seed(self.seed)

        texts = list(texts)
        y = [self._to_binary(lab) for lab in labels]
        enc = self._tok(texts, truncation=True, max_length=self.max_len,
                        padding=False, return_tensors=None)
        ds = [{"input_ids": enc["input_ids"][i],
               "attention_mask": enc["attention_mask"][i],
               "labels": y[i]} for i in range(len(texts))]

        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding

        collate = DataCollatorWithPadding(tokenizer=self._tok, return_tensors="pt")
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            collate_fn=collate, num_workers=0, pin_memory=False)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        self._model.train()

        for ep in range(self.epochs):
            total, seen = 0.0, 0
            for batch in loader:
                batch = {k: v.to(self.device_name) for k, v in batch.items()}
                opt.zero_grad()
                # 干净分支不需要梯度：detach 掉能省一半显存。
                # 8GB 的统一内存在 MPS 上就是这么抠出来的——不 detach 直接 OOM。
                with torch.no_grad():
                    clean = self._model(**batch)
                loss = clean.loss.detach() * (1.0 - self.mix)

                if self.mix > 0:
                    if self.adv_mode == "fgsm":
                        adv_emb = fgsm_embeddings(self, batch["input_ids"],
                                                  batch["attention_mask"],
                                                  batch["labels"], eps=self.eps)
                    else:
                        adv_emb = pgd_embeddings(self, batch["input_ids"],
                                                 batch["attention_mask"], batch["labels"],
                                                 eps=self.eps, alpha=self.alpha,
                                                 steps=self.steps)
                    adv_out = self._model(inputs_embeds=adv_emb,
                                          attention_mask=batch["attention_mask"],
                                          labels=batch["labels"])
                    loss = loss + adv_out.loss * self.mix

                loss.backward()
                opt.step()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                total += float(loss.item()) * len(batch["labels"])
                seen += len(batch["labels"])
            self._log(f"[AT {self.adv_mode}] epoch {ep + 1}/{self.epochs} "
                      f"loss={total / max(1, seen):.4f}")
        self._model.eval()
        try:
            self.save()
        except Exception as e:  # noqa: BLE001
            self._log(f"保存权重失败: {e}")
        return self
