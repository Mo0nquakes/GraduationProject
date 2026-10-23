# -*- coding: utf-8 -*-
"""生成开题报告 docs/开题报告.docx

用法：python scripts/make_proposal.py
依赖：python-docx
"""
from __future__ import annotations

import datetime
import pathlib

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "开题报告.docx"

BODY_LATIN = "Times New Roman"
BODY_EA = "宋体"
HEAD_EA = "黑体"


# ---------------------------------------------------------------- 基础排版
def set_run_font(run, latin=BODY_LATIN, ea=BODY_EA, size=None, bold=None, color=None):
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    rfonts.set(qn("w:eastAsia"), ea)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_style_font(style, latin, ea, size, bold=False):
    style.font.name = latin
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    rfonts.set(qn("w:eastAsia"), ea)


def setup_document() -> Document:
    doc = Document()

    sec = doc.sections[0]
    sec.page_height = Cm(29.7)
    sec.page_width = Cm(21.0)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.17)
    sec.right_margin = Cm(3.17)

    normal = doc.styles["Normal"]
    set_style_font(normal, BODY_LATIN, BODY_EA, 12)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)

    for lvl, size, before, after in ((1, 16, 18, 10), (2, 14, 12, 6), (3, 12, 10, 4)):
        st = doc.styles[f"Heading {lvl}"]
        set_style_font(st, BODY_LATIN, HEAD_EA, size, bold=True)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        st.paragraph_format.keep_with_next = True

    # 页脚页码
    footer_p = sec.footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer_p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9)
    return doc


def h1(doc, text):
    return doc.add_heading(text, level=1)


def h2(doc, text):
    return doc.add_heading(text, level=2)


def h3(doc, text):
    return doc.add_heading(text, level=3)


def p(doc, text, indent=True, bold=False, size=12, align=None, space_after=0):
    """正文段落。text 中可用 **粗体** 标记局部加粗。"""
    para = doc.add_paragraph()
    if indent:
        para.paragraph_format.first_line_indent = Pt(size * 2)
    if align is not None:
        para.alignment = align
    if space_after:
        para.paragraph_format.space_after = Pt(space_after)

    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        run = para.add_run(seg)
        set_run_font(run, size=size, bold=(bold or i % 2 == 1))
    return para


def bullet(doc, text, size=12):
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Pt(size * 2)
    para.paragraph_format.first_line_indent = Pt(-size)
    para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        run = para.add_run(seg)
        set_run_font(run, size=size, bold=(i % 2 == 1))
    return para


def caption(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(3)
    para.paragraph_format.space_after = Pt(9)
    run = para.add_run(text)
    set_run_font(run, size=10.5, bold=False)
    return para


def placeholder(doc, text):
    """图表占位：最后统一绘制，先留位置，避免正文写完再回头找。"""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(0)
    run = para.add_run(text)
    set_run_font(run, size=10.5)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    return para


def table(doc, headers, rows, widths=None, size=10.5, caption_text=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        para = hdr[i].paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        run = para.add_run(htext)
        set_run_font(run, size=size, bold=True)

    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            para = cells[i].paragraphs[0]
            para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            if i == 0:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for j, seg in enumerate(str(val).split("**")):
                if not seg:
                    continue
                run = para.add_run(seg)
                set_run_font(run, size=size, bold=(j % 2 == 1))

    if widths:
        total = sum(widths)
        widths = [w * 14.4 / total for w in widths]  # 正文可用宽 14.66cm，留一点余量
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
            for cell in t.columns[i].cells:
                cell.width = Cm(w)

    if caption_text:
        caption(doc, caption_text)
    else:
        doc.add_paragraph()
    return t


# ---------------------------------------------------------------- 正文
def build():
    doc = setup_document()
    today = datetime.date(2026, 10, 23)

    # ---------- 封面 ----------
    for _ in range(3):
        doc.add_paragraph()
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run("本科毕业设计（论文）开题报告")
    set_run_font(run, ea=HEAD_EA, size=26, bold=True)

    doc.add_paragraph()
    doc.add_paragraph()

    blank = "_" * 22
    info = [
        ("论文题目", "面向大模型提示注入的检测器及其对抗鲁棒性研究"),
        ("", "—— 从文本空间攻击到红队回路对抗训练"),
        ("学    院", blank),
        ("专    业", blank),
        ("班    级", blank),
        ("学生姓名", blank),
        ("学    号", blank),
        ("指导教师", blank),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for label, value in info:
        cells = t.add_row().cells
        cells[0].width = Cm(3.4)
        cells[1].width = Cm(10.4)

        pl = cells[0].paragraphs[0]
        pl.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pl.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        if label:
            r1 = pl.add_run(label + "：")
            set_run_font(r1, size=13, bold=True)

        pv = cells[1].paragraphs[0]
        pv.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        if label == "论文题目":
            r2 = pv.add_run(value)
            set_run_font(r2, size=14, bold=True)
        elif label == "":
            pv.paragraph_format.space_before = Pt(2)
            r2 = pv.add_run(value)
            set_run_font(r2, size=11.5)
        else:
            r2 = pv.add_run(value)
            set_run_font(r2, size=13)

    for _ in range(6):
        doc.add_paragraph()
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run(f"{today.year} 年 {today.month} 月")
    set_run_font(run, size=13)

    doc.add_page_break()

    # ---------- 摘要 ----------
    h1(doc, "摘  要")
    p(doc,
      "大模型正在从“对话工具”变成“应用底座”，检索增强生成（RAG）与智能体（Agent）把网页、文档、"
      "邮件和工具返回值直接拼进提示词。这些内容由第三方写入，却与开发者指令共享同一条文本通道，"
      "模型无法从类型上区分“数据”与“指令”，由此产生提示注入（prompt injection）攻击。"
      "目前工业界主流的防线是在外围挂一个检测器，但绝大多数检测器只在固定的干净测试集上报告准确率，"
      "几乎没有工作系统刻画“攻击者针对检测器优化之后”检测器还剩多少能力。")
    p(doc,
      "本课题不追求训练一个更强的分类器，而是围绕**检测器的对抗鲁棒性**做一篇评测驱动的研究："
      "（1）构建带难负样本与混淆变体的注入检测基准，区分直接注入、间接注入与良性文本；"
      "（2）建立**文本空间**攻击谱系（零宽字符、全角、同形字、编码包裹、语义改写、截断），"
      "在同等黑盒查询预算下评测各检测器族；（3）提出**红队回路对抗训练**，用攻击者与检测器交替迭代的方式"
      "提升鲁棒性，并与图像领域沿用而来的嵌入空间对抗训练（FGSM/PGD）做对照，检验后者对文本空间攻击是否真的有效；"
      "（4）实现一个可运行的三级防护原型系统（规则→轻量分类器→神经检测器），配最小 RAG 靶子测量端到端攻击成功率，"
      "并用哈希链审计日志保证检测记录可追溯。")
    p(doc,
      "课题的预期结论包含一个**证伪性**的判断：嵌入空间对抗训练几乎不能降低真实文本改写攻击的攻击成功率，"
      "它更像一种正则化；真正有效的是在输入分布上做红队回路训练，配合分块滑窗与输入规范化。"
      "该判断无论成立与否都有价值：成立，则纠正了一个被广泛误用的方法；不成立，则说明迁移性比预期更强，同样值得报告。")

    # ---------- 一 ----------
    h1(doc, "一、选题背景与研究意义")

    h2(doc, "1.1 研究背景")
    p(doc,
      "2022 年底，Perez 与 Ribeiro 通过手工构造的输入就能让 GPT-3 偏离既定目标，"
      "并把“忽略以上指令”这类攻击命名为提示注入[1]。2023 年，Greshake 等人指出更严重的一类："
      "攻击者不必直接与模型对话，只要把自己的指令塞进“将来会被检索到”的网页或文档里，"
      "就能在用户毫不知情时劫持整个应用，称为间接提示注入[2]，并给出了数据窃取、蠕虫式传播等影响分类。"
      "此后，Liu 等人给出了提示注入的形式化框架与统一基准[3]，Yi 等人构建了间接注入基准 BIPIA[4]，"
      "AgentDojo 在 97 个真实任务、629 个安全测试用例上评测智能体，结论是现有模型与防御都远未解决问题[10]。"
      "在工程侧，OWASP 2025 年 LLM 应用十大风险把提示注入列在首位（LLM01）。")
    p(doc,
      "与之相对，防御研究大致分成两条路线。一条是**系统级**的：在模型训练中引入指令优先级[5]，"
      "把提示与数据拆成两条通道（StruQ[7]），给不可信文本打来源标记（Spotlighting[6]），"
      "用偏好优化让模型偏好“安全回答”（SecAlign[8]），或者干脆用程序把控制流与数据流分离（CaMeL[9]）。"
      "另一条是**检测级**的：在输入进入模型之前用一个分类器判断“这段文本里有没有注入指令”，"
      "代表工作是 WildGuard 这类审核模型[15]。系统级方案效果好，但要求改模型或改应用架构，"
      "对本科课题和大多数存量系统都不现实；检测级方案可以即插即用，因此是工业界的主流选择。")
    p(doc,
      "问题恰恰出在检测级方案上。检测器本身是一个机器学习模型，它同样会被对抗样本欺骗。"
      "而现有的注入检测工作报告的是“在某个固定测试集上的准确率/F1”，几乎没有回答一个安全研究必须回答的问题："
      "**当攻击者知道你在用检测器，并针对它优化输入时，它还能挡住多少？**")

    h2(doc, "1.2 问题的提出")
    p(doc,
      "自然语言处理领域早就研究过这个问题，并且形成了一个容易被误用的结论。Goodfellow 等人提出的"
      "快速梯度符号法（FGSM）[18] 通过在输入上叠加梯度方向的扰动来构造对抗样本，随后 FreeLB[20]、"
      "ALUM[21] 把这一思路搬到词嵌入空间，通过“在 embedding 上加扰动再做对抗训练”提升鲁棒性。"
      "这套方法在图像领域是标准做法，于是不少提示注入相关工作直接照搬。")
    p(doc,
      "但文本与图像有一个本质差别：**嵌入空间的扰动不可解码回文本**。你在 768 维向量上加 0.01 的扰动，"
      "没有任何一个真实用户能生成这样一段输入。因此，在嵌入空间做对抗训练，防的是一个现实中不存在的攻击者；"
      "它能起的作用，更接近一种数据增强式的正则化。而真实攻击者手里的武器是**文本空间**的："
      "插入零宽字符、把字母换成全角或西里尔同形字、把载荷 base64 编码、用另一个大模型做语义改写、"
      "把指令藏在 512 个 token 之后的文档末尾。TextFooler[19] 早在 2019 年就证明了：只靠查询目标模型的输出标签，"
      "就能在文本空间稳定地攻破 BERT。")
    p(doc,
      "由此引出本课题的核心问题：**在文本空间攻击下，各类提示注入检测器还剩多少鲁棒性？"
      "什么样的训练方式能真正提升这种鲁棒性？**")

    h2(doc, "1.3 研究意义")
    p(doc, "**理论意义**：")
    bullet(doc, "把“对抗鲁棒性”这一在图像与文本分类中成熟、却在提示注入检测中被含混处理的问题，落实为可测量的实验设计；")
    bullet(doc, "给出文本空间攻击谱系与检测器鲁棒性之间的定量关系，特别是对“嵌入空间对抗训练具有可迁移性”这一隐含假设给出可证伪的检验；")
    bullet(doc, "把红队（red teaming）从一次性的评测动作，变成训练回路中的一个组成部分，为“攻防协同演进”提供一个可复现的小规模样例。")
    p(doc, "**工程意义**：")
    bullet(doc, "产出一个可直接部署的三级防护原型（规则→轻量分类器→神经检测器），并给出延迟与误报的实测数据；")
    bullet(doc, "配套一个最小的 RAG 靶子，用“密钥有没有泄露出去”这种端到端指标衡量防护效果，而不是只看分类指标；")
    bullet(doc, "检测记录以哈希链方式落库，任何一条历史判定被篡改都能被检出，满足安全审计的基本要求。")

    # ---------- 二 ----------
    h1(doc, "二、国内外研究现状")

    h2(doc, "2.1 提示注入攻击")
    p(doc,
      "**直接注入**：攻击者在用户输入中写入“忽略之前的指令”“你现在是……”等语句，"
      "使模型放弃开发者设定的行为。Perez 与 Ribeiro 的 PromptInject 框架系统化了这类攻击的构造方式[1]。")
    p(doc,
      "**间接注入**：载荷位于模型将要读取的外部数据中，包括网页、PDF、邮件、代码注释、"
      "甚至工具调用的返回值。Greshake 等人证明了可以借此实现远程利用、数据外泄与蠕虫式扩散[2]。"
      "这类攻击是当前 RAG 与 Agent 系统面临的主要威胁，也是本课题的重点。")
    p(doc,
      "**形式化与基准**：Liu 等人提出统一框架，把已有攻击视为该框架的特例，并在 5 种攻击、10 种防御、"
      "10 个模型、7 个任务上做了系统评测[3]；BIPIA 专门针对间接注入，指出模型失败的两个原因——"
      "不能区分“信息性上下文”与“可执行指令”，以及缺乏“不要执行外部内容中指令”的意识[4]。")
    p(doc,
      "**自动化攻击**：GCG 通过梯度搜索构造对抗后缀，并发现其可迁移到黑盒模型[11]；"
      "PAIR 用攻击者大模型迭代改写，在不到 20 次查询内完成越狱[12]；"
      "Crescendo 则通过多轮渐进诱导，让每一轮单独看都无害[13]。"
      "后两者说明：真实的攻击者是有查询预算、会迭代的黑盒对手，而不是一次性提交固定字符串。")

    h2(doc, "2.2 防御：系统级与检测级")
    table(doc,
          ["路线", "代表工作", "核心思想", "对本科课题的可行性"],
          [
              ["系统级", "Instruction Hierarchy[5]", "训练模型区分系统/用户/第三方指令的优先级", "需重训模型，不可行"],
              ["系统级", "StruQ[7]", "提示与数据分两条通道，结构化查询 + 专门微调", "需重训模型，不可行"],
              ["系统级", "Spotlighting[6]", "用变换标记不可信文本的来源", "可用作对照 baseline"],
              ["系统级", "SecAlign[8]", "用偏好优化让模型偏好安全输出", "需重训模型，不可行"],
              ["系统级", "CaMeL[9]", "程序层分离控制流与数据流 + 能力约束", "思想可借鉴，实现超范围"],
              ["检测级", "WildGuard[15]", "轻量审核模型，判断输入/输出的安全风险", "**可直接作为强 baseline**"],
              ["检测级", "Llama Guard 系列", "安全分类器，覆盖多类风险", "**可作为 baseline**"],
          ],
          widths=[2.2, 3.4, 5.6, 4.0],
          caption_text="表 1  提示注入防御的两条路线对比")
    p(doc,
      "需要指出一个常被忽视的差别：WildGuard 这类审核模型判断的是“内容是否有害”，"
      "而提示注入检测要判断的是“文本中是否含试图覆盖指令的内容”，后者的良性难例（例如用户正常地说"
      "“忽略我上一条消息，我改主意了”）恰恰会被前者误报。因此把审核模型直接当注入检测器用，"
      "其误报率在真实业务中往往不可接受——这也是本课题要专门构造难负样本的原因。")

    h2(doc, "2.3 对抗鲁棒性与对抗训练")
    p(doc,
      "FGSM[18] 奠定了“沿着梯度方向施加最坏扰动”的范式；FreeLB[20] 把它引入自然语言理解的微调阶段，"
      "通过在词嵌入附近多次累积扰动提升泛化；ALUM[21] 进一步把它用于预训练与持续预训练。"
      "这些工作的共同前提是：扰动发生在连续的表示空间。")
    p(doc,
      "另一条线是文本空间攻击。TextFooler[19] 采用同义词替换 + 查询反馈的贪心策略，"
      "在保持语义与语法正确的前提下攻破 BERT，且计算复杂度与文本长度线性相关。"
      "它的意义在于：**攻击产物是真实可读的文本**，因此可以用来评测任何检测器，"
      "而不要求检测器暴露梯度。")
    p(doc,
      "两条线的差别正是本课题的切入点。对本课题而言，文本空间攻击对规则、TF-IDF、"
      "预训练模型三类检测器是“对等”的——都只需要黑盒查询；而 FGSM 只对神经网络有效，"
      "对 TF-IDF+逻辑回归这类模型根本无法定义，因此无法支撑“传统机器学习 vs 预训练模型”的公平对比。")

    h2(doc, "2.4 可用数据与评测基准")
    table(doc,
          ["资源", "性质", "在本课题中的用途"],
          [
              ["WildGuardMix[15]", "92K 条安全审核数据，含对抗越狱样本", "良性/恶意候选池，跨集泛化测试"],
              ["ToxicChat[16]", "真实用户与 AI 的对话日志，标注毒性", "真实分布良性样本（难负样本来源）"],
              ["XSTest[17]", "250 条“看似危险实则安全”的提示", "**良性难负样本**，压误报"],
              ["BIPIA[4]", "间接注入基准，覆盖多类任务", "间接注入样本构造参考"],
              ["HarmBench[14]", "标准化红队评测框架（18 方法 / 33 模型）", "攻击方法与评测协议参考"],
              ["AgentDojo[10]", "97 任务 / 629 安全用例的智能体环境", "威胁模型与任务设定参考"],
              ["Lakera PINT", "提示注入基准数据集", "跨集泛化测试集"],
              ["AEGIS 2.0", "NVIDIA 安全数据集（NeurIPS 2024）", "补充评测集"],
          ],
          widths=[3.4, 6.2, 5.6],
          caption_text="表 2  主要公开资源及其用途")
    p(doc,
      "必须说明：PromptInject[1] 是攻击模板框架，HarmBench[14] 是红队评测框架（需要靶子模型才能产出攻击成功率），"
      "二者都**不是**现成的二分类检测语料。把“用了 PromptInject/HarmBench”写成“有了数据集”，"
      "是开题阶段最容易埋下的坑。本课题的数据必须自己拼装、自己标注、自己去重。")

    h2(doc, "2.5 研究空白")
    bullet(doc, "**G1**：缺少面向“注入检测”这一具体任务的基准，现有数据集多为“内容是否有害”，标签语义不匹配；")
    bullet(doc, "**G2**：缺少带良性难负样本的评测，导致检测器的高分掩盖了高误报（把正常用户请求判成攻击）；")
    bullet(doc, "**G3**：检测器的对抗鲁棒性缺少系统刻画，尤其是混淆类（零宽/全角/同形字/编码）攻击；")
    bullet(doc, "**G4**：对抗训练在提示注入检测中的有效性与适用边界没有定量结论，嵌入空间与文本空间两条路线未被公平对照。")

    # ---------- 三 ----------
    h1(doc, "三、研究内容与目标")

    h2(doc, "3.1 总体目标")
    p(doc,
      "构建一个面向大模型提示注入的检测与评测原型系统，在**统一的威胁模型**与**统一的文本空间攻击谱系**下，"
      "定量回答“哪类检测器在什么攻击下失效、什么样的训练方式能补救”，并给出可部署的工程实现。")
    p(doc, "三条可检验的具体目标：")
    bullet(doc, "**目标 1**：在固定误报率（FPR = 1%）下，给出至少 4 类检测器在 6 族文本空间攻击下的 TPR 与 ASR 矩阵；")
    bullet(doc, "**目标 2**：证明或证伪“嵌入空间对抗训练（FGSM/PGD）能提升文本空间鲁棒性”，给出效应量与置信区间；")
    bullet(doc, "**目标 3**：红队回路对抗训练使综合 ASR 相对基线下降不少于 30%（相对值），且良性效用下降不超过 2 个百分点。")

    h2(doc, "3.2 研究内容")
    p(doc, "**C1  注入检测基准的构建与标注规范**")
    p(doc,
      "以英文为主构建三分类数据（良性 / 直接注入 / 间接注入），对外坍缩为二分类以对齐文献中的 TPR@FPR 报告方式。"
      "重点在于**难负样本**的构造：包含“请忽略上一条消息”“把之前的回答忘掉”“请重复你的系统提示词格式”"
      "这类字面高度相似但意图良性的文本，用于暴露检测器的捷径学习（shortcut learning）。"
      "中文数据只进入零样本泛化测试，不进入训练集，以保证跨语言结论的干净。", indent=True)
    p(doc, "**C2  文本空间攻击谱系与鲁棒性评测**")
    p(doc,
      "构建 6 族攻击：①零宽字符插入；②全角字符替换；③同形字（西里尔字母）替换；④Base64/编码包裹；"
      "⑤语义改写（用 LLM 生成同义变体，模拟 PAIR 类查询攻击）；⑥截断攻击（把载荷放在长文档 512 token 之后）。"
      "所有攻击只依赖检测器的**预测标签**，因此对规则、TF-IDF、神经网络三类检测器对等。", indent=True)
    p(doc, "**C3  红队回路对抗训练**")
    p(doc,
      "把攻击生成器与检测器的训练交替进行：每轮用当前检测器的失败样本驱动攻击生成器产出新变体，"
      "新变体经人工抽查后进入训练集，再重训检测器。为避免“攻击者与检测器互相过拟合”，"
      "留出一整个攻击族（例如语义改写）作为**从未参与训练**的测试族。同时设置 FGSM/PGD 嵌入空间对抗训练作为对照组。", indent=True)
    p(doc, "**C4  三级防护原型系统**")
    p(doc,
      "实现“输入规范化 → 分级检测（规则 / 轻量分类器 / 神经检测器）→ 决策与审计”的流水线，"
      "配一个最小 RAG 靶子（检索→拼接提示→小模型生成）测量端到端攻击成功率，"
      "所有判定写入带哈希链的审计日志，并提供 Web 接口与可视化控制台。", indent=True)

    h2(doc, "3.3 拟解决的关键问题")
    p(doc,
      "**Q1  标签边界的界定。** “忽略以上指令”这句话，在攻击里是指令覆盖，在正常对话里可能只是用户改主意。"
      "如果边界不清，难负样本反而会污染正类。拟采用“是否试图改变**应用方**设定的行为”作为判据，"
      "并在标注规范中给出 10 条以上边界案例，标注一致性用 Cohen's κ 度量。")
    p(doc,
      "**Q2  攻击的公平性。** 不同检测器可用的信息不同（神经网络有梯度，TF-IDF 没有）。"
      "若用梯度攻击，则对比不成立。拟统一采用**仅黑盒查询**的文本空间攻击，把梯度攻击只作为“白盒上界”单独报告。")
    p(doc,
      "**Q3  训练-测试泄漏的控制。** 红队回路容易让训练集与测试集共享攻击模板，从而虚高。"
      "拟在数据层做三级隔离：样本文本去重（SHA-256 精确 + SimHash 近似）、攻击模板隔离、"
      "以及留出一个完整攻击族作为未见攻击测试。")

    # ---------- 四 ----------
    h1(doc, "四、研究方法与技术路线")

    h2(doc, "4.1 威胁模型")
    p(doc,
      "本课题首先明确“防谁”。详细文档见仓库 docs/威胁模型.md，核心假设如下表。"
      "没有威胁模型的检测器研究，本质上只是一个分类实验；把它写清楚，是安全方向论文的基本要求。")
    table(doc,
          ["编号", "攻击者", "能力假设", "是否覆盖"],
          [
              ["T1", "恶意终端用户", "可任意构造输入，黑盒，可无限次查询", "是"],
              ["T2", "内容投毒者", "可让一条文档进入检索库，不知模型权重", "是（重点）"],
              ["T3", "有预算的优化攻击者", "黑盒，可用 LLM 迭代改写（PAIR/Crescendo 类）", "是（有限预算）"],
              ["T4", "白盒攻击者", "知道权重与梯度，可做 GCG/FGSM", "仅作对照"],
              ["T5", "供应链攻击者", "篡改权重或训练数据", "否，列为未来工作"],
          ],
          widths=[1.4, 3.6, 7.4, 3.2],
          caption_text="表 3  攻击者能力假设")
    p(doc,
      "这张表最重要的作用是给出一条自我约束：**真实世界的注入攻击几乎都是 T1–T3（黑盒、文本空间）**。"
      "因此只在嵌入空间做对抗训练，防的是一个现实中不存在的攻击者——这正是原方案需要修正的地方。")

    h2(doc, "4.2 总体技术路线")
    p(doc, "技术路线分为五个阶段，前后依赖关系如下：")
    bullet(doc, "**阶段一 数据**：公开语料筛选 → 清洗去重 → 难负样本构造 → 三分类标注 → 版本化入库（dataset_version + sample）；")
    bullet(doc, "**阶段二 基线**：规则检测器与 TF-IDF+逻辑回归，跑通“训练集校准阈值 → 测试集报告 TPR@FPR”的完整评测链路；")
    bullet(doc, "**阶段三 神经检测**：DistilBERT 微调 + 分块滑窗推理，处理长文档；")
    bullet(doc, "**阶段四 攻防**：文本空间攻击谱系 → 鲁棒性矩阵 → 红队回路对抗训练 → 与 FGSM 对抗训练对照；")
    bullet(doc, "**阶段五 系统**：三级流水线封装为服务，接 RAG 靶子测端到端 ASR，接审计链与控制台。")
    placeholder(doc, "【图 1 位置：技术路线图（最后绘制）】")
    caption(doc, "图 1  技术路线")

    h2(doc, "4.3 关键方法")

    h3(doc, "4.3.1 阈值校准与指标口径")
    p(doc,
      "安全场景下 Accuracy 与 F1 没有意义：一个把所有样本判为“安全”的分类器准确率可能高达 90%。"
      "本课题统一报告 **TPR@FPR=1%**（以及 0.1% 作为参考），其含义是“在误报不超过 1% 的前提下能挡住多少攻击”。"
      "实现上，阈值**只在训练集上校准**（取训练集良性分数的 99% 分位数），再固定该阈值到测试集上评估，"
      "避免在同一批数据上既选阈值又报指标。")
    p(doc,
      "同时必须报告负样本量。当测试集只有 18 条良性样本时，FPR=1% 这个点根本不可分辨（最小分辨率约 5.6%），"
      "此时给出的 TPR@FPR=1% 是没有统计意义的数字——这一点在论文中要显式说明，而不是藏起来。")

    h3(doc, "4.3.2 文本空间攻击谱系")
    table(doc,
          ["攻击族", "变换方式", "对应现实威胁", "朴素缓解"],
          [
              ["零宽字符", "在词内插入 U+200B 等不可见字符", "绕过关键词与分词", "剥离零宽/格式字符"],
              ["全角替换", "ASCII + 0xFEE0 转全角", "绕过精确匹配", "NFKC 归一化"],
              ["同形字", "用西里尔字母替换拉丁字母", "伪装成合法词", "Unicode 同形字映射表"],
              ["编码包裹", "整段载荷 Base64 / ROT13", "让语义检测失效", "检测可疑编码串并解码复检"],
              ["语义改写", "LLM 生成同义变体，保留攻击意图", "对抗训练后的迁移攻击", "红队回路训练（本文方法）"],
              ["截断攻击", "载荷置于长文档 512 token 之后", "利用模型输入长度限制", "分块滑窗 + 块级分数取最大"],
          ],
          widths=[2.4, 4.6, 4.6, 4.0],
          caption_text="表 4  文本空间攻击谱系")

    h3(doc, "4.3.3 红队回路对抗训练")
    p(doc,
      "设检测器为 D，攻击生成器为 G（可用一个小参数量的本地模型或调用 API）。每一轮的流程是：")
    bullet(doc, "① 用 D 对当前攻击集打分，挑出“被判为良性”的失败样本（假阴性）；")
    bullet(doc, "② 把这些失败样本作为 in-context 示例喂给 G，要求它生成语义等价但措辞不同的新变体；")
    bullet(doc, "③ 对新变体做人工抽查（每轮 20 条），剔除语义已损坏的；")
    bullet(doc, "④ 新变体并入训练集，重训 D；")
    bullet(doc, "⑤ 在**留出攻击族**上评估，记录 ASR 曲线，直到连续两轮不再下降或达到预算。")
    p(doc,
      "该方法的代价是训练成本随轮数线性增长，因此需要设置早停。其风险是攻击生成器与检测器形成"
      "“两个人互相模仿”的循环，所以留出整族攻击做检验是方法成立的前提，而不是可选项。")

    h3(doc, "4.3.4 长文档分块滑窗检测")
    p(doc,
      "标准做法是把输入截断到模型最大长度（如 512 token）。攻击者只要把指令放在第 600 个 token，"
      "检测器就“看不见”，而生成模型（上下文窗口更大）却“看得见”——这是一个纯粹的工程漏洞，"
      "不需要任何机器学习技巧就能利用。")
    p(doc,
      "缓解方案是分块滑窗：以 256 token 为块长、128 token 为步长切分，逐块打分后取最大值作为文档分数，"
      "并记录命中的字符区间，用于界面高亮。代价是推理时间随文档长度线性增长，需要在论文中给出延迟-召回权衡曲线。")

    h3(doc, "4.3.5 输入规范化")
    p(doc,
      "在检测之前做确定性变换：Unicode NFKC 归一化、剥离零宽与格式控制字符、同形字映射、"
      "把连续重复字符折叠、还原常见编码。这一步几乎是零成本的，且对人类可读性无影响，"
      "实验上应能消掉大部分混淆类攻击。它与机器学习方法是正交的——这也是“防御要做纵深”的具体体现。")

    # ---------- 五 ----------
    h1(doc, "五、系统设计")

    h2(doc, "5.1 总体架构")
    p(doc,
      "系统分为四层，自下而上为：存储层（SQLite + 文件）、算法层（检测器、攻击、指标）、"
      "服务层（FastAPI）、交互层（Web 控制台）。所有跨层调用都通过明确定义的接口，"
      "保证“换掉某个检测器不影响其他部分”。")
    bullet(doc, "**存储层**：8 张表，覆盖数据集版本、样本、攻击、攻防关联、防御策略、模型版本、实验记录与检测日志；")
    bullet(doc, "**算法层**：检测器统一实现 fit/predict_proba 协议，攻击统一实现 apply(text)→text，指标手写实现；")
    bullet(doc, "**服务层**：/detect 返回逐块分数供界面高亮，/audit/verify 校验哈希链完整性；")
    bullet(doc, "**交互层**：控制台提供提交检测、查看详情、发起红队攻击、对比实验、审计校验五个页面。")
    placeholder(doc, "【图 2 位置：系统总体架构图（最后绘制）】")
    caption(doc, "图 2  系统总体架构")

    h2(doc, "5.2 数据库设计")
    table(doc,
          ["表名", "作用", "关键字段"],
          [
              ["dataset_version", "数据集版本，保证实验可复现", "name / split_rule / created_at"],
              ["sample", "单条样本", "text / label / sha256 / is_hard_negative"],
              ["attack", "攻击方法元信息", "family / params_json"],
              ["attack_sample", "攻击产物（样本 × 攻击）", "attack_id / sample_id / text"],
              ["defense_policy", "防御配置（规范化+阈值+分块）", "norm_rules / threshold / chunk_size"],
              ["model_version", "模型快照", "name / params / path"],
              ["experiment_run", "一次实验的全部配置与结果", "config_json / metrics_json / git_commit"],
              ["detection_log", "在线判定记录（哈希链）", "prev_hash / cur_hash / score / decision"],
          ],
          widths=[3.6, 6.2, 5.8],
          caption_text="表 5  数据库表设计")
    p(doc,
      "需要强调设计动机：把“数据集版本”“模型版本”“防御策略”“实验记录”都做成独立的表，"
      "是为了让任意一条实验结论都能回答“用的是哪版数据、哪个模型、什么阈值”。"
      "如果只建一张结果表，三个月后自己都说不清某个数字是怎么来的——这是复现性最实际的敌人。")

    h2(doc, "5.3 哈希链审计")
    p(doc,
      "每条检测记录在写入时计算 cur_hash = SHA-256(prev_hash ‖ 规范化后的载荷)，"
      "其中 prev_hash 指向上一条记录的 cur_hash，形成一条链；第一条的 prev_hash 取常量 \"GENESIS\"。"
      "任何一条历史记录被修改，其后所有哈希都对不上，/audit/verify 会定位到第一个断点。"
      "这在取证场景中是刚需：你不能既用日志当证据，又允许日志被悄悄改写。")

    h2(doc, "5.4 接口与界面")
    p(doc,
      "服务层接口包括：健康检查、单条检测（返回总分、逐块分数、命中区间）、批量检测、"
      "审计链校验、统计概览。控制台把“检测结果”渲染成带高亮的原文，"
      "让使用者直接看到“是哪一段被判成了攻击”，而不是只给一个分数。"
      "这一交互细节对安全运营很重要：分析师需要判断是模型错了还是真的有攻击。")

    # ---------- 六 ----------
    h1(doc, "六、实验方案")

    h2(doc, "6.1 数据构建")
    p(doc,
      "训练集以英文为主。良性样本来源：ToxicChat 中的人工正常对话、XSTest 的安全提示、"
      "公开文档与表格数据的自然语言描述。恶意样本：由公开注入模板改写生成，"
      "并用 LLM 生成语义变体以降低模板痕迹。难负样本单独构造并打标，用于 A6 消融。"
      "所有样本入库前做 SHA-256 精确去重与 SimHash 近似去重，避免训练/测试泄漏。")
    p(doc,
      "规模上不做无意义的“越大越好”：目标训练集约 4000 条、测试集 1000 条、"
      "其中良性不少于 40%，确保 FPR=1% 这个工作点有足够分辨率（至少 500 条良性测试样本，分辨率 0.2%）。")

    h2(doc, "6.2 实验矩阵")
    table(doc,
          ["编号", "实验", "自变量", "观测指标", "预期结论"],
          [
              ["A1", "基线对比", "规则 / TF-IDF+LR / DistilBERT / WildGuard", "TPR@FPR=1%, 延迟", "神经模型最优但差距不大"],
              ["A2", "跨集泛化", "训练集→PINT/XSTest", "TPR@FPR, 误报率", "跨集掉点明显"],
              ["A3", "FGSM-AT 对照", "无 AT / FGSM-AT / PGD-AT", "文本攻击下 ASR", "**ASR 几乎不降（证伪核心创新）**"],
              ["A4", "攻击谱系", "6 族攻击", "各族 ASR", "混淆类攻击 ASR 最高"],
              ["A5", "红队回路 AT", "轮数 0/1/2/3", "留出族 ASR", "ASR 显著下降"],
              ["A6", "难负样本消融", "有/无难负样本", "FPR, TPR", "无难负样本时 FPR 虚低"],
              ["A7", "端到端", "RAG 靶子 + 各检测器", "端到端 ASR, P99 延迟", "拦截有效但延迟上升"],
          ],
          widths=[1.4, 3.0, 4.8, 3.6, 4.2],
          caption_text="表 6  实验矩阵")
    p(doc,
      "其中 A3 是本课题的“主线实验”。它检验的是一个被广泛默认、却少有直接证据的假设："
      "嵌入空间对抗训练能提升对文本攻击的鲁棒性。若结果如预期——ASR 几乎不降——"
      "那么这一结论本身就构成了对现有做法的方法论批评，价值高于“又训练了一个更好的模型”。")

    h2(doc, "6.3 评价指标")
    bullet(doc, "**TPR@FPR=1%**：主指标，固定误报预算下的检出率；同时报告 FPR=0.1% 作为参考；")
    bullet(doc, "**ASR（攻击成功率）**：恶意样本被判为良性的比例，跨攻击族分别报告；")
    bullet(doc, "**端到端 ASR**：RAG 靶子中机密是否真的被输出，而不是“检测器判对了吗”；")
    bullet(doc, "**延迟**：单条 P50/P99，以及分块后的线性增长系数；")
    bullet(doc, "**效用**：良性样本被误拦的比例（即 FPR），以及良性文档问答正确率，确保防护没有把功能废掉。")

    h2(doc, "6.4 统计与可复现")
    p(doc,
      "所有实验固定随机种子并记录 git commit；每个指标给出 Wilson 置信区间；"
      "不同检测器的对比使用配对检验（同一批测试样本上的 McNemar 检验），而不是只看两个点估计的差值。"
      "所有原始预测写入 predictions.jsonl，指标可由它重算，避免“报告的数字无法回溯”。")

    h2(doc, "6.5 预实验：链路已跑通")
    p(doc,
      "截至开题，代码骨架已完成并在一个 170 条的模板小数据集上跑通全链路（此为冒烟测试，"
      "数字不代表最终结论，仅证明流程可用）。下表为规则基线在攻击谱系下的表现：")
    table(doc,
          ["指标/攻击", "数值", "说明"],
          [
              ["TPR @ FPR=1%", "0.697", "训练集校准阈值为 0.5000"],
              ["AUC", "0.848", "排序能力尚可"],
              ["单条延迟", "0.0021 ms", "纯规则，几乎零成本"],
              ["identity（自检）", "ASR 0.303", "恰等于 1 − TPR，验证链路自洽"],
              ["zero_width", "ASR 0.970", "**零宽字符几乎全逃逸**"],
              ["fullwidth", "ASR 1.000", "**全角替换完全逃逸**"],
              ["homoglyph", "ASR 0.788", "同形字部分逃逸"],
              ["base64_wrap", "ASR 1.000", "**编码包裹完全逃逸**"],
              ["难负样本消融", "TPR→0.000", "规则基线被难负样本彻底打死"],
              ["截断攻击", "0.5→1.0", "朴素截断漏检；分块滑窗检出（命中第 3840 词）"],
              ["RAG 端到端", "泄漏→拦截", "无防御时输出密钥；加检测器后拦截"],
          ],
          widths=[4.2, 3.4, 8.4],
          caption_text="表 7  预实验结果（模板小数据集，仅验证链路）")
    p(doc,
      "这组数字虽然样本很小，但已经显现出本课题的核心叙事：**一个在干净数据上 TPR 0.697、AUC 0.848 的检测器，"
      "在简单字符混淆下 ASR 高达 0.97–1.00**。混淆攻击的实现成本几乎为零，防御成本也很低（NFKC + 剥零宽），"
      "但前提是你先把它测出来。这正是“评测驱动”的价值。")

    # ---------- 七 ----------
    h1(doc, "七、可行性分析")

    h2(doc, "7.1 已完成的工作")
    p(doc,
      "代码骨架已实现并验证，包含：数据生成与分层切分、8 张表的数据库与哈希链审计、"
      "三种检测器（规则/TF-IDF 双粒度）、5 族字符级攻击、手写指标（TPR@FPR / AUC / PRF / ASR）、"
      "分块滑窗检测、RAG 靶子、FastAPI 接口与命令行工具，共 31 个文件已提交至 git。"
      "三条演示命令（demo / truncation / rag）均可直接运行，输出即为表 7 的来源。")

    h2(doc, "7.2 技术可行性")
    bullet(doc, "所用技术栈（Python + scikit-learn + PyTorch/Transformers）均为成熟开源方案，社区资料充分；")
    bullet(doc, "检测模型选用 DistilBERT（约 66M 参数），在 M 系列芯片上微调可在数十分钟内完成，无需 GPU 集群；")
    bullet(doc, "文本空间攻击中，字符级攻击完全本地实现、零成本；语义改写调用 API 的量可控（每轮 100–200 次）；")
    bullet(doc, "所有公开数据集均可免费下载，不存在数据获取障碍。")

    h2(doc, "7.3 时间可行性")
    p(doc,
      "本课题需要正视一个现实约束：作者同期在准备研究生入学考试（2026 年 12 月）。"
      "因此进度安排必须把“重实验”放在寒假，而把考试前的时间用于**低认知负荷、可中断**的数据工作。"
      "这也是把交付物设计为“开题报告 + 可运行骨架 + 可复现实验脚本”的原因："
      "骨架的跑通把后续工作的不确定性降到了“跑实验”这一件事上。")

    # ---------- 八 ----------
    h1(doc, "八、进度安排")
    table(doc,
          ["时间", "任务", "产出"],
          [
              ["2026.10 下 – 12 中", "数据拼装与清洗；难负样本构造；基线跑通（每周 ≤ 3 小时）", "数据集 v1、基线结果"],
              ["2026.12 下", "考试周，项目暂停", "—"],
              ["2027.01 中 – 02 底", "DistilBERT 微调；攻击谱系实现；A1–A4 实验（寒假重仓）", "鲁棒性矩阵初稿"],
              ["2027.03 上", "红队回路对抗训练；A5–A6；系统整合与控制台", "系统 v1、中期检查材料"],
              ["2027.03 下 – 04 上", "A7 端到端实验；补实验；论文初稿", "论文初稿"],
              ["2027.04 中 – 05 上", "论文修改、查重、图表定稿", "论文定稿"],
              ["2027.05", "答辩准备与答辩", "答辩 PPT"],
          ],
          widths=[3.8, 8.2, 4.0],
          caption_text="表 8  进度安排")
    p(doc,
      "关于图表：用例图、ER 图、架构图三张图安排在论文定稿阶段统一绘制。"
      "其中 ER 图已可由数据库 schema 自动生成（仓库内 injlab/db/er.py，输出 Mermaid 源码），"
      "避免手工画图与实现不一致——图与代码不同步是评审最容易抓的问题之一。")

    # ---------- 九 ----------
    h1(doc, "九、预期成果与创新点")

    h2(doc, "9.1 预期成果")
    bullet(doc, "一篇本科毕业论文；")
    bullet(doc, "一套可复现的实验代码与数据构造脚本（含基线、攻击、训练、评测）；")
    bullet(doc, "一个可运行的三级防护原型系统（API + 控制台 + 审计日志）；")
    bullet(doc, "一份注入检测基准（含难负样本与 6 族攻击变体）及其标注规范。")

    h2(doc, "9.2 创新点")
    p(doc,
      "**创新点 1：文本空间攻击谱系下的检测器鲁棒性基准。** 现有工作多在固定攻击集上报告准确率，"
      "本课题把攻击组织成有现实对应物的谱系，并在固定误报率下给出可比矩阵，"
      "同时用“identity 攻击的 ASR 应等于 1 − TPR”作为链路自检，保证评测框架本身没有算错。")
    p(doc,
      "**创新点 2：对“嵌入空间对抗训练可迁移”的证伪性检验。** 把 FGSM/PGD 对抗训练作为对照组，"
      "与红队回路训练在同一条文本空间攻击谱系上比较。这是一个“无论正负都有结论”的实验设计，"
      "避免了“我们的方法更好”这类无法证伪的表述。")
    p(doc,
      "**创新点 3：把工程漏洞纳入安全评测。** 截断攻击与编码混淆这两类问题不需要任何机器学习技巧，"
      "却能让检测器整体失效。把它们与前两类方法学问题放在同一张表里报告，"
      "体现“安全是系统工程”而非“模型精度”的基本立场。")
    p(doc,
      "**创新点 4：带审计链的检测服务。** 用哈希链保证检测记录不可抵赖，"
      "把“模型可解释”推进到“判定可追溯”，在同类本科工作中较少见，且实现成本很低。")

    # ---------- 十 ----------
    h1(doc, "十、风险与应对")
    table(doc,
          ["风险", "可能性", "影响", "应对"],
          [
              ["真实数据标注工作量超预期", "高", "中", "先用模板+LLM 生成打底，人工只标注难负样本与边界案例"],
              ["红队回路效果不显著", "中", "高", "按证伪设计：负结果同样成文，改为“对抗训练在注入检测中无效”的结论"],
              ["语义改写依赖 API，成本/稳定性受限", "中", "中", "本地小模型兜底；固定随机种子缓存生成结果"],
              ["MPS 后端兼容问题导致训练中断", "中", "低", "训练脚本支持 --device cpu 回退，先小规模验证再放大"],
              ["考研期间进度停滞", "高", "中", "考试前只做可中断的数据工作；核心实验集中在寒假"],
              ["与已有工作撞车", "低", "中", "开题后每月检索一次 arXiv cs.CR，及时调整叙事侧重"],
          ],
          widths=[5.4, 1.8, 1.6, 7.4],
          caption_text="表 9  风险与应对")

    # ---------- 十一 ----------
    h1(doc, "十一、参考文献")
    refs = [
        "PEREZ F, RIBEIRO I. Ignore Previous Prompt: Attack Techniques For Language Models[J]. arXiv preprint arXiv:2211.09527, 2022. (NeurIPS 2022 ML Safety Workshop)",
        "GRESHAKE K, ABDELNABI S, MISHRA S, et al. Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection[J]. arXiv preprint arXiv:2302.12173, 2023.",
        "LIU Y, JIA Y, GENG R, et al. Formalizing and Benchmarking Prompt Injection Attacks and Defenses[C]//USENIX Security Symposium, 2024. arXiv:2310.12815.",
        "YI J, XIE Y, ZHU B, et al. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models[C]//KDD, 2025. arXiv:2312.14197.",
        "WALLACE E, XIAO K, LEIKE R, et al. The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions[J]. arXiv preprint arXiv:2404.13208, 2024.",
        "HINES K, LOPEZ G, HALL M, et al. Defending Against Indirect Prompt Injection Attacks With Spotlighting[J]. arXiv preprint arXiv:2403.14720, 2024.",
        "CHEN S, PIET J, SITAWARIN C, et al. StruQ: Defending Against Prompt Injection with Structured Queries[C]//USENIX Security Symposium, 2025. arXiv:2402.06363.",
        "CHEN S, ZHARMAGAMBETOV A, MAHLOUJIFAR S, et al. SecAlign: Defending Against Prompt Injection with Preference Optimization[C]//ACM CCS, 2025. arXiv:2410.05451.",
        "DEBENEDETTI E, SHUMAILOV I, FAN T, et al. Defeating Prompt Injections by Design[J]. arXiv preprint arXiv:2503.18813, 2025.",
        "DEBENEDETTI E, ZHANG J, BALUNOVIĆ M, et al. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents[J]. arXiv preprint arXiv:2406.13352, 2024.",
        "ZOU A, WANG Z, CARLINI N, et al. Universal and Transferable Adversarial Attacks on Aligned Language Models[J]. arXiv preprint arXiv:2307.15043, 2023.",
        "CHAO P, ROBEY A, DOBRIBAN E, et al. Jailbreaking Black Box Large Language Models in Twenty Queries[J]. arXiv preprint arXiv:2310.08419, 2023.",
        "RUSSINOVICH M, SALEM A, ELDAN R. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack[C]//USENIX Security Symposium, 2025. arXiv:2404.01833.",
        "MAZEIKA M, PHAN L, YIN X, et al. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal[J]. arXiv preprint arXiv:2402.04249, 2024.",
        "HAN S, RAO K, ETTINGER A, et al. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs[C]//NeurIPS, 2024. arXiv:2406.18495.",
        "LIN Z, WANG Z, TONG Y, et al. ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation[C]//EMNLP Findings, 2023. arXiv:2310.17389.",
        "RÖTTGER P, KIRK H R, VIDGEN B, et al. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models[C]//NAACL, 2024. arXiv:2308.01263.",
        "GOODFELLOW I J, SHLENS J, SZEGEDY C. Explaining and Harnessing Adversarial Examples[C]//ICLR, 2015. arXiv:1412.6572.",
        "JIN D, JIN Z, ZHOU J T, et al. Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment[C]//AAAI, 2020. arXiv:1907.11932.",
        "ZHU C, CHENG Y, GAN Z, et al. FreeLB: Enhanced Adversarial Training for Natural Language Understanding[C]//ICLR, 2020. arXiv:1909.11764.",
        "LIU X, CHENG H, HE P, et al. Adversarial Training for Large Neural Language Models[J]. arXiv preprint arXiv:2004.08994, 2020.",
        "OWASP. OWASP Top 10 for LLM Applications (LLM01: Prompt Injection)[EB/OL]. 2025. https://genai.owasp.org/",
    ]
    for i, ref in enumerate(refs, 1):
        para = doc.add_paragraph()
        para.paragraph_format.left_indent = Cm(0.85)
        para.paragraph_format.first_line_indent = Cm(-0.85)
        para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        para.paragraph_format.space_after = Pt(6)
        run = para.add_run(f"[{i}]\t{ref}")
        set_run_font(run, size=10.5)

    doc.save(OUT)
    print(f"已生成 {OUT}")


if __name__ == "__main__":
    build()
