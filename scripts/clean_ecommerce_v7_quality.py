import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "LLaMA-Factory" / "data"
REPORT_DIR = ROOT / "项目一" / "数据处理"

INPUT_PATH = DATA_DIR / "ecommerce_customer_sft_v6_cleaned.json"
OUTPUT_PATH = DATA_DIR / "ecommerce_customer_sft_v7_quality.json"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"
REPORT_PATH = REPORT_DIR / "ecommerce_customer_sft_v7_quality_report.md"

DATASET_NAME = "ecommerce_customer_sft_v7_quality"


LEADING_FILLERS = re.compile(
    r"^(?:亲亲|亲爱的|亲爱哒|亲|您好呀|您好呢|您好|你好呀|你好|好的呢|好哒|好的|尊敬的客户|客户您好)"
    r"[\s，,。.!！~、]*"
)

DROP_SENTENCE_PATTERNS = [
    "请问还有什么",
    "还有什么能够帮",
    "还有其他",
    "有什么问题我可以帮",
    "有什么可以帮您",
    "欢迎咨询京东物流商家在线客服",
    "尊敬的商家朋友",
    "祝您生活愉快",
    "祝您购物愉快",
    "感谢您对京东的支持",
    "有问题欢迎随时召唤",
    "一直与您同在",
    "赴汤蹈火",
    "家大卖",
    "订单接到手软",
    "么么",
    "mua",
]

CUTESY_PATTERNS = [
    (re.compile(r"亲爱哒|亲爱的|亲亲"), ""),
    (re.compile(r"(?<!相)亲(?!属|人|自)"), ""),
    (re.compile(r"小妹|妹子"), "我"),
    (re.compile(r"这边帮您"), "我会帮您"),
    (re.compile(r"这边为您"), "我会为您"),
    (re.compile(r"这边会"), "我会"),
    (re.compile(r"辛苦亲爱的"), "请您"),
    (re.compile(r"辛苦您"), "请您"),
    (re.compile(r"辛苦"), "请"),
    (re.compile(r"哈[~～]*"), ""),
    (re.compile(r"哦[~～]*"), ""),
    (re.compile(r"呢[~～]*"), ""),
    (re.compile(r"[~～]+"), ""),
]

BAD_OUTPUT_PATTERNS = [
    "尊敬的商家",
    "欢迎咨询京东物流商家在线客服",
    "家大卖",
    "订单接到手软",
    "赴汤蹈火",
    "么么",
    "mua",
    "辛苦亲爱的",
    "有什么问题我可以帮",
    "有什么可以帮您",
    "请问还有什么",
    "还有其他",
    "审核下再投",
    "一体机发车",
    "分拣中心",
    "已经解锁",
    "系统下架商品",
    "上架",
    "商家在线客服",
    "尊敬的商家",
    "欢迎光临",
    "私人服务专员",
    "请问您看中哪款商品",
    "正为您查询",
    "稍微等待",
    "稍微等等",
    "请问是这个商品吗",
    "请问是这个订单的问题吗",
    "此单有什么问题吗",
    "外呼号码",
    "反馈站点",
    "联系站点",
    "站点尽快",
    "商品编码",
    "商品快照",
    "条形码",
    "是不是这个商品",
    "这个商品吗",
    "请问是这个商品",
    "这个商品的么",
    "这个手机吗",
    "这个订单吗",
    "妹纸",
    "不要不要",
    "特别特别伤心",
    "拼购店铺",
    "贵司",
    "商家缴费",
    "on 您好",
    "非常有眼光",
    "性价比真是超高",
    "建议您留着使用",
]

LOW_CONTEXT_INSTRUCTION = re.compile(
    r"^(?:好|好的|好吧|嗯|嗯嗯|哦|OK|Ok|ok|可以|行|也行吧|谢谢|是的|对|不用了|没有了)[。.!！?？\s]*$"
)

CODE_LIKE_INSTRUCTION = re.compile(
    r"^(?:ID:|#[A-Za-z]-|[A-Z]{1,4}\[数字x\]|[A-Z]{1,4}\d|\[数字x\]\s*;;)"
)

ACTION_WORDS = (
    "提供",
    "查看",
    "核实",
    "确认",
    "申请",
    "提交",
    "上传",
    "联系",
    "处理",
    "退款",
    "退货",
    "换货",
    "取消",
    "拦截",
    "拒收",
    "配送",
    "发货",
    "下单",
    "补发",
    "补开",
    "修改",
    "查询",
    "反馈",
)

INTERNAL_CONTEXT_PATTERNS = [
    "顾客通过",
    "系统下架",
    "审核上架",
    "商品编码",
    "商家",
    "外单",
    "换单打印",
    "催投",
    "顾客手机号",
]

DOMAIN_KEYWORDS = [
    "订单",
    "发票",
    "价保",
    "差价",
    "优惠",
    "优惠券",
    "活动",
    "退款",
    "退货",
    "换货",
    "售后",
    "服务单",
    "快递",
    "物流",
    "配送",
    "派送",
    "签收",
    "发货",
    "取消",
    "地址",
    "安装",
    "维修",
    "质保",
    "保质期",
    "运费",
    "包装",
    "破损",
    "质量",
    "正品",
    "会员",
    "PLUS",
    "客服",
    "投诉",
    "人工",
    "补发",
    "漏发",
    "拒收",
]


def normalize_space(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([，。！？；：、])\s*", r"\1", text)
    text = re.sub(r"([，。！？；：、]){2,}", r"\1", text)
    text = text.replace("，。", "。").replace("。，", "。")
    return text.strip(" ，,")


def ensure_terminal_punctuation(text: str) -> str:
    if text and not re.search(r"[。！？?!]$", text):
        return text + "。"
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？?；;])", text)
    return [p.strip() for p in parts if p.strip()]


def strip_template_sentences(text: str) -> tuple[str, list[str]]:
    removed = []
    kept = []
    for sentence in split_sentences(text):
        if any(pat in sentence for pat in DROP_SENTENCE_PATTERNS):
            removed.append(sentence)
        else:
            kept.append(sentence)
    return normalize_space("".join(kept)), removed


def clean_answer(text: str) -> tuple[str, list[str]]:
    before = text or ""
    notes = []
    text = normalize_space(before)

    while True:
        new_text = LEADING_FILLERS.sub("", text).strip()
        if new_text == text:
            break
        text = new_text
        notes.append("remove_leading_filler")

    text, removed_sentences = strip_template_sentences(text)
    if removed_sentences:
        notes.append("remove_template_sentence")

    for pattern, replacement in CUTESY_PATTERNS:
        new_text = pattern.sub(replacement, text)
        if new_text != text:
            notes.append("normalize_cutesy_tone")
            text = new_text

    text = re.sub(r"您好[，,。!！\s]*", "", text)
    text = text.replace("这边", "")
    text = re.sub(r"我我会", "我会", text)
    text = re.sub(r"([，。！？；：])的[。！？]?$", r"\1", text)
    text = re.sub(r"([。！？；])+", r"\1", text)
    text = ensure_terminal_punctuation(normalize_space(text))

    if before.strip() != text:
        notes.append("answer_changed")
    return text, sorted(set(notes))


def clean_history(history):
    cleaned = []
    changed = False
    if not isinstance(history, list):
        return [], bool(history)

    for item in history:
        if not isinstance(item, list) or len(item) != 2:
            changed = True
            continue
        user_turn = normalize_space(str(item[0]))
        assistant_turn, _ = clean_answer(str(item[1]))
        if not user_turn or not assistant_turn:
            changed = True
            continue
        cleaned.append([user_turn, assistant_turn])
        changed = changed or user_turn != item[0] or assistant_turn != item[1]
    return cleaned, changed


def has_enough_actionable_content(text: str) -> bool:
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if len(chinese_chars) >= 28:
        return True
    return len(chinese_chars) >= 16 and any(word in text for word in ACTION_WORDS)


def should_drop(sample, cleaned_output: str) -> str | None:
    instruction = normalize_space(str(sample.get("instruction", "")))
    history = sample.get("history") or []
    source = sample.get("metadata", {}).get("source", "")

    if source == "synthetic_v5":
        return "drop_synthetic_v5_misalignment_risk"
    if not instruction or not cleaned_output:
        return "drop_empty_instruction_or_output"
    if any(pat in instruction for pat in INTERNAL_CONTEXT_PATTERNS):
        return "drop_internal_or_merchant_context"
    if LOW_CONTEXT_INSTRUCTION.match(instruction):
        return "drop_low_context_instruction"
    instruction_without_placeholders = re.sub(r"\[[^\]]+x\]", "", instruction)
    instruction_without_placeholders = re.sub(r"[A-Z]{1,4}\d*[A-Z0-9]*", "", instruction_without_placeholders)
    natural_chars = re.findall(r"[\u4e00-\u9fff]", instruction_without_placeholders)
    domain_text = instruction_without_placeholders + cleaned_output
    if source != "curated_sop_v2" and not any(keyword in domain_text for keyword in DOMAIN_KEYWORDS):
        return "drop_no_customer_service_domain_signal"
    if CODE_LIKE_INSTRUCTION.match(instruction) and len(natural_chars) < 6:
        return "drop_code_like_instruction"
    if instruction.startswith("[数字x]") and len(natural_chars) <= 6:
        return "drop_placeholder_heavy_low_context_instruction"
    if instruction.startswith("[订单编号:") and len(natural_chars) <= 10:
        return "drop_order_card_without_customer_question"
    if instruction.startswith("[链接x]") and len(natural_chars) <= 8:
        return "drop_link_only_or_greeting_instruction"
    if any(pat in cleaned_output for pat in BAD_OUTPUT_PATTERNS):
        return "drop_remaining_bad_template"
    if not has_enough_actionable_content(cleaned_output):
        return "drop_too_short_or_not_actionable"
    return None


def update_dataset_info():
    info = json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8"))
    info[DATASET_NAME] = {
        "file_name": OUTPUT_PATH.name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
            "history": "history",
        },
    }
    DATASET_INFO_PATH.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    output = []
    drop_reasons = Counter()
    change_reasons = Counter()
    before_source = Counter()
    after_source = Counter()
    seen_outputs = {}
    examples = []

    for idx, sample in enumerate(data):
        before_source[sample.get("metadata", {}).get("source", "")] += 1
        new_sample = deepcopy(sample)
        cleaned_output, notes = clean_answer(str(sample.get("output", "")))
        cleaned_history, history_changed = clean_history(sample.get("history", []))
        reason = should_drop(sample, cleaned_output)
        if reason:
            drop_reasons[reason] += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "action": "drop",
                        "reason": reason,
                        "instruction": sample.get("instruction", ""),
                        "before": sample.get("output", ""),
                        "after": cleaned_output,
                    }
                )
            continue

        duplicate_key = normalize_space(cleaned_output)
        if duplicate_key in seen_outputs:
            drop_reasons["drop_duplicate_cleaned_output"] += 1
            continue
        seen_outputs[duplicate_key] = idx

        new_sample["instruction"] = normalize_space(str(sample.get("instruction", "")))
        new_sample["input"] = normalize_space(str(sample.get("input", "")))
        new_sample["output"] = cleaned_output
        new_sample["history"] = cleaned_history
        metadata = dict(new_sample.get("metadata", {}))
        metadata["quality_cleaned_from"] = "ecommerce_customer_sft_v6_cleaned"
        if notes or history_changed:
            metadata["quality_clean_notes"] = notes + (["history_changed"] if history_changed else [])
        new_sample["metadata"] = metadata
        output.append(new_sample)
        after_source[metadata.get("source", "")] += 1

        for note in notes:
            change_reasons[note] += 1
        if history_changed:
            change_reasons["history_changed"] += 1

        if sample.get("output", "").strip() != cleaned_output and len(examples) < 20:
            examples.append(
                {
                    "action": "change",
                    "reason": ",".join(notes),
                    "instruction": sample.get("instruction", ""),
                    "before": sample.get("output", ""),
                    "after": cleaned_output,
                }
            )

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    update_dataset_info()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    duplicate_counter = Counter(item["output"] for item in output)
    template_counts = {
        pat: sum(1 for item in output if pat in item["output"])
        for pat in ["亲亲", "您好", "还有其他", "请问还有", "有什么问题我可以帮", "小妹", "么么", "mua"]
    }

    lines = [
        "# ecommerce_customer_sft_v7_quality 清洗报告",
        "",
        "## 总览",
        "",
        f"- 输入文件：`{INPUT_PATH.relative_to(ROOT)}`",
        f"- 输出文件：`{OUTPUT_PATH.relative_to(ROOT)}`",
        f"- 输入条数：{len(data)}",
        f"- 输出条数：{len(output)}",
        f"- 删除条数：{len(data) - len(output)}",
        f"- 清洗后 exact duplicate output 数：{sum(1 for _, v in duplicate_counter.items() if v > 1)}",
        "",
        "## 删除原因",
        "",
    ]
    for reason, count in drop_reasons.most_common():
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## 修改原因", ""])
    for reason, count in change_reasons.most_common():
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## 来源分布", "", "### 清洗前", ""])
    for source, count in before_source.most_common():
        lines.append(f"- {source}: {count}")
    lines.extend(["", "### 清洗后", ""])
    for source, count in after_source.most_common():
        lines.append(f"- {source}: {count}")

    lines.extend(["", "## 清洗后模板词残留", ""])
    for pat, count in template_counts.items():
        lines.append(f"- {pat}: {count}")

    lines.extend(["", "## 样例", ""])
    for i, example in enumerate(examples, 1):
        lines.extend(
            [
                f"### {i}. {example['action']} / {example['reason']}",
                "",
                f"- Q: {example['instruction']}",
                f"- Before: {example['before']}",
                f"- After: {example['after']}",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"input_count={len(data)}")
    print(f"output_count={len(output)}")
    print(f"removed_count={len(data) - len(output)}")
    print("drop_reasons=", dict(drop_reasons.most_common()))
    print("change_reasons=", dict(change_reasons.most_common()))
    print("after_source=", dict(after_source.most_common()))
    print(f"output={OUTPUT_PATH}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
