import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "LLaMA-Factory" / "data"
REPORT_DIR = ROOT / "项目一" / "数据处理"

SFT_PATH = DATA_DIR / "ecommerce_customer_sft_v7_quality.json"
DPO_PATH = DATA_DIR / "ecommerce_customer_dpo_v1.json"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"
REPORT_PATH = REPORT_DIR / "ecommerce_customer_dpo_v1_report.md"

DATASET_NAME = "ecommerce_customer_dpo_v1"
MAX_SAMPLES = 300


CATEGORY_KEYWORDS = {
    "物流配送": ["物流", "快递", "配送", "派送", "签收", "发货", "送货", "地址", "拒收", "站点"],
    "发票": ["发票", "抬头", "税号", "专票", "电子票", "增票"],
    "售后退换": ["售后", "退货", "换货", "退款", "服务单", "维修", "质量", "破损", "包装", "检测"],
    "价保优惠": ["价保", "差价", "优惠", "优惠券", "活动", "降价", "价格保护"],
    "安装预约": ["安装", "师傅", "预约", "上门"],
    "人工投诉": ["人工", "投诉", "客服", "转人工"],
}

REJECTED_TEMPLATES = {
    "物流配送": [
        "亲亲，物流问题请您耐心等待，快递会尽快处理的，请问还有其他可以帮您的吗？",
        "您好，这个要看物流更新，暂时没有办法确认，建议您再等等。",
    ],
    "发票": [
        "亲亲，发票问题请您在页面自己看一下，一般都是可以处理的，还有其他问题吗？",
        "您好，发票以系统显示为准，具体能不能改我这边也不清楚。",
    ],
    "售后退换": [
        "亲亲，售后问题请您提交申请后等待审核，具体结果以页面为准，请问还有其他可以帮您的吗？",
        "您好，这个需要售后处理，我这边无法判断，您先等通知就可以。",
    ],
    "价保优惠": [
        "亲亲，优惠和价保都是系统判断的，页面能申请就申请，不能申请就没有办法。",
        "您好，活动规则比较复杂，建议您自己看页面说明，请问还有其他问题吗？",
    ],
    "安装预约": [
        "亲亲，安装问题师傅会联系您的，您耐心等待就可以，还有其他可以帮您的吗？",
        "您好，安装时间这个说不准，等师傅电话通知即可。",
    ],
    "人工投诉": [
        "亲亲，请不要着急，您的心情我理解，请问还有其他可以帮您的吗？",
        "您好，这个问题我也处理不了，建议您自己找人工。",
    ],
}


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"\s*([，。！？；：、])\s*", r"\1", text)
    return text


def detect_category(sample: dict) -> str:
    text = normalize(sample.get("instruction", "")) + normalize(sample.get("output", ""))
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "通用"


def is_good_candidate(sample: dict) -> bool:
    instruction = normalize(sample.get("instruction", ""))
    output = normalize(sample.get("output", ""))
    if len(re.findall(r"[\u4e00-\u9fff]", instruction)) < 6:
        return False
    if len(re.findall(r"[\u4e00-\u9fff]", output)) < 28:
        return False
    if len(output) > 280:
        return False
    bad_fragments = [
        "请问还有",
        "还有其他",
        "亲亲",
        "您好",
        "欢迎光临",
        "正为您查询",
        "不是很懂",
        "不太懂",
        "耐心等待",
        "烦请等待",
        "请您等待",
        "这个单子",
        "您看可以吗",
        "可以吗?",
        "可以吗？",
        "很快为您送达",
    ]
    return not any(fragment in output for fragment in bad_fragments)


def build_rejected(category: str, index: int) -> str:
    templates = REJECTED_TEMPLATES[category]
    return templates[index % len(templates)]


def update_dataset_info():
    info = json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8"))
    info[DATASET_NAME] = {
        "file_name": DPO_PATH.name,
        "ranking": True,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "chosen": "chosen",
            "rejected": "rejected",
            "system": "system",
            "history": "history",
        },
    }
    DATASET_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    sft_data = json.loads(SFT_PATH.read_text(encoding="utf-8"))
    candidates = [sample for sample in sft_data if is_good_candidate(sample)]
    buckets: dict[str, list[dict]] = {category: [] for category in CATEGORY_KEYWORDS}
    for sample in candidates:
        category = detect_category(sample)
        if category in buckets:
            buckets[category].append(sample)

    selected = []
    # Round-robin sampling keeps DPO from being dominated by one business category.
    while len(selected) < MAX_SAMPLES:
        progressed = False
        for category in buckets:
            if buckets[category]:
                selected.append((category, buckets[category].pop(0)))
                progressed = True
                if len(selected) >= MAX_SAMPLES:
                    break
        if not progressed:
            break

    dpo_data = []
    category_counter = Counter()
    for idx, (category, sample) in enumerate(selected):
        rejected = build_rejected(category, idx)
        chosen = normalize(sample["output"])
        if chosen == rejected:
            continue
        category_counter[category] += 1
        dpo_data.append(
            {
                "instruction": normalize(sample["instruction"]),
                "input": normalize(sample.get("input", "")),
                "chosen": chosen,
                "rejected": rejected,
                "system": sample.get("system", ""),
                "history": sample.get("history", []),
                "metadata": {
                    "source_dataset": "ecommerce_customer_sft_v7_quality",
                    "source": sample.get("metadata", {}).get("source", ""),
                    "category": category,
                    "preference_goal": "prefer actionable customer-service answers over repetitive template replies",
                },
            }
        )

    DPO_PATH.write_text(json.dumps(dpo_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_dataset_info()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    repeated_rejected = Counter(item["rejected"] for item in dpo_data)
    lines = [
        "# ecommerce_customer_dpo_v1 构建报告",
        "",
        "## 总览",
        "",
        f"- 来源 SFT 数据：`{SFT_PATH.relative_to(ROOT)}`",
        f"- 输出 DPO 数据：`{DPO_PATH.relative_to(ROOT)}`",
        f"- SFT 输入条数：{len(sft_data)}",
        f"- 候选条数：{len(candidates)}",
        f"- DPO 输出条数：{len(dpo_data)}",
        "- chosen：来自 v7_quality 中保留的真实/精修可执行客服回答",
        "- rejected：构造为模板复读、空泛安抚、只让等待或答非所问的低质量回答",
        "",
        "## 类别分布",
        "",
    ]
    for category, count in category_counter.most_common():
        lines.append(f"- {category}: {count}")

    lines.extend(["", "## rejected 模板复用统计", ""])
    for rejected, count in repeated_rejected.most_common():
        lines.append(f"- {count} x {rejected}")

    lines.extend(["", "## 样例", ""])
    for i, item in enumerate(dpo_data[:12], 1):
        lines.extend(
            [
                f"### {i}. {item['metadata']['category']}",
                "",
                f"- Q: {item['instruction']}",
                f"- Chosen: {item['chosen']}",
                f"- Rejected: {item['rejected']}",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"sft_count={len(sft_data)}")
    print(f"candidate_count={len(candidates)}")
    print(f"dpo_count={len(dpo_data)}")
    print(f"category_counter={dict(category_counter)}")
    print(f"output={DPO_PATH}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
