#!/usr/bin/env python
"""Build a cleaner ecommerce customer-service SFT dataset.

The first dataset was converted from turn-level JDDC chats. It is usable for
pipeline validation, but it over-represents greetings, closing questions and
short acknowledgements. This script creates v2 by keeping business-solving
answers, limiting repeated templates, and adding a small curated SOP-style set
without leaking the fixed eval set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "LLaMA-Factory" / "data" / "ecommerce_customer_sft.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "LLaMA-Factory" / "data" / "ecommerce_customer_sft_v2.json"
DEFAULT_DATASET_INFO = PROJECT_ROOT / "LLaMA-Factory" / "data" / "dataset_info.json"
DEFAULT_REPORT = PROJECT_ROOT / "项目一" / "数据处理" / "ecommerce_customer_sft_v2_report.md"

SYSTEM_PROMPT = (
    "你是京东电商场景的中文客服助手。请根据顾客问题和已有对话上下文，"
    "给出礼貌、准确、可执行的客服回复。回答应先解决用户问题，再给出必要的安抚和兜底说明，"
    "不要重复同一句话。"
)

BUSINESS_KEYWORDS = {
    "订单",
    "物流",
    "快递",
    "配送",
    "发货",
    "签收",
    "收货",
    "地址",
    "电话",
    "退货",
    "退款",
    "换货",
    "售后",
    "服务单",
    "发票",
    "购物清单",
    "优惠",
    "优惠券",
    "满减",
    "活动",
    "价保",
    "破损",
    "坏了",
    "损坏",
    "漏发",
    "少发",
    "补发",
    "配件",
    "取消",
    "拒收",
    "运费",
    "质量",
    "保修",
    "维修",
    "仓库",
    "商品",
    "货物",
    "支付",
    "到账",
    "审核",
    "检测",
    "投诉",
    "人工",
    "客服",
}

ACTION_KEYWORDS = {
    "提供",
    "申请",
    "提交",
    "查看",
    "确认",
    "核实",
    "反馈",
    "联系",
    "处理",
    "修改",
    "取消",
    "拒收",
    "上传",
    "保留",
    "查询",
    "留意",
    "关注",
    "下载",
    "补充",
    "协助",
}

TEMPLATE_ONLY_PATTERNS = [
    r"还有.*帮.*吗",
    r"其他.*帮.*吗",
    r"有什么问题.*帮您",
    r"感谢您对京东的支持",
    r"生活愉快",
    r"再见",
    r"评价",
    r"打赏",
    r"期待再次为您服务",
    r"祝福您",
    r"怠慢",
    r"善解人意",
    r"宽容的客户",
    r"太客气",
    r"客气啦",
    r"分内之事",
    r"包容心",
    r"乐意为您服务",
    r"随时为您解答",
    r"随时提问",
    r"问题是否完全",
    r"咨询之前的问题",
    r"稍等",
    r"马上.*查询",
    r"正在.*查询",
    r"正在.*核实",
]

LOW_INFO_INPUTS = {
    "你好",
    "您好",
    "谢谢",
    "谢谢你",
    "好的",
    "好",
    "可以",
    "嗯",
    "嗯嗯",
    "没有了",
    "没了",
    "不用",
    "不需要",
    "是的",
    "对",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def has_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def is_placeholder_only(text: str) -> bool:
    stripped = normalize_space(text)
    if not stripped:
        return True

    scrubbed = re.sub(r"\[[^\]]+x\]", "", stripped)
    scrubbed = re.sub(r"[订单编号金额下单时间日期数字商品ID咨询配送商品信息:：,，\-\s]+", "", scrubbed)
    return len(scrubbed) <= 1


def has_repeated_sentence(text: str) -> bool:
    segments = [seg.strip() for seg in re.split(r"[。！？!?；;\n]+", text) if len(seg.strip()) >= 6]
    return bool(segments and Counter(segments).most_common(1)[0][1] >= 2)


def is_template_only_output(output: str) -> bool:
    compact = normalize_space(output)
    if len(compact) < 16:
        return True
    return any(re.search(pattern, compact) for pattern in TEMPLATE_ONLY_PATTERNS)


def sample_text(sample: dict[str, Any]) -> str:
    history = sample.get("history") or []
    history_text = "\n".join(f"{user}\n{assistant}" for user, assistant in history)
    return "\n".join([sample.get("instruction", ""), sample.get("input", ""), history_text, sample.get("output", "")])


def drop_reason(sample: dict[str, Any]) -> str | None:
    instruction = normalize_space(sample.get("instruction", ""))
    output = normalize_space(sample.get("output", ""))
    combined = sample_text(sample)

    if instruction in LOW_INFO_INPUTS or is_placeholder_only(instruction):
        return "low_info_input"
    if is_template_only_output(output):
        return "template_or_short_output"
    if has_repeated_sentence(output):
        return "self_repeated_output"
    if not has_any(combined, BUSINESS_KEYWORDS):
        return "no_business_signal"
    if len(output) < 28 and not has_any(output, ACTION_KEYWORDS):
        return "short_without_action"
    return None


def cap_repeated_outputs(samples: list[dict[str, Any]], max_per_exact_output: int) -> tuple[list[dict[str, Any]], int]:
    seen: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for sample in samples:
        output = normalize_space(sample.get("output", ""))
        if seen[output] >= max_per_exact_output:
            dropped += 1
            continue
        seen[output] += 1
        kept.append(sample)

    return kept, dropped


def make_sample(
    instruction: str,
    output: str,
    history: list[list[str]] | None = None,
    category: str = "",
    source: str = "curated_sop_v2",
) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": SYSTEM_PROMPT,
        "history": history or [],
        "metadata": {
            "source": source,
            "category": category,
        },
    }


def make_generated_output(category: str, query: str, action: str, boundary: str, closing: str, variant: int) -> str:
    brief = re.sub(r"[？?。.!！]+$", "", normalize_space(query))
    if len(brief) > 30:
        brief = brief[:30]

    openers = [
        f"亲亲，关于您咨询的“{brief}”，我先帮您说明一下。",
        f"针对“{brief}”这种情况，可以按下面方式处理。",
        f"您提到“{brief}”，这个需要结合订单状态和页面规则确认。",
        f"很抱歉给您带来不便，关于“{brief}”我会尽量帮您核实处理。",
        f"理解您的情况，“{brief}”需要先确认具体订单和处理节点。",
        f"这个问题我帮您梳理一下，“{brief}”通常需要按规则核实。",
        f"亲亲，您这个“{brief}”问题可以先这样处理。",
        f"关于“{brief}”，建议先按页面规则和订单状态来判断。",
    ]
    transitions = [
        "建议您先",
        "您可以先",
        "为了更快处理，建议您",
        "我这边可以协助您",
        "请您先",
        "处理时需要先",
        "您可以在页面先",
        "我会根据您提供的信息协助",
    ]
    opener = openers[variant % len(openers)]
    transition = transitions[variant % len(transitions)]
    output = f"{opener}{transition}{action}{boundary}{closing}"
    output = output.replace("先先", "先")
    output = output.replace("协助进入", "协助您进入")
    return output


def make_generated_query(query: str, variant: int) -> str:
    query = normalize_space(query)
    variants = [
        query,
        f"麻烦帮我看一下，{query}",
        f"您好，{query}",
        f"这个订单有点着急，{query}",
        f"我想确认一下，{query}",
        f"请问一下，{query}",
        f"帮我处理一下可以吗？{query}",
        f"我不太清楚规则，{query}",
    ]
    return variants[variant % len(variants)]


def build_generated_sop_samples() -> list[dict[str, Any]]:
    """Generate broad business-solving SOP samples across ecommerce scenarios.

    These are intentionally moderate-sized and rule-based. They are used to
    broaden scenario coverage, not to replace real customer-service data.
    """
    scenarios = [
        {
            "category": "地址修改",
            "queries": [
                "收货地址写成老家了，现在能不能改成公司？",
                "地址少写了门牌号，会不会影响配送？",
                "我想把收货电话换成我家人的号码。",
                "快递到一半了，能不能帮我改到附近驿站？",
                "订单还没出库，我要改一下省市区信息。",
                "地址填错小区了，配送员会联系我吗？",
                "预售订单的地址现在可以修改吗？",
                "我搬家了，之前下的订单能不能改地址？",
            ],
            "action": "提供正确的收货信息和订单号，我会根据订单是否出库、是否进入配送环节来提交修改或拦截申请；如果页面支持自助修改，也可以在订单详情页直接操作。",
            "boundary": "如果商品已经出库、进入派送或新旧地址跨度较大，可能无法保证修改成功，具体以仓库和物流处理结果为准。",
            "closing": "请您保持电话畅通，避免影响后续配送。",
        },
        {
            "category": "物流催促",
            "queries": [
                "快递两天没更新了，能帮我催一下吗？",
                "预计今天送达，但是现在还没有派送。",
                "物流卡在中转站不动了怎么办？",
                "明天要用这个商品，可以加急吗？",
                "配送员一直不联系我，你们能催吗？",
                "包裹到了本地但不派送，怎么回事？",
                "物流显示异常，帮我看一下。",
                "我想知道这个订单大概什么时候能到。",
            ],
            "action": "提供订单号或物流单号，我会帮您查询当前物流节点，并反馈物流核实或催促配送；若页面有配送员电话，也建议您同步尝试联系确认派送时间。",
            "boundary": "实际送达时间会受天气、站点派送量和物流运输影响，是否能加急需要以物流处理结果为准。",
            "closing": "后续请您留意物流更新和电话通知。",
        },
        {
            "category": "签收异常",
            "queries": [
                "物流显示签收了，但我没有收到货。",
                "快递被别人签收了，我该怎么办？",
                "显示放快递柜了，可是我没有取件码。",
                "门卫说没收到这个包裹，帮我查一下。",
                "订单签收地址不对，是不是送错了？",
                "快递员说放门口了，现在找不到了。",
                "系统显示本人签收，但不是我签的。",
                "代收点查不到我的包裹。",
            ],
            "action": "先确认家人、门卫、快递柜或代收点是否代收，同时提供订单号和物流信息，我会帮您反馈物流核实签收记录、派送位置和签收凭证。",
            "boundary": "是否丢件或误签需要物流核查后确认，处理方案也会以核查结果为准。",
            "closing": "建议您保留相关沟通记录，方便后续售后跟进。",
        },
        {
            "category": "退货退款",
            "queries": [
                "我不想要了，怎么申请退货退款？",
                "退货申请在哪里提交？",
                "拆封看了一下还能退吗？",
                "超过七天了还能不能退？",
                "退款一直没到账，帮我查一下。",
                "退货被拒了，我想复核。",
                "我选错退款原因了，能修改吗？",
                "收到货不满意，想退回去。",
            ],
            "action": "在订单详情页提交售后申请，选择退货退款并按页面提示填写原因、上传凭证和寄回商品；如果已经有服务单，请提供服务单号，我可以帮您查询审核和退款进度。",
            "boundary": "是否支持退货会根据商品类型、签收时间、包装配件和是否影响二次销售判断，退款到账时间以原支付渠道为准。",
            "closing": "请您先不要丢弃商品、包装和配件，以免影响售后审核。",
        },
        {
            "category": "换货",
            "queries": [
                "商品有问题，我想换一件新的。",
                "买错型号了，可以换型号吗？",
                "换货的新商品什么时候发出？",
                "换回来的商品还是坏的怎么办？",
                "颜色买错了，能不能换颜色？",
                "换货需要我先寄回去吗？",
                "申请换货后一直没有进展。",
                "配件不能用，可以只换配件吗？",
            ],
            "action": "在订单售后页面提交换货申请，并上传商品问题照片、包装照片或检测说明；如果需要更换型号、颜色或配件，我会帮您根据库存和售后规则核实是否支持。",
            "boundary": "换货通常需要经过审核、寄回、验收和重新发出，是否能换指定款式要以商品规则和库存为准。",
            "closing": "请您关注服务单进度，页面有更新时会同步显示处理结果。",
        },
        {
            "category": "商品破损",
            "queries": [
                "外包装压坏了，里面商品也破了。",
                "收到的玻璃杯碎了，怎么售后？",
                "液体商品漏出来了，包装都是湿的。",
                "签收后发现商品裂开了。",
                "快递盒破损严重，我担心商品有问题。",
                "商品边角磕碰了，可以换吗？",
                "运输导致商品变形了怎么办？",
                "收到时包装已经被拆过了。",
            ],
            "action": "保留商品、外包装、快递面单和破损部位照片，在订单售后页面提交退换货申请；我也可以帮您反馈售后尽快核实。",
            "boundary": "是否需要寄回、补发或退款，要根据售后审核和商品破损情况确认。",
            "closing": "建议您尽快提交凭证，避免超过售后处理时效。",
        },
        {
            "category": "漏发少件",
            "queries": [
                "买了两件只收到一件。",
                "包装里少了说明书和配件。",
                "套装里面缺一个小部件。",
                "订单显示全部签收，但实际少了一箱。",
                "赠品没有收到，可以补发吗？",
                "拆箱发现少发了一个颜色。",
                "商品主体到了，充电器没在盒子里。",
                "多件商品是不是分开发的？",
            ],
            "action": "先核对订单是否分包裹发出，再提供订单号、收到的商品照片、外包装照片和缺少配件名称，我会帮您反馈仓库或售后核实。",
            "boundary": "是否属于漏发、是否支持补发或退款，需要以仓库复核和售后审核结果为准。",
            "closing": "请您暂时保留外包装和面单，方便核实包裹信息。",
        },
        {
            "category": "发票",
            "queries": [
                "我要开发票，入口在哪里？",
                "电子发票下载不了怎么办？",
                "发票抬头写错了能修改吗？",
                "税号填错了，可以重开吗？",
                "单位要求纸质发票，能寄吗？",
                "发票金额和订单金额不一致。",
                "发票一直没开出来，帮我催一下。",
                "购物清单能不能盖章？",
            ],
            "action": "进入订单详情页查看发票入口，按页面提示填写抬头、税号和发票类型；如果已开票或页面无法操作，请提供订单号和正确开票信息，我会帮您核实是否支持补开、冲红重开或购物清单申请。",
            "boundary": "发票类型、重开规则和寄送方式会受订单、商品和财务规则影响，最终以页面规则和财务审核为准。",
            "closing": "请您确认抬头和税号准确后再提交，避免后续反复修改。",
        },
        {
            "category": "优惠券活动",
            "queries": [
                "优惠券为什么结算用不了？",
                "平台券和店铺券能叠加吗？",
                "满减活动没有生效怎么办？",
                "活动价和页面宣传不一样。",
                "券过期了能补发吗？",
                "商品参加活动但购物车没减价。",
                "新人券领取后找不到了。",
                "秒杀价没抢到，能按活动价补吗？",
            ],
            "action": "查看优惠券使用规则、适用商品、金额门槛、活动时间和结算页提示；如果仍有疑问，可以提供券信息、商品链接和页面截图，我会帮您核实具体原因。",
            "boundary": "优惠是否生效以结算页展示为准，过期券、限品类券或不可叠加活动通常无法强制使用。",
            "closing": "建议您下单前先在结算页确认最终应付金额。",
        },
        {
            "category": "价保差价",
            "queries": [
                "刚买完就降价了，能退差价吗？",
                "价保入口在哪里找？",
                "为什么我的订单不能申请价保？",
                "活动价比我买的时候低，能补吗？",
                "价保申请失败了怎么回事？",
                "赠品变化算不算价保？",
                "优惠券导致便宜了，可以价保吗？",
                "超过价保时间还能补差价吗？",
            ],
            "action": "在订单详情页查看是否有价格保护入口，并按页面提示提交申请；如果页面提示不支持，请提供订单号和降价截图，我可以帮您核实规则。",
            "boundary": "价保会受商品类型、活动规则、购买时间、赠品和券后价格影响，是否通过以系统审核结果为准。",
            "closing": "建议您在价保有效期内尽快提交，避免超过申请时间。",
        },
        {
            "category": "订单取消",
            "queries": [
                "刚下单不想要了，怎么取消？",
                "订单取消一直审核中。",
                "已经出库了还能取消吗？",
                "预售订单能不能取消？",
                "取消订单后多久退款？",
                "取消失败了怎么办？",
                "我想撤销取消申请。",
                "订单被自动取消了，优惠会退吗？",
            ],
            "action": "进入订单详情页查看取消入口，若订单尚未出库通常可以申请取消；如果取消申请审核中，我可以帮您反馈催促并查询订单状态。",
            "boundary": "已出库、已发货或预售定金类订单可能无法直接取消，退款和优惠返还要以订单规则和页面结果为准。",
            "closing": "请您关注订单状态变化，避免重复提交影响处理。",
        },
        {
            "category": "支付退款",
            "queries": [
                "付款失败但银行卡扣钱了怎么办？",
                "退款显示成功但我没到账。",
                "用了白条支付，退款退到哪里？",
                "银行卡支付退款要多久？",
                "我想换一种支付方式可以吗？",
                "订单重复支付了怎么处理？",
                "支付时提示风控拦截。",
                "退款能不能退到别的账户？",
            ],
            "action": "提供订单号和支付信息，我会帮您核实订单支付和退款状态；退款通常按原支付路径退回，您也可以同步查看支付渠道账单。",
            "boundary": "不同支付方式到账时间不同，是否能更换支付方式或退款账户需要以支付规则为准。",
            "closing": "如果支付渠道已经扣款但订单未成功，建议您先保留扣款记录便于核查。",
        },
        {
            "category": "售后时效",
            "queries": [
                "服务单提交后多久审核？",
                "售后一直没人处理。",
                "商品寄回签收了但服务单没更新。",
                "检测时间太久了，能催吗？",
                "商家不同意退款，我可以申诉吗？",
                "售后审核失败了还能重新申请吗？",
                "服务单被关闭了怎么办？",
                "平台介入入口在哪里？",
            ],
            "action": "提供服务单号，我会帮您查询当前审核、入库、检测或退款节点，并反馈催促；如果您不认可处理结果，可以查看服务单页面是否支持申诉或平台介入。",
            "boundary": "售后处理会受商品类型、物流入库、检测排队和商家审核影响，最终结果以售后审核为准。",
            "closing": "建议您补充清晰凭证，这有助于售后更快判断问题。",
        },
        {
            "category": "安装预约",
            "queries": [
                "大家电什么时候上门安装？",
                "空调安装师傅一直没联系我。",
                "能改安装时间吗？",
                "安装要不要额外收费？",
                "送货和安装能安排同一天吗？",
                "师傅上门迟到了怎么办？",
                "我想取消安装预约。",
                "安装地址写错了能改吗？",
            ],
            "action": "提供订单号和期望安装时间，我会帮您查看安装服务状态并反馈预约或改约；如果页面支持自助预约，也可以在订单服务入口操作。",
            "boundary": "上门时间会受服务网点排班、地区和商品类型影响，额外收费项目需要以安装服务规则和师傅现场确认结果为准。",
            "closing": "请您保持电话畅通，方便师傅上门前联系确认。",
        },
        {
            "category": "维修保修",
            "queries": [
                "商品用了几个月坏了，还能保修吗？",
                "保修期从什么时候开始算？",
                "维修需要寄到哪里？",
                "维修要收费吗？",
                "保修卡丢了怎么办？",
                "返修后多久能寄回？",
                "检测说人为损坏，我不认可。",
                "商品过保了还能维修吗？",
            ],
            "action": "在订单售后页面提交维修或返修申请，并上传故障说明、商品照片和购买凭证；如果已有服务单，请提供服务单号，我帮您查询检测和维修进度。",
            "boundary": "是否免费保修会根据保修期限、故障原因和检测结论判断，人为损坏或过保可能产生费用。",
            "closing": "建议您寄修前备份个人数据并保留寄回物流凭证。",
        },
        {
            "category": "商品咨询",
            "queries": [
                "这个商品适合什么型号使用？",
                "尺寸参数在哪里看？",
                "页面写的规格我没看懂。",
                "商品是不是正品？",
                "这个版本和另一个版本有什么区别？",
                "包装里包含哪些配件？",
                "食品保质期大概多久？",
                "衣服尺码怎么选？",
            ],
            "action": "先查看商品详情页的规格参数、包装清单和售后说明；如果页面信息不清楚，您可以提供商品链接或截图，我会帮您反馈商家或根据页面信息进一步核实。",
            "boundary": "商品参数、适配型号和保质期等信息需要以商品详情页、实物包装或商家确认为准。",
            "closing": "建议您下单前确认规格型号，避免买错影响使用。",
        },
        {
            "category": "库存预售",
            "queries": [
                "这个商品什么时候补货？",
                "预售尾款什么时候付？",
                "预售商品什么时候发货？",
                "缺货了能不能帮我锁库存？",
                "到货提醒怎么设置？",
                "预售定金能退吗？",
                "商品显示采购中是什么意思？",
                "门店有货但线上没货怎么办？",
            ],
            "action": "查看商品页库存、预售规则和预计发货时间；如果页面支持到货提醒，您可以先开启提醒，我也可以帮您根据商品信息核实当前状态。",
            "boundary": "补货时间、预售发货和定金退还会受供应和活动规则影响，具体以商品页和订单页展示为准。",
            "closing": "建议您关注页面更新，库存恢复后尽快下单。",
        },
        {
            "category": "账户会员",
            "queries": [
                "我的会员权益怎么没生效？",
                "积分在哪里查看？",
                "账号绑定的手机号不用了怎么办？",
                "会员券为什么没有到账？",
                "账号登录异常怎么办？",
                "实名认证信息能修改吗？",
                "PLUS会员可以退款吗？",
                "积分兑换失败了怎么处理？",
            ],
            "action": "进入账户或会员中心查看权益、积分和券包状态；如果页面显示异常，请提供账号信息截图或相关订单号，我会帮您反馈核实。",
            "boundary": "会员权益、积分有效期和账号信息修改会受账户规则影响，部分信息需要您在安全验证后自行操作。",
            "closing": "请您注意保护账户隐私，不要在聊天中发送完整密码或验证码。",
        },
        {
            "category": "安全风控",
            "queries": [
                "下单提示账户有风险怎么办？",
                "为什么我的订单被风控取消了？",
                "支付时让我验证身份。",
                "系统提示交易异常，能解除吗？",
                "优惠券使用被限制了。",
                "账号被锁定了怎么处理？",
                "收不到验证码怎么办？",
                "别人用我账号下单了怎么办？",
            ],
            "action": "根据页面提示完成身份验证，或通过账户安全中心查看异常原因；如果涉及订单或支付限制，请提供订单号和页面提示截图，我会帮您反馈核实。",
            "boundary": "风控规则涉及账户安全，客服无法直接绕过系统限制，是否解除需要以平台安全审核为准。",
            "closing": "如果怀疑账号被盗，请尽快修改密码并检查绑定手机和收货地址。",
        },
        {
            "category": "投诉安抚",
            "queries": [
                "你们处理太慢了，我很生气。",
                "客服一直没有解决我的问题。",
                "这个结果我不接受。",
                "我要投诉商家服务态度。",
                "物流和售后互相推脱怎么办？",
                "我已经说了很多遍了还是没人管。",
                "这次购物体验太差了。",
                "我要求平台介入处理。",
            ],
            "action": "记录您的问题和诉求，并根据订单或服务单状态反馈对应部门核实；如果页面支持投诉、申诉或平台介入，也建议您同步提交相关凭证。",
            "boundary": "赔付、退款或责任认定需要根据订单规则、凭证和审核结果判断，客服不能直接承诺超出规则的处理。",
            "closing": "很抱歉给您带来不好的体验，我会尽量协助您推进处理。",
        },
        {
            "category": "转人工",
            "queries": [
                "我要找人工客服。",
                "这个问题你处理不了，帮我转人工。",
                "我不想跟机器人说了。",
                "能不能安排专员联系我？",
                "我要投诉，需要人工处理。",
                "页面没有转人工入口怎么办？",
                "帮我记录一下让人工回电。",
                "我需要人工确认售后结果。",
            ],
            "action": "如果当前页面支持转人工入口，您可以直接发起人工服务；我也可以帮您整理订单号、问题描述和诉求，方便人工客服接入后快速处理。",
            "boundary": "是否能立即接入人工会受当前排队量和服务时间影响，回电也需要以页面或人工客服安排为准。",
            "closing": "建议您提前准备好订单号和相关截图，减少重复沟通。",
        },
    ]

    samples: list[dict[str, Any]] = []
    for scenario in scenarios:
        for query_idx, query in enumerate(scenario["queries"]):
            for variant in range(8):
                generated_query = make_generated_query(query, variant)
                output = make_generated_output(
                    scenario["category"],
                    generated_query,
                    scenario["action"],
                    scenario["boundary"],
                    scenario["closing"],
                    variant + query_idx,
                )
                samples.append(
                    make_sample(
                        instruction=generated_query,
                        output=output,
                        category=scenario["category"],
                        source="generated_sop_v2",
                    )
                )

    return samples


def build_curated_samples() -> list[dict[str, Any]]:
    """Small SOP-like supplement. Keep it broad and non-identical to eval items."""
    rows = [
        (
            "地址修改",
            "我下单后发现地址写错了，现在还能改吗？",
            "亲亲，可以先帮您查看订单状态。请您提供正确的收货地址和联系电话，如果订单还未出库或仍支持修改，我会为您提交地址修改申请；如果已经发货，则需要根据物流节点确认是否还能拦截或转寄。",
        ),
        (
            "地址修改",
            "快递已经发出来了，我想换到另一个城市收货可以吗？",
            "亲亲，商品发出后跨城市改址通常限制比较多，是否能处理需要看物流当前节点和承运方规则。若不支持转寄，建议您关注派送进度，必要时可拒收后重新用正确地址下单。",
        ),
        (
            "物流催促",
            "快递好几天没动了，是不是丢了？",
            "亲亲，先别着急。请您提供订单号或物流单号，我会帮您查询最新物流节点；如果确实长时间没有更新，会为您反馈物流核实，是否丢件需要以物流核查结果为准。",
        ),
        (
            "物流催促",
            "显示签收了但我没有拿到东西，怎么处理？",
            "很抱歉给您带来不便。建议您先确认家人、门卫、快递柜或代收点是否代收；如果仍未找到，请提供订单号，我会帮您反馈物流核实签收情况，后续以核查结果为准。",
        ),
        (
            "发票",
            "公司报销要发票，在哪里申请？",
            "亲亲，您可以进入订单详情页查看发票申请入口，按页面提示填写发票类型、抬头和税号等信息。如果页面无法申请，请提供订单号，我帮您查看是否支持补开发票。",
        ),
        (
            "发票",
            "发票抬头写错了，能不能重新开？",
            "可以帮您查看。请您提供订单号和正确抬头信息，若订单规则支持冲红重开，我会协助提交处理；是否可以重开以及处理时效，需要以财务审核结果为准。",
        ),
        (
            "退货退款",
            "商品收到后不想要了，想退掉可以吗？",
            "亲亲，如果商品符合退货政策且不影响二次销售，一般可以在订单详情页申请退货退款。请您按页面提示提交售后并寄回商品，最终是否通过以售后审核结果为准。",
        ),
        (
            "退货退款",
            "退货已经寄回去了，为什么钱还没退？",
            "亲亲，退货退款需要商家或仓库收到商品并完成验收后处理。审核通过后退款会按原支付路径退回，具体到账时间以支付渠道为准。您可以提供服务单号，我帮您查询进度。",
        ),
        (
            "换货",
            "我收到的东西有问题，能换一个新的吗？",
            "很抱歉影响您的使用。您可以在订单详情页提交换货售后，并上传商品问题照片、外包装照片等凭证；售后审核通过后，会根据库存和规则为您安排换货。",
        ),
        (
            "换货",
            "买错颜色了，可以直接换颜色吗？",
            "亲亲，是否能更换颜色需要看商品是否支持换货以及目标颜色是否有库存。您可以先在售后页面尝试申请换货；如果不支持换色，可能需要退货后重新下单。",
        ),
        (
            "优惠券活动",
            "为什么优惠券结算的时候用不了？",
            "优惠券无法使用通常和适用商品、订单金额、活动时间、账户限制或叠加规则有关。建议您先查看优惠券使用规则和结算页提示；如果仍不清楚，可以提供券信息和商品链接，我帮您核实。",
        ),
        (
            "优惠券活动",
            "刚买完就降价了，能不能退差价？",
            "亲亲，是否支持退差价需要看商品和订单是否符合价保规则。您可以在订单详情页查看是否有价保入口，或提供订单号，我帮您核实能否申请价格保护。",
        ),
        (
            "破损漏发",
            "我收到的商品外包装破了，里面也坏了怎么办？",
            "很抱歉给您带来不好的体验。请您先保留商品、外包装和快递面单，并拍摄破损照片，在订单售后页面提交退换货申请；我也会帮您反馈售后尽快核实处理。",
        ),
        (
            "破损漏发",
            "买了三件只收到两件，是不是漏发了？",
            "亲亲，建议您先核对包裹数量和物流信息，部分订单可能会分包裹发出。如果确认只收到两件，请提供订单号、收到商品照片和外包装照片，我会帮您反馈仓库或售后核实。",
        ),
        (
            "订单取消",
            "刚下单不想买了，在哪里取消？",
            "亲亲，您可以进入订单详情页查看是否有取消订单入口。若订单尚未出库，通常可以申请取消；若已出库或发货，可能无法直接取消，建议您关注物流并按规则拒收或申请售后。",
        ),
        (
            "订单取消",
            "订单显示发货了，还能直接取消退款吗？",
            "订单已发货后一般无法直接取消，需要根据物流状态处理。您可以尝试拒收，商品退回并验收后再按流程退款；具体处理结果以订单页面和售后审核为准。",
        ),
        (
            "售后时效",
            "我提交售后好几天了没人处理，能催一下吗？",
            "亲亲，很抱歉让您久等了。请您提供服务单号，我会帮您查询当前处理进度并反馈催促；如果超过页面承诺时效，也会尽量协助您继续跟进。",
        ),
        (
            "售后时效",
            "售后让我等检测，检测到底要多久？",
            "检测时效会受商品类型、物流到件和售后排队影响。请您提供服务单号，我帮您查看当前检测状态并反馈催促，最终结果以售后检测结论为准。",
        ),
        (
            "转人工投诉",
            "你说这些没用，我要找人工客服投诉。",
            "亲亲，很抱歉没能及时解决您的问题，也理解您的心情。我可以继续帮您记录并反馈问题；如果当前页面支持转人工或投诉入口，建议您按页面提示发起人工服务或提交投诉申请。",
        ),
        (
            "转人工投诉",
            "你一直答非所问，帮我转人工。",
            "非常抱歉给您带来不好的体验。您可以通过当前页面的转人工入口联系人工客服，建议您准备好订单号和问题描述，方便人工客服更快为您处理。",
        ),
    ]

    samples: list[dict[str, Any]] = []
    for category, instruction, output in rows:
        samples.append(make_sample(instruction, output, category=category))

    history_rows = [
        (
            "地址修改",
            [["我想把地址从上海改到安徽", "请问新旧地址是否在同一个城市呢？"]],
            "不在一个城市，这种还能帮我转寄吗？",
            "亲亲，新旧地址不在同一个城市时通常无法直接转站点再投或转寄。建议您关注物流进度，如果无法送达可尝试拒收，后续商品退回后再重新用正确地址下单。",
        ),
        (
            "退货退款",
            [["我申请退货被拒了", "页面提示商品影响二次销售"]],
            "我只是拆开看了一下，为什么不能退？",
            "亲亲，退货审核会根据商品状态、包装、配件和是否影响二次销售来判断。您可以补充商品实拍图和具体说明，我帮您反馈复核，但最终是否支持退货需要以售后审核结果为准。",
        ),
        (
            "破损漏发",
            [["我收到的杯子碎了", "外包装也有挤压痕迹"]],
            "这种情况还要我寄回去吗？",
            "亲亲，是否需要寄回要以售后审核要求为准。建议您先上传破损商品、外包装和快递面单照片，售后审核后会告知是否需要寄回以及后续处理方式。",
        ),
        (
            "优惠券活动",
            [["页面写满1000减150", "但是结算没有减免"]],
            "是不是你们活动骗人？",
            "亲亲，很抱歉让您有这样的感受。活动优惠通常需要满足指定商品、活动时间、金额门槛和券规则，结算页未减免可能是不满足条件。您可以提供商品链接和页面截图，我帮您核实具体原因。",
        ),
    ]
    for category, history, instruction, output in history_rows:
        samples.append(make_sample(instruction, output, history=history, category=category))

    samples.extend(build_generated_sop_samples())

    return samples


def update_dataset_info(dataset_info_path: Path, dataset_name: str, output_path: Path) -> None:
    with dataset_info_path.open("r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    dataset_info[dataset_name] = {
        "file_name": output_path.name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
            "history": "history",
        },
    }

    with dataset_info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_report(
    report_path: Path,
    source_count: int,
    final_samples: list[dict[str, Any]],
    drop_counts: Counter[str],
    cap_dropped: int,
    curated_count: int,
    max_per_exact_output: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = [normalize_space(sample["output"]) for sample in final_samples]
    sources = Counter(sample.get("metadata", {}).get("source", "unknown") for sample in final_samples)
    exact_dup_rate = 1 - len(set(outputs)) / len(outputs)
    short_outputs = sum(len(output) < 30 for output in outputs)
    template_hits = sum(any(re.search(pattern, output) for pattern in TEMPLATE_ONLY_PATTERNS) for output in outputs)

    lines = [
        "# ecommerce_customer_sft_v2 数据处理报告",
        "",
        "## 处理目标",
        "",
        "v2 的目标是降低纯寒暄、收尾、等待查询和短回复样本占比，提升“业务问题 -> 可执行客服答复”的样本比例，缓解 SFT 后模型在推理时重复输出模板句直到 token 上限的问题。",
        "",
        "## 样本规模",
        "",
        f"- 原始 v1 样本数：{source_count}",
        f"- v2 最终样本数：{len(final_samples)}",
        f"- 其中原始 JDDC 过滤保留：{sources.get('jddc_chat_0.1per', 0)}",
        f"- 其中 curated SOP 补充：{curated_count}",
        f"- 精确重复 output 比例：{exact_dup_rate:.2%}",
        f"- 单个完全相同 output 最大保留数：{max_per_exact_output}",
        f"- 短 output（<30 字）数量：{short_outputs}",
        f"- 命中模板收尾/等待规则的 output 数量：{template_hits}",
        "",
        "## 删除原因统计",
        "",
        "| 原因 | 数量 |",
        "| --- | ---: |",
    ]
    for reason, count in drop_counts.most_common():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            f"| exact_output_cap | {cap_dropped} |",
            "",
            "## v2 样本设计原则",
            "",
            "1. 业务解决型样本为主体，回答要包含判断、操作步骤、限制条件或后续处理路径。",
            "2. 允许温柔话术，例如“亲亲”“很抱歉”，但不允许纯话术成为主要答案。",
            "3. 高频收尾句和寒暄句做限频，避免模型把它们学成默认输出。",
            "4. 不把固定评测集原样加入训练，避免评测泄漏。",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    with args.source.open("r", encoding="utf-8") as f:
        source_samples = json.load(f)

    drop_counts: Counter[str] = Counter()
    filtered: list[dict[str, Any]] = []
    for sample in source_samples:
        reason = drop_reason(sample)
        if reason:
            drop_counts[reason] += 1
            continue

        cleaned = dict(sample)
        cleaned["system"] = SYSTEM_PROMPT
        cleaned.setdefault("metadata", {})
        filtered.append(cleaned)

    capped, cap_dropped = cap_repeated_outputs(filtered, args.max_per_exact_output)
    curated = build_curated_samples() if args.include_curated else []
    final_samples = capped + curated

    rng = random.Random(args.seed)
    rng.shuffle(final_samples)
    final_samples.sort(key=lambda item: (item.get("metadata", {}).get("source", ""), item.get("instruction", "")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if not args.skip_register:
        update_dataset_info(args.dataset_info, args.dataset_name, args.output)

    write_report(
        report_path=args.report,
        source_count=len(source_samples),
        final_samples=final_samples,
        drop_counts=drop_counts,
        cap_dropped=cap_dropped,
        curated_count=len(curated),
        max_per_exact_output=args.max_per_exact_output,
    )

    return {
        "source": str(args.source),
        "output": str(args.output),
        "dataset_name": args.dataset_name,
        "source_samples": len(source_samples),
        "final_samples": len(final_samples),
        "filtered_from_source": len(capped),
        "curated_samples": len(curated),
        "drop_counts": dict(drop_counts),
        "exact_output_cap_dropped": cap_dropped,
        "report": str(args.report),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--dataset-name", default="ecommerce_customer_sft_v2")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-per-exact-output", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--no-curated", dest="include_curated", action="store_false")
    parser.set_defaults(include_curated=True)
    args = parser.parse_args()

    summary = build_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
