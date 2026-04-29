# ecommerce_customer_sft_5000 生成报告

## 基本信息

- dataset name: ecommerce_customer_sft_5000
- file: LLaMA-Factory/data/ecommerce_customer_sft_5000.json
- count: 5000
- single: 3500
- multi: 1500
- ratio: 7:3

## 分类分布

- 人工投诉: 100
- 优惠活动: 350
- 发票: 250
- 售后服务: 1400
- 商品咨询: 1500
- 物流跟踪: 450
- 订单查询: 400
- 退换货: 550

## 轮次分布

- multi: 1500
- single: 3500

## 回答风格分布

- boundary: 597
- confirm: 925
- direct: 1545
- empathy: 226
- risk: 501
- short: 1206

## 生成约束

- 参考 `数据规范.md` 和 `电商客服合成数据风格参考.md`。
- 重点生成商品咨询和售后服务。
- 多轮历史使用真实追问，不使用规则总结式接话。
- 扫描禁用词：亲亲、小妹、么么、mua、还有其他可以帮您的吗、您好，我来帮您查询一下。
