# 对话迁移实验：课程队友交接包

**先读 [实验结果报告](reports/RESULTS_ZH.md)，再读 [交接日志](docs/EXPERIMENT_LOG.md)。** 想直接看原文，可读[三个具体例子](reports/CASE_STUDIES_ZH.md)。这里保留三类能看清楚的数据/指标尝试，不是整个研究工程，也不是投稿结论。

我们研究：同一份历史从多轮消息变为整体粘贴，下一轮的编辑结果、回应规则或表达方式是否变化？目前固定的是处理流程；尚未确定的是最合适的数据集和主要评测目标。

|保留的尝试|为什么保留|主要结果|
|---|---|---|
|书页/代码/表格局部编辑|任务和答案可直接检查，区分修改与保留|24题：B15、C17、提示19、单历史向量11题完整正确|
|公开MT-Eval回应规则|有公开材料、三组对照、明确规则|双版本首批A/B4/4、C3/4；扩展B/C均9/10|
|缩略语这一具体表达|无需把embedding余弦当“味道”|旧批C密度更高6/6，新批2/4；未启动新批steering|

KL补充放在第一类中，展示旧稀疏取点为什么可能漏掉真正分叉。三类任务分开分析，不相加成“总成功率”。交接选择发生在看到结果之后，目的为易读；不用于估计普遍迁移损失。

## 三组到底是什么

- **AOnline：** Qwen逐轮生成历史，保留实际KV缓存，再接最终请求。
- **BReplay：** 相同历史正文，按user/assistant角色从零重放。
- **CPasted：** 相同历史加文字角色标签，整体放入一条user消息。

主要格式比较为B/C；A/B控制重新计算和分段路线。每个追问从冻结历史独立启动。公开问题只提供用户材料，主实验的assistant历史由Qwen生成；数据集原作者的回复不能充当Qwen在线A组。

**不是每个历史试验都有A组。** 编辑旧名N/P分别对应B/C；扩展双版本问题只有B/C、分段和提示。不得把旧N直接改名为A。不同阶段粘贴包装和生成上限也不完全相同，原始配置各自保留，不能跨阶段混成一个受控消融。

## 不需要GPU：复算已保存结果

Python 3.10+，仅标准库，不安装任何模型，不需要API key：

```bash
python scripts/rebuild_report.py
python -m unittest discover -s tests -v
python scripts/verify_release.py
```

第一条核对原有checker结果，重算表格并生成`reports/summary.json`与中文报告。它复算文本、缩略语和已存逐位置KL/JS的均值；**没有仅凭摘要重建完整词表概率**。结果文本和token未经改写，重复调用不会启动GPU。

代码AST的冻结记录来自较新Python，交接重算对3.11/3.12只兼容新增的空`type_params=[]`字段；不忽略非空类型参数或正文字符串，不改变原有案例的通过/失败。首个Linux CI发现的版本不兼容及修复提交保留在仓库历史中。

## 可选：在新目录重跑三组

```bash
pip install -r requirements-gpu.txt
python scripts/run_new_threeway.py \
  --experiment strategy \
  --model /path/to/local/Qwen2.5-7B-Instruct \
  --max-cases 1 \
  --out runs/strategy-new
```

也支持`--experiment style`。它读取选定案例的原始用户轮，**重新生成新的历史**，再跑A/B/C。新历史与旧记录可能不同，结果另存`runs/`，不能称原结果逐token复现，也不能与旧回答混合计分。开始前把`--max-cases`设小，不会自动扫参数。需要本地模型、CUDA、足够显存；没有下载权重、租卡或后台循环。

这个入口复用`reference_code/`中原有OnlineEngine，历史/当前请求会实际记录input IDs、缓存长度和停止类型。新入口已通过语法和命令行检查，指标另有CPU测试；**没有为交接再跑GPU验证**。历史中出现异常角色token、截断或token重分词不一致时，保存失败并终止，不悄悄删例重试。

旧编辑干预、KL计算及原始生成程序在`reference_code/`，是逐字源码快照，依赖旧目录布局、预先生成的向量和本地模型配置；不宣称直接在这个裁剪仓库执行旧`main()`就能重现。其原始配置、代码SHA、已保存数据和结果均可检查。队友优先用上面的CPU复算和三组新批入口。

## 文件导航

```text
docs/EXPERIMENT_LOG.md          全部路线的压缩概览＋本次选择理由
docs/PROPOSAL_OUTLINE.md        一页课程proposal框架（待做与已做分开）
reports/RESULTS_ZH.md           三类选定实验的结果、解释和局限
reports/summary.json            机器可读的重算结果
experiments/01_editing/         12历史、144选定方法输出、60分布记录
experiments/02_strategy/        8历史、80首批输出、40扩展输出
experiments/03_style/           6旧口语历史/48输出、4新历史/52输出
reference_code/                 18个历史脚本及其本地依赖快照
third_party/mt_eval/            原checker、MIT许可、来源记录
provenance/original_files.json  源文件SHA、投影与选择范围
selection.json                 为什么保留/省略这些数据
```

364条最终输出和60条分布记录不是424个独立样本。编辑有12份历史，策略首批8份（扩展复用其中2份），风格旧6份/新4份。跨方法、同历史追问都相关。

实验模型为Qwen2.5-7B-Instruct BF16，greedy，单5090。当前仓库为离线交接产物，没有正在运行的实验或后台监控。整理日期2026-11-04；保存结果来自2026-10-10至2026-10-14，不代表此后服务器实时状态。

代码许可见[LICENSE](LICENSE)，数据与第三方来源见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。未发布权重、KV张量、账号凭据、服务器登录信息、真实WildChat聊天或全部旧实验目录。
