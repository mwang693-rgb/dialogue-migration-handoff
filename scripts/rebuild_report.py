"""Recompute every headline without a GPU, torch, API or model download."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handoff import ROOT, analyze_all


def main():
    result = analyze_all()
    out = ROOT/'reports'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    e, s, t = result['editing'], result['strategy'], result['style']
    lines = ['# 队友交接：三类探索实验结果', '', '整理：2026-11-04；实验记录：2026-10-10至2026-10-14。本报告由已保存输出离线重算，新增模型调用0。', '',
             '## 先看结论', '', '这次探索主要在选数据和选指标。已确定共享历史的对照流程，尚未确定一个稳定、跨题材的恢复目标。最适合课程proposal主线的是“回应规则的连续性”；局部编辑提供清楚的正确性检查；具体风格表达作为负结果附录。没有必要把所有steering变体写进proposal。', '',
             '## 1. 局部编辑：完整修改与内容保留', '', '12份Qwen生成的冻结成果：书页、函数、表格各4份；每份两个独立编辑请求，共24题。早期N/P实验的N表示角色重放B，不是在线A。此批未测在线A。各方法使用相同历史和当前请求。', '',
             '|方法|完整编辑正确|与原生全文相同|保留条目代理（书页/表格）|', '|---|---:|---:|---:|']
    labels = {'N':'原生角色重放 B（旧名N）','P':'整体粘贴 C（旧名P）','Hint':'C＋固定续接提示','D':'C＋单历史格式差向量','RandomD':'C＋同范数随机方向','ReverseD':'C＋反向格式差向量'}
    for m, v in e['metrics'].items():
        lines.append(f"|{labels[m]}|{v['joint_correct']}/24|{v['exact_native_text']}/24|{v['retained_items']}/{v['expected_retained_items']}|")
    lines += ['', '保留条目计数只检查应保留页面/整行是否仍存在，不自动检查顺序或每个字段。代码以AST整体比较，未执行生成代码。完整编辑分数同时要求指定修改与其余结构正确。', '',
              '可读例子：artifact-book-0 / replace_page，原生保留了全部7页非目标内容，却漏掉替换句末尾要求的数字1，所以完整修改失败。该失败不是“保留内容失败”。', '',
              '本批粘贴正确17题，超过原生15题；不能把原生文本当质量金标准。固定提示19题，简单差向量11题，因此不支持差向量优于提示。六个方法是交接时从原九方法中挑选的完整列；没有按题目成功与否筛选。', '',
              '### KL补充：必须在共同前缀上逐位置检查', '', '同3份旧开发历史×2请求×N/P两条参考路径×5条件，共60条分布记录。参考前缀使用真实生成token IDs，各条件强制读同一前缀，逐位置比较下一token分布。KL(原生分布‖条件分布)；不是整序列KL、不是自由文本正确性。', '',
              '|题目（N参考/P条件）|旧稀疏均值KL|逐位置均值KL|首次自由分叉位置（零起始）|该位置KL|', '|---|---:|---:|---:|---:|']
    for r in e['dense_examples']:
        lines.append(f"|{r['case_id']}/{r['query_id']}|{r['old_sparse_mean_recomputed64']:.6g}|{r['dense_mean_kl']:.6g}|{r['first_native_paste_free_divergence']}|{r['kl_at_first_free_divergence']}|")
    lines += ['', '旧稀疏点会漏掉真正分叉，不能用接近零的稀疏KL证明回答相同。离线程序复算保存的逐位置KL/JS均值，不宣称能仅凭这些摘要重新生成全部词表概率；重新计算概率需要模型前向。', '',
              '## 2. 回应策略：公开MT-Eval改造', '', '8份公开历史，四类规则各两个版本：JSON、列表、双版本回答、固定结尾。原始确认和前三个任务回复均由Qwen实际生成；问题4/5各自独立续聊。全部80条最终输出公开，历史7/8在类别checker上连续合规，未按最终掉分筛选。', '',
              '|类别|在线A|角色重放B|粘贴C|B分段空对照|C＋明确规则提示|', '|---|---:|---:|---:|---:|---:|']
    names = {'format:json_format':'JSON','format:bullet_list':'列表','combination:two_responses':'双版本回答','startend:end_phrase':'固定结尾'}
    for category, metrics in s['groups'].items():
        vals=[f"{metrics[m]['passed']}/{metrics[m]['n']}" for m in ['AOnline','BReplay','CPasted','BSplitNull','CSpecificRule']]
        lines.append('|'+names[category]+'|'+'|'.join(vals)+'|')
    lines += ['', '主要例子：mteval-24的疫苗请求，A/B保持双版本，C只给一个；明确规则提示修复。这个请求及历史任务满足额外的200词要求。完整8份都保留，不能只展示这一个例子。评分是原作者类别checker：双版本主要检查分隔形式，不保证两版本语义不同或事实正确；主表也不自动包含第二条字数要求。', '',
              '### 同源扩展', '', '对两份双版本历史的原剩余问题6—10全部继续测试，共10个相关请求、40个输出。这个扩展由首批缺口触发，未新增独立历史，没有在线A组。', '', '|条件|双版本合规|','|---|---:|']
    for m, v in s['extension']['methods'].items(): lines.append(f"|{m}|{v['passed']}/{v['n']}|")
    lines += ['', '扩展B/C都是9/10，没有新增B对C错的缺口；共同失败是邮件问题。首批与扩展合并只能描述B13/14、C12/14、提示14/14，独立历史只有2份，不能称14个独立样本或稳定总体效应。扩展v1有一次未写盘生成后实现失败；v2的40条是本仓库公开结果，原失败不计入成功率。', '',
              '## 3. 具体风格：缩略语变化能否在新问题确认', '', '借MT-Eval问题流，不用其参考assistant回复；首轮加入口语要求，Qwen逐轮实际作答形成历史。旧批六份口语历史，保留/删除首轮风格子句×B/C×两个追问，48条输出。新批四份此前未在已检查60个历史文件中出现的问题流，3轮历史后独立续聊；只保证记录范围内的问题文本去重，不保证全新语义领域。', '',
              '固定30个缩略语的无歧义形式计数。密度=100×命中次数/正则词数；先对同历史两个普通追问平均，再对历史等权。另看去Markdown后的前120空白分词窗口，窗口内正则词数可能略超过120。', '',
              '|批次|B实际缩略语总数|C实际缩略语总数|C密度更高的历史|平均C−B密度|','|---|---:|---:|---:|---:|']
    for label, batch in [('旧开发批（6历史）',t['old']),('新确认批（4历史）',t['new'])]:
        v=batch['full'];lines.append(f"|{label}|{v['counts']['BKeep']}|{v['counts']['CKeep']}|{v['positive_sources']}/{len(v['source_rows'])}|{v['mean_c_minus_b']:.5f}|")
    lines += ['', '新批正常回答A/B/C总次数6/7/6；只有2/4历史C密度更高（另1条相同），均值被一份历史拉高。旧批6/6同向未稳定复现。删除首轮要求后，新批B/C均无命中；明确覆盖成正式拼写时A/B/C各4/4无命中。这个指标只覆盖已列缩略语，不代表完整风格或内容质量。', '',
              '**新批未通过预设继续门槛，steering输出为0。** 门槛包括至少3/4历史同向和C至少多4个短形式等工作性要求，非显著性检验。不得把准备过的代码说成已运行的干预，也不能换分母后宣布成功。', '',
              '## 如何用于课程proposal', '', '问题：共享历史改变输入结构后，历史回应规则是否延续？数据处理：公开问题→Qwen逐轮生成→冻结→A/B/C独立续聊。主要终点建议用明确规则的合规率，并加入用户合法覆盖请求；编辑实验说明如何同时评价目标修改与非目标保留。全文exact/KL用于补充，不作为唯一成功标准。风格确认作为我们为什么没有直接采用embedding总分的探索记录。', '',
              '这三类试验不是同一任务，不能把正确数相加报一个总恢复率；以上数据全是已看过的开发探索。下一次新实验需重新冻结数据与指标。课程目标是跑清楚一项比较，不承诺发表、新方法或稳赢的steering效果。', '',
              '## 代码与来源', '', '`experiments/`是选择后的原文/原token/原评分；`provenance/original_files.json`记录源文件SHA和筛选；`reference_code/`是对应历史程序与本地依赖的逐字快照，不是开箱即用的部署目录。`scripts/rebuild_report.py`可在CPU重算本报告全部表格；GPU入口另见README。原始工程和其他探索结果未被修改。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    from handoff import rows
    edit_rows = rows('experiments/01_editing/outputs.jsonl')
    history = {h['case_id']: h for h in rows('experiments/01_editing/histories.jsonl')}
    pair = next(r for r in edit_rows if r['method']=='N' and not r['checker']['all_pass'] and any(x['case_id']==r['case_id'] and x['query_id']==r['query_id'] and x['method']=='P' and x['checker']['all_pass'] for x in edit_rows))
    query = next(q for q in history[pair['case_id']]['queries'] if q['query_id']==pair['query_id'])
    cases = ['# 三个可以直接读的例子', '', '这些是看过结果后为说明评分选的例子，不是独立测试或总体比例。完整小批次均在experiments中。', '', '## 编辑：粘贴也可能比原生正确', '', f"案例：{pair['case_id']} / {pair['query_id']}", '', '当前请求：', '', query['text'], '', '指标依据是实际冻结成果加本次修改要求；下面保持原文。']
    for m in ['N','P','Hint','D']:
        r = next(x for x in edit_rows if x['case_id']==pair['case_id'] and x['query_id']==pair['query_id'] and x['method']==m)
        cases += ['', f"### {m}：完整修改{'通过' if r['checker']['all_pass'] else '失败'}", '', '~~~~text', r['text'], '~~~~']
    gap = s['first_batch_gaps'][0]
    hs = {h['case_id']:h for h in rows('experiments/02_strategy/histories.jsonl')}
    h = hs[gap['case_id']]
    cases += ['', '## 策略：双版本变成单版本', '', '历史约定：'+h['rule_text'], '', '当前请求：'+h['queries'][gap['query']], '', '前三个实际任务回复都通过类别checker；这条最终请求的具体例子如下。']
    for m in ['AOnline','BReplay','CPasted','CSpecificRule']:
        cases += ['', '### '+m, '', '~~~~text', gap['outputs'][m], '~~~~']
    cases += ['', '后续同两历史的新请求B/C都9/10，所以这个单点不能写成稳定大批现象。', '', '## 风格：新四历史方向不一致', '', '|新历史|完整文本C−B缩略语密度|', '|---|---:|']
    for r in t['new']['full']['source_rows']:
        cases.append(f"|{r['case_id']}|{r['c_minus_b']:+.5f}|")
    cases += ['', '新批平均仍为正，但只有两条为正。必须同时看分历史结果和实际命中总量，不能只看平均值。']
    (out/'CASE_STUDIES_ZH.md').write_text('\n'.join(cases)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(verified=True, editing_outputs=144, dense_records=60, strategy_outputs=120, style_outputs=100, new_model_calls=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
