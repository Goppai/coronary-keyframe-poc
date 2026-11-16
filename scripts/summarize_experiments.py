#!/usr/bin/env python3
"""Build a human-readable summary from measured JSON; no accuracy inference."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('docs/evaluation.md'))
    args=parser.parse_args()
    matrix=json.loads(args.matrix.read_text())
    native=json.loads(args.native.read_text())
    comparison=json.loads(args.comparison.read_text())
    headers='| 原始输入 | 分析最大边 | 重复 | 均值s | P50s | P95s | FPS均值 | CPU% | 峰值RSS MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|'
    lines=[headers]
    for data in (matrix,native):
        for group in data['groups']:
            s=group['summary']
            lines.append(f"| {group['dicom_id']} | {data['config']['preprocessing']['max_size']} | {s['repeat']} | {s['mean_seconds']:.3f} | {s['p50_seconds']:.3f} | {s['p95_seconds']:.3f} | {s['mean_fps']:.2f} | {s['mean_cpu_percent']:.1f} | {s['peak_rss_mb_sampled']:.1f} |")
    default=next(g for g in matrix['groups'] if g['dicom_id']=='phantom_512_100')['summary']
    full=native['groups'][0]['summary']
    stages=['| 阶段 | 缩放256均值s | 原生512均值s |','|---|---:|---:|']
    for key in default['stage_mean_seconds']:
        stages.append(f"| {key} | {default['stage_mean_seconds'][key]:.4f} | {full['stage_mean_seconds'][key]:.4f} |")
    selections=['| 模式 | 候选帧（0起始） |','|---|---|']
    for key,data in comparison['methods'].items():
        selections.append(f"| {key} | {', '.join(str(f['frame_index']) for f in data['selected_frames']) or '拒绝'} |")
    text=f'''# 实验记录与结论

日期：2026-11-06。数据：确定性分支血管合成phantom，随机种子7，12位存储、16位分配、未压缩XA DICOM。**没有真实患者DICOM或医生标注，不报告临床准确率。** 测试是代码机制与机器性能验证。

## 性能测量

Python {matrix['environment']['python']}，{matrix['environment']['platform']}，{matrix['environment']['machine']}，逻辑CPU {matrix['environment']['logical_cpus']}，OpenCV线程配置 {matrix['environment'].get('opencv_threads','未记录')}。库版本与完整参数见原始JSON。每组预热{matrix['warmup']}次、记录3次；同一进程暖缓存，运行于开发工作站，未独占CPU，后台负载未控制。3次重复的P95只能作初步描述，不能视为稳定尾延迟基线。

模式multi_feature，包括Frangi和Sato；增强仅处理充盈候选。原始输入尺寸与分析尺寸分别列出，256/512/1024并不是都按原分辨率运行。原生512另测一组；sigma仍为[1,2,3]处理像素，不据此比较不同尺寸下的图像质量。

{chr(10).join(lines)}

总耗时为DICOM读取/解码至选择，不包括进程启动、样本生成、报告与增强缩略图导出。CPU百分比以单核100%计，可超过100%。RSS为10ms采样整个进程峰值，不是Python堆，也不是独立算法增量；可能漏掉短峰值，受重复运行的内存分配器缓存影响。

## 512×512、100帧分阶段耗时

{chr(10).join(stages)}

上述每阶段时间直接来自重复测量，不是估算。默认路径中Frangi/Sato多尺度增强是主要开销；输入变大还会增加解码和全分辨率归一化/展示数组的内存与时间。没有现有ONNX模型的同机基线，不能宣称加速倍数。

## 同一合成样本的模式比较

充盈区间：{comparison['candidate_intervals']}。

{chr(10).join(selections)}

这只是不同启发式选择位置的可视比较，不是人工首选帧命中率，也不能用合成注射峰等同于临床最佳帧。无有效区间时可拒绝，单元测试也验证了明确负样本标注的拒绝评价。

## 最终十个问题的当前答案

1. **最适合的传统算法**：尚不能在临床意义上确定；现有系统提供充盈门控加血管可见性/质量的可比较基线。
2. **Frangi还是Sato更好**：合成样本均产生管状响应；阶段耗时可比较，但临床效果需要相同病例与医生标签。不能由本次实验判定胜者。
3. **仅充盈是否足够**：目前没有支持其足够的证据；暗化不评价分支完整性、重叠或运动，可用于廉价候选生成。
4. **清晰度与运动是否改善**：改变了本样本排序，但无医生标注，不能称为准确性改善。需要消融试验及跟踪失败分层分析。
5. **512×512、100帧耗时**：本机合成样本默认缩放256分析，均值{default['mean_seconds']:.3f}s；原生512分析{full['mean_seconds']:.3f}s。均为多指标模式核心流程，真实病例需重新测量。
6. **主要耗时阶段**：见分阶段实测表；应优先优化多尺度Hessian增强，其次检查全序列复制和预处理。
7. **能否替代深度学习模型**：尚未验证。缺少真实标注、现有模型输出和同机CPU耗时，不能作替代结论。
8. **能否减少模型推理帧数**：有实验可行性。本合成样本将100帧门控至{sum(b-a+1 for a,b in comparison['candidate_intervals'])}帧；这不是已验证的临床召回或加速收益，必须评估被排除的可接受帧和混合流程总耗时。
9. **Linux Electron移植方向**：优先C++ OpenCV后台模块配合Worker Threads；Hessian滤波需移植并验证数值一致性。WASM另测解码、内存和线程开销，详见algorithm_design.md。
10. **风险与局限**：导管/肋骨/文字伪响应、曝光变化、已充盈参考、亮血管/特殊减影、严重噪声、分支重叠和缩短、注射不完整、光流失败、权重过拟合。当前不含ECG相位、投照角或病变特异性评价，不能保证诊断适用性。

## 下一轮验证与性能目标

先建立病例级开发/测试集、冻结归一化/阈值/权重、医生首选及可接受集合/无效序列标签；按设备、左右冠脉、投照角、帧率、压缩方式、图像质量和异常类型分层。与现有模型在同机同输入对比Top-1/Top-3、帧距、拒绝率/错误拒绝率和端到端时间，并统计病例级不确定性。测试集不参与调权。

依据当前实测，优先将候选门控和单一增强模式作为优化实验，目标是相对于本次multi_feature基线减少Hessian计算和峰值RSS，再在真实标签上确认质量损失；暂不设临床可接受的硬性延迟/准确率门槛。性能复验建议repeat≥10，并记录独占/后台负载、冷暖缓存、固定线程和CPU型号。大序列应验证流式解码与按需导出，避免多份全序列数组。

## 可复现产物

- `{args.matrix}`：9组尺寸/帧数，每组各次运行和统计。
- `{args.native}`：原生512像素分析单独测量。
- `{args.comparison}`：算法参数、候选和元数据。
- `outputs/demo_final/report.html`：可视对比与逐帧CSV。
- `tests/`：像素格式、极性、Rescale、RLE压缩、多峰、拒绝、清晰度、光流、时间去重、参考评价、CLI和批处理失败测试。

早期 `outputs/phantom`、`outputs/validated_phantom`、`outputs/benchmark_matrix` 为开发中间结果；最终结论仅使用上述明确列出的产物。
'''
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text,encoding='utf-8')
    print(args.output)

if __name__=='__main__':
    main()
