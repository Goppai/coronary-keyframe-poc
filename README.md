# 冠脉 DSA 传统关键帧提取 PoC

纯 CPU、无训练模型的研究工具。输入灰度多帧 DICOM，输出逐帧指标、充盈区间、Top-K 候选、Frangi/Sato 对比和静态 HTML 报告。**尚未通过真实临床影像与医生标注验证，不能据此替代生产模型或诊断。**

## 当前交付结果

- [示例HTML报告](outputs/demo_final/report.html)（合成数据）
- [实测性能与十个问题的结论](docs/evaluation.md)
- 27项测试通过；9组原始尺寸/帧数性能矩阵及原生512分析另测一组。

## 安装

Python 3.10+，建议独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
# 压缩像素按需安装；安装并不表示所有编码都已验证
pip install -e '.[compressed]'
pytest -q
```

无 PyTorch、TensorFlow、ONNX Runtime、CUDA 依赖。`requirements-tested.txt` 为当前实验环境锁定快照；跨平台建议优先使用 pyproject 中的依赖。环境信息写入 benchmark.json。

## 本地网页工作台

macOS可直接在Finder中双击项目目录的 **启动网页.command**。也可使用下方命令：

```bash
cd /Users/zhaobo/Desktop/work/coronary-keyframe-poc
.venv/bin/python scripts/web.py --open
```

浏览器打开 `http://127.0.0.1:8765`，终端保持运行，Ctrl+C停止服务。可用 `--port 8766` 更换端口，`--config config.yaml` 调整底层算法参数，`--output outputs/web_runs` 更换输出根目录。网页不需要Node.js、前端构建或额外Python依赖，仅监听本机。

1. 点击“导入文件夹”，选择 `/Users/zhaobo/Downloads/harbin_useful_data`，也支持单独选择多个文件。目录会递归列出待检查文件，跳过常见非影像附属文件；未知类型在分析时明确报错。
2. 选择分析方法、分析最大边和 Top-K，点击左侧 DICOM 自动分析。默认使用第二轮“充盈 + 优化 Frangi”，第一轮算法仍可从同一菜单选择。仅当前文件传给本机服务，临时输入在完成或失败后删除。
3. 勾选“对比第二轮五种算法”可在“算法对比”页查看并排关键帧和各方法 CPU 耗时，点击方法名切换候选帧。“阅片对比”左侧可播放、拖动或逐帧查看 DICOM 序列，右侧可切换候选和滤波响应。点击右侧候选，左侧跳到同一帧。序列使用报告的灰度窗口和 ROI 设置。逐帧数据展示运动状态和选帧窗口，报告与原始 JSON/CSV 可下载。帧号从 0 开始，候选最多 K 张。
4. 在“第二轮实验参数”可选择优化 Frangi 尺度、Black-hat 核和参数矩阵。参数矩阵默认关闭，开启后单独统计。需要平均耗时、P50/P95 时选择“预热 + 重复 3/5/10 次”，重复测量涵盖解码到选帧。
5. “下载全部输出”打包报告、JSON/CSV、关键帧、增强图和曲线；选择重复测量时另含repeated_benchmark.json。

网页当前会话内缓存结果，点击已分析文件可直接切回；修改参数后需点“重新分析”。刷新页面/重启服务后需重新选择文件夹，已导出成果仍保存在 `outputs/web_runs/<任务编号>/artifacts/` 中，可离线打开report.html。仅在本地页面显示原文件名，不把原文件名或路径写进报告；输出按随机任务编号存放。单文件默认上限1GiB，每次仅运行一个任务，避免多个DICOM同时占用大量内存。CSV页面预览前100帧的主要指标，下载含全部帧和特征。重复性能测量为本机进程暖缓存测量，非临床准确性评价。

## 第二轮算法对比

“第二轮实验参数”提供充盈补检和舒张末期排序。网页默认开启充盈补检，舒张末期排序可手动勾选。修改后点“重新分析”对照结果。命令行以配置为准。

- **用局部血管响应校验充盈区间**。通过局部血管样结构的面积、强度和时序变化补检，并校验全图检测。如果局部区间全部早于全图区间，且其局部响应峰值超过全图区间内峰值的 1.5 倍，就用较早的区间选帧。网页与报告会标明来源。`results.json` 可对照两种检测的区间。
- **舒张末期辅助排序**。根据造影影像估计相位。充盈区间内重复出现血管区域由扩张转为收缩时，附近帧最高加 0.015。只有周期低运动标记时，可在阅片器跳转查看，排序不加分。相位参考为实验性结果。

命令行可在下方命令追加 `--filling-fallback --end-diastole`。逐帧证据和相位接近度见 `scores.csv`，相位参考帧号和估计状态见 `results.json`。

```bash
.venv/bin/python scripts/compare_round2.py \
    --input /Users/zhaobo/Downloads/harbin_useful_data \
    --output outputs/my_round2 \
    --sample-count 3 --matrix-repeat 2
```

同一序列共用解码、归一化、充盈候选区间和选帧窗口，对比原版 Frangi、Sobel、Black-hat、Black-hat 加运动和 OpenCV Frangi。`--input` 也接受单个 DICOM。省略 `--sample-count` 会处理目录中全部候选文件。省略 `--matrix-repeat` 只运行五路对比。指定 `--method contrast_frangi_fast` 或 `--method contrast_blackhat` 可单独运行该方法。

各序列的 `report.html` 展示并排候选帧、滤波响应、逐帧曲线、CPU 耗时和参数。`scores.csv` 包含平滑前后评分、局部帧差、显影变化、运动置信状态和选帧窗口编号。`results.json` 含选帧、失败原因及可选实验矩阵。`benchmark.json` 区分阶段实测耗时、共享阶段和滤波矩阵。根目录的 `summary.json` 索引每份报告。

使用 `--config config.yaml` 调整 `round2` 段和通用参数。Black-hat 核尺寸以处理后像素为单位。Sobel 背景掩膜可通过将 `sobel_background_threshold` 设为 `null` 关闭。分辨率实验固定同一充盈区间，以便比较滤波器。脚本为无扩展名、`.dcm`、`.dicom` 和 `.ima` 文件按路径排序分配 `sample_NNN`，再按帧数分层抽样。`--annotations` 对单个 DICOM 接受 JSON 文件，对目录接受包含 `sample_NNN.json` 的文件夹。

真实数据的测量条件、结果与迁移判断见 [第二轮实验报告](docs/round2_evaluation.md)。

## 快速运行

```bash
# 没有真实样本时，生成明确标识为合成的分支血管 phantom
python scripts/generate_synthetic.py --output data/samples/phantom.dcm --size 512 --frames 100
python scripts/analyze.py --input data/samples/phantom.dcm --compare-all --output outputs/demo
# 真实输入示例
python scripts/analyze.py --input data/samples/sample.dcm --method multi_feature --top-k 3 --config config.yaml --output outputs/sample
python scripts/batch_analyze.py --input data/samples --output outputs/batch
python scripts/benchmark.py --input data/samples --repeat 10 --output outputs/benchmark
# 9 组原始尺寸/帧数组合，实际分析默认缩放至最大边256
python scripts/benchmark.py --input synthetic --repeat 3 --output outputs/matrix
# 单独测试实际512像素分析，注意与原始512输入/缩放256的区别
python scripts/benchmark.py --input synthetic --sizes 512 --frame-counts 100 --analysis-size 512 --repeat 3 --output outputs/native512
```

所有脚本均支持 `--help`。输出目录须为空，避免旧关键帧混入新报告。报告直接打开 `report.html`，无需 Web 服务。

输出：`results.json`、`scores.csv`、`keyframes/*.png`、`visualizations/*.png`、`benchmark.json`、`report.html`。CSV 保存直方图、均值、分位数、背景差分、充盈、Frangi/Sato 原始响应、清晰度、光流、各模式最终评分。未计算的增强指标用 `*_computed=false` 标记，值为占位零。

## 配置与语义

- 默认用全序列共享的 1–99 百分位归一化，不使用显示 WindowCenter/WindowWidth；MONOCHROME1 统一成暗血管极性。原始像素仅在读取/预处理阶段使用，展示图不参与重新计算。
- `preprocessing.roi` 为原始像素 `[x,y,width,height]`；分析最大边默认256，保持长宽比、不放大。运动 ROI 在缩放后的分析坐标中指定。所有时间索引从 **0** 开始。
- 默认前5帧中位数背景，可切换 `low_motion`、`adaptive`。后两种仅为实验策略；已充盈背景、曝光变化和运动可导致失败。
- 8种模式：`contrast_only`、`frangi_only`、`sato_only`、`contrast_frangi`、`contrast_sato`、`multi_feature`、`contrast_frangi_motion`、`contrast_sato_motion`。
- 所有模式默认经过充盈候选门控，`frangi_only` 表示候选内只按 Frangi 排序。它不是“全序列无门控 Frangi”消融。`candidate_only: false` 可让增强计算覆盖全序列，但选择仍经过门控。
- 多指标模式将各增强响应、两种清晰度、稀疏光流做序列相对归一化，使用配置初始权重；不是经验证最优权重，不是概率。Frangi使用固定 `frangi_gamma`，避免单帧自动gamma干扰跨帧可比性。Frangi/Sato 数值尺度不同，原始响应阈值需分别检查。
- Top-K 为上限；局部峰、最小时间距离、相似度过滤可导致少于K张。优先为不同充盈区间保留峰，但距离/相似度约束仍可阻止保留。
- 没有显著充盈变化或血管响应不足时拒绝返回候选，要求人工复核；“有候选”同样不是临床有效性保证。

## 人工参考评价

单文件 `--sample-id sample_001 --annotations data/annotations/sample_001.json`。示例：

```json
{"dicom_id":"sample_001","acceptable_frames":[34,35,36,37,38],"preferred_frame":36}
```

明确无有效帧的标签用 `"acceptable_frames": []`。不提供标注时不输出准确率。`preferred_frame` 必须属于可接受集合；区分首选帧距离和可接受集合最小距离。批处理按输入文件排序分配 sample_NNN，标注目录按同名JSON读取；排序变动会改变编号，正式实验应冻结输入清单。汇总各命中率时只使用对应的适用样本，并输出分母。

## 输入与安全边界

支持单样本 MONOCHROME1/2、8/16位分配（含12位存储）、有符号/无符号、单/多帧、Modality LUT / Rescale、FrameTime / FrameTimeVector / CineRate。压缩解码委托 pydicom 插件；失败明确报错。增强 DICOM 功能组变换、彩色和浮点像素不支持。目录表示多个独立DICOM文件，不自动将分散的单帧文件按Series重组成多帧序列。

不写患者标签和源路径到报告/日志；请自行使用脱敏样本编号。**影像可能有烧录文字，输出仍是潜在敏感医疗数据，去标签不等于匿名。** 输出目录不要提交到 Git。

## 测试与实验说明

见 [算法设计](docs/algorithm_design.md)、[文献核查](docs/literature.md)、[实验与结论](docs/evaluation.md)。性能统计区分单次和重复运行，报告每阶段、P50/P95、FPS、CPU时间和10ms采样进程RSS。内存采样可能遗漏短峰值；同进程重复的RSS受分配器保留影响。默认总耗时从DICOM解码到选择，不含进程启动/报告导出；导出单独计时。
