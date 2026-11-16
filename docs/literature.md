# 文献核查与实现溯源

核查日期：2026-11-06。论文算法与本项目启发式分开记录，不将文献中的性能数值当作本项目实验结果。

## 1. ICPR 2010

Automatic selection of keyframes from angiogram videos. DOI: 10.1109/ICPR.2010.975。

来源：[IBM作者机构页面](https://research.ibm.com/publications/automatic-selection-of-keyframes-from-angiogram-videos)。已读取原始摘要；DOI全文入口无法通过当前工具获取。

摘要可确认两条不同路线：Frangi滤波，以及从影像边缘检测平行曲线；用医生标注的冠脉帧作比较。无法核对全文中的具体参数、时间规则、样本数、评价公式或性能，不据此补写细节。

本项目只借鉴血管可见性评分方向，没有实现平行边缘算法，也不是该论文完整复现。

## 2. ICPRAM 2023

Automatic Coronary Angiogram Keyframe Extraction. DOI: 10.5220/0011850700003411。

来源：[出版方详情](https://www.scitepress.org/PublicationsDetail.aspx?ID=QhrTWXiEsD8%3D)、[原文PDF](https://www.scitepress.org/Papers/2023/118507/118507.pdf)。浏览工具超时后通过公开URL下载，已提取并阅读8页全文，重点核对第4节伪代码、第5节实验与表1。

原文步骤：灰度帧逐一滤波、聚合滤后强度、取强度最大位置为pilot，再从其附近截取固定数量帧；也试验Gaussian前后级联及固定/可变裁边。摘要称Contrast Enhanced Sato，但表1列名为Sato，并没有给出足以逐项复现的独立contrast增强参数；不可自动等同于CLAHE。

伪代码存在值得保留的歧义：`nb_k=7`，Python形式半开区间会取6个位置且包含pilot，正文则称六个邻帧；边界条件排除了索引0。Algorithm 3按人工一致程度进行结果“improvement/rounding”，不能等同于改进了独立预测器，也不能把其结果等同本项目Top-1/Top-3。表1并未提供可直接采用的Frangi对比行。本项目不复制这些有歧义的边界/评价规则。

实验使用从45段造影抽取的1434帧，由两名心脏科医生标注；论文描述按帧随机划分，不能据此确认病例级独立性。原文的汇总百分比不移用到本项目，不作为我们选择Sato优于Frangi的证据。

本项目的共享归一化、背景暗化门控、响应面积×强度、固定Frangi gamma、清晰度/光流融合和Top-K时间去重均是自行设计；不是该论文的完整复现。

## 3. JAPSC 综述

用户提供题名为 Computational Methods for Analysing X-ray-guided Coronary Angiography: Key Frame Selection, Vessel Segmentation and Stenosis Detection。检索出现扩展题名 From Key Frame Selection to Vessel Segmentation, Stenosis Detection and Classification，DOI 10.15420/japsc.2024.46；目录也出现 Survey on Computational Analysis of X-ray-guided Coronary Angiography。

[出版方页面](https://www.japscjournal.com/articles/computational-methods-analysing-x-ray-guided-coronary-angiography-key-frame-selection)、[期刊目录](https://www.japscjournal.com/articles?page=5)。正文访问403，元数据尚有差异，未据此提取公式、量化结果或断言其具体方法结论。后续需取得全文核对正式题名、版本及参考文献。

## 撤稿排除

检索中发现题名相近的另一篇文献 Extraction of the Best Frames in Coronary Angiograms for Diagnosis and Analysis 存在[撤稿公告](https://pmc.ncbi.nlm.nih.gov/articles/PMC6601230/)。本项目不采用该文献的研究结论或性能证据。不能把该撤稿状态误归给上述三篇不同论文；同时当前没有完成所有出版方全文版本的撤稿状态审计。

## 下一步阅读缺口

取得原文后应核对：输入极性、滤波尺度和增强步骤、帧强度聚合方式、真值定义、病例划分、评价分母、硬件和计时边界。当前实现是可检验的工程基线，不是经全文逐项复现的论文基准。
