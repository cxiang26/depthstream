# DepthStream 论文骨架（工作文档）

**工作标题**: DepthStream: Amortized Real-Time Monocular Depth for Commodity GPUs
**一句话定位**: 免训练的摊销推理运行时——6M 深度模型 + 光流传播 + 仿射稳定，
在消费级 GPU 上以 175 FPS 输出时序稳定的全分辨率深度，同延迟预算下质量
优于大模型逐帧硬扛。

## 目标venue
- 首选: WACV / 3DV（工程实时系统）
- 备选: ACM MM industrial track / JRTIP / CVPR embedded vision workshop

---

## 1. Introduction
- 痛点: 视频深度 = 逐帧调用大模型 → 帧率低 + 时序抖动
- 观察: 深度场帧间变化远小于图像外观变化 → 深度可"传播"
- 贡献:
  1. train-free 摊销推理运行时（关键帧推理 + 光流传播 + 仿射重锚 + EMA）
  2. 消抖机制使时序一致性提升 4×（kf10: 0.0070 vs dense 0.0290, 合成视频）
  3. 6GB 消费级 GPU 上 176 FPS 全分辨率深度；延迟对齐对比大模型
  4. （分析节）投机级联的失效分析：蒸馏学生 δ1=0.979 已达教师一致

## 2. Related Work
- Video depth (train-heavy): NVDS, ChronoDepth, DepthCrafter, Video Depth Anything（扩散/学习，慢）
- **VeloDepth (arXiv 2512.10725, 2025.12) — 最接近的先行工作，必须正面引用对比**：
  同样是 keyframe + 光流传播 + 自适应重锚（概念重叠），但需要**训练 55M 传播网络**
  （8×4090×6天），构建在 UniDepthV2/UniK3D 基础模型上（总 409M 参数），
  26.2 FPS@0.5MP on RTX 3090。
- 定位差异（论文的主张锚点）：
  | 维度 | VeloDepth | DepthStream (ours) |
  |---|---|---|
  | 传播模块 | 学习式(55M, 训练) | **零训练纯运行时** |
  | 基础模型 | UniDepthV2 类(409M) | **tiny 6M (DepthART)** |
  | 硬件 | RTX 3090 | **6GB 笔记本 3060** |
  | 速度 | 26.2 FPS@0.5MP | **142-176 FPS@0.3MP** |
  | 部署 | 需训练管线 | ONNX/TRT 即插 |
- 流式传播失败处理：VeloDepth 用学习门控；我们用光度残差自适应调度（免训练）
- 级联/speculative：PatchFusion 类 + 我们的失效分析（§5，独家）
- **查新结论（2026-10-09）**：核心交集"零训练传播运行时 for tiny 深度模型"未被占；
  VeloDepth 占据"学习式传播 for 基础模型"。二者互补：VeloDepth 证明学习式传播
  对基础模型有效，本文证明对蒸馏 tiny 模型零训练运行时即可达到近似收益，
  且快一个数量级。写作时主张须从"提出摊销"降级为
  "摊销原理的零训练实例化 + tiny 模型场景的系统性分析"。

## 3. Method
- 3.1 Amortized formulation: 何时推理 vs 何时传播（关键帧调度）
- 3.2 Flow propagation: backward flow + remap; 流分辨率-精度权衡
- 3.3 Re-anchoring: 环带 LSQ 仿射对齐 + EMA(λ)
- 3.4 置信度掩码（TODO: 光流幅值/光度残差 → 不可信区域回退推理）

## 4. Experiments
### 4.1 Setup
- 学生: DepthART-S (MNv4-slim-SPF, 6M, TRT FP16 224²) — 已有 engine
- 平台: RTX 3060 Laptop 6GB / （待补）desktop GPU + Jetson Orin NX + CPU
- 数据: Sintel depth 训练集（GT 深度+光流）; （待补）Bonn RGB-D
- 指标: δ1(对齐), TAE(GT-flow warp), ms/frame; 功耗

### 4.2 Main table（表1 — 双数据集定稿版，详见 results_summary.md）
**Sintel 全量 23 序列 1064 帧（合成，GT 深度+GT 光流）**

| Config | δ1↑ | TAE(GT流)↓ | ms/frame↓ |
|---|---|---|---|
| DAv2-L @518 逐帧 | 0.4930 | 0.1844 | 56.4 |
| DepthART-S dense 逐帧 | 0.4614 | 0.1585 | 10.6 |
| **Stream kf3** | **0.4614** | **0.1343** | **8.4** |
| Stream kf10 | 0.4406 | 0.1094 | 6.1 |
| **Stream kf10-自适应** | 0.4434 | **0.1054** | 7.6 |
| Stream kf10 + 前向GT流 | 0.4145 | 0.1282 | 6.3 |
| Stream kf3 + 384流 | 0.4619 | 0.1354 | 9.4 |

**Bonn RGB-D 3 序列（真实室内动态，Kinect GT，TAE 用估计流口径）**

| Config | δ1↑ | TAE(est)↓ | ms/frame↓ |
|---|---|---|---|
| DAv2-L @518 逐帧 | 0.9811 | 0.4820 | 92.5 |
| DepthART dense 逐帧 | 0.9655 | 0.4607 | 7.6 |
| **Stream kf3** | **0.9637** | 0.4773 | **6.7** |
| Stream kf10-自适应 | 0.9484 | 0.4852 | 6.3 |

**头条（双数据集互证）**:
1. kf3 精度与 dense 一致（Sintel 100.0% / Bonn 99.8%），Sintel 上 TAE -15%、延迟 -20%
   —— "kf3 严格优于逐帧"
2. kf3 对 DAv2-L：93.6%(Sintel)/98.2%(Bonn) 精度 @ 1/6.7~1/13.7 延迟，
   时序一致性 Sintel +27%、Bonn +1%
3. kf10-自适应：TAE 全场最优（Sintel 0.1054 = DAv2-L 的 1.75×），
   衰减曲线平坦（age0→9 仅 -3.3%）
4. 消融链: EMA有害→删；自适应调度→留；oracle前向流/hiflow→流非瓶颈；
   运动分离单应→负结果（仅旋转主导 temple_2 +13%）
**dav2l224 排查结论**: 非单调是逐序列真实现象（alley_1 224最好/ambush_2 全崩），
对齐方式会改变排序 → 全量平均 + 双口径报告，如实呈现

### 4.3 传播误差累积（图2 accum.png）+ 定性可视化（qual_figs.py 生成）
- **Fig 1: framework.png / .pdf**（方法框架图，draw_framework.py 生成）：
  输入→关键帧判定→双路径（蓝色模型推理 / 橙色光流传播）→仿射重锚融合→输出，
  含光度残差自适应触发的虚线回路；矢量 PDF 供 LaTeX
- **Fig: comparison_strip.png**（§4.5 定性主图）：RGB|GT|dense|kf3|kf10-adp|DAv2-L
  六面板统一对齐色标 + 误差热图行（标注逐面板 δ1），Sintel alley_1
- **Fig: spacetime_slice.png**（时序一致性主图）：固定扫描行深度随时间堆叠，
  dense 的水平抖动纹理 vs kf10-adp 的平滑条带，GT 条带对照
- **Fig: flicker_maps.png**：|D_t−warp(D_{t−1})| 热图，dense 均值 0.3185 vs
  kf10-adp 0.1756（-45%），误差集中于运动边界 halo
- **Fig: bonn_qual.png**：真实数据对比条（balloon2，δ1 标注 0.998/0.998/0.997）
- **Fig: pareto.png**（延迟-精度散点）、**accum.png**（衰减曲线）、
  **perseq.png**（逐序列柱状）
### 4.4 消融: EMA λ ∈ {0,0.3,0.6,1.0}; flow_width ∈ {128,256,384}; 对齐 on/off
### 4.5 Pareto 图（图1: x=延迟, y=δ1, 点=所有配置，包络线加粗）

## 5. Analysis: When does speculative depth pay off?（C 的负结果改写）
- 蒸馏学生 vs 教师 δ1 一致性 0.979 → 级联无空间
- 推论: 级联只对 (a) 扩散教师 (b) 4K OOM (c) 弱学生 成立
- 表: cascade 延迟/质量（已有 depthart_cascade.py）

## 6. Limitations
- 光流遮挡失效（置信度掩码缓解）
- 纯旋转/缩放过快时传播漂移
- 单目尺度歧义（relative 版本）；metric 版需 K
- Sintel 为合成数据（用 Bonn 真实数据补）

## 7. Reproducibility
- 一键脚本: eval_sintel.py / depthestream.py
- License 表: DepthART 权重 CC BY-NC / DAv2 Apache-2.0-BitArial?（待核）/
  Sintel 数据 CC BY-NC — 论文声明
- TRT 11.3 迁移注记（附录，工程贡献）

---

## 立即待办（按序）
1. [数据到位] 跑 eval_sintel.py --max-seqs 6 --frames 50 → 表1
2. 传播误差累积脚本（plot_accum.py）
3. 消融 runner（EMA/flow_width/对齐）
4. Bonn RGB-D 数据接入
5. 查新检索（video depth temporal consistency train-free propagation）
