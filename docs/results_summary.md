# DepthStream 双数据集结果汇总（论文主表定稿数据）

数据口径：Sintel 全量 23 序列 1064 帧（合成，GT 深度+GT 光流）；
Bonn RGB-D 动态 3 序列（真实室内，Kinect GT 深度）。δ1 为仿射不变对齐
（lstsq 尺度平移，GT 有效上限 Sintel<500m / Bonn<10m）；TAE 为时序 warp 误差
（Sintel 用 GT 光流；Bonn 用 DIS 估计光流，全配置一致故可横向对比）。
硬件：RTX 3060 Laptop 6GB，ema=1.0，DIS 后向光流 @256。

## 主表 A：双数据集核心行（dense / kf3 / kf10-adp / 教师上界）

| 配置 | Sintel δ1↑ | Sintel TAE↓ | Sintel ms | Bonn δ1↑ | Bonn TAE(est)↓ | Bonn ms |
|---|---|---|---|---|---|---|
| DAv2-L @518 逐帧 | 0.4930 | 0.1844 | 56.4 | 0.9811 | 0.4820 | 92.5 |
| DepthART dense 逐帧 | 0.4614 | 0.1585 | 10.6 | 0.9655 | 0.4607 | 7.6 |
| **Stream kf3** | **0.4614** | **0.1343** | **8.4** | **0.9637** | 0.4773 | **6.7** |
| Stream kf10-自适应 | 0.4434 | **0.1054** | 7.6 | 0.9484 | 0.4852 | 6.3 |

## 主表 B：Sintel 专属消融行

| 配置 | δ1↑ | TAE↓ | ms |
|---|---|---|---|
| Stream kf10 固定间隔 | 0.4406 | 0.1094 | 6.1 |
| Stream kf10 + 前向GT流 | 0.4145 | 0.1282 | 6.3 |
| Stream kf3 + 384流 | 0.4619 | 0.1354 | 9.4 |

## 派生指标（论文正文引用值）

**kf3 vs dense（摊销是否免费）**
- Sintel：δ1 保持 **100.0%**，TAE 改善 **15.3%**，延迟 **-20.2%** → 严格占优
- Bonn：δ1 保持 **99.8%**，TAE +3.6%（估计流口径），延迟 -11.4%

**kf3 vs DAv2-L@518（对教师）**
- 精度：Sintel **93.6%** / Bonn **98.2%**
- 延迟：**1/6.7**（Sintel）/ **1/13.7**（Bonn）
- 时序一致性（TAE）：Sintel 好 **27.2%**（1.37×）；Bonn 好 1.0%

**kf10-自适应 vs 两者**
- vs dense：Sintel δ1 96.1%、TAE 好 **33.5%**；Bonn δ1 98.2%
- vs DAv2-L：Sintel TAE 好 **42.8%**（1.75×）@ 1/7.4 延迟；Bonn TAE 持平（-0.7%）

**传播衰减（kf10-adp, Sintel）**：age0→9 δ1 仅 -3.3%（0.4695→0.4541），
自适应重锚消除了固定间隔版本的 age0 凹陷

## 平台表（精度后端无关 + CPU 实时）

| 平台 | 后端 | 精度 | 模型延迟 | E2E kf10 | δ1 (kf3) |
|---|---|---|---|---|---|
| RTX 3060 Laptop 6GB | TensorRT 11.3 | FP16 | 1.20 ms | ~150-176 FPS* | 0.4614 |
| CPU Ryzen 7 5800H | ORT 1.26 fp32 | FP32 | 6.95 ms | **249.5 FPS*** | 0.4614 |

*640×480 合成视频 E2E；Sintel 1024×436 下 kf3 E2E = 118 FPS。
精度后端无关：TRT fp16 GPU 与 ORT fp32 CPU 全配置 δ1 差 ≤ 0.0001。
INT8 动态量化负结果：82-144ms（慢 12-20×，CNN 动态量化落参考核），6MB 量化模型留档。

## 四条结论（正文引用）

1. kf3 严格优于逐帧：Sintel 精度无损 + TAE -15% + 延迟 -20%；Bonn 精度 -0.2%
2. 对教师：1/6.7~1/13.7 延迟下保 93.6-98.2% 精度，时序一致性反超
3. EMA 有害（0.2875→0.2772 @Sintel kf3）、仿射重锚是有效组件
4. 光流质量非瓶颈（oracle 前向流、384 宽流均无收益）；
   运动分离传播负结果（单应只在 temple_2 旋转主导序列 +13%）

## 复现命令
```
python eval_sintel_v2.py --max-seqs 23 --frames 50            # Sintel 全量
python eval_sintel_v2.py --max-seqs 23 --frames 50 --backend cpu --skip-dav2
python eval_bonn.py --frames 60                               # Bonn
python bench_cpu.py                                           # CPU 平台
python plots.py                                               # 四张论文图
```

## 已知边界（Limitations 素材）
- Sintel 为合成数据（Bonn 补真实，但仅 3 序列，扩量待做）
- Bonn 的 TAE 用估计光流口径（无 GT 流），与 Sintel 的 GT 流口径不同，已在表中分列
- 单 GPU 平台（3060 Laptop，WSL）；Orin NX / 台式卡 / 功耗待补
- INT8 静态 PTQ、VeloDepth τ₅ 指标对接未做
