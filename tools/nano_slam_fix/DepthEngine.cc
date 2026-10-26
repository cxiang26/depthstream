// DepthEngine 实现 — 前后处理严格对齐 depthstream_sdk/trt_backend.py(只读参考, 未改动)
// preprocess: BGR2RGB -> /255 -> INTER_AREA 224 -> ImageNet mean/std -> CHW
// 输出: (1,1,224,224) float32 相对深度 -> INTER_LINEAR 上采样回原尺寸
#include "DepthEngine.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>

#include "KeyFrame.h"
#include "MapPoint.h"

namespace ORB_SLAM3 {

// ---------------- TensorRT ----------------

namespace {
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
    }
};
TrtLogger g_trt_logger;
}  // namespace

bool TrtDepthEngine::load(const std::string& engine_path, int size) {
    size_ = size;
    std::ifstream f(engine_path.c_str(), std::ios::binary);
    if (!f) {
        fprintf(stderr, "[深度] engine 打不开: %s\n", engine_path.c_str());
        return false;
    }
    f.seekg(0, f.end);
    size_t n = static_cast<size_t>(f.tellg());
    f.seekg(0, f.beg);
    std::vector<char> data(n);
    f.read(data.data(), n);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime) return false;
    engine_ = runtime->deserializeCudaEngine(data.data(), n);
    if (!engine_) {
        fprintf(stderr, "[深度] engine 反序列化失败\n");
        return false;
    }
    context_ = static_cast<nvinfer1::ICudaEngine*>(engine_)
                   ->createExecutionContext();
    if (!context_) return false;

    const size_t in_bytes = sizeof(float) * 3 * size_ * size_;
    const size_t out_bytes = sizeof(float) * size_ * size_;
    if (cudaMalloc(&d_in_, in_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_out_, out_bytes) != cudaSuccess) return false;
    h_in_ = static_cast<float*>(malloc(in_bytes));
    h_out_ = static_cast<float*>(malloc(out_bytes));
    loaded_ = true;
    printf("[深度] engine 加载成功: %s\n", engine_path.c_str());
    return true;
}

static void preprocess(const cv::Mat& bgr, int size, float* dst) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    cv::resize(rgb, rgb, cv::Size(size, size), 0, 0, cv::INTER_AREA);
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < size; ++r) {
            const float* row = rgb.ptr<float>(r);
            float* ch = dst + c * size * size + r * size;
            for (int col = 0; col < size; ++col)
                ch[col] = (row[col * 3 + c] - mean[c]) / stdv[c];
        }
}

cv::Mat TrtDepthEngine::infer(const cv::Mat& bgr) {
    if (!loaded_ || bgr.empty()) return cv::Mat();
    preprocess(bgr, size_, h_in_);
    cudaMemcpy(d_in_, h_in_, sizeof(float) * 3 * size_ * size_,
               cudaMemcpyHostToDevice);
    void* bindings[2] = {d_in_, d_out_};
    static_cast<nvinfer1::IExecutionContext*>(context_)->executeV2(bindings);
    cudaMemcpy(h_out_, d_out_, sizeof(float) * size_ * size_,
               cudaMemcpyDeviceToHost);
    cv::Mat d224(size_, size_, CV_32F, h_out_);
    cv::Mat full;
    cv::resize(d224, full, bgr.size(), 0, 0, cv::INTER_LINEAR);
    return full;
}

// ---------------- 彩色化 ----------------

cv::Mat colorize_depth(const cv::Mat& depth) {
    if (depth.empty()) return cv::Mat();
    std::vector<float> v;
    v.reserve(depth.total());
    for (int r = 0; r < depth.rows; ++r) {
        const float* row = depth.ptr<float>(r);
        for (int c = 0; c < depth.cols; ++c)
            if (std::isfinite(row[c])) v.push_back(row[c]);
    }
    if (v.empty()) return cv::Mat::zeros(depth.size(), CV_8UC3);
    std::nth_element(v.begin(), v.begin() + v.size() * 2 / 100, v.end());
    float lo = v[v.size() * 2 / 100];
    std::nth_element(v.begin(), v.begin() + v.size() * 98 / 100, v.end());
    float hi = v[v.size() * 98 / 100];
    cv::Mat dn = (depth - lo) / std::max(hi - lo, 1e-6f);
    cv::min(dn, 1.0f, dn);
    cv::max(dn, 0.0f, dn);
    cv::Mat u8;
    dn.convertTo(u8, CV_8U, 255.0);
    cv::Mat out;
    cv::applyColorMap(u8, out, cv::COLORMAP_JET);
    return out;
}

// ---------------- DepthAnnotator ----------------

DepthAnnotator* DepthAnnotator::instance() {
    static DepthAnnotator inst;  // 首次调用时按环境变量初始化
    return &inst;
}

DepthAnnotator::DepthAnnotator() {
    const char* e = getenv("ORB_DEPTH_ENGINE");
    const char* o = getenv("ORB_DEPTH_OUT");
    if (!e || !o) { enabled_ = false; return; }
    eng_.reset(new TrtDepthEngine());
    if (!eng_->load(e)) { enabled_ = false; return; }
    out_dir_ = o;
    mkdir(out_dir_.c_str(), 0755);
    enabled_ = true;
    // FIX(2026-10-13): engine 预热。Nano 上首次 CUDA/TRT 推理冷启动可达数十
    // 秒, 若发生在 worker 处理第一个真实任务期间, 后续关键帧会因队列拥塞被大
    // 量丢弃(旧版 3 上限时丢过近 200 个)。这里在跟踪开始前用一帧全零图预热,
    // 冷初始化成本转移到 SLAM 构造阶段, 结果直接丢弃。
    {
        cv::Mat dummy(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        eng_->infer(dummy);
        printf("[深度] engine 预热完成(冷初始化已提前)\n");
    }
    th_ = std::thread(&DepthAnnotator::worker, this);
    printf("[深度] 注解器已启用  engine=%s  out=%s\n", e, o);
}

DepthAnnotator::~DepthAnnotator() {
    // FIX(2026-10-13): worker 是死循环线程, 静态单例析构时若仍 joinable,
    // std::thread 析构会直接 std::terminate (旧版退出时打印的
    // "terminate called without an active exception" 即来源于此)。
    requestStopAndJoin();
}

void DepthAnnotator::requestStopAndJoin() {
    // FIX(2026-10-13 晚): 持锁设置停止标志并 notify, 否则 cv_.wait 中的
    // worker 可能永远等不到唤醒(无超时)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
        cv_.notify_all();
    }
    if (th_.joinable()) th_.join();
}

void DepthAnnotator::stashColor(const cv::Mat& bgr) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lk(mtx_);
    last_bgr_ = bgr.clone();
}

void DepthAnnotator::submit(KeyFrame* kf, const cv::Mat& K) {
    if (!enabled_ || stop_) return;
    std::lock_guard<std::mutex> lk(mtx_);
    if (jobs_.size() >= kMaxJobs) {
        // 丢最旧, 不反压跟踪; 但计数并在首次丢弃时告警(旧版静默丢弃)
        if (dropped_ == 0)
            fprintf(stderr, "[深度] 警告: 标注队列积压(%zu 个), 开始丢弃最旧任务\n",
                    jobs_.size());
        jobs_.erase(jobs_.begin());
        ++dropped_;
    }
    Job j;
    j.kf = kf;
    j.bgr = last_bgr_.clone();
    j.fx = K.at<double>(0, 0); j.fy = K.at<double>(1, 1);
    j.cx = K.at<double>(0, 2); j.cy = K.at<double>(1, 2);
    jobs_.push_back(j);
    cv_.notify_one();   // FIX(2026-10-13 晚): 条件变量化, 通知 worker 取任务
}

static float sample_depth(const cv::Mat& d, float x, float y) {
    int xi = static_cast<int>(x + 0.5f), yi = static_cast<int>(y + 0.5f);
    if (xi < 0 || yi < 0 || xi >= d.cols || yi >= d.rows) return 0.f;
    float v = d.at<float>(yi, xi);
    return std::isfinite(v) ? v : 0.f;
}

void DepthAnnotator::worker() {
    double t_infer_sum = 0;
    long n = 0;
    // FIX(2026-10-13 晚): 空闲等待从"持锁 sleep 4ms 紧密轮询"改为条件变量。
    // 旧写法中 worker 持 mtx_ 睡 4ms 后立刻重锁, 与主线程每帧 stashColor 抢
    // 同一把锁; 地图未初始化(0 关键帧)时 worker 永远困在空轮询, 主线程 futex
    // 饥饿实测卡 2.5s/帧(0.35fps; gdb 栈证实主线程阻塞在 inst.mtx_ 的
    // __lll_lock_wait)。cv_.wait 在睡眠期间释放锁, 且仅在有任务时被唤醒。
    // flush 语义保留: stop_ 后继续处理完队列剩余任务再退出。
    std::unique_lock<std::mutex> lk(mtx_);
    while (true) {
        cv_.wait(lk, [&] { return stop_ || !jobs_.empty(); });
        if (stop_ && jobs_.empty()) break;
        Job job = jobs_.front();
        jobs_.erase(jobs_.begin());
        lk.unlock();
        auto t0 = std::chrono::steady_clock::now();
        cv::Mat depth = eng_->infer(job.bgr);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        t_infer_sum += ms;
        ++n;
        if (depth.empty()) { lk.lock(); continue; }

        const long id = job.kf->mnId;

        // --- SLAM 锚定尺度: 预测深度 / 该帧地图点真实深度 的中位数 ---
        std::vector<double> ratios;
        Sophus::SE3f Tcw = job.kf->GetPose();
        const std::vector<cv::KeyPoint>& kps = job.kf->mvKeysUn;
        const std::vector<MapPoint*> mps = job.kf->GetMapPointMatches();
        for (size_t i = 0; i < mps.size() && i < kps.size(); ++i) {
            MapPoint* mp = mps[i];
            if (!mp || mp->isBad()) continue;
            float d = sample_depth(depth, kps[i].pt.x, kps[i].pt.y);
            if (d <= 0) continue;
            Eigen::Vector3f xc = Tcw * mp->GetWorldPos();
            if (xc.z() <= 0.05f) continue;
            ratios.push_back(static_cast<double>(d) / xc.z());
        }
        double scale = 1.0;
        if (!ratios.empty()) {
            std::nth_element(ratios.begin(),
                             ratios.begin() + ratios.size() / 2, ratios.end());
            scale = ratios[ratios.size() / 2];
        }
        {
            std::lock_guard<std::mutex> lk(ply_mtx_);
            scale_by_kf_.push_back(std::make_pair(id, scale));
            depths_[id] = depth;
        }
        printf("[深度] KF %ld  推理 %.0fms  锚定尺度 %.4f  (样本 %zu)\n",
               id, ms, scale, ratios.size());

        // --- 可视化输出 ---
        cv::Mat vis = colorize_depth(depth);
        if (!vis.empty())
            cv::imwrite(out_dir_ + "/depth_KF" + std::to_string(id) + ".jpg",
                        vis);

        // --- 点云入缓冲(经锚定尺度; 点色暂 0, 后续可按 colormap 填充) ---
        Sophus::SE3f Twc = Tcw.inverse();
        // 改名 lkp: 避免遮蔽 worker 函数级的 unique_lock lk(mtx_)(否则
        // while 体末尾的 lk.lock() 会解析到这把 lock_guard 上)
        std::lock_guard<std::mutex> lkp(ply_mtx_);
        const int step = 4;
        for (int v = 0; v < depth.rows; v += step) {
            const float* row = depth.ptr<float>(v);
            for (int u = 0; u < depth.cols; u += step) {
                float drel = row[u];
                if (!std::isfinite(drel) || drel <= 0) continue;
                float z = static_cast<float>(scale) * drel;
                Eigen::Vector3f xc((u - job.cx) * z / job.fx,
                                   (v - job.cy) * z / job.fy, z);
                Eigen::Vector3f xw = Twc * xc;
                ply_pts_.push_back(xw.x());
                ply_pts_.push_back(xw.y());
                ply_pts_.push_back(xw.z());
                ply_pts_.push_back(0);
                ply_pts_.push_back(0);
                ply_pts_.push_back(0);
            }
        }
        lk.lock();
    }
}

void DepthAnnotator::exportAll(const std::string& ply_path,
                               const std::string& stats_path) {
    if (!enabled_) return;
    // FIX(2026-10-13): 先停止并 join worker, 再落盘。旧版导出时 worker 仍在
    // 跑(日志里出现过导出之后还打印 "[深度] KF 291"), 点云/尺度统计可能缺最
    // 后几个关键帧; join 之后数据才是完整的。
    requestStopAndJoin();
    if (dropped_ > 0)
        printf("[深度] 注意: %d 个标注任务因队列拥塞被丢弃(无深度图/尺度)\n",
               dropped_);
    std::lock_guard<std::mutex> lk(ply_mtx_);
    FILE* f = fopen(ply_path.c_str(), "w");
    if (f) {
        const size_t npts = ply_pts_.size() / 6;
        fprintf(f,
                "ply\nformat ascii 1.0\nelement vertex %zu\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\n"
                "property uchar blue\nend_header\n",
                npts);
        for (size_t i = 0; i < npts; ++i) {
            const float* p = &ply_pts_[i * 6];
            fprintf(f, "%.3f %.3f %.3f %d %d %d\n", p[0], p[1], p[2],
                    int(p[3]), int(p[4]), int(p[5]));
        }
        fclose(f);
        printf("[深度] 点云导出: %s (%zu 点)\n", ply_path.c_str(), npts);
    }
    FILE* s = fopen(stats_path.c_str(), "w");
    if (s) {
        fprintf(s, "# kf_id anchored_scale\n");
        for (auto& kv : scale_by_kf_)
            fprintf(s, "%ld %.5f\n", kv.first, kv.second);
        fclose(s);
        printf("[深度] 尺度统计: %s (%ld 关键帧)\n", stats_path.c_str(),
               (long)scale_by_kf_.size());
    }
}

}  // namespace ORB_SLAM3
