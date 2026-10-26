#pragma once
// DepthEngine — 深度增强 SLAM 组件(路线A)
//  TrtDepthEngine : TensorRT 相对深度推理, 前后处理与 depthstream_sdk/trt_backend.py 严格一致
//  DepthAnnotator : 关键帧异步标注线程(不阻塞跟踪), 产出:
//    - 每关键帧彩色深度图 out_dir/depth_KF*.jpg
//    - SLAM 锚定尺度统计(预测深度/地图点深度 的中位数) out_dir/scale_stats.txt
//    - 全部关键帧的深度点云 PLY(尺度经锚定校正)
// 通过环境变量启用: ORB_DEPTH_ENGINE=engine路径 ORB_DEPTH_OUT=输出目录
// 未设置时全部调用为空操作, 零开销零风险。
#include <opencv2/opencv.hpp>
#include <atomic>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace ORB_SLAM3 {

class KeyFrame;

class TrtDepthEngine {
public:
    bool load(const std::string& engine_path, int size = 224);
    // BGR 输入 -> 与输入同尺寸的 float32 相对深度图
    cv::Mat infer(const cv::Mat& bgr);
    bool ok() const { return loaded_; }

private:
    int size_ = 224;
    bool loaded_ = false;
    void* engine_ = nullptr;    // nvinfer1::ICudaEngine*
    void* context_ = nullptr;   // nvinfer1::IExecutionContext*
    void* d_in_ = nullptr;
    void* d_out_ = nullptr;
    float* h_in_ = nullptr;
    float* h_out_ = nullptr;
};

// 彩色化(与 SDK utils.colorize 一致: 2/98 百分位 + TURBO)
cv::Mat colorize_depth(const cv::Mat& depth);

class DepthAnnotator {
public:
    // 依据环境变量构造的单例; 未配置 ORB_DEPTH_ENGINE/ORB_DEPTH_OUT 时 disabled
    static DepthAnnotator* instance();
    bool enabled() const { return enabled_; }

    // 每帧调用(GrabImageMonocular): 保存最近的彩色帧供关键帧使用
    void stashColor(const cv::Mat& bgr);
    // 关键帧创建时调用: 入队异步标注(队列超限自动丢最旧, 不反压跟踪)
    void submit(KeyFrame* kf, const cv::Mat& K);

    // 退出时导出: 深度锚定点云 PLY + 每关键帧尺度统计
    // (内部会先停止并 join worker 线程, 保证导出数据完整)
    void exportAll(const std::string& ply_path, const std::string& stats_path);

    ~DepthAnnotator();  // join worker, 避免静态析构时 terminate

private:
    DepthAnnotator();
    void worker();
    void requestStopAndJoin();  // 设置停止标志并 join worker

    struct Job {
        KeyFrame* kf;
        cv::Mat bgr;
        double fx, fy, cx, cy;
    };

    // FIX(2026-10-13): 队列上限由 3 提高到 64。首个推理在 Nano 上冷启动可达
    // 数十秒, 期间持续提交的关键帧会把队列冲爆, 旧策略静默丢弃近 200/291 个
    // 标注。配合构造函数中的 engine 预热, 正常跑动时队列深度只有个位数。
    static const size_t kMaxJobs = 64;

    bool enabled_ = false;
    std::string out_dir_;
    std::unique_ptr<TrtDepthEngine> eng_;
    cv::Mat last_bgr_;
    std::vector<Job> jobs_;   // 队列上限 kMaxJobs, 丢最旧不反压
    std::mutex mtx_;
    // FIX(2026-10-13 晚): worker 空闲等待改用条件变量。原实现持锁 sleep_for(4ms)
    // 紧密轮询, 与主线程每帧 stashColor 抢同一把锁导致 futex 饥饿
    // (实测 0.35fps, gdb 证实主线程阻塞在 inst.mtx_)。cv_.wait 睡眠期间放锁。
    std::condition_variable cv_;
    std::thread th_;
    std::atomic<bool> stop_{false};
    int dropped_ = 0;         // 因队列拥塞被丢弃的任务数(导出时报告)
    std::map<long, cv::Mat> depths_;                   // kfId -> 相对深度
    std::vector<std::pair<long, double>> scale_by_kf_; // kfId -> 锚定尺度
    std::vector<float> ply_pts_;                       // x y z r g b
    std::mutex ply_mtx_;
};

}  // namespace ORB_SLAM3
