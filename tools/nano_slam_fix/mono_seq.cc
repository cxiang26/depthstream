// mono_seq.cc — 离线图像序列单目 ORB-SLAM3 运行器 (WSL/x86 本地版, 无 GStreamer)
// 从 mono_live.cc 派生 (2026-10-14): 拉流循环替换为逐帧读图, 其余逻辑
// (计时/诊断/保存顺序/深度注解导出) 与 mono_live 保持一致, 保证与 Nano
// 实验可比。
//
// 用法 1(目录模式):  ./mono_seq voc.yaml settings.yaml <帧目录> [fps]
//   枚举 <帧目录>/seq_%04d.png (从 1 起), 时间戳 t = i/fps (默认 30)
// 用法 2(TUM 模式):  ./mono_seq voc.yaml settings.yaml <rgb.txt> [fps]
//   rgb.txt 为 TUM 格式 "timestamp path"(# 开头行为注释), 直接用采集
//   真时间戳, 轨迹第一列即采集时刻, 评测无需换算; path 相对于 rgb.txt 所在目录
// 深度注解器: 环境变量 ORB_DEPTH_ENGINE / ORB_DEPTH_OUT (未设则纯单目)
#include <csignal>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <System.h>
#include "DepthEngine.h"

static volatile bool g_stop = false;
static void on_sig(int) { g_stop = true; }

static void install_signal_handlers() {
    struct sigaction sa;
    sa.sa_handler = on_sig;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

struct Frame {
    std::string path;
    double t;
};

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0]
                  << " vocabulary.txt settings.yaml <帧目录|rgb.txt> [fps]"
                  << std::endl;
        return 1;
    }
    install_signal_handlers();

    const double fps = (argc > 4) ? atof(argv[4]) : 30.0;
    const std::string src = argv[3];

    // ---- 构建帧列表 ----
    std::vector<Frame> frames;
    if (src.size() > 4 && src.substr(src.size() - 4) == ".txt") {
        // TUM rgb.txt: "timestamp filename"
        std::ifstream fin(src);
        if (!fin) { std::cerr << "打不开 " << src << std::endl; return 1; }
        const std::string base =
            src.substr(0, src.find_last_of("/\\") + 1);
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            double ts; std::string rel;
            if (ss >> ts >> rel)
                frames.push_back({base + rel, ts});
        }
        std::cout << "[mono_seq] TUM 模式: " << frames.size()
                  << " 帧(真时间戳), 清单=" << src << std::endl;
    } else {
        for (int i = 1;; ++i) {
            char buf[512];
            snprintf(buf, sizeof(buf), "%s/seq_%04d.png", src.c_str(), i);
            std::ifstream probe(buf, std::ios::binary | std::ios::ate);
            if (!probe) break;
            frames.push_back({buf, i / fps});
        }
        std::cout << "[mono_seq] 目录模式: " << frames.size()
                  << " 帧, fps=" << fps << std::endl;
    }
    if (frames.empty()) { std::cerr << "帧列表为空" << std::endl; return 1; }

    // ---- ORB-SLAM3: 纯运算模式, headless ----
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    std::cout << "SLAM 跟踪开始: " << frames.size() << " 帧" << std::endl;

    long n = 0;
    auto t0 = std::chrono::steady_clock::now();
    double s_read = 0, s_track = 0;
    // 诊断: 前 80 帧 ORB 特征数(与 mono_live 相同口径)
    cv::Ptr<cv::ORB> dbg_orb = cv::ORB::create(1000, 1.2f, 8, 20, 0, 2,
                                               cv::ORB::HARRIS_SCORE, 31, 7);

    for (const Frame &f : frames) {
        if (g_stop) break;
        auto w0 = std::chrono::steady_clock::now();
        cv::Mat bgr = cv::imread(f.path, cv::IMREAD_COLOR);
        auto w1 = std::chrono::steady_clock::now();
        if (bgr.empty()) {
            std::cerr << "[警告] 读图失败: " << f.path << std::endl;
            continue;
        }
        if (n < 80) {
            std::vector<cv::KeyPoint> dbg_kps;
            dbg_orb->detect(bgr, dbg_kps);
            if (n % 8 == 0)
                std::cout << "[诊断] 帧" << n << " ORB特征=" << dbg_kps.size()
                          << " 亮度均值=" << (int)cv::mean(bgr)[0] << std::endl;
        }
        auto w2 = std::chrono::steady_clock::now();
        SLAM.TrackMonocular(bgr, f.t);
        auto w3 = std::chrono::steady_clock::now();
        ++n;
        s_read  += std::chrono::duration<double>(w1 - w0).count();
        s_track += std::chrono::duration<double>(w3 - w2).count();
        if (n % 20 == 0)
            std::cout << "[计时] n=" << n << "/" << frames.size()
                      << " read=" << (int)(s_read / n * 1000) << "ms"
                      << " track=" << (int)(s_track / n * 1000) << "ms"
                      << std::endl;
    }

    std::cout << "停止, 保存轨迹... (处理 " << n << " 帧)" << std::endl;
    if (n > 0)
        std::cout << "[计时汇总] read均=" << (int)(s_read / n * 1000) << "ms"
                  << " track均=" << (int)(s_track / n * 1000) << "ms"
                  << std::endl;
    // 保存顺序与 mono_live 一致: Shutdown 等线程真正停止 → 轨迹 → 深度导出
    SLAM.Shutdown();
    SLAM.SaveTrajectoryTUM("frame_trajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("keyframe_trajectory.txt");
    ORB_SLAM3::DepthAnnotator::instance()->exportAll(
        "depth_out/map_points.ply", "depth_out/scale_stats.txt");
    std::cout << "共 " << n << " 帧, 输出 frame_trajectory.txt / "
              << "keyframe_trajectory.txt" << std::endl;
    return 0;
}
