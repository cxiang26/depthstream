// mono_live.cc — CSI 相机实时单目 ORB-SLAM3 (Jetson Nano, headless)
// 采集用原生 GStreamer C API(系统 OpenCV 无 GStreamer 支持, cv::VideoCapture 不可用)
// 用法: ./mono_live vocabulary.txt settings.yaml  (Ctrl+C/超时结束并保存轨迹)
// 2026-10-13 晚: 加分阶段计时(pull/cvt/track)+ PTS 间隔, 用于定位深度模式下
//   主循环 0.3fps 的瓶颈(纯单目 37fps, 带 TRT 深度 0.3fps, 疑似 CUDA 上下文
//   与 nvvidconv 争用; 计时数据可判定 3s/帧 花在拉流还是跟踪)
#include <csignal>
#include <iostream>
#include <thread>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <System.h>
#include "DepthEngine.h"

static volatile bool g_stop = false;
static void on_sig(int) { g_stop = true; }

// FIX(2026-10-13): 用 sigaction 且不设 SA_RESTART。std::signal 默认带
// SA_RESTART, 相机断流时 appsink 的阻塞 pull 会被自动重启, g_stop 永远
// 等不到检查点, 进程挂死。去掉 SA_RESTART 后 pull 返回 EINTR/失败即退出
// 循环, 走保存流程。
static void install_signal_handlers() {
    struct sigaction sa;
    sa.sa_handler = on_sig;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // no SA_RESTART
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " vocabulary.txt settings.yaml" << std::endl;
        return 1;
    }
    install_signal_handlers();

    // ---- 原生 GStreamer: NVMM 硬缩放 -> BGR 内存帧(640x480, 丢旧帧) ----
    // 诊断后门(2026-10-13 晚): ORB_GST_PIPELINE 可整条覆盖默认管线。
    // 相机被 nvargus wedge 时, 用 videotestsrc 合成源验证 SLAM 主循环/
    // 深度注解器(与相机硬件无关)。覆盖串里必须包含 name=sink 的 appsink。
    // 注意: 覆盖检查必须在创建默认管线之前——nvarguscamerasrc 元素在
    // 创建/销毁时会连接卡死的 daemon, 阻塞约 30s, 会拖垮整个测试时序。
    gst_init(&argc, &argv);
    GError *err = nullptr;
    std::string pipe;
    if (const char* p = getenv("ORB_GST_PIPELINE")) {
        pipe = p;
        std::cout << "[诊断] 使用覆盖管线: " << pipe << std::endl;
    } else {
        pipe =
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx,width=640,height=480 ! "
            "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true";
    }
    GstElement *pipeline = gst_parse_launch(pipe.c_str(), &err);
    if (!pipeline || err) {
        std::cerr << "管线创建失败: " << (err ? err->message : "?") << std::endl;
        return 1;
    }
    GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    {   // 诊断: 确认管线真的到了 PLAYING(合成源无主循环时可能卡在 PAUSED)
        GstState st = GST_STATE_VOID_PENDING, pend = GST_STATE_VOID_PENDING;
        GstStateChangeReturn r = gst_element_get_state(pipeline, &st, &pend, 5 * GST_SECOND);
        const char* rs = r == GST_STATE_CHANGE_SUCCESS ? "SUCCESS" :
                         r == GST_STATE_CHANGE_ASYNC ? "ASYNC" : "FAILURE";
        std::cout << "[诊断] set PLAYING -> " << rs << " state=" << (int)st
                  << " pending=" << (int)pend << std::endl;
    }

    // ---- ORB-SLAM3: 纯运算模式, headless ----
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    std::cout << "SLAM 跟踪开始: 手持相机缓慢平移... Ctrl+C 保存并退出" << std::endl;

    const int W = 640, H = 480;
    long n = 0, dropped = 0;
    long last_pts = -1;
    auto t0 = std::chrono::steady_clock::now();
    // 分阶段计时(2026-10-13 诊断): pull=拉流等待, cvt=BGRA2BGR, track=SLAM跟踪
    double s_pull = 0, s_cvt = 0, s_track = 0, s_pts = 0;
    long prev_pts = -1;
    // 诊断(2026-10-13 晚): 前80帧独立统计ORB特征数, 判断单目初始化失败
    // 是特征不足(场景/曝光)还是视差不足(运动)。与SLAM内部提取器参数一致
    cv::Ptr<cv::ORB> dbg_orb = cv::ORB::create(1000, 1.2f, 8, 20, 0, 2, cv::ORB::HARRIS_SCORE, 31, 7);

    while (!g_stop) {
        auto w0 = std::chrono::steady_clock::now();
        GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        auto w1 = std::chrono::steady_clock::now();
        if (n < 5 || n % 500 == 0)
            std::cout << "[诊断] pull#" << n << " -> " << (sample ? "sample" : "NULL")
                      << " 等待" << (int)(std::chrono::duration<double, std::milli>(
                             w1 - w0).count()) << "ms" << std::endl;
        if (!sample) {                      // 管线停止(EOS)
            break;
        }
        GstBuffer *buf = gst_sample_get_buffer(sample);
        GstMapInfo map;
        if (gst_buffer_map(buf, &map, GST_MAP_READ) && map.size >= (gsize)(W * H * 4)) {
            GstClockTime pts = GST_BUFFER_PTS(buf);
            // FIX(2026-10-13 晚): multifilesrc 等源不产生时间戳(pts=NONE),
            // 原判断 (long)pts != last_pts 会把全部帧当重复帧丢弃(表现为
            // "处理 0 帧")。pts=NONE 时一律按新帧处理。
            bool is_new = (pts == GST_CLOCK_TIME_NONE) ||
                          ((long)pts != last_pts);
            if (is_new) {           // 只处理新帧
                last_pts = (long)pts;
                if (prev_pts >= 0) s_pts += ((double)pts - (double)prev_pts) / 1e9;
                prev_pts = (long)pts;
                cv::Mat bgra(H, W, CV_8UC4, (void *)map.data);
                cv::Mat bgr;
                cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
                // 诊断: 前80帧特征数(初始化失败归因: 特征不足 vs 视差不足)
                if (n < 80) {
                    std::vector<cv::KeyPoint> dbg_kps;
                    dbg_orb->detect(bgr, dbg_kps);
                    if (n % 8 == 0)
                        std::cout << "[诊断] 帧" << n << " ORB特征=" << dbg_kps.size()
                                  << " 亮度均值=" << (int)cv::mean(bgr)[0] << std::endl;
                }
                auto w2 = std::chrono::steady_clock::now();
                double t = std::chrono::duration<double>(w2 - t0).count();
                SLAM.TrackMonocular(bgr, t);
                auto w3 = std::chrono::steady_clock::now();
                ++n;
                s_pull  += std::chrono::duration<double>(w1 - w0).count();
                s_cvt   += std::chrono::duration<double>(w2 - w1).count();
                s_track += std::chrono::duration<double>(w3 - w2).count();
                if (n % 10 == 0) {
                    std::cout << "[计时] n=" << n
                              << " pull=" << (int)(s_pull / n * 1000) << "ms"
                              << " cvt=" << (int)(s_cvt / n * 1000) << "ms"
                              << " track=" << (int)(s_track / n * 1000) << "ms"
                              << " pts间隔=" << (int)(s_pts / (n - 1) * 1000) << "ms"
                              << " 墙钟均=" << (int)(std::chrono::duration<double>(
                                     w3 - t0).count() / n * 1000) << "ms"
                              << std::endl;
                }
            } else {
                ++dropped;
            }
        }
        gst_buffer_unmap(buf, &map);
        gst_sample_unref(sample);
    }

    std::cout << "停止, 保存轨迹... (处理 " << n << " 帧, 跳过旧帧 " << dropped
              << ")" << std::endl;
    if (n > 0) {
        std::cout << "[计时汇总] pull均=" << (int)(s_pull / n * 1000) << "ms"
                  << " cvt均=" << (int)(s_cvt / n * 1000) << "ms"
                  << " track均=" << (int)(s_track / n * 1000) << "ms"
                  << " pts间隔均=" << (int)(s_pts / (n > 1 ? n - 1 : 1) * 1000) << "ms"
                  << std::endl;
    }
    // 顺序说明: Shutdown() 现在会等 LocalMapping/LoopClosing 真正停止(带超时),
    // 因此轨迹保存不再与活动线程竞态; exportAll 内部会先 join 深度注解 worker,
    // 保证点云/尺度统计覆盖全部已处理关键帧。frame_trajectory.txt 对本 fork 的
    // 单目模式同样有效(System::SaveTrajectoryTUM 的单目限制已移除)。
    SLAM.Shutdown();
    SLAM.SaveTrajectoryTUM("frame_trajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("keyframe_trajectory.txt");
    // 深度注解器(未设置环境变量时为空操作)
    ORB_SLAM3::DepthAnnotator::instance()->exportAll(
        "depth_out/map_points.ply", "depth_out/scale_stats.txt");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    std::cout << "共 " << n << " 帧, 输出 frame_trajectory.txt / "
              << "keyframe_trajectory.txt" << std::endl;
    return 0;
}
