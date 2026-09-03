// SPDX-License-Identifier: Apache-2.0
#include "rx_robot_motion_zmq_subscriber.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>   // YAML is a JSON superset; used to parse the ZMQ header.
#include <zmq.h>

#include "common/log.hpp"
#include "sonic/sonic_robot_motion_buffer.hpp"

namespace rx101 {

namespace {

constexpr size_t kHeaderSize = 1280;  // wire contract; must match Python side.
constexpr int    kMaxFramesPerPacket = 64;  // GR00T ships ≤16; we cap defensively.

struct SubCtx {
    Config                                cfg;
    void*                                 zmq_ctx = nullptr;
    void*                                 zmq_sock = nullptr;
    std::atomic<SonicRobotMotionBuffer*>  buffer{nullptr};
    std::atomic<bool>                     running{false};
    std::thread                           thread;

    std::atomic<uint64_t> accepted{0};
    std::atomic<uint64_t> dropped{0};
    std::mutex            drop_reason_mu;
    const char*           last_drop_reason = "";  // static string only

    void note_drop(const char* reason) {
        dropped.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(drop_reason_mu);
        last_drop_reason = reason;
    }
};

SubCtx g_ctx;

// Return micro-seconds since some fixed but unspecified epoch (monotonic).
uint64_t mono_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Field {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t offset = 0;    // start in payload
    size_t nbytes = 0;
};

int dtype_itemsize(std::string_view d) {
    if (d == "f32" || d == "i32")  return 4;
    if (d == "f64" || d == "i64")  return 8;
    if (d == "bool" || d == "u8")  return 1;
    return 0;
}

// Parse the 1280 B JSON header. Return true on success; on false, *reason is set.
bool parse_header(const std::string& hdr_txt,
                  std::vector<Field>* out,
                  int* version,
                  const char** reason) {
    try {
        YAML::Node root = YAML::Load(hdr_txt);
        *version = root["v"].as<int>(1);
        // We accept mismatched endian to be forgiving, but we always assume LE
        // on the wire — sender is required to emit LE.
        if (root["endian"] && root["endian"].as<std::string>() != "le") {
            *reason = "endian != le";
            return false;
        }
        auto fields = root["fields"];
        if (!fields || !fields.IsSequence()) {
            *reason = "fields missing";
            return false;
        }
        size_t offset = 0;
        for (auto f : fields) {
            Field fv;
            fv.name  = f["name"].as<std::string>();
            fv.dtype = f["dtype"].as<std::string>();
            auto shp = f["shape"];
            if (!shp || !shp.IsSequence()) {
                *reason = "shape missing";
                return false;
            }
            size_t count = 1;
            for (auto s : shp) {
                int64_t n = s.as<int64_t>();
                if (n < 0) { *reason = "negative shape"; return false; }
                fv.shape.push_back(n);
                count *= static_cast<size_t>(n);
            }
            const int isz = dtype_itemsize(fv.dtype);
            if (isz == 0) { *reason = "unknown dtype"; return false; }
            fv.offset = offset;
            fv.nbytes = count * isz;
            offset += fv.nbytes;
            out->push_back(std::move(fv));
        }
        return true;
    } catch (const std::exception& e) {
        *reason = "header parse error";
        return false;
    }
}

const Field* find_field(const std::vector<Field>& fields, std::string_view name) {
    for (const auto& f : fields) if (f.name == name) return &f;
    return nullptr;
}

// Copy [N, D] float32 payload into a caller-owned std::vector<float> of size N*D.
bool copy_matrix_f32(const std::vector<Field>& fields,
                     std::string_view name, int expected_inner,
                     const uint8_t* payload, size_t payload_size,
                     int* out_n, std::vector<float>* out,
                     const char** reason) {
    const Field* f = find_field(fields, name);
    if (!f) { *reason = "missing field"; return false; }
    if (f->dtype != "f32") { *reason = "dtype != f32"; return false; }
    if (f->shape.size() != 2 || f->shape[1] != expected_inner) {
        *reason = "shape mismatch"; return false;
    }
    if (f->offset + f->nbytes > payload_size) { *reason = "payload truncated"; return false; }
    *out_n = static_cast<int>(f->shape[0]);
    out->resize(static_cast<size_t>(*out_n) * expected_inner);
    std::memcpy(out->data(), payload + f->offset, f->nbytes);
    return true;
}

bool copy_vec_i64(const std::vector<Field>& fields,
                  std::string_view name, int expected_n,
                  const uint8_t* payload, size_t payload_size,
                  std::vector<int64_t>* out,
                  const char** reason) {
    const Field* f = find_field(fields, name);
    if (!f) { *reason = "missing field"; return false; }
    if (f->dtype != "i64") { *reason = "dtype != i64"; return false; }
    if (f->shape.size() != 1 || f->shape[0] != expected_n) {
        *reason = "shape mismatch"; return false;
    }
    if (f->offset + f->nbytes > payload_size) { *reason = "payload truncated"; return false; }
    out->resize(expected_n);
    std::memcpy(out->data(), payload + f->offset, f->nbytes);
    return true;
}

// One recv iteration; blocks up to `timeout_ms`. Returns:
//   true  -> handled a packet (accepted or dropped-with-reason)
//   false -> socket error or timeout with no packet
bool recv_and_dispatch(std::vector<uint8_t>* rxbuf, int timeout_ms) {
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    // Wait up to timeout for the next PUB frame.
    zmq_setsockopt(g_ctx.zmq_sock, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    int rc = zmq_msg_recv(&msg, g_ctx.zmq_sock, 0);
    if (rc < 0) {
        int err = zmq_errno();
        zmq_msg_close(&msg);
        if (err == EAGAIN || err == EINTR) return false;
        LOG_WARN("[rx_robot_motion_zmq] zmq_msg_recv err=%d %s", err, zmq_strerror(err));
        return false;
    }
    const size_t total = zmq_msg_size(&msg);
    const uint8_t* data = static_cast<const uint8_t*>(zmq_msg_data(&msg));

    // Strip topic prefix.
    const size_t topic_len = g_ctx.cfg.topic.size();
    if (total < topic_len + kHeaderSize) {
        g_ctx.note_drop("packet too short");
        zmq_msg_close(&msg);
        return true;
    }
    if (std::memcmp(data, g_ctx.cfg.topic.data(), topic_len) != 0) {
        g_ctx.note_drop("topic mismatch");
        zmq_msg_close(&msg);
        return true;
    }

    const uint8_t* header_bytes = data + topic_len;
    const uint8_t* payload      = data + topic_len + kHeaderSize;
    const size_t   payload_size = total - topic_len - kHeaderSize;

    // Find header end (null-padded to fixed size).
    size_t hdr_len = 0;
    while (hdr_len < kHeaderSize && header_bytes[hdr_len] != 0) ++hdr_len;
    std::string hdr_txt(reinterpret_cast<const char*>(header_bytes), hdr_len);

    std::vector<Field> fields;
    int version = 0;
    const char* reason = "";
    if (!parse_header(hdr_txt, &fields, &version, &reason)) {
        g_ctx.note_drop(reason);
        zmq_msg_close(&msg);
        return true;
    }
    if (version != 1) {
        g_ctx.note_drop("protocol version");
        zmq_msg_close(&msg);
        return true;
    }

    // Decode required fields.
    const int dof = g_ctx.cfg.expected_dof;
    int n_pos = 0, n_vel = 0, n_quat = 0;
    std::vector<float> jp, jv, bq;
    std::vector<int64_t> fi;
    if (!copy_matrix_f32(fields, "joint_pos", dof, payload, payload_size, &n_pos, &jp, &reason)) {
        g_ctx.note_drop(reason);
        zmq_msg_close(&msg);
        return true;
    }
    if (!copy_matrix_f32(fields, "joint_vel", dof, payload, payload_size, &n_vel, &jv, &reason)) {
        g_ctx.note_drop(reason);
        zmq_msg_close(&msg);
        return true;
    }
    if (!copy_matrix_f32(fields, "body_quat", 4, payload, payload_size, &n_quat, &bq, &reason)) {
        g_ctx.note_drop(reason);
        zmq_msg_close(&msg);
        return true;
    }
    if (n_pos != n_vel || n_pos != n_quat) {
        g_ctx.note_drop("N mismatch across fields");
        zmq_msg_close(&msg);
        return true;
    }
    if (!copy_vec_i64(fields, "frame_index", n_pos, payload, payload_size, &fi, &reason)) {
        g_ctx.note_drop(reason);
        zmq_msg_close(&msg);
        return true;
    }
    if (n_pos <= 0 || n_pos > kMaxFramesPerPacket) {
        g_ctx.note_drop("N out of range");
        zmq_msg_close(&msg);
        return true;
    }

    SonicRobotMotionBuffer* buf = g_ctx.buffer.load(std::memory_order_acquire);
    if (buf) {
        const uint64_t now = mono_us();
        for (int f = 0; f < n_pos; ++f) {
            RobotMotionFrame frame;
            frame.frame_index = fi[f];
            std::memcpy(frame.joint_pos.data(),      jp.data() + f * dof, dof * sizeof(float));
            std::memcpy(frame.joint_vel.data(),      jv.data() + f * dof, dof * sizeof(float));
            std::memcpy(frame.body_quat_wxyz.data(), bq.data() + f * 4,   4   * sizeof(float));
            buf->ingest(frame, /*recv_mono_us=*/now);
        }
    }
    g_ctx.accepted.fetch_add(1, std::memory_order_relaxed);
    zmq_msg_close(&msg);
    return true;
}

void sub_thread_main() {
    LOG_INFO("[rx_robot_motion_zmq] sub thread up (connect=%s topic=%s dof=%d)",
             g_ctx.cfg.connect_addr.c_str(), g_ctx.cfg.topic.c_str(),
             g_ctx.cfg.expected_dof);
    std::vector<uint8_t> rxbuf;
    while (g_ctx.running.load(std::memory_order_acquire)) {
        // 100 ms poll so Shutdown() wakes us within one cycle.
        recv_and_dispatch(&rxbuf, /*timeout_ms=*/100);
    }
    LOG_INFO("[rx_robot_motion_zmq] sub thread exiting (accepted=%llu dropped=%llu)",
             (unsigned long long)g_ctx.accepted.load(),
             (unsigned long long)g_ctx.dropped.load());
}

}  // namespace

bool RxRobotMotionZmqSubscriber::Init(const Config& cfg)
{
    if (g_ctx.running.load(std::memory_order_acquire)) {
        LOG_WARN("[rx_robot_motion_zmq] Init called twice; ignoring");
        return true;
    }
    g_ctx.cfg = cfg;
    g_ctx.zmq_ctx = zmq_ctx_new();
    if (!g_ctx.zmq_ctx) {
        LOG_WARN("[rx_robot_motion_zmq] zmq_ctx_new failed");
        return false;
    }
    g_ctx.zmq_sock = zmq_socket(g_ctx.zmq_ctx, ZMQ_SUB);
    if (!g_ctx.zmq_sock) {
        LOG_WARN("[rx_robot_motion_zmq] zmq_socket failed");
        zmq_ctx_term(g_ctx.zmq_ctx);
        g_ctx.zmq_ctx = nullptr;
        return false;
    }
    if (cfg.conflate) {
        int one = 1;
        zmq_setsockopt(g_ctx.zmq_sock, ZMQ_CONFLATE, &one, sizeof(one));
    }
    int hwm = 1;
    zmq_setsockopt(g_ctx.zmq_sock, ZMQ_RCVHWM,  &hwm, sizeof(hwm));
    zmq_setsockopt(g_ctx.zmq_sock, ZMQ_SUBSCRIBE, cfg.topic.data(), cfg.topic.size());
    int linger = 0;
    zmq_setsockopt(g_ctx.zmq_sock, ZMQ_LINGER, &linger, sizeof(linger));
    if (zmq_connect(g_ctx.zmq_sock, cfg.connect_addr.c_str()) != 0) {
        LOG_WARN("[rx_robot_motion_zmq] zmq_connect failed: %s", zmq_strerror(zmq_errno()));
        zmq_close(g_ctx.zmq_sock);
        zmq_ctx_term(g_ctx.zmq_ctx);
        g_ctx.zmq_sock = nullptr;
        g_ctx.zmq_ctx = nullptr;
        return false;
    }
    g_ctx.running.store(true, std::memory_order_release);
    g_ctx.thread = std::thread(sub_thread_main);
    return true;
}

void RxRobotMotionZmqSubscriber::BindBuffer(SonicRobotMotionBuffer* buffer)
{
    g_ctx.buffer.store(buffer, std::memory_order_release);
}

void RxRobotMotionZmqSubscriber::Shutdown()
{
    if (!g_ctx.running.load(std::memory_order_acquire)) return;
    g_ctx.running.store(false, std::memory_order_release);
    if (g_ctx.thread.joinable()) g_ctx.thread.join();
    if (g_ctx.zmq_sock) { zmq_close(g_ctx.zmq_sock); g_ctx.zmq_sock = nullptr; }
    if (g_ctx.zmq_ctx)  { zmq_ctx_term(g_ctx.zmq_ctx); g_ctx.zmq_ctx = nullptr; }
    g_ctx.buffer.store(nullptr, std::memory_order_release);
}

uint64_t RxRobotMotionZmqSubscriber::AcceptedCount() { return g_ctx.accepted.load(std::memory_order_relaxed); }
uint64_t RxRobotMotionZmqSubscriber::DroppedCount()  { return g_ctx.dropped.load(std::memory_order_relaxed); }
const char* RxRobotMotionZmqSubscriber::LastDropReason() {
    std::lock_guard<std::mutex> lk(g_ctx.drop_reason_mu);
    return g_ctx.last_drop_reason;
}

}  // namespace rx101
