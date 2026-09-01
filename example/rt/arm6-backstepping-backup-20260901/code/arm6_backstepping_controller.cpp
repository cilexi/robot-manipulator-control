/**
 * @file arm6_backstepping_controller.cpp
 * @brief Compact no-learning backstepping experiment for physical joint 6.
 *
 * This file is a clean No-RL baseline derived from optimal_backstepping_torque:
 *   1) optional scalar output-barrier coordinate z1;
 *   2) first backstepping virtual velocity alpha;
 *   3) z2=dq6-alpha and robust adaptive term nu_r;
 *   4) scalar nominal inertia mapping tau6=J6*v.
 *
 * Actor/critic networks, RBF features, Bellman residuals, pseudo-Huber loss,
 * weights, and every learning update are intentionally absent.  Joints 1-5
 * use a small engineering PD hold around q_init so the joint-6 scalar reduction
 * is tested with the remaining joints approximately fixed.
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include "rokae/robot.h"

namespace {

constexpr std::size_t kDof = 6;
constexpr std::size_t kJoint6 = 5;
constexpr double kDt = 0.001;
constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1.0e-9;

using Vec6 = std::array<double, kDof>;

struct Config {
  bool hardware_enable = false;
  bool ready_pose_confirmed = false;
  bool perform_ready_move = false;
  bool power_off_on_exit = true;
  std::string robot_ip = "192.168.2.200";
  std::string local_ip = "192.168.2.100";
  Vec6 ready_q{};
  double ready_move_speed = 0.05;

  // qd6=q_init6+A*sin(omega*t), followed by a center hold.
  double reference_amplitude6 = 0.02;
  double reference_omega6 = 0.523598775598;
  double reference_cycles = 2.0;
  double final_hold_time = 3.0;

  // Optional absolute joint-angle constraint: -output_bound6<q6<output_bound6.
  bool enable_output_barrier = true;
  double output_bound6 = 0.10;
  double barrier_b0 = 2.0;
  double barrier_bf = 1.0;
  double barrier_abort_ratio = 0.90;

  // No-RL two-layer backstepping gains, used directly in each layer.
  double k1_6 = 6.0;
  double k2_6 = 10.0;
  double alpha_limit6 = 0.30;
  double alpha_dot_limit6 = 3.0;
  double alpha_dot_filter = 0.02;
  double virtual_accel_limit6 = 8.0;

  // Core-function robust adaptation retained from the original No-RL branch.
  bool enable_theta_adaptation = true;
  bool freeze_theta_on_virtual_saturation = true;
  double iota = 1.0;
  double theta_initial = 0.05;
  double gamma_theta = 0.05;
  double sigma_theta = 0.10;
  double theta_max = 5.0;
  double adaptation_start_time = 1.0;

  // Explicit scalar input map for joint 6.  This replaces the large embedded
  // URDF calculator and must be identified/checked for the actual setup.
  double nominal_inertia6 = 0.02;//论文控制率中的B的倒数，即惯性矩阵

  // Engineering PD hold for joints 1-5; not part of the joint-6 control law.
  double hold_kp = 2.0;
  double hold_kd = 0.10;

  double command_ramp_time = 1.0;
  double stop_ramp_time = 1.0;
  double sdk_filter_cutoff_hz = 30.0;
  Vec6 tau_limit{0.40, 0.40, 0.40, 0.40, 0.40, 0.30};
  Vec6 tau_rate_limit{10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
  double joint6_abort_error = 0.08;
  double joint6_displacement_limit = 0.10;
  Vec6 abort_velocity{0.30, 0.30, 0.30, 0.30, 0.30, 0.30};
  Vec6 hold_joint_drift_limit{0.02, 0.02, 0.02, 0.02, 0.02, 0.0};
  double log_decimation = 10.0;
};

double clamp(double value, double lo, double hi) {//夹住，将值限制在hi和lo之间
  return std::max(lo, std::min(hi, value));
}

bool isFinite(double value) { return std::isfinite(value); }//判断是否为有限数

std::string trimCopy(const std::string &text) {//去掉字符串前后的空格
  std::size_t first = 0;
  while (first < text.size() &&
         std::isspace(static_cast<unsigned char>(text[first]))) {
    ++first;
  }
  std::size_t last = text.size();
  while (last > first &&
         std::isspace(static_cast<unsigned char>(text[last - 1]))) {
    --last;
  }
  return text.substr(first, last - first);
}

double parseDouble(const std::string &text, const std::string &key,
                   int line_no) {//将字符串转换为double类型，用于解析配置文件中的数值
  std::size_t used = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &used);
  } catch (...) {
    throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                             " 行 " + key + " 不是有效数字");
  }
  if (used != text.size() || !isFinite(value)) {
    throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                             " 行 " + key + " 必须是有限数字");
  }
  return value;
}

bool parseBool(const std::string &text, const std::string &key, int line_no) {//将字符串转换为bool类型，用于解析配置文件中的布尔值
  std::string lowered = text;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  if (lowered == "true" || lowered == "1") {
    return true;
  }
  if (lowered == "false" || lowered == "0") {
    return false;
  }
  throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                           " 行 " + key + " 必须是 true/false");
}

Vec6 parseVec6(const std::string &text, const std::string &key, int line_no) {//将字符串解析为6维向量，用于解析配置文件中的向量值
  std::istringstream stream(text);
  Vec6 values{};
  for (double &value : values) {
    if (!(stream >> value) || !isFinite(value)) {
      throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                               " 行 " + key + " 必须包含 6 个有限数字");
    }
  }
  std::string extra;
  if (stream >> extra) {
    throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                             " 行 " + key + " 只能包含 6 个数字");
  }
  return values;
}

void applyConfig(Config &cfg, const std::string &key, const std::string &value,
                 int line_no) {//将配置文件中的键值对应用到配置结构体中
#define SET_BOOL(name)                                                         \
  if (key == #name) {                                                         \
    cfg.name = parseBool(value, key, line_no);                                \
    return;                                                                   \
  }
#define SET_SCALAR(name)                                                       \
  if (key == #name) {                                                         \
    cfg.name = parseDouble(value, key, line_no);                              \
    return;                                                                   \
  }
#define SET_VEC(name)                                                          \
  if (key == #name) {                                                         \
    cfg.name = parseVec6(value, key, line_no);                                \
    return;                                                                   \
  }

  SET_BOOL(hardware_enable)
  SET_BOOL(ready_pose_confirmed)
  SET_BOOL(perform_ready_move)
  SET_BOOL(power_off_on_exit)
  if (key == "robot_ip") {
    cfg.robot_ip = value;
    return;
  }
  if (key == "local_ip") {
    cfg.local_ip = value;
    return;
  }
  SET_VEC(ready_q)
  SET_SCALAR(ready_move_speed)
  SET_SCALAR(reference_amplitude6)
  SET_SCALAR(reference_omega6)
  SET_SCALAR(reference_cycles)
  SET_SCALAR(final_hold_time)
  SET_BOOL(enable_output_barrier)
  SET_SCALAR(output_bound6)
  SET_SCALAR(barrier_b0)
  SET_SCALAR(barrier_bf)
  SET_SCALAR(barrier_abort_ratio)
  SET_SCALAR(k1_6)
  SET_SCALAR(k2_6)
  SET_SCALAR(alpha_limit6)
  SET_SCALAR(alpha_dot_limit6)
  SET_SCALAR(alpha_dot_filter)
  SET_SCALAR(virtual_accel_limit6)
  SET_BOOL(enable_theta_adaptation)
  SET_BOOL(freeze_theta_on_virtual_saturation)
  SET_SCALAR(iota)
  SET_SCALAR(theta_initial)
  SET_SCALAR(gamma_theta)
  SET_SCALAR(sigma_theta)
  SET_SCALAR(theta_max)
  SET_SCALAR(adaptation_start_time)
  SET_SCALAR(nominal_inertia6)
  SET_SCALAR(hold_kp)
  SET_SCALAR(hold_kd)
  SET_SCALAR(command_ramp_time)
  SET_SCALAR(stop_ramp_time)
  SET_SCALAR(sdk_filter_cutoff_hz)
  SET_VEC(tau_limit)
  SET_VEC(tau_rate_limit)
  SET_SCALAR(joint6_abort_error)
  SET_SCALAR(joint6_displacement_limit)
  SET_VEC(abort_velocity)
  SET_VEC(hold_joint_drift_limit)
  SET_SCALAR(log_decimation)

#undef SET_BOOL
#undef SET_SCALAR
#undef SET_VEC
  throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                           " 行存在未知参数: " + key);
}

void requireFinite(double value, const std::string &name) {//检查数值是否为有限数，如果不是则抛出异常
  if (!isFinite(value)) {
    throw std::runtime_error("配置项 " + name + " 必须是有限数字");
  }
}

void requirePositive(double value, const std::string &name) {//检查数值是否为正数，如果不是则抛出异常
  requireFinite(value, name);
  if (value <= 0.0) {
    throw std::runtime_error("配置项 " + name + " 必须大于 0");
  }
}

void requireNonNegative(double value, const std::string &name) {//检查数值是否为非负数，如果不是则抛出异常
  requireFinite(value, name);
  if (value < 0.0) {
    throw std::runtime_error("配置项 " + name + " 不能小于 0");
  }
}

double sineDuration(const Config &cfg) {//计算正弦轨迹的持续时间
  return cfg.reference_cycles * 2.0 * kPi / cfg.reference_omega6;
}

double totalTime(const Config &cfg) {//计算总时间，包括正弦轨迹的持续时间和最终保持时间
  return sineDuration(cfg) + cfg.final_hold_time;
}

void validateConfig(const Config &cfg) {//验证配置项的有效性，如果不符合要求则抛出异常
  if (cfg.robot_ip.empty() || cfg.local_ip.empty()) {
    throw std::runtime_error("robot_ip 和 local_ip 不能为空");
  }
  for (std::size_t joint = 0; joint < kDof; ++joint) {
    requireFinite(cfg.ready_q[joint],
                  "ready_q[" + std::to_string(joint + 1) + "]");
    requirePositive(cfg.tau_limit[joint],
                    "tau_limit[" + std::to_string(joint + 1) + "]");
    requirePositive(cfg.tau_rate_limit[joint],
                    "tau_rate_limit[" + std::to_string(joint + 1) + "]");
    requirePositive(cfg.abort_velocity[joint],
                    "abort_velocity[" + std::to_string(joint + 1) + "]");
    if (joint != kJoint6) {
      requirePositive(cfg.hold_joint_drift_limit[joint],
                      "hold_joint_drift_limit[" +
                          std::to_string(joint + 1) + "]");
    }
  }
  requirePositive(cfg.ready_move_speed, "ready_move_speed");
  requirePositive(cfg.reference_amplitude6, "reference_amplitude6");
  requirePositive(cfg.reference_omega6, "reference_omega6");
  requirePositive(cfg.reference_cycles, "reference_cycles");
  if (std::abs(cfg.reference_cycles - std::round(cfg.reference_cycles)) >
      kEps) {
    throw std::runtime_error("reference_cycles 必须是正整数");
  }
  requirePositive(cfg.final_hold_time, "final_hold_time");
  requirePositive(cfg.output_bound6, "output_bound6");
  requirePositive(cfg.barrier_b0, "barrier_b0");
  requirePositive(cfg.barrier_bf, "barrier_bf");
  if (cfg.barrier_b0 <= cfg.barrier_bf) {
    throw std::runtime_error("barrier_b0 必须大于 barrier_bf");
  }
  if (!(cfg.barrier_abort_ratio > 0.0 && cfg.barrier_abort_ratio < 1.0)) {
    throw std::runtime_error("barrier_abort_ratio 必须位于 (0,1)");
  }
  if (cfg.enable_output_barrier &&
      cfg.reference_amplitude6 >=
          cfg.barrier_abort_ratio * cfg.output_bound6) {
    throw std::runtime_error("第六关节参考轨迹可能触及障碍停机边界");
  }
  requirePositive(cfg.k1_6, "k1_6");
  requirePositive(cfg.k2_6, "k2_6");
  requirePositive(cfg.alpha_limit6, "alpha_limit6");
  requirePositive(cfg.alpha_dot_limit6, "alpha_dot_limit6");
  if (!(cfg.alpha_dot_filter > 0.0 && cfg.alpha_dot_filter <= 1.0)) {
    throw std::runtime_error("alpha_dot_filter 必须位于 (0,1]");
  }
  requirePositive(cfg.virtual_accel_limit6, "virtual_accel_limit6");
  requirePositive(cfg.iota, "iota");
  requireNonNegative(cfg.theta_initial, "theta_initial");
  requireNonNegative(cfg.gamma_theta, "gamma_theta");
  requireNonNegative(cfg.sigma_theta, "sigma_theta");
  requirePositive(cfg.theta_max, "theta_max");
  if (cfg.theta_initial > cfg.theta_max) {
    throw std::runtime_error("theta_initial 不能大于 theta_max");
  }
  requireNonNegative(cfg.adaptation_start_time, "adaptation_start_time");
  requirePositive(cfg.nominal_inertia6, "nominal_inertia6");
  requireNonNegative(cfg.hold_kp, "hold_kp");
  requireNonNegative(cfg.hold_kd, "hold_kd");
  if (cfg.hold_kp + cfg.hold_kd <= 0.0) {
    throw std::runtime_error("hold_kp 与 hold_kd 不能同时为 0");
  }
  requireNonNegative(cfg.command_ramp_time, "command_ramp_time");
  requireNonNegative(cfg.stop_ramp_time, "stop_ramp_time");
  if (cfg.stop_ramp_time > cfg.final_hold_time) {
    throw std::runtime_error("stop_ramp_time 不能大于 final_hold_time");
  }
  requirePositive(cfg.sdk_filter_cutoff_hz, "sdk_filter_cutoff_hz");
  if (cfg.sdk_filter_cutoff_hz > 1000.0) {
    throw std::runtime_error("sdk_filter_cutoff_hz 不能超过 1000 Hz");
  }
  requirePositive(cfg.joint6_abort_error, "joint6_abort_error");
  requirePositive(cfg.joint6_displacement_limit,
                  "joint6_displacement_limit");
  if (cfg.reference_amplitude6 >= 0.8 * cfg.joint6_displacement_limit) {
    throw std::runtime_error(
        "reference_amplitude6 必须小于 joint6_displacement_limit 的 80%");
  }
  requirePositive(cfg.log_decimation, "log_decimation");
  if (std::abs(cfg.log_decimation - std::round(cfg.log_decimation)) > kEps) {
    throw std::runtime_error("log_decimation 必须是整数");
  }
}

Config loadConfig(const std::string &path) {//
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("无法打开配置文件: " + path);
  }
  Config cfg;
  std::string line;
  int line_no = 0;
  while (std::getline(input, line)) {
    ++line_no;
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
      line.erase(comment);
    }
    line = trimCopy(line);
    if (line.empty()) {
      continue;
    }
    const auto equal = line.find('=');
    if (equal == std::string::npos) {
      throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                               " 行必须写成 key = value");
    }
    const std::string key = trimCopy(line.substr(0, equal));
    const std::string value = trimCopy(line.substr(equal + 1));
    if (key.empty() || value.empty()) {
      throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                               " 行的 key 或 value 为空");
    }
    applyConfig(cfg, key, value, line_no);
  }
  validateConfig(cfg);
  return cfg;
}

bool fileReadable(const std::string &path) {//检查文件是否可读
  std::ifstream input(path);
  return input.good();
}

std::string defaultConfigPath() {//返回默认配置文件路径
  const std::array<std::string, 5> candidates{
      "arm6_backstepping.cfg", "config/arm6_backstepping.cfg",
      "../config/nomal_backstepping.cfg", "example/rt/nomal_backstepping.cfg",
      "../example/rt/nomal_backstepping.cfg"};
  for (const auto &candidate : candidates) {
    if (fileReadable(candidate)) {
      return candidate;
    }
  }
  return candidates.front();
}

struct Desired {//期望轨迹结构体，包含位置、速度、加速度和正弦激活标志
  double q = 0.0;
  double dq = 0.0;
  double ddq = 0.0;
  bool sine_active = false;
};

Desired desiredTrajectory(double time, double q_init6, const Config &cfg) {//计算期望轨迹，根据时间和配置参数生成正弦轨迹
  Desired desired;
  desired.q = q_init6;
  if (time < sineDuration(cfg)) {
    const double angle = cfg.reference_omega6 * time;//计算当前时间对应的角度
    desired.q += cfg.reference_amplitude6 * std::sin(angle);//计算期望位置
    desired.dq =
        cfg.reference_amplitude6 * cfg.reference_omega6 * std::cos(angle);
    desired.ddq = -cfg.reference_amplitude6 * cfg.reference_omega6 *
                  cfg.reference_omega6 * std::sin(angle);
    desired.sine_active = true;
  }//参考轨迹形式：q_init + A × sin(ω × t)
  return desired;
}

struct BarrierScalar {//输出障碍变换结果，包含s、h1、h2、ds/dt和比率
  double transformed = 0.0;
  double jacobian = 1.0;
  double time_term = 0.0;
  double derivative = 0.0;
  double ratio = 0.0;
};

BarrierScalar barrierTransform(double x, double x_dot, double lambda,
                               double beta, double beta_dot,
                               double b0) {//输出障碍变换函数，将输入x和x_dot进行变换，返回变换后的结果
  const double root = std::sqrt(x * x + lambda * lambda + kEps);//
  const double eta = b0 * x / root;
  const double raw_ratio = eta / (beta + kEps);
  const double ratio = clamp(raw_ratio, -0.9995, 0.9995);//被限制的比例系数（-1，1）
  const double one_minus = 1.0 - ratio * ratio;
  const double transformed = ratio / (one_minus + kEps);//s
  const double ds_dratio =
      (1.0 + ratio * ratio) / (one_minus * one_minus + kEps);//s对ratio偏导
  const double deta_dx = b0 * lambda * lambda /
                         std::pow(x * x + lambda * lambda + kEps, 1.5);//η对x偏导
  const double dratio_dbeta =
      -eta / ((beta + kEps) * (beta + kEps));
  const double jacobian =
      std::max(1.0e-5, ds_dratio * deta_dx / (beta + kEps));//求h1的式子，正确
  const double time_term = ds_dratio * dratio_dbeta * beta_dot;//论文中的h_beta_x（h2）
  return {transformed, jacobian, time_term,
          jacobian * x_dot + time_term, raw_ratio};
}

struct BacksteppingOutput {
  Desired desired{};
  double error = 0.0;
  double z1 = 0.0;
  double z2 = 0.0;
  double hx = 1.0;
  double hbeta_x = 0.0;
  double barrier_ratio = 0.0;//输出障碍变换的比率？
  double alpha = 0.0;
  double alpha_dot = 0.0;
  double mu_s = 0.0;
  double phi = 1.0;
  double nu_r = 0.0;
  double term_z2 = 0.0;
  double term_mu = 0.0;
  double term_nu = 0.0;
  double virtual_accel_unsaturated = 0.0;//未饱和的虚拟加速度？
  double virtual_accel = 0.0;
  double theta_hat = 0.0;
  double torque6_raw = 0.0;
  bool alpha_saturated = false;
  bool alpha_dot_saturated = false;
  bool virtual_accel_saturated = false;
  bool finite = true;
};

class NoRlBacksteppingController {
 public:
  NoRlBacksteppingController(const Config &cfg, double q_init6)
      : cfg_(cfg), q_init6_(q_init6), theta_hat_(cfg.theta_initial) {}

  BacksteppingOutput compute(double time, double q6, double dq6) {
    BacksteppingOutput output;
    output.desired = desiredTrajectory(time, q_init6_, cfg_);//计算期望轨迹
    output.error = q6 - output.desired.q;

    double sd_dot = output.desired.dq;
    if (cfg_.enable_output_barrier) {//如果启用输出障碍变换，则计算变换后的值
      // 当前为固定绝对约束，因此beta=bf且beta_dot=0；仍完整保留论文h_beta项。
      const double beta = cfg_.barrier_bf;
      const double beta_dot = 0.0;
      const double lambda =//计算lambda值，固定边界情况下正确
          cfg_.output_bound6 *
          std::sqrt(cfg_.barrier_b0 * cfg_.barrier_b0 -
                    cfg_.barrier_bf * cfg_.barrier_bf) /
          cfg_.barrier_bf;
      const BarrierScalar actual =//计算实际值的障碍变换，包含s,h1,h2,ds/dt,ratio
          barrierTransform(q6, dq6, lambda, beta, beta_dot,
                           cfg_.barrier_b0);
      const BarrierScalar desired = barrierTransform(
          output.desired.q, output.desired.dq, lambda,
          beta, beta_dot, cfg_.barrier_b0);
      output.z1 = actual.transformed - desired.transformed;
      output.hx = actual.jacobian;
      output.hbeta_x = actual.time_term;
      output.barrier_ratio = actual.ratio;
      sd_dot = desired.derivative;//期望变换导数h_d*dq_d+h_beta_d
    } else {
      output.z1 = output.error;
    }

    const double alpha_raw =
        (-cfg_.k1_6 * output.z1 - output.hbeta_x + sd_dot) /
        (output.hx + kEps);//论文虚拟控制律：Hx^{-1}(-K1*z1-h_beta_x+s_dot_d)
    output.alpha = clamp(alpha_raw, -cfg_.alpha_limit6, cfg_.alpha_limit6);
    output.alpha_saturated = std::abs(output.alpha - alpha_raw) > kEps;

    if (!alpha_initialized_) {//如果alpha的导数未初始化，则初始化
      alpha_dot_filtered_ = output.desired.ddq;
      alpha_initialized_ = true;
    } else {
      const double raw_alpha_dot = (output.alpha - previous_alpha_) / kDt;
      alpha_dot_filtered_ +=
          cfg_.alpha_dot_filter * (raw_alpha_dot - alpha_dot_filtered_);
    }
    previous_alpha_ = output.alpha;
    output.alpha_dot = clamp(alpha_dot_filtered_, -cfg_.alpha_dot_limit6,
                             cfg_.alpha_dot_limit6);//限制后的滤波alpha导数
    output.alpha_dot_saturated =//
        std::abs(output.alpha_dot - alpha_dot_filtered_) > kEps;

    output.z2 = dq6 - output.alpha;
    output.phi = 1.0 + std::abs(q6) + std::abs(dq6) + 0.25 * dq6 * dq6;
    output.mu_s = output.hx * output.z1;//论文中的mu_s=Hx*z1
    output.nu_r = theta_hat_ * output.phi * output.phi * output.z2 /
                  (2.0 * cfg_.iota * cfg_.iota);//论文中的nu_r=theta_hat*phi^2*z2/(2*iota^2)

    output.term_z2 = -cfg_.k2_6 * output.z2;//论文中的term_z2=-K2*z2
    output.term_mu = -output.mu_s;
    output.term_nu = -output.nu_r;//论文中的term_nu=-nu_r
    output.virtual_accel_unsaturated =//论文中B(***)括号中项
        output.alpha_dot + output.term_z2 + output.term_mu + output.term_nu;
    output.virtual_accel =//做个限制
        clamp(output.virtual_accel_unsaturated, -cfg_.virtual_accel_limit6,
              cfg_.virtual_accel_limit6);
    output.virtual_accel_saturated =//留个判断是否到达限制
        std::abs(output.virtual_accel - output.virtual_accel_unsaturated) >
        kEps;
    output.torque6_raw = cfg_.nominal_inertia6 * output.virtual_accel;

    if (cfg_.enable_theta_adaptation &&
        time >= cfg_.adaptation_start_time &&
        !(cfg_.freeze_theta_on_virtual_saturation &&
          output.virtual_accel_saturated)) {
      const double denominator = 2.0 * cfg_.iota * cfg_.iota;
      theta_hat_ +=
          kDt * (cfg_.gamma_theta / denominator * output.phi * output.phi *
                     output.z2 * output.z2 -
                 cfg_.sigma_theta * theta_hat_);
      theta_hat_ = clamp(theta_hat_, 0.0, cfg_.theta_max);
    }
    output.theta_hat = theta_hat_;
    output.finite =
        isFinite(output.desired.q) && isFinite(output.desired.dq) &&
        isFinite(output.desired.ddq) && isFinite(output.error) &&
        isFinite(output.z1) && isFinite(output.z2) && isFinite(output.hx) &&
        isFinite(output.hbeta_x) &&
        isFinite(output.alpha) && isFinite(output.alpha_dot) &&
        isFinite(output.mu_s) && isFinite(output.phi) &&
        isFinite(output.nu_r) && isFinite(output.virtual_accel) &&
        isFinite(output.theta_hat) && isFinite(output.torque6_raw);
    return output;
  }

 private:
  const Config &cfg_;
  double q_init6_ = 0.0;
  double theta_hat_ = 0.0;
  double previous_alpha_ = 0.0;
  double alpha_dot_filtered_ = 0.0;
  bool alpha_initialized_ = false;
};

struct LimitedTorque {
  Vec6 raw{};
  Vec6 command{};
  Vec6 amplitude_saturated{};
  Vec6 rate_saturated{};
  double ramp_factor = 1.0;
};

LimitedTorque buildAndLimitTorque(const BacksteppingOutput &control,
                                  const Vec6 &q, const Vec6 &dq,
                                  const Vec6 &q_init, const Vec6 &previous,
                                  double time, const Config &cfg) {//构建并限制关节力矩输出，根据控制器输出、关节状态和配置参数生成最终的力矩命令
  LimitedTorque output;
  for (std::size_t joint = 0; joint < kJoint6; ++joint) {
    output.raw[joint] = -cfg.hold_kp * (q[joint] - q_init[joint]) -
                        cfg.hold_kd * dq[joint];
  }
  output.raw[kJoint6] = control.torque6_raw;

  double ramp = 1.0;
  if (cfg.command_ramp_time > kEps && time < cfg.command_ramp_time) {
    ramp = 0.5 * (1.0 - std::cos(kPi * time / cfg.command_ramp_time));
  }
  const double stop_start = totalTime(cfg) - cfg.stop_ramp_time;
  if (cfg.stop_ramp_time > kEps && time > stop_start) {
    const double u = clamp((time - stop_start) / cfg.stop_ramp_time, 0.0, 1.0);
    ramp *= 0.5 * (1.0 + std::cos(kPi * u));
  }
  output.ramp_factor = ramp;
  for (std::size_t joint = 0; joint < kDof; ++joint) {
    const double requested = ramp * output.raw[joint];
    const double amplitude_limited =
        clamp(requested, -cfg.tau_limit[joint], cfg.tau_limit[joint]);
    output.amplitude_saturated[joint] =
        std::abs(amplitude_limited - requested) > kEps ? 1.0 : 0.0;
    const double step = cfg.tau_rate_limit[joint] * kDt;
    output.command[joint] =
        clamp(amplitude_limited, previous[joint] - step,
              previous[joint] + step);
    output.rate_saturated[joint] =
        std::abs(output.command[joint] - amplitude_limited) > kEps ? 1.0
                                                                   : 0.0;
  }
  return output;
}

struct SafetyStatus {
  bool abort = false;
  std::string reason;
};

SafetyStatus checkSafety(const Vec6 &q, const Vec6 &dq, const Vec6 &q_init,
                         const BacksteppingOutput &control,
                         bool state_read_ok, const Config &cfg) {//检查安全状态，包括关节状态、控制器输出和配置参数
  if (!state_read_ok) {
    return {true, "实时状态读取失败"};
  }
  if (!control.finite) {
    return {true, "反步控制器出现非有限数值"};
  }
  for (std::size_t joint = 0; joint < kDof; ++joint) {
    if (!isFinite(q[joint]) || !isFinite(dq[joint])) {
      return {true, "关节状态出现非有限数值"};
    }
    if (std::abs(dq[joint]) > cfg.abort_velocity[joint]) {
      return {true, "关节 " + std::to_string(joint + 1) + " 速度超限"};
    }
    if (joint != kJoint6 &&
        std::abs(q[joint] - q_init[joint]) >
            cfg.hold_joint_drift_limit[joint]) {
      return {true, "保持关节 " + std::to_string(joint + 1) +
                        " 偏离实验起点"};
    }
  }
  if (std::abs(control.error) > cfg.joint6_abort_error) {
    return {true, "关节 6 跟踪误差超限"};
  }
  if (std::abs(q[kJoint6] - q_init[kJoint6]) >
      cfg.joint6_displacement_limit) {
    return {true, "关节 6 偏离实验起点超过允许范围"};
  }
  if (cfg.enable_output_barrier &&
      std::abs(control.barrier_ratio) > cfg.barrier_abort_ratio) {
    return {true, "关节 6 接近输出约束边界"};
  }
  return {};
}

void writeVectorHeader(std::ofstream &log, const std::string &name) {//写入向量表头到日志文件
  for (std::size_t joint = 0; joint < kDof; ++joint) {
    log << ',' << name << (joint + 1);
  }
}

void writeVector(std::ofstream &log, const Vec6 &values) {//写入向量数据到日志文件
  for (double value : values) {
    log << ',' << value;
  }
}

void writeLogHeader(std::ofstream &log) {//写入日志文件的表头，包括时间、关节状态、控制器输出和力矩命令
  log << "time,sine_active,ramp_factor";
  writeVectorHeader(log, "q");
  writeVectorHeader(log, "dq");
  log << ",qd6,dqd6,ddqd6,error6,z1,z2,hx,hbeta_x6,barrier_ratio6,alpha6"
         ",alpha_dot6,mu_s6,phi6,nu_r6,term_z2_6,term_mu_6,term_nu_6"
         ",virtual_accel_unsat6,virtual_accel6,theta_hat,tau6_raw";
  writeVectorHeader(log, "tau_raw");
  writeVectorHeader(log, "tau_cmd");
  log << ",alpha_saturated,alpha_dot_saturated,virtual_accel_saturated"
         ",amplitude_saturated6,rate_saturated6,abort\n";
}

void writeLogRow(std::ofstream &log, double time, const Vec6 &q,
                 const Vec6 &dq, const BacksteppingOutput &control,
                 const LimitedTorque &torque, bool abort) {//写入日志行，包括时间、关节状态、控制器输出和力矩命令
  log << std::setprecision(10) << time << ','
      << (control.desired.sine_active ? 1 : 0) << ',' << torque.ramp_factor;
  writeVector(log, q);
  writeVector(log, dq);
  log << ',' << control.desired.q << ',' << control.desired.dq << ','
      << control.desired.ddq << ',' << control.error << ',' << control.z1
      << ',' << control.z2 << ',' << control.hx << ','
      << control.hbeta_x << ','
      << control.barrier_ratio << ',' << control.alpha << ','
      << control.alpha_dot << ',' << control.mu_s << ',' << control.phi
      << ',' << control.nu_r << ',' << control.term_z2 << ','
      << control.term_mu << ',' << control.term_nu << ','
      << control.virtual_accel_unsaturated << ',' << control.virtual_accel
      << ',' << control.theta_hat << ',' << control.torque6_raw;
  writeVector(log, torque.raw);
  writeVector(log, torque.command);
  log << ',' << (control.alpha_saturated ? 1 : 0) << ','
      << (control.alpha_dot_saturated ? 1 : 0) << ','
      << (control.virtual_accel_saturated ? 1 : 0) << ','
      << torque.amplitude_saturated[kJoint6] << ','
      << torque.rate_saturated[kJoint6] << ',' << (abort ? 1 : 0) << '\n';
}

void throwIfError(const std::error_code &ec, const std::string &operation) {//检查错误码，如果有错误则抛出异常
  if (ec) {
    throw std::runtime_error(operation + "失败: " + ec.message());
  }
}

void printConfigSummary(const Config &cfg, const std::string &path) {//打印配置摘要信息，包括关节数、是否启用输出障碍、控制参数和参考轨迹参数
  std::cout << "arm6_backstepping 配置检查通过: " << path << '\n'
            << "  joint=6, RL/NN=absent, barrier="
            << (cfg.enable_output_barrier ? "true" : "false") << '\n'
            << "  K1=" << cfg.k1_6 << ", K2=" << cfg.k2_6 << '\n'
            << "  reference A=" << cfg.reference_amplitude6
            << " rad, omega=" << cfg.reference_omega6 << " rad/s\n"
            << "  nominal_inertia6=" << cfg.nominal_inertia6
            << ", v_limit=" << cfg.virtual_accel_limit6
            << ", tau_limit6=" << cfg.tau_limit[kJoint6] << " Nm\n"
            << "  sine_duration=" << sineDuration(cfg)
            << " s, total_time=" << totalTime(cfg) << " s" << std::endl;
}

#ifndef NOMAL_BACKSTEPPING_OFFLINE_TEST
bool runExperiment(rokae::xMateRobot &robot, const Config &cfg) {//运行实验函数，连接机械臂并执行实时控制
  using namespace rokae;
  using namespace rokae::RtSupportedFields;
  auto rt_controller = robot.getRtMotionController().lock();
  if (!rt_controller) {
    throw std::runtime_error("无法获取实时运动控制器");
  }
  std::error_code ec;
  robot.stopReceiveRobotState();
  robot.startReceiveRobotState(std::chrono::milliseconds(1),
                               {jointPos_m, jointVel_m});
  if (cfg.perform_ready_move) {
    ec.clear();
    const Vec6 current_q = robot.jointPos(ec);
    throwIfError(ec, "读取 MoveJ 起点");
    rt_controller->MoveJ(cfg.ready_move_speed, current_q, cfg.ready_q);
  } else {
    std::cout << "跳过 MoveJ，当前姿态作为第六关节正弦中心和其余关节保持位置。"
              << std::endl;
  }
  ec.clear();
  const Vec6 q_init = robot.jointPos(ec);
  throwIfError(ec, "读取实时控制初始位姿");
  NoRlBacksteppingController controller(cfg, q_init[kJoint6]);
  if (!rt_controller->setFilterLimit(true, cfg.sdk_filter_cutoff_hz)) {
    throw std::runtime_error("SDK 力矩滤波参数设置失败");
  }
  std::ofstream log("arm6_backstepping_log.csv", std::ios::trunc);
  if (!log.is_open()) {
    throw std::runtime_error("无法创建 arm6_backstepping_log.csv");
  }
  writeLogHeader(log);
  rt_controller->startMove(RtControllerMode::torque);
  std::cout << "已进入第六关节 No-RL 反步力矩实验。" << std::endl;

  double time = 0.0;
  std::size_t tick = 0;
  Vec6 previous_torque{};
  bool experiment_aborted = false;
  std::string abort_reason;
  const std::size_t log_decimation =
      static_cast<std::size_t>(std::llround(cfg.log_decimation));

  std::function<Torque(void)> callback = [&]() -> Torque {
    time += kDt;
    ++tick;
    Vec6 q{};
    Vec6 dq{};
    const bool state_read_ok =
        robot.getStateData(jointPos_m, q) == 0 &&
        robot.getStateData(jointVel_m, dq) == 0;
    BacksteppingOutput control;
    LimitedTorque torque;
    SafetyStatus safety;
    if (state_read_ok) {
      control = controller.compute(time, q[kJoint6], dq[kJoint6]);
      torque = buildAndLimitTorque(control, q, dq, q_init, previous_torque,
                                   time, cfg);
      safety = checkSafety(q, dq, q_init, control, true, cfg);
    } else {
      safety = {true, "实时状态读取失败"};
    }
    if (safety.abort) {
      torque.command.fill(0.0);
      experiment_aborted = true;
      abort_reason = safety.reason;
    }
    previous_torque = torque.command;
    if (tick % log_decimation == 0 || safety.abort) {
      writeLogRow(log, time, q, dq, control, torque, safety.abort);
    }
    if (tick % 500 == 0) {
      std::cout << "t=" << time << " e6=" << control.error
                << " z1=" << control.z1 << " z2=" << control.z2
                << " tau6=" << torque.command[kJoint6]
                << " theta=" << control.theta_hat << std::endl;
      log.flush();
    }
    Torque command(kDof, 0.0);
    for (std::size_t joint = 0; joint < kDof; ++joint) {
      command.tau[joint] = torque.command[joint];
    }
    if (safety.abort || time >= totalTime(cfg)) {
      command.tau.assign(kDof, 0.0);
      command.setFinished();
    }
    return command;
  };
  rt_controller->setControlLoop(callback, 0, true);
  rt_controller->startLoop(true);
  log.flush();
  log.close();
  if (experiment_aborted) {
    std::cerr << "实验由安全保护中止: " << abort_reason << std::endl;
  } else {
    std::cout << "实验完成。日志: arm6_backstepping_log.csv" << std::endl;
  }
  return !experiment_aborted;
}
#endif

void printUsage(const char *program) {
  std::cout << "用法:\n"
            << "  " << program << " --check-config [cfg路径]\n"
            << "  " << program
            << " --run [cfg路径] [机器人IP] [本机实时网卡IP]\n"
            << "无参数运行只检查配置，不连接机械臂。" << std::endl;
}

}  // namespace

#ifndef NOMAL_BACKSTEPPING_OFFLINE_TEST
int main(int argc, char *argv[]) {//主函数，处理命令行参数，加载配置文件，连接机械臂并运行实验
  const std::string command = argc > 1 ? argv[1] : "--check-config";
  if (command != "--check-config" && command != "--run") {
    printUsage(argv[0]);
    return 2;
  }
  const std::string config_path = argc > 2 ? argv[2] : defaultConfigPath();
  Config cfg;
  try {
    cfg = loadConfig(config_path);
    if (command == "--run") {
      if (argc > 3) cfg.robot_ip = argv[3];
      if (argc > 4) cfg.local_ip = argv[4];
      validateConfig(cfg);
    }
    printConfigSummary(cfg, config_path);
  } catch (const std::exception &exception) {
    std::cerr << "配置检查失败: " << exception.what() << std::endl;
    return 1;
  }
  if (command == "--check-config") {
    if (argc == 1) printUsage(argv[0]);
    return 0;
  }
  if (!cfg.hardware_enable || !cfg.ready_pose_confirmed) {
    std::cerr << "拒绝连接机械臂：hardware_enable 与 ready_pose_confirmed "
                 "必须同时为 true。"
              << std::endl;
    return 3;
  }
  try {
    std::error_code ec;
    rokae::xMateRobot robot(cfg.robot_ip, cfg.local_ip);
    bool entered_automatic = false;
    bool entered_realtime = false;
    bool powered = false;
    auto cleanup = [&]() {
      if (entered_realtime) {
        ec.clear();
        robot.setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec);
      }
      if (powered && cfg.power_off_on_exit) {
        ec.clear();
        robot.setPowerState(false, ec);
      }
      if (entered_automatic) {
        ec.clear();
        robot.setOperateMode(rokae::OperateMode::manual, ec);
      }
    };
    try {
      ec.clear();
      const auto info = robot.robotInfo(ec);
      throwIfError(ec, "读取机器人信息");
      std::cout << "已连接机型: " << info.type << std::endl;
      ec.clear();
      robot.setOperateMode(rokae::OperateMode::automatic, ec);
      throwIfError(ec, "切换自动模式");
      entered_automatic = true;
      ec.clear();
      robot.setMotionControlMode(rokae::MotionControlMode::RtCommand, ec);
      throwIfError(ec, "切换实时命令模式");
      entered_realtime = true;
      ec.clear();
      robot.setPowerState(true, ec);
      throwIfError(ec, "机器人上电");
      powered = true;
      const bool success = runExperiment(robot, cfg);
      cleanup();
      return success ? 0 : 4;
    } catch (const rokae::RealtimeMotionException &exception) {
      std::cerr << "实时运动异常: " << exception.what() << std::endl;
      cleanup();
      return 4;
    } catch (...) {
      cleanup();
      throw;
    }
  } catch (const std::exception &exception) {
    std::cerr << "实验启动/运行失败: " << exception.what() << std::endl;
    return 1;
  }
}
#endif
