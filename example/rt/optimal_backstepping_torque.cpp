/**
 * @file optimal_backstepping_torque.cpp
 * @brief xCore SDK 6 轴协作机械臂的最优反步/SARL 实物实验骨架。
 *
 * 本文件有意保持为“一个 cpp + 一个 cfg”：控制律、配置解析、实时循环、
 * 安全保护和 CSV 日志都集中在这里，便于实验阶段对照论文与调参。
 *
 * 论文第二步需要 B^{-1}(q)=M(q)。本版已将厂家 xMateCR7.urdf 的关节轴、
 * 质量、质心和惯性参数固化为无动态分配的 6x6 质量矩阵计算器；部分关节
 * 控制器固定只控制关节 4、5、6，并提取对应 M_AA。厂家惯性数据存在物理一致性疑点，
 * 因而它是待厂家/辨识验证的候选模型，真机启用另有显式确认锁。
 *
 * ======================== 与原参考 CPP 的关系 ========================
 * 原文件 negotiable_ppc_torque.cpp 实现的是：
 *   Y0*theta_hat + Slotine-Li 滤波误差反馈 + 可协商 PPC + 模拟人力矩。
 * 本文件没有在原力矩后面简单叠加一个 RL 补偿，而是按论文的 HJB 反步顺序
 * 重新组织整个控制器，主要改动是：
 *   1. 删除缺失的 slotine_li_y0.hpp、Pinocchio 和 Y0 参数辨识依赖，并用
 *      固定维数代码实现厂家 URDF 的 M(q)，不在实时回调中解析 XML；
 *   2. 用论文的输出状态障碍变换 sx、sd 和 z1=sx-sd 替换原 PPC 误差漏斗；
 *   3. 用 HJB 引导的虚拟控制 alpha 和实际力矩 tau 替换原 PPC 力矩公式；
 *   4. 加入 core-function 鲁棒参数 theta_hat 及其 sigma 修正自适应律；
 *   5. 加入两层 actor-critic、Bellman 残差和联合 pseudo-Huber 学习律；
 *   6. 将受控对象固定为关节 4、5、6；网络也缩减为三输出、14/16 维输入；
 *   7. 保留并加强原参考程序的 MoveJ、1 ms 状态读取、力矩限幅和异常停机。
 *
 * ======================== 与论文公式的对应 ===========================
 * 为方便与 Safe_RL_pseudo_Huber_regularization.tex 对照，后文注释直接使用
 * 论文 LaTeX 标签，例如 eq:alpha_output、eq:tau_output。代码变量的对应关系：
 *   hx       <-> H_x 的对角元；       hbeta <-> h_beta；
 *   actor1   <-> hat h_a1；           actor2 <-> hat h_a2；
 *   hc1/hc2  <-> hat h_c1/hat h_c2；  theta_hat_ <-> hat vartheta；
 *   e1/e2    <-> Bellman 残差向量；   ep1/ep2 <-> actor-critic 一致性误差；
 *   lambda1/2<-> Lambda_c1/Lambda_c2；第二层为 M_AA 引起的完整活动块。
 *
 * 工程约定：数组始终按关节 1~6 排列，C++ 下标为 0~5；角度 rad，速度
 * rad/s，加速度 rad/s^2，力矩 Nm。即使只控制一个关节，kDof 和 Torque
 * 的长度仍必须保持 6。
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "rokae/robot.h"

namespace {

// xMateRobot 和 SDK 命令仍为 6 轴；控制算法固定只作用于物理关节 4、5、6。
constexpr std::size_t kDof = 6;
// 当前实验只允许物理关节 5 进入控制律；其余关节命令恒为 0 Nm。
constexpr std::size_t kControlledDof = 1;
constexpr std::array<std::size_t, kControlledDof> kControlledJoints{4};

// 对固定三轴降阶系统恢复完整网络输入：
// Z1_A=[z1_A,q_A,qd_A,qd_dot_A,beta,beta_dot]，维数 14；
// Z2_A=[z2_A,q_A,dq_A,alpha_A,alpha_dot_A,phi_A]，维数 16。
// 不采用指数规模笛卡尔网格，而是每层固定 64 个确定性高维稀疏中心。
constexpr std::size_t kZ1Dim = 4 * kControlledDof + 2;
constexpr std::size_t kZ2Dim = 5 * kControlledDof + 1;
constexpr std::size_t kRbfNodes = 64;
// 仿真每层使用 25 个节点。实物保留其前 25 个 Halton 中心，并继续使用
// 同一序列扩展到 64 个；初始化按 25/N 缩放，使节点增多不放大初始 Actor。
constexpr double kSimulationRbfNodes = 25.0;

// xCore 实时回调按 1 ms 规划。所有 theta/NN 权重的 Euler 更新都乘 kDt。
// 原 Python 验证程序 DT=0.004，不能把“每步增量”原样复制到这里。
constexpr double kDt = 0.001;
constexpr double kPi = 3.1415926535897932384626433832795;

// 仅用于避免除零和根号边界，不是论文中的设计参数 epsilon_g。
constexpr double kEps = 1.0e-9;

using Vec6 = std::array<double, kDof>;// 6 轴向量
using Mat6 = std::array<Vec6, kDof>;// 6x6 矩阵
using Vec3 = std::array<double, 3>;// 3 轴向量
using Mat3 = std::array<Vec3, 3>;// 3x3 矩阵
using Feature = std::array<double, kRbfNodes>;// 64 维稀疏 RBF 特征向量
using WeightMatrix = std::array<Feature, kControlledDof>;// 3x64 权重矩阵，按 [输出关节][RBF 节点] 排布
using Z1Input = std::array<double, kZ1Dim>;// 14 维第一层输入各使用一个素数基；确定性生成保证实验可复现。
using Z2Input = std::array<double, kZ2Dim>;// 16 维第二层输入各使用一个素数基；确定性生成保证实验可复现。

// 16 维第二层输入各使用一个素数基；确定性生成保证实验可复现。
constexpr std::array<unsigned int, kZ2Dim> kHaltonBases{
    2, 3, 5, 7, 11, 13};//生成 RBF 神经网络的中心位置

/**
 * @brief CPP 中所有可调参数的内存表示，启动时由同名 CFG 键覆盖。
 *
 * 增加一个参数必须同步完成四件事：
 *   Config 增加成员 -> applyConfig() 增加键 -> validateConfig() 检查
 *   -> 控制律实际使用。只在 CFG 中写一行不会自动进入控制器。
 *
 * 相对原 ControlConfig 的主要变化：删除 Y0、模拟人力矩和协商 PPC 参数；
 * 新增输出状态障碍、HJB 两层代价、鲁棒自适应、actor-critic、
 * pseudo-Huber、固定三轴模型和双重硬件解锁参数。
 */
struct Config {
  // -------------------------- 真机双重解锁 ---------------------------
  // 这三项不是论文控制律。hardware_enable 防止普通运行误连真机；
  // ready_pose_confirmed 要求操作者另外确认 MoveJ 位姿。程序还要求命令行
  // 显式给出 --run，因此默认 cfg 只允许离线参数检查。
  bool hardware_enable = false;
  bool ready_pose_confirmed = false;
  // 直接上机排查版新增：false 时不执行 MoveJ，直接把启动时的当前关节角
  // 作为 q_init、静止参考和约束中心。只有 ready_q 已经示教确认后才设 true。
  bool perform_ready_move = false;
  bool power_off_on_exit = true;
  std::string robot_ip = "192.168.2.160";
  std::string local_ip = "192.168.2.100";

  // ----------------------- 实时周期与轨迹接口 -------------------------
  // total_time/start_delay 对应实验时序；trajectory_ramp_time 使 qd、dqd、
  // ddqd 平滑进入参考轨迹。command_ramp_time 另对最终力矩做渐入，两者不要混淆。
  // 原参考 cfg 的 start_delay=0、ramp_time=0 较激进，本文件改为非零默认值。
  double total_time = 20.0;
  double start_delay = 1.0;
  double trajectory_ramp_time = 2.0;
  double command_ramp_time = 2.0;
  double ready_move_speed = 0.10;
  double sdk_filter_cutoff_hz = 30.0;

  // ready_q 是 MoveJ 目标；实际参考轨迹围绕 MoveJ 后重新测得的 q_init，
  // 而不是盲目假设机械臂准确到达 ready_q。
  Vec6 ready_q{};
  Vec6 traj_offset{};
  Vec6 traj_amp{0, 0, 0, 0.02, 0.02, 0.015};
  Vec6 traj_omega{0, 0, 0, 0.35, 0.35, 0.35};
  Vec6 traj_phase{0, 0, 0, 0, kPi / 2.0, kPi};

  // true：使用与仿真同形、但幅值和变化速度都更保守的梯形包络参考；
  // false：回退到上面的单一正弦参考。四个时刻均相对力矩模式开始。
  bool use_slow_trapezoid_reference = true;
  double slow_ref_rise_start = 4.0;
  double slow_ref_rise_end = 8.0;
  double slow_ref_fall_start = 11.0;
  double slow_ref_fall_end = 15.0;
  // 位移均相对 q_init：low=base+low_amp*cos，high=base+lift+
  // high_ripple_amp*cos，再由 C1 分段梯形包络在 low/high 之间混合。
  Vec6 slow_ref_base{0, 0, 0, 0.005, -0.004, 0.003};
  Vec6 slow_ref_low_amp{0, 0, 0, 0.008, 0.007, 0.006};
  Vec6 slow_ref_lift{0, 0, 0, 0.035, -0.030, 0.025};
  Vec6 slow_ref_high_ripple_amp{0, 0, 0, 0.006, -0.005, 0.004};
  double slow_ref_low_period = 12.0;
  double slow_ref_high_period = 26.0;

  // ------------------- 论文：不规则输出约束变换 ----------------------
  // 对应论文 eq:eta_zeta_x ~ eq:z1_output_dynamics。
  // enable_output_barrier=false 时退化为 z1=q-qd、Hx=I、hbeta=0，便于先
  // 验证无障碍的反步基线；它不是原协商 PPC 的 rho0/rho_inf 漏斗。
  bool enable_output_barrier = true;

  // 开启后可选论文 F-C 或原工程 C-F-C；关闭时 kappa=0，固定有限约束。
  bool enable_constraint_schedule = false;
  // true：论文 F-C；false：原工程 C-F-C。总开关关闭时仍为固定有限约束。
  bool constraint_schedule_start_free = true;
  double constraint_to_free_start = 3.0;
  double constraint_to_free_end = 4.0;
  double free_to_constraint_start = 15.0;
  double free_to_constraint_end = 16.0;

  // 论文写绝对输出 x1。实物不同准备姿态可能远离零位，因此这里把
  // x1 工程化为 q-(q_init+offset)，使安全区围绕本次实验起点。
  Vec6 constraint_center_offset{};

  // 有限约束阶段要求 |q-center| < output_bound。由它反算论文 lambda_i；
  // barrier_abort_ratio 是提前停机阈值，不属于理论障碍函数。
  Vec6 output_bound{0.20, 0.20, 0.20, 0.15, 0.15, 0.12};
  double barrier_b0 = 2.0;
  double barrier_bf = 1.0;
  double barrier_abort_ratio = 0.90;

  // ------------------- 论文：两步 HJB 反步控制 -----------------------
  // k1/k2 对应论文 K1/K2 的对角元；varrho1/2 严格保留为论文的正标量，
  // 实际控制律使用 Pi_i=K_i+varrho_i*I。
  Vec6 k1{0, 0, 0, 2, 2, 2};
  Vec6 k2{0, 0, 0, 3, 3, 3};
  double varrho1 = 1.0;
  double varrho2 = 1.0;

  // Q1/R1 与 Q2/R2 是两个 HJB 积分性能指标的对角权重，进入 e1/e2
  // Bellman 残差；它们不是普通反馈增益。
  Vec6 q1_cost{0, 0, 0, 4, 4, 4};
  Vec6 q2_cost{0, 0, 0, 0.5, 0.5, 0.5};
  Vec6 r1_cost{0, 0, 0, 0.1, 0.1, 0.1};
  Vec6 r2_cost{0, 0, 0, 0.02, 0.02, 0.02};

  // 下面三组是实物数值保护，不改变论文公式定义，但实际发生饱和时闭环
  // 已不再严格等于未饱和理论系统，分析实验数据时必须同时检查饱和占比。
  Vec6 alpha_limit{0, 0, 0, 0.25, 0.25, 0.25};
  Vec6 alpha_dot_limit{0, 0, 0, 1.2, 1.2, 1.2};
  Vec6 virtual_accel_limit{0, 0, 0, 1.2, 1.2, 1.2};

  // 论文假定 alpha 可微且 alpha_dot 可得；真机实现用有限差分加一阶脏微分。
  // alpha_dot_filter 越小越平滑但相位滞后越大，属于工程近似。
  double alpha_dot_filter = 0.05;

  // ---------------- 论文：core-function 鲁棒自适应 ------------------
  // 对应 eq:robust_output 与 eq:vartheta_update_output。
  // theta_initial/theta_max 是 hat vartheta 的初值/数值保护；iota 控制
  // nu_r = hat_vartheta*phi^2*z2/(2*iota^2) 的强度。
  double iota = 1.0;
  double theta_initial = 0.05;
  bool enable_theta_adaptation = true;
  double gamma_theta = 0.02;
  double sigma_theta = 0.10;
  double theta_max = 5.0;

  // ---------------- 论文：actor-critic 与 pseudo-Huber ---------------
  // enable_rl=false：actor/critic 输出均为 0，得到共享同一障碍和鲁棒项的
  // No-RL 基线；enable_learning 单独控制权重是否在线更新。
  // 原参考 PPC 没有这两层网络，这整组均为本论文控制器新增。
  bool enable_rl = false;
  bool enable_learning = false;
  double learning_start_time = 5.0;

  // gamma_ci/gamma_ai 对应 eq:dc0_output/eq:da0_output 的输出学习方向增益。
  // 每轴独立设置，顺序始终为关节 1~6。
  Vec6 gamma_c1{0, 0, 0, 0.15, 0.15, 0.15};
  Vec6 gamma_a1{0, 0, 0, 0.18, 0.18, 0.18};
  Vec6 gamma_c2{0, 0, 0, 0.12, 0.12, 0.12};
  Vec6 gamma_a2{0, 0, 0, 0.15, 0.15, 0.15};

  // weight_gain_* 是权重空间中的 Gamma_ci/Gamma_ai 标量实现；leak_*
  // 对应论文 ell_ci/ell_ai。输出方向增益与权重增益分开，便于控制尺度。
  double weight_gain_c1 = 0.20;
  double weight_gain_a1 = 0.20;
  double weight_gain_c2 = 0.15;
  double weight_gain_a2 = 0.15;
  double leak_c1 = 0.03;
  double leak_a1 = 0.03;
  double leak_c2 = 0.03;
  double leak_a2 = 0.03;

  // delta_norm 对应 Sbar 分母的 delta_i；delta_ci 对应 Omega_ci 正则项。
  // pseudo_huber_eps1/2 是论文 epsilon_gi，作用于
  // r_i=col{Bellman_i,e_Pi}；越大越早衰减异常残差引起的学习更新。
  double delta_norm = 0.08;
  double delta_ci = 0.08;
  double pseudo_huber_eps1 = 0.04;
  double pseudo_huber_eps2 = 0.08;

  // 对应论文 Upsilon_ai=(1/varrho_i+bar_upsilon_ai)I。CFG 只填写严格为正的
  // bar_upsilon_ai，updateLayer 内部再计算有效耗散系数。
  double bar_upsilon_a1 = 0.70;
  double bar_upsilon_a2 = 0.75;
  // actor_limit 和初始斜率是真机输出保护/初始化策略，不是论文理想网络假设。
  Vec6 actor_limit1{0, 0, 0, 0.10, 0.10, 0.10};
  Vec6 actor_limit2{0, 0, 0, 0.30, 0.30, 0.30};
  // false 与仿真/论文一致，Actor 直接使用限幅后的 W^T S；true 才启用
  // 真机可选的 z^T h_a>=0 半空间安全投影。
  bool enable_actor_direction_projection = false;
  Vec6 initial_actor_gain1{0, 0, 0, 0.005, 0.005, 0.005};
  Vec6 initial_actor_gain2{0, 0, 0, 0.005, 0.005, 0.005};
  // 与仿真相同，Actor 初值由“小幅可复现噪声 + 中心斜率”构成。
  // seed 以整数形式写入标量 CFG；自定义 SplitMix64/Box-Muller 避免平台差异。
  double initial_weight_noise = 0.001;
  double initial_weight_seed = 21.0;

  // 与仿真统一的无量纲网络几何：Halton 中心 [-1.5,1.5]、截断 [-4,4]。
  // CPP 分母为 rbf_width^2*dim，因此 1.02 在 14/16 维上的有效宽度约
  // 3.82/4.08，接近仿真的 3.2/3.4，并补偿实物多一个受控关节。
  double rbf_center_min = -1.5;
  double rbf_center_max = 1.5;
  double rbf_width = 1.02;
  Vec6 rbf_z1_scale{0, 0, 0, 0.3, 0.3, 0.3};
  double rbf_beta_dot_scale = 0.5;
  double rbf_phi_scale = 1.5;
  double rbf_standardized_clip = 4.0;

  // ---------------------- 输入映射与真机保护 --------------------------
  // true：使用厂家 xMateCR7.urdf 中的六轴质量/质心/惯量在线计算完整 M(q)；
  // false：回退到下面的常对角 nominal_inertia_diag。
  bool use_xmate_cr7_urdf_mass_matrix = true;
  // 固定控制关节 4～6 时，M_AA 主子矩阵成立的前提是关节 1～3 由独立
  // 位置伺服、制动器或机械约束可靠保持 qdot_f=qddot_f=0。本开关是
  // 实验条件声明，软件不能仅凭它真的锁住关节。
  bool inactive_joints_fixed = true;
  // URDF 自身标注“inertia in the model is not correct”，六组惯性主值
  // 也均违反刚体惯量三角不等式。该标志现在只记录操作者是否已经确认风险：
  // false 会在真机启动前打印醒目警告，但不再阻止程序运行；true 表示已确认。
  // 无论取值如何，质量矩阵的有限性/正定性检查和运行时安全停机仍然有效。
  bool acknowledge_unverified_urdf_inertia = false;

  // 仅在关闭 URDF 模型时使用的安全回退。torque_bias 是经验证补偿的预留，
  // 默认 0；它不属于 M(q)，也不能凭感觉当作重力项填写。
  Vec6 nominal_inertia_diag{0, 0, 0, 1, 1, 1};
  Vec6 torque_bias{};

  // 以下均为理论控制律外的工程保护：最终力矩幅值、力矩变化率、跟踪误差、
  // 关节速度阈值。安全检查仍覆盖关节 1～3，以发现 0 Nm 轴意外漂移。
  Vec6 tau_limit{0, 0, 0, 1.5, 1.5, 1};
  Vec6 tau_rate_limit{0, 0, 0, 35, 35, 25};
  Vec6 abort_error{0.15, 0.15, 0.15, 0.12, 0.12, 0.10};
  Vec6 abort_velocity{0.5, 0.5, 0.5, 0.6, 0.6, 0.6};
  // 固定轴 1~3 的参考被强制为静止；若其位置偏离 q_init 超过该值，
  // 说明“固定关节”假设已经失效，必须立即中止。单位 rad。
  Vec6 inactive_joint_drift_limit{0.01, 0.01, 0.01, 0, 0, 0};

  // 1 ms 回调每 log_decimation 周期写一行；10 即约 100 Hz 日志。
  double log_decimation = 10.0;
};

double clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

bool isFinite(double value) { return std::isfinite(value); }

// 算法固定只控制物理关节 4、5、6；保留 Config 形参以减少调用点噪声。
bool isActive(const Config &, std::size_t joint) {
  return joint == kControlledJoints.front();
}

// -------------------- xMateCR7 固定尺寸 URDF 质量矩阵 --------------------
// 下列常量逐项抄自厂家 xMateCR7_description/urdf/xMateCR7.urdf：六个
// revolute joint 的 origin/axis，以及 link1~link6 的 mass、COM、inertia。
// 不在 1 kHz 回调中解析 XML，也不分配动态内存；计算结果与 CRBA/Jacobian
// 刚体公式等价。源 URDF 自身标注“inertia in the model is not correct”；
// 这里仅按文件数值计算，不会修复它的物理异常。

constexpr Mat3 symmetricInertia(double ixx, double ixy, double ixz,
                                double iyy, double iyz, double izz) {
  return Mat3{{Vec3{ixx, ixy, ixz}, Vec3{ixy, iyy, iyz},
               Vec3{ixz, iyz, izz}}};
}

constexpr std::array<Vec3, kDof> kXmateJointTranslation{
    Vec3{0.0, 0.0, 0.0},    Vec3{0.0, 0.0, 0.296},
    Vec3{0.0, 0.0, 0.49},   Vec3{0.0, 0.0, 0.36},
    Vec3{0.0, -0.151, 0.0}, Vec3{0.0, 0.0, 0.1265}};

constexpr std::array<Vec3, kDof> kXmateJointAxis{
    Vec3{0.0, 0.0, 1.0},  Vec3{0.0, 1.0, 0.0},
    Vec3{0.0, -1.0, 0.0}, Vec3{0.0, 0.0, 1.0},
    Vec3{0.0, -1.0, 0.0}, Vec3{0.0, 0.0, 1.0}};

constexpr Vec6 kXmateLinkMass{3.44074938, 5.024380503, 2.439584706,
                              2.439584706, 2.422757896, 1.229800857};

constexpr std::array<Vec3, kDof> kXmateLinkCom{
    Vec3{-0.000072769597, -0.01679187012, 0.2986854502},
    Vec3{0.0, 0.003491209523, 0.2410487354},
    Vec3{0.0, 0.023, 0.097}, Vec3{0.0, -0.020, -0.046},
    Vec3{0.0, 0.012, 0.096}, Vec3{0.0, 0.017, -0.043}};

constexpr std::array<Mat3, kDof> kXmateLinkInertia{
    symmetricInertia(0.01764837, 0.0, 0.0, 0.016617391, -0.002218013,
                     1.120163944),
    symmetricInertia(0.136759927, 0.0, 0.0, 1.747453457, -0.024189444,
                     0.012535321),
    symmetricInertia(0.026159338, 0.0, 0.0, 0.024282179, -0.00615637,
                     1.11989835),
    symmetricInertia(0.012, 0.0, 0.0, 0.221, -0.002, 0.004),
    symmetricInertia(0.016, 0.0, 0.0, 0.015, -0.002, 0.213),
    symmetricInertia(0.002, 0.0, 0.0, 0.212, 0.0, 0.002)};

Vec3 add3(const Vec3 &a, const Vec3 &b) {
  return Vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 subtract3(const Vec3 &a, const Vec3 &b) {
  return Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

double dot3(const Vec3 &a, const Vec3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross3(const Vec3 &a, const Vec3 &b) {
  return Vec3{a[1] * b[2] - a[2] * b[1],
              a[2] * b[0] - a[0] * b[2],
              a[0] * b[1] - a[1] * b[0]};
}

Mat3 identity3() {
  return Mat3{{Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0},
               Vec3{0.0, 0.0, 1.0}}};
}

Vec3 mat3Vector(const Mat3 &matrix, const Vec3 &vector) {
  Vec3 result{};
  for (std::size_t row = 0; row < 3; ++row) {
    result[row] = dot3(matrix[row], vector);
  }
  return result;
}

Mat3 mat3Multiply(const Mat3 &a, const Mat3 &b) {
  Mat3 result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        result[row][column] += a[row][inner] * b[inner][column];
      }
    }
  }
  return result;
}

Mat3 transpose3(const Mat3 &matrix) {
  Mat3 result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      result[row][column] = matrix[column][row];
    }
  }
  return result;
}

Mat3 axisAngle3(const Vec3 &axis, double angle) {
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  const double one_minus_cosine = 1.0 - cosine;
  const double x = axis[0];
  const double y = axis[1];
  const double z = axis[2];
  return Mat3{{
      Vec3{x * x * one_minus_cosine + cosine,
           x * y * one_minus_cosine - z * sine,
           x * z * one_minus_cosine + y * sine},
      Vec3{y * x * one_minus_cosine + z * sine,
           y * y * one_minus_cosine + cosine,
           y * z * one_minus_cosine - x * sine},
      Vec3{z * x * one_minus_cosine - y * sine,
           z * y * one_minus_cosine + x * sine,
           z * z * one_minus_cosine + cosine}}};
}

struct Transform3 {// 3D 刚体变换，旋转矩阵 + 平移向量
  Mat3 rotation{identity3()};
  Vec3 translation{};
};

Transform3 composeTransform(const Transform3 &parent,
                            const Transform3 &local) {// 计算父子坐标系变换
  return Transform3{mat3Multiply(parent.rotation, local.rotation),
                    add3(parent.translation,
                         mat3Vector(parent.rotation, local.translation))};
}

Mat6 xmateCr7UrdfMassMatrix(const Vec6 &q) {//计算机械臂的惯性矩阵
  std::array<Transform3, kDof> link_transform{};
  std::array<Vec3, kDof> joint_position{};
  std::array<Vec3, kDof> joint_axis_world{};
  Transform3 parent;

  for (std::size_t joint = 0; joint < kDof; ++joint) {
    const Transform3 origin{identity3(), kXmateJointTranslation[joint]};
    const Transform3 joint_frame = composeTransform(parent, origin);
    joint_position[joint] = joint_frame.translation;
    joint_axis_world[joint] =
        mat3Vector(joint_frame.rotation, kXmateJointAxis[joint]);
    const Transform3 rotation{axisAngle3(kXmateJointAxis[joint], q[joint]),
                              Vec3{}};
    link_transform[joint] = composeTransform(joint_frame, rotation);
    parent = link_transform[joint];
  }

  Mat6 mass_matrix{};
  for (std::size_t link = 0; link < kDof; ++link) {
    const Vec3 com_world =
        add3(link_transform[link].translation,
             mat3Vector(link_transform[link].rotation, kXmateLinkCom[link]));
    const Mat3 inertia_world =
        mat3Multiply(mat3Multiply(link_transform[link].rotation,
                                  kXmateLinkInertia[link]),
                     transpose3(link_transform[link].rotation));
    std::array<Vec3, kDof> linear_jacobian{};
    std::array<Vec3, kDof> angular_jacobian{};
    for (std::size_t joint = 0; joint <= link; ++joint) {
      angular_jacobian[joint] = joint_axis_world[joint];
      linear_jacobian[joint] =
          cross3(joint_axis_world[joint],
                 subtract3(com_world, joint_position[joint]));
    }
    for (std::size_t row = 0; row <= link; ++row) {
      for (std::size_t column = 0; column <= link; ++column) {
        mass_matrix[row][column] +=
            kXmateLinkMass[link] *
                dot3(linear_jacobian[row], linear_jacobian[column]) +
            dot3(angular_jacobian[row],
                 mat3Vector(inertia_world, angular_jacobian[column]));
      }
    }
  }

  // 理论结果应对称；平均上下三角消除浮点累加顺序产生的末位误差。
  for (std::size_t row = 0; row < kDof; ++row) {
    for (std::size_t column = row + 1; column < kDof; ++column) {
      const double symmetric =
          0.5 * (mass_matrix[row][column] + mass_matrix[column][row]);
      mass_matrix[row][column] = symmetric;
      mass_matrix[column][row] = symmetric;
    }
  }
  return mass_matrix;//计算返回惯性矩阵
}

Mat6 selectedMassMatrix(const Vec6 &q, const Config &cfg) {//计算选择的惯性矩阵
  Mat6 selected{};
  if (cfg.use_xmate_cr7_urdf_mass_matrix) {
    const Mat6 full = xmateCr7UrdfMassMatrix(q);
    for (std::size_t row = 0; row < kDof; ++row) {
      for (std::size_t column = 0; column < kDof; ++column) {
        if (isActive(cfg, row) && isActive(cfg, column)) {
          selected[row][column] = full[row][column];
        }
      }
    }
  } else {
    for (std::size_t joint = 0; joint < kDof; ++joint) {
      if (isActive(cfg, joint)) {
        selected[joint][joint] = cfg.nominal_inertia_diag[joint];
      }
    }
  }
  return selected;
}

bool activeCholesky(const Mat6 &matrix, const Config &cfg, Mat6 &lower) {//计算下三角矩阵（辅助后文矩阵运算）
  lower = Mat6{};
  for (std::size_t row = 0; row < kDof; ++row) {
    if (!isActive(cfg, row)) {
      continue;
    }
    for (std::size_t column = 0; column <= row; ++column) {
      if (!isActive(cfg, column)) {
        continue;
      }
      double value = matrix[row][column];
      if (!isFinite(value)) {
        return false;
      }
      for (std::size_t inner = 0; inner < column; ++inner) {
        if (isActive(cfg, inner)) {
          value -= lower[row][inner] * lower[column][inner];
        }
      }
      if (row == column) {
        if (!(value > 1.0e-10) || !isFinite(value)) {
          return false;
        }
        lower[row][column] = std::sqrt(value);
      } else {
        lower[row][column] = value / lower[column][column];
      }
    }
  }
  return true;
}

bool solveActiveSpd(const Mat6 &matrix, const Vec6 &rhs, const Config &cfg,//求解线性方程组（用于 Actor–Critic 更新中的正则化学习矩阵求解）
                    Vec6 &solution) {
  Mat6 lower{};
  if (!activeCholesky(matrix, cfg, lower)) {
    return false;
  }
  Vec6 intermediate{};
  for (std::size_t row = 0; row < kDof; ++row) {
    if (!isActive(cfg, row)) {
      continue;
    }
    double value = rhs[row];
    for (std::size_t column = 0; column < row; ++column) {
      if (isActive(cfg, column)) {
        value -= lower[row][column] * intermediate[column];
      }
    }
    intermediate[row] = value / lower[row][row];
  }
  solution = Vec6{};
  for (std::size_t reverse = kDof; reverse-- > 0;) {
    if (!isActive(cfg, reverse)) {
      continue;
    }
    double value = intermediate[reverse];
    for (std::size_t column = reverse + 1; column < kDof; ++column) {
      if (isActive(cfg, column)) {
        value -= lower[column][reverse] * solution[column];
      }
    }
    solution[reverse] = value / lower[reverse][reverse];
  }
  return true;
}

Vec6 mat6Vector(const Mat6 &matrix, const Vec6 &vector) {//矩阵与向量相乘运算
  Vec6 result{};
  for (std::size_t row = 0; row < kDof; ++row) {
    for (std::size_t column = 0; column < kDof; ++column) {
      result[row] += matrix[row][column] * vector[column];
    }
  }
  return result;
}

Mat6 secondLayerGInverse(const Mat6 &mass_matrix, const Config &cfg) {//计算第二层控制律的逆矩阵
  // B=M^{-1}，G2=B R2^{-1} B^T，因此 G2^{-1}=M^T R2 M。
  // 刚体质量矩阵对称，但仍按 M^T R2 M 的一般形式逐项计算。
  Mat6 result{};
  for (std::size_t row = 0; row < kDof; ++row) {
    if (!isActive(cfg, row)) {
      continue;
    }
    for (std::size_t column = 0; column < kDof; ++column) {
      if (!isActive(cfg, column)) {
        continue;
      }
      for (std::size_t inner = 0; inner < kDof; ++inner) {
        if (isActive(cfg, inner)) {
          result[row][column] += mass_matrix[inner][row] *
                                 cfg.r2_cost[inner] *
                                 mass_matrix[inner][column];
        }
      }
    }
  }
  return result;
}

std::string trimCopy(const std::string &text) {//去除字符串首尾空白
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

std::string normalizeKey(std::string key) {//将配置文件的键名标准化为小写并替换连字符为下划线
  key = trimCopy(key);
  for (char &c : key) {
    if (c == '-') {
      c = '_';
    } else {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
  }
  return key;
}

std::vector<double> parseNumbers(std::string text, int line_no,//解析数字列表
                                 const std::string &key) {
  // CFG 数组既允许“1 2 3”，也允许“[1,2,3]”或用分号分隔；最终统一按
  // 空白读取。这里只处理文本格式，不对参数物理意义做判断。
  for (char &c : text) {
    if (c == ',' || c == '[' || c == ']' || c == ';') {
      c = ' ';
    }
  }
  std::istringstream input(text);
  std::vector<double> values;
  double value = 0.0;
  while (input >> value) {
    values.push_back(value);
  }
  if (!input.eof()) {
    throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                             " 行参数 " + key + " 含有非数字内容");
  }
  return values;
}

double parseScalar(const std::string &text, int line_no,
                   const std::string &key) {
  const auto values = parseNumbers(text, line_no, key);
  if (values.size() != 1) {
    throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                             " 行参数 " + key + " 必须有 1 个数字");
  }
  return values.front();
}

Vec6 parseVec6(const std::string &text, int line_no, const std::string &key) {
  // 本项目的 SDK 与控制器状态均固定为 6 轴，因此所有逐关节参数必须写满
  // 6 个值。本版本固定控制关节 4、5、6；仅用于控制律的数组在 1～3 轴填 0，
  // 但安全阈值等监控全部关节的数组仍应填写有效数值。
  const auto values = parseNumbers(text, line_no, key);
  if (values.size() != kDof) {
    throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                             " 行参数 " + key + " 必须有 6 个数字");
  }
  Vec6 result{};
  std::copy(values.begin(), values.end(), result.begin());
  return result;
}

bool parseBool(std::string text, int line_no, const std::string &key) {
  text = normalizeKey(text);
  if (text == "true" || text == "yes" || text == "on" || text == "1") {
    return true;
  }
  if (text == "false" || text == "no" || text == "off" || text == "0") {
    return false;
  }
  throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                           " 行参数 " + key + " 必须是 true/false");
}

void applyConfig(Config &cfg, const std::string &raw_key,//将配置文件的键值对应用到 Config 对象中（对应处理）
                 const std::string &value, int line_no) {
  // CFG 键与 Config 成员的一一映射入口。与原参考 cfg 的宽松读取方式相比，
  // 本实现遇到未知键会在本函数末尾直接报错，避免参数拼写错误被悄悄忽略。
  // 若以后新增可调参数，应同时完成三处：Config 成员、本函数 SET_*、
  // validateConfig() 的范围检查；否则新参数不会真正进入控制器。
  const std::string key = normalizeKey(raw_key);
#define SET_BOOL(name)                                                          \
  if (key == #name) {                                                           \
    cfg.name = parseBool(value, line_no, key);                                  \
    return;                                                                     \
  }
#define SET_SCALAR(name)                                                        \
  if (key == #name) {                                                           \
    cfg.name = parseScalar(value, line_no, key);                                \
    return;                                                                     \
  }
#define SET_VEC(name)                                                           \
  if (key == #name) {                                                           \
    cfg.name = parseVec6(value, line_no, key);                                  \
    return;                                                                     \
  }

  SET_BOOL(hardware_enable)
  SET_BOOL(ready_pose_confirmed)
  SET_BOOL(perform_ready_move)
  SET_BOOL(power_off_on_exit)
  if (key == "robot_ip") {
    cfg.robot_ip = trimCopy(value);
    return;
  }
  if (key == "local_ip") {
    cfg.local_ip = trimCopy(value);
    return;
  }
  SET_SCALAR(total_time)
  SET_SCALAR(start_delay)
  SET_SCALAR(trajectory_ramp_time)
  SET_SCALAR(command_ramp_time)
  SET_SCALAR(ready_move_speed)
  SET_SCALAR(sdk_filter_cutoff_hz)
  SET_VEC(ready_q)
  SET_VEC(traj_offset)
  SET_VEC(traj_amp)
  SET_VEC(traj_omega)
  SET_VEC(traj_phase)
  SET_BOOL(use_slow_trapezoid_reference)
  SET_SCALAR(slow_ref_rise_start)
  SET_SCALAR(slow_ref_rise_end)
  SET_SCALAR(slow_ref_fall_start)
  SET_SCALAR(slow_ref_fall_end)
  SET_VEC(slow_ref_base)
  SET_VEC(slow_ref_low_amp)
  SET_VEC(slow_ref_lift)
  SET_VEC(slow_ref_high_ripple_amp)
  SET_SCALAR(slow_ref_low_period)
  SET_SCALAR(slow_ref_high_period)

  SET_BOOL(enable_output_barrier)
  SET_BOOL(enable_constraint_schedule)
  SET_BOOL(constraint_schedule_start_free)
  SET_SCALAR(constraint_to_free_start)
  SET_SCALAR(constraint_to_free_end)
  SET_SCALAR(free_to_constraint_start)
  SET_SCALAR(free_to_constraint_end)
  SET_VEC(constraint_center_offset)
  SET_VEC(output_bound)
  SET_SCALAR(barrier_b0)
  SET_SCALAR(barrier_bf)
  SET_SCALAR(barrier_abort_ratio)

  SET_VEC(k1)
  SET_VEC(k2)
  SET_SCALAR(varrho1)
  SET_SCALAR(varrho2)
  SET_VEC(q1_cost)
  SET_VEC(q2_cost)
  SET_VEC(r1_cost)
  SET_VEC(r2_cost)
  SET_VEC(alpha_limit)
  SET_VEC(alpha_dot_limit)
  SET_VEC(virtual_accel_limit)
  SET_SCALAR(alpha_dot_filter)

  SET_SCALAR(iota)
  SET_SCALAR(theta_initial)
  SET_BOOL(enable_theta_adaptation)
  SET_SCALAR(gamma_theta)
  SET_SCALAR(sigma_theta)
  SET_SCALAR(theta_max)

  SET_BOOL(enable_rl)
  SET_BOOL(enable_learning)
  SET_SCALAR(learning_start_time)
  SET_VEC(gamma_c1)
  SET_VEC(gamma_a1)
  SET_VEC(gamma_c2)
  SET_VEC(gamma_a2)
  SET_SCALAR(weight_gain_c1)
  SET_SCALAR(weight_gain_a1)
  SET_SCALAR(weight_gain_c2)
  SET_SCALAR(weight_gain_a2)
  SET_SCALAR(leak_c1)
  SET_SCALAR(leak_a1)
  SET_SCALAR(leak_c2)
  SET_SCALAR(leak_a2)
  SET_SCALAR(delta_norm)
  SET_SCALAR(delta_ci)
  SET_SCALAR(pseudo_huber_eps1)
  SET_SCALAR(pseudo_huber_eps2)
  SET_SCALAR(bar_upsilon_a1)
  SET_SCALAR(bar_upsilon_a2)
  SET_VEC(actor_limit1)
  SET_VEC(actor_limit2)
  SET_BOOL(enable_actor_direction_projection)
  SET_VEC(initial_actor_gain1)
  SET_VEC(initial_actor_gain2)
  SET_SCALAR(initial_weight_noise)
  SET_SCALAR(initial_weight_seed)
  SET_SCALAR(rbf_center_min)
  SET_SCALAR(rbf_center_max)
  SET_SCALAR(rbf_width)
  SET_VEC(rbf_z1_scale)
  SET_SCALAR(rbf_beta_dot_scale)
  SET_SCALAR(rbf_phi_scale)
  SET_SCALAR(rbf_standardized_clip)

  SET_BOOL(use_xmate_cr7_urdf_mass_matrix)
  SET_BOOL(inactive_joints_fixed)
  SET_BOOL(acknowledge_unverified_urdf_inertia)
  SET_VEC(nominal_inertia_diag)
  SET_VEC(torque_bias)
  SET_VEC(tau_limit)
  SET_VEC(tau_rate_limit)
  SET_VEC(abort_error)
  SET_VEC(abort_velocity)
  SET_VEC(inactive_joint_drift_limit)
  SET_SCALAR(log_decimation)

#undef SET_BOOL
#undef SET_SCALAR
#undef SET_VEC
  throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                           " 行存在未知参数: " + raw_key);
}

void requireFinite(double value, const std::string &name) {//检查数值是否为有限数字
  if (!isFinite(value)) {
    throw std::runtime_error("配置项 " + name + " 必须是有限数字");
  }
}

void requirePositive(double value, const std::string &name) {//检查数值是否为正数
  requireFinite(value, name);
  if (value <= 0.0) {
    throw std::runtime_error("配置项 " + name + " 必须大于 0");
  }
}

void requireNonNegative(double value, const std::string &name) {//检查数值是否为非负数
  requireFinite(value, name);
  if (value < 0.0) {
    throw std::runtime_error("配置项 " + name + " 不能小于 0");
  }
}

void validateVec(const Vec6 &values, const std::string &name,
                 bool strictly_positive, bool non_negative) {//验证向量中的每个元素是否满足条件
  for (std::size_t i = 0; i < kDof; ++i) {
    const std::string item = name + "[" + std::to_string(i + 1) + "]";
    if (strictly_positive) {
      requirePositive(values[i], item);
    } else if (non_negative) {
      requireNonNegative(values[i], item);
    } else {
      requireFinite(values[i], item);
    }
  }
}

void validateControlledVec(const Vec6 &values, const std::string &name,
                           bool strictly_positive, bool non_negative) {//验证受控关节的向量参数
  // 控制律参数只在物理关节 4、5、6 上有意义。CFG 仍写六个值以保持 SDK
  // 的六轴格式，因此这里只验证受控三轴，允许 1～3 轴用 0 明确表示“不使用”。
  for (const std::size_t joint : kControlledJoints) {
    const std::string item =
        name + "[" + std::to_string(joint + 1) + "]";
    if (strictly_positive) {
      requirePositive(values[joint], item);
    } else if (non_negative) {
      requireNonNegative(values[joint], item);
    } else {
      requireFinite(values[joint], item);
    }
  }
}

void validateFixedJointVec(const Vec6 &values, const std::string &name,
                           bool strictly_positive, bool non_negative) {//验证固定关节的向量参数
  // inactive_joint_drift_limit 只用于受外部机构固定的关节 1、2、3。
  for (std::size_t joint = 0; joint < kControlledJoints.front(); ++joint) {
    const std::string item =
        name + "[" + std::to_string(joint + 1) + "]";
    if (strictly_positive) {
      requirePositive(values[joint], item);
    } else if (non_negative) {
      requireNonNegative(values[joint], item);
    } else {
      requireFinite(values[joint], item);
    }
  }
}

void validateConfig(const Config &cfg) {
  // 真机运行前的静态参数审查。它只能发现数值/组合明显不合法，不能证明
  // 控制器对具体机械臂稳定，也不能替代厂家关节限位、额定力矩和负载核对。
  if (cfg.robot_ip.empty() || cfg.local_ip.empty()) {
    throw std::runtime_error("robot_ip 和 local_ip 不能为空");
  }
  requirePositive(cfg.total_time, "total_time");
  requireNonNegative(cfg.start_delay, "start_delay");
  requireNonNegative(cfg.trajectory_ramp_time, "trajectory_ramp_time");
  requireNonNegative(cfg.command_ramp_time, "command_ramp_time");
  requirePositive(cfg.ready_move_speed, "ready_move_speed");
  requirePositive(cfg.sdk_filter_cutoff_hz, "sdk_filter_cutoff_hz");
  if (cfg.sdk_filter_cutoff_hz > 1000.0) {
    throw std::runtime_error("sdk_filter_cutoff_hz 不能超过 SDK 上限 1000 Hz");
  }
  // 固定三轴降阶模型 M_AA 的推导前提是关节 1～3 的速度、加速度为零。
  // 该条件不能通过发 0 Nm 自动实现，因此不允许在 CFG 中关闭此实验声明。
  if (!cfg.inactive_joints_fixed) {
    throw std::runtime_error(
        "本版本固定只控制关节 4、5、6，必须设置 inactive_joints_fixed=true，"
        "并由外部位置伺服、制动器或机械约束保持关节 1、2、3");
  }
  validateVec(cfg.ready_q, "ready_q", false, false);
  validateVec(cfg.traj_offset, "traj_offset", false, false);
  validateVec(cfg.traj_amp, "traj_amp", false, false);
  validateVec(cfg.traj_omega, "traj_omega", false, true);
  validateVec(cfg.traj_phase, "traj_phase", false, false);
  validateVec(cfg.slow_ref_base, "slow_ref_base", false, false);
  validateVec(cfg.slow_ref_low_amp, "slow_ref_low_amp", false, false);
  validateVec(cfg.slow_ref_lift, "slow_ref_lift", false, false);
  validateVec(cfg.slow_ref_high_ripple_amp, "slow_ref_high_ripple_amp", false,
              false);
  requirePositive(cfg.slow_ref_low_period, "slow_ref_low_period");
  requirePositive(cfg.slow_ref_high_period, "slow_ref_high_period");
  if (cfg.use_slow_trapezoid_reference) {
    if (!(cfg.start_delay + cfg.trajectory_ramp_time <=
              cfg.slow_ref_rise_start &&
          cfg.slow_ref_rise_start < cfg.slow_ref_rise_end &&
          cfg.slow_ref_rise_end <= cfg.slow_ref_fall_start &&
          cfg.slow_ref_fall_start < cfg.slow_ref_fall_end &&
          cfg.slow_ref_fall_end <= cfg.total_time)) {
      throw std::runtime_error(
          "慢速梯形参考时序必须满足 start_delay+trajectory_ramp_time <= "
          "rise_start < rise_end <= fall_start < fall_end <= total_time");
    }
    const double rise_duration =
        cfg.slow_ref_rise_end - cfg.slow_ref_rise_start;
    const double fall_duration =
        cfg.slow_ref_fall_end - cfg.slow_ref_fall_start;
    if (std::abs(rise_duration - fall_duration) > 1.0e-9) {
      throw std::runtime_error(
          "slow_ref 上升和下降时长必须相等，以保持等腰梯形和时间对称性");
    }
  }
  for (std::size_t joint = 0; joint < kDof; ++joint) {
    if (!isActive(cfg, joint) &&
        (std::abs(cfg.traj_offset[joint]) > kEps ||
         std::abs(cfg.traj_amp[joint]) > kEps ||
         std::abs(cfg.slow_ref_base[joint]) > kEps ||
         std::abs(cfg.slow_ref_low_amp[joint]) > kEps ||
         std::abs(cfg.slow_ref_lift[joint]) > kEps ||
         std::abs(cfg.slow_ref_high_ripple_amp[joint]) > kEps)) {
      throw std::runtime_error(
          "固定关节 " + std::to_string(joint + 1) +
          " 的正弦和慢速梯形参考参数必须为 0；固定关节不能要求跟踪轨迹");
    }
  }

  requirePositive(cfg.barrier_b0, "barrier_b0");
  requirePositive(cfg.barrier_bf, "barrier_bf");
  if (cfg.barrier_b0 <= cfg.barrier_bf) {
    throw std::runtime_error("barrier_b0 必须大于 barrier_bf");
  }
  if (!(cfg.barrier_abort_ratio > 0.0 && cfg.barrier_abort_ratio < 1.0)) {
    throw std::runtime_error("barrier_abort_ratio 必须位于 (0,1)");
  }
  validateVec(cfg.constraint_center_offset, "constraint_center_offset", false,
              false);
  validateVec(cfg.output_bound, "output_bound", true, false);
  if (cfg.enable_output_barrier && !cfg.enable_constraint_schedule) {
    // 固定约束模式下先用轨迹幅值的最坏上界做保守检查，防止期望轨迹本身
    // 已越过 barrier_abort_ratio。F-C/C-F-C 的 beta 随时间变化，无法仅用
    // 这条静态不等式完整验证，因此仍依赖实时 barrier_ratio 停机检查。
    for (std::size_t i = 0; i < kDof; ++i) {
      const double reference_offset =
          cfg.use_slow_trapezoid_reference
              ? std::abs(cfg.slow_ref_base[i]) +
                    std::abs(cfg.slow_ref_low_amp[i]) +
                    std::abs(cfg.slow_ref_lift[i]) +
                    std::abs(cfg.slow_ref_high_ripple_amp[i])
              : std::abs(cfg.traj_offset[i]) + std::abs(cfg.traj_amp[i]);
      const double worst_reference_offset =
          std::abs(cfg.constraint_center_offset[i]) + reference_offset;
      if (worst_reference_offset >=
          cfg.barrier_abort_ratio * cfg.output_bound[i]) {
        throw std::runtime_error(
            "关节 " + std::to_string(i + 1) +
            " 的参考轨迹可能触及固定输出约束/停机边界，请减小当前轨迹"
            "幅值参数或调整 output_bound");
      }
    }
  }
  if (cfg.enable_constraint_schedule && cfg.constraint_schedule_start_free) {
    if (!(cfg.free_to_constraint_start < cfg.free_to_constraint_end)) {
      throw std::runtime_error("自由到约束 F-C 的起止时刻顺序不合法");
    }
  } else if (cfg.enable_constraint_schedule &&
             !(cfg.constraint_to_free_start < cfg.constraint_to_free_end &&
               cfg.constraint_to_free_end <= cfg.free_to_constraint_start &&
               cfg.free_to_constraint_start < cfg.free_to_constraint_end)) {
    throw std::runtime_error("约束 C-F-C 的四个时刻顺序不合法");
  }
  if (cfg.use_slow_trapezoid_reference && cfg.enable_constraint_schedule &&
      !cfg.constraint_schedule_start_free &&
      !(cfg.constraint_to_free_end <= cfg.slow_ref_rise_start &&
        cfg.slow_ref_fall_end <= cfg.free_to_constraint_start)) {
    throw std::runtime_error(
        "C-F-C 模式下慢速梯形抬升/回落必须完整位于自由阶段内部");
  }

  validateControlledVec(cfg.k1, "k1", true, false);
  validateControlledVec(cfg.k2, "k2", true, false);
  requirePositive(cfg.varrho1, "varrho1");
  requirePositive(cfg.varrho2, "varrho2");
  validateControlledVec(cfg.q1_cost, "q1_cost", true, false);
  validateControlledVec(cfg.q2_cost, "q2_cost", true, false);
  validateControlledVec(cfg.r1_cost, "r1_cost", true, false);
  validateControlledVec(cfg.r2_cost, "r2_cost", true, false);
  validateControlledVec(cfg.alpha_limit, "alpha_limit", true, false);
  validateControlledVec(cfg.alpha_dot_limit, "alpha_dot_limit", true, false);
  validateControlledVec(cfg.virtual_accel_limit, "virtual_accel_limit", true,
                        false);
  if (!(cfg.alpha_dot_filter > 0.0 && cfg.alpha_dot_filter <= 1.0)) {
    throw std::runtime_error("alpha_dot_filter 必须位于 (0,1]");
  }

  requirePositive(cfg.iota, "iota");
  requireNonNegative(cfg.theta_initial, "theta_initial");
  requireNonNegative(cfg.gamma_theta, "gamma_theta");
  requireNonNegative(cfg.sigma_theta, "sigma_theta");
  requirePositive(cfg.theta_max, "theta_max");
  if (cfg.theta_initial > cfg.theta_max) {
    throw std::runtime_error("theta_initial 不能大于 theta_max");
  }
  requireNonNegative(cfg.learning_start_time, "learning_start_time");
  validateControlledVec(cfg.gamma_c1, "gamma_c1", false, true);
  validateControlledVec(cfg.gamma_a1, "gamma_a1", false, true);
  validateControlledVec(cfg.gamma_c2, "gamma_c2", false, true);
  validateControlledVec(cfg.gamma_a2, "gamma_a2", false, true);
  requireNonNegative(cfg.weight_gain_c1, "weight_gain_c1");
  requireNonNegative(cfg.weight_gain_a1, "weight_gain_a1");
  requireNonNegative(cfg.weight_gain_c2, "weight_gain_c2");
  requireNonNegative(cfg.weight_gain_a2, "weight_gain_a2");
  requireNonNegative(cfg.leak_c1, "leak_c1");
  requireNonNegative(cfg.leak_a1, "leak_a1");
  requireNonNegative(cfg.leak_c2, "leak_c2");
  requireNonNegative(cfg.leak_a2, "leak_a2");
  requirePositive(cfg.delta_norm, "delta_norm");
  requirePositive(cfg.delta_ci, "delta_ci");
  requireNonNegative(cfg.pseudo_huber_eps1, "pseudo_huber_eps1");
  requireNonNegative(cfg.pseudo_huber_eps2, "pseudo_huber_eps2");
  requirePositive(cfg.bar_upsilon_a1, "bar_upsilon_a1");
  requirePositive(cfg.bar_upsilon_a2, "bar_upsilon_a2");
  validateControlledVec(cfg.actor_limit1, "actor_limit1", true, false);
  validateControlledVec(cfg.actor_limit2, "actor_limit2", true, false);
  validateControlledVec(cfg.initial_actor_gain1, "initial_actor_gain1", false,
                        true);
  validateControlledVec(cfg.initial_actor_gain2, "initial_actor_gain2", false,
                        true);
  requireNonNegative(cfg.initial_weight_noise, "initial_weight_noise");
  requireNonNegative(cfg.initial_weight_seed, "initial_weight_seed");
  if (std::floor(cfg.initial_weight_seed) != cfg.initial_weight_seed ||
      cfg.initial_weight_seed > 4294967295.0) {
    throw std::runtime_error(
        "initial_weight_seed 必须是 [0,4294967295] 内的整数");
  }
  requirePositive(cfg.rbf_width, "rbf_width");
  if (cfg.rbf_center_max <= cfg.rbf_center_min) {
    throw std::runtime_error("rbf_center_max 必须大于 rbf_center_min");
  }
  validateControlledVec(cfg.rbf_z1_scale, "rbf_z1_scale", true, false);
  requirePositive(cfg.rbf_beta_dot_scale, "rbf_beta_dot_scale");
  requirePositive(cfg.rbf_phi_scale, "rbf_phi_scale");
  requirePositive(cfg.rbf_standardized_clip, "rbf_standardized_clip");

  validateControlledVec(cfg.nominal_inertia_diag, "nominal_inertia_diag", true,
                        false);
  validateControlledVec(cfg.torque_bias, "torque_bias", false, false);
  // acknowledge_unverified_urdf_inertia 只用于在配置摘要处给出一次警告，
  // 不再是 validateConfig() 的拒绝条件。质量矩阵数值检查仍在下方和实时循环中。
  if (cfg.perform_ready_move) {
    // 只有确实要执行 MoveJ 时，ready_q 才是本次启动姿态，提前检查该姿态
    // 的 M_456,456。跳过 MoveJ 时，控制器构造函数会用实测 q_init 再检查。
    Mat6 ready_mass_lower{};
    const Mat6 ready_mass = selectedMassMatrix(cfg.ready_q, cfg);
    if (!activeCholesky(ready_mass, cfg, ready_mass_lower)) {
      throw std::runtime_error("ready_q 处所选关节质量矩阵不是有限正定矩阵");
    }
  }
  validateControlledVec(cfg.tau_limit, "tau_limit", true, false);
  validateControlledVec(cfg.tau_rate_limit, "tau_rate_limit", true, false);
  validateVec(cfg.abort_error, "abort_error", true, false);
  validateVec(cfg.abort_velocity, "abort_velocity", true, false);
  validateFixedJointVec(cfg.inactive_joint_drift_limit,
                        "inactive_joint_drift_limit", true, false);
  requirePositive(cfg.log_decimation, "log_decimation");
  if (std::abs(cfg.log_decimation - std::round(cfg.log_decimation)) > kEps) {
    throw std::runtime_error("log_decimation 必须是整数");
  }
}

Config loadConfig(const std::string &path) {//从指定路径加载配置文件并解析为 Config 对象
  // 每行格式为 key=value；# 后内容在解析前被删除，所以 CFG 可以大量添加
  // 教学注释而不影响数值。读取完成后统一 validateConfig()，不边读边运行。
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

std::string defaultConfigPath() {//返回默认配置文件路径，涉及到本机路径，迁移需注意
  const std::array<std::string, 7> candidates{
      "optimal_backstepping_torque.cfg",
      "config/optimal_backstepping_torque.cfg",
      "../config/optimal_backstepping_torque.cfg",
      "../../config/optimal_backstepping_torque.cfg",
      "../optimal_backstepping_xcore/config/optimal_backstepping_torque.cfg",
      "实物实验/optimal_backstepping_xcore/config/optimal_backstepping_torque.cfg",
      "../实物实验/optimal_backstepping_xcore/config/optimal_backstepping_torque.cfg"};
  for (const auto &candidate : candidates) {
    if (fileReadable(candidate)) {
      return candidate;
    }
  }
  return candidates[1];
}

std::string boolText(bool value) { return value ? "true" : "false"; }

void printConfigSummary(const Config &cfg, const std::string &path) {//打印配置文件摘要信息
  std::cout << "配置检查通过: " << path << '\n'
            << "  hardware_enable=" << boolText(cfg.hardware_enable)
            << ", ready_pose_confirmed=" << boolText(cfg.ready_pose_confirmed)
            << ", perform_ready_move=" << boolText(cfg.perform_ready_move)
            << '\n'
            << "  robot_ip=" << cfg.robot_ip << ", local_ip=" << cfg.local_ip
            << '\n'
             << "  barrier=" << boolText(cfg.enable_output_barrier)
             << ", constraint_schedule="
             << boolText(cfg.enable_constraint_schedule)
             << ", RL=" << boolText(cfg.enable_rl)
             << ", learning=" << boolText(cfg.enable_learning) << '\n'
             << "  network=joint Z1/Z2(" << kZ1Dim << '/' << kZ2Dim
             << "), rbf_nodes=" << kRbfNodes
             << ", Halton=simulation first 25 + continuation"
             << ", actor_direction_projection="
             << boolText(cfg.enable_actor_direction_projection) << '\n'
             << "  rbf_center=[" << cfg.rbf_center_min << ','
             << cfg.rbf_center_max << "], rbf_width=" << cfg.rbf_width
             << ", standardized_clip=" << cfg.rbf_standardized_clip
             << ", initial_weight_seed=" << cfg.initial_weight_seed << '\n'
             << "  pseudo_huber=residual col(Bellman,eP), eps=("
             << cfg.pseudo_huber_eps1 << ',' << cfg.pseudo_huber_eps2 << ")"
             << '\n'
             << "  mass_model="
             << (cfg.use_xmate_cr7_urdf_mass_matrix
                     ? "xMateCR7 URDF full M(q) / fixed M_456,456"
                     : "nominal diagonal fallback")
             << ", controlled_joints=5"
             << ", inactive_joints_fixed="
             << boolText(cfg.inactive_joints_fixed) << '\n'
             << "  urdf_inertia_acknowledged="
             << boolText(cfg.acknowledge_unverified_urdf_inertia) << '\n'
             << "  注意：未确认的 URDF 惯性参数会产生警告，但不再阻止运行；"
                "质量矩阵数值健康检查仍保持启用。"
             << std::endl;
  if (cfg.hardware_enable && cfg.use_xmate_cr7_urdf_mass_matrix &&
      !cfg.acknowledge_unverified_urdf_inertia) {
    // 放在摘要函数中而非 validateConfig()，避免 --run 覆盖 IP 后第二次校验时
    // 重复打印；该分支只告警，不抛出异常、不改变控制参数。
    std::cerr
        << "警告：厂家 URDF 惯性参数尚未通过物理一致性确认；程序将按当前配置继续运行。\n"
        << "      当前 M(q) 只能视为候选模型，请保持保守力矩/加速度限制并确保急停可用。\n"
        << "      若已完成人工核实，可设置 acknowledge_unverified_urdf_inertia=true 以消除此警告。"
        << std::endl;
  }
}

struct Desired {
  // 论文控制律需要 q_d、q_dot_d，并在 alpha_dot 初始化时使用 q_ddot_d；
  // 因而这里同时解析生成三阶参考量，避免在 1 kHz 回调中再数值微分 q_d。
  Vec6 q{};
  Vec6 dq{};
  Vec6 ddq{};
};

struct ScalarReference {
  double value = 0.0;
  double derivative = 0.0;
  double second_derivative = 0.0;
};

ScalarReference slowTrapezoidEnvelope(double time, const Config &cfg) { //参考轨迹函数梯形包络函数
  // 与仿真 _trapezoid_envelope() 同形：0 -> raised-cosine -> 1 ->
  // raised-cosine -> 0。端点的一阶导数严格为 0，二阶导数分段有界。
  if (time <= cfg.slow_ref_rise_start ||
      time >= cfg.slow_ref_fall_end) {
    return {};
  }
  if (time < cfg.slow_ref_rise_end) {
    const double duration =
        cfg.slow_ref_rise_end - cfg.slow_ref_rise_start;
    const double u = (time - cfg.slow_ref_rise_start) / duration;
    const double w = kPi / duration;
    return {0.5 * (1.0 - std::cos(kPi * u)),
            0.5 * w * std::sin(kPi * u),
            0.5 * w * w * std::cos(kPi * u)};
  }
  if (time <= cfg.slow_ref_fall_start) {
    return {1.0, 0.0, 0.0};
  }
  const double duration = cfg.slow_ref_fall_end - cfg.slow_ref_fall_start;
  const double u = (time - cfg.slow_ref_fall_start) / duration;
  const double w = kPi / duration;
  return {0.5 * (1.0 + std::cos(kPi * u)),
          -0.5 * w * std::sin(kPi * u),
          -0.5 * w * w * std::cos(kPi * u)};
}

Desired desiredTrajectory(double time, const Vec6 &q_init, const Config &cfg) {
  // 所有模式都以 q_init 为基准，并在 start_delay 后用半余弦启动渐入。
  // slow 模式复现仿真的 low -> 梯形抬升 -> high -> 梯形回落 -> low 形状，
  // 但采用更长周期和更小真机幅值；false 时保留原单正弦回退。
  Desired out;
  out.q = q_init;
  if (time <= cfg.start_delay) {
    return out;
  }

  const double t = time - cfg.start_delay;
  double ramp = 1.0;
  double ramp_dot = 0.0;
  double ramp_ddot = 0.0;
  if (cfg.trajectory_ramp_time > kEps && t < cfg.trajectory_ramp_time) {//正常三角函数轨迹
    const double a = kPi * t / cfg.trajectory_ramp_time;
    const double w = kPi / cfg.trajectory_ramp_time;
    ramp = 0.5 * (1.0 - std::cos(a));
    ramp_dot = 0.5 * w * std::sin(a);
    ramp_ddot = 0.5 * w * w * std::cos(a);
  }

  if (cfg.use_slow_trapezoid_reference) {
    const ScalarReference envelope = slowTrapezoidEnvelope(time, cfg);
    const double center =
        0.5 * (cfg.slow_ref_rise_start + cfg.slow_ref_fall_end);
    const double relative_time = time - center;
    const double low_w = 2.0 * kPi / cfg.slow_ref_low_period;
    const double high_w = 2.0 * kPi / cfg.slow_ref_high_period;
    const double low_cos = std::cos(low_w * relative_time);
    const double low_sin = std::sin(low_w * relative_time);
    const double high_cos = std::cos(high_w * relative_time);
    const double high_sin = std::sin(high_w * relative_time);

    for (std::size_t i = 0; i < kDof; ++i) {
      const double low =
          cfg.slow_ref_base[i] + cfg.slow_ref_low_amp[i] * low_cos;
      const double low_dot =
          -cfg.slow_ref_low_amp[i] * low_w * low_sin;
      const double low_ddot =
          -cfg.slow_ref_low_amp[i] * low_w * low_w * low_cos;
      const double high = cfg.slow_ref_base[i] + cfg.slow_ref_lift[i] +
                          cfg.slow_ref_high_ripple_amp[i] * high_cos;
      const double high_dot =
          -cfg.slow_ref_high_ripple_amp[i] * high_w * high_sin;
      const double high_ddot = -cfg.slow_ref_high_ripple_amp[i] * high_w *
                               high_w * high_cos;
      const double carrier =
          (1.0 - envelope.value) * low + envelope.value * high;
      const double carrier_dot =
          (1.0 - envelope.value) * low_dot +
          envelope.value * high_dot +
          envelope.derivative * (high - low);
      const double carrier_ddot =
          (1.0 - envelope.value) * low_ddot +
          envelope.value * high_ddot +
          2.0 * envelope.derivative * (high_dot - low_dot) +
          envelope.second_derivative * (high - low);
      out.q[i] += ramp * carrier;
      out.dq[i] = ramp_dot * carrier + ramp * carrier_dot;
      out.ddq[i] = ramp_ddot * carrier +
                   2.0 * ramp_dot * carrier_dot + ramp * carrier_ddot;
    }
  } else {
    for (std::size_t i = 0; i < kDof; ++i) {
      const double w = cfg.traj_omega[i];
      const double angle = w * t + cfg.traj_phase[i];
      const double carrier =
          cfg.traj_offset[i] + cfg.traj_amp[i] * std::sin(angle);
      const double carrier_dot = cfg.traj_amp[i] * w * std::cos(angle);
      const double carrier_ddot =
          -cfg.traj_amp[i] * w * w * std::sin(angle);
      out.q[i] += ramp * carrier;
      out.dq[i] = ramp_dot * carrier + ramp * carrier_dot;
      out.ddq[i] = ramp_ddot * carrier +
                   2.0 * ramp_dot * carrier_dot + ramp * carrier_ddot;
    }
  }
  return out;
}

struct LinkValue {
  // value 是论文决策连接函数 kappa(t)，derivative 是其解析时间导数。
  double value = 0.0;
  double derivative = 0.0;
};

LinkValue smoothDecisionLink(double time, double start, double end,
                             bool increasing) {
  // 论文约束-自由切换中的光滑 C-infinity 决策连接函数。指数限制到 [-80,80]
  // 只是防止 exp 溢出，不改变远离数值极限时的论文表达式。
  if (time <= start) {
    return increasing ? LinkValue{0.0, 0.0} : LinkValue{1.0, 0.0};
  }
  if (time >= end) {
    return increasing ? LinkValue{1.0, 0.0} : LinkValue{0.0, 0.0};
  }

  double chi = 0.0;
  double chi_dot = 0.0;
  if (increasing) {
    chi = 1.0 / (end - time) - 1.0 / (time - start);
    chi_dot = 1.0 / ((end - time) * (end - time)) +
              1.0 / ((time - start) * (time - start));
  } else {
    chi = 1.0 / (time - start) - 1.0 / (end - time);
    chi_dot = -1.0 / ((time - start) * (time - start)) -
              1.0 / ((end - time) * (end - time));
  }
  const double exponent = clamp(-2.0 * chi, -80.0, 80.0);
  const double value = 1.0 / (1.0 + std::exp(exponent));
  return {value, 2.0 * value * (1.0 - value) * chi_dot};
}

LinkValue decisionKappa(double time, const Config &cfg) {
  // start_free=true 为论文 F-C：kappa 由 1 降至 0；false 为原工程
  // C-F-C：kappa 依次为 0->1->0。beta=(b0-bf)kappa+bf。
  // 关闭 schedule 时始终返回有限约束状态。
  if (!cfg.enable_constraint_schedule) {
    return {0.0, 0.0};
  }
  if (cfg.constraint_schedule_start_free) {
    if (time <= cfg.free_to_constraint_start) {
      return {1.0, 0.0};
    }
    if (time < cfg.free_to_constraint_end) {
      return smoothDecisionLink(time, cfg.free_to_constraint_start,
                                cfg.free_to_constraint_end, false);
    }
    return {0.0, 0.0};
  }
  if (time <= cfg.constraint_to_free_start) {
    return {0.0, 0.0};
  }
  if (time < cfg.constraint_to_free_end) {
    return smoothDecisionLink(time, cfg.constraint_to_free_start,
                              cfg.constraint_to_free_end, true);
  }
  if (time <= cfg.free_to_constraint_start) {
    return {1.0, 0.0};
  }
  if (time < cfg.free_to_constraint_end) {
    return smoothDecisionLink(time, cfg.free_to_constraint_start,
                              cfg.free_to_constraint_end, false);
  }
  return {0.0, 0.0};
}

struct BarrierScalar {
  // transformed = s_x(x,beta)，即论文 eq:sx_definition；
  // jacobian    = h_x = partial s_x / partial x；
  // time_term   = h_beta = partial s_x/partial beta * beta_dot；
  // derivative  = h_x*x_dot+h_beta；
  // ratio       = 未钳位的 zeta=eta/beta，仅供真机越界监视。
  double transformed = 0.0;
  double jacobian = 1.0;
  double time_term = 0.0;
  double derivative = 0.0;
  double ratio = 0.0;
};

BarrierScalar barrierTransform(double x, double x_dot, double lambda,
                               double beta, double beta_dot,
                               double b0) {
  // 论文 eq:eta_zeta_x、eq:sx_definition 和 eq:Hx_hbeta_definition 的标量化
  // 实现。六个关节各自调用一次，等价于对角 H_x/H_beta。
  const double root = std::sqrt(x * x + lambda * lambda + kEps);
  const double eta = b0 * x / root;
  const double raw_ratio = eta / (beta + kEps);
  // 仅计算变换和导数时将 zeta 限到 0.995，避免 1-zeta^2 接近 0 导致
  // 浮点爆炸；安全判断仍使用 raw_ratio，因此该钳位不会掩盖真实越界。
  const double ratio = clamp(raw_ratio, -0.995, 0.995);
  const double one_minus = 1.0 - ratio * ratio;
  const double transformed = ratio / (one_minus + kEps);
  const double ds_dratio =
      (1.0 + ratio * ratio) / (one_minus * one_minus + kEps);
  const double deta_dx =
      b0 * lambda * lambda /
      std::pow(x * x + lambda * lambda + kEps, 1.5);
  const double dratio_dx = deta_dx / (beta + kEps);
  const double dratio_dbeta = -eta / ((beta + kEps) * (beta + kEps));
  const double jacobian = std::max(1.0e-5, ds_dratio * dratio_dx);
  const double time_term = ds_dratio * dratio_dbeta * beta_dot;
  return {transformed, jacobian, time_term,
          jacobian * x_dot + time_term, raw_ratio};
}

// 完整 Z1/Z2 使用固定 64 个确定性 Halton 高维稀疏中心；节点数不会随
// 输入维数指数增长。输入先做仿射标准化，Gaussian 距离按维数归一。
double radicalInverse(std::size_t index, unsigned int base) {
  double value = 0.0;
  double factor = 1.0 / static_cast<double>(base);
  while (index > 0) {
    value += factor * static_cast<double>(index % base);
    index /= base;
    factor /= static_cast<double>(base);
  }
  return value;
}

double sparseCenter(std::size_t node, std::size_t dimension,// 计算稀疏 RBF 中心，使用 Halton 序列生成
                    const Config &cfg) {
  // 与仿真 _halton_centers() 完全相同：序号从 1 开始，两层共用同一
  // 低差异序列。实物的前 25 个中心因此与仿真一致，后 39 个只是续点。
  const std::size_t sequence_index = node + 1;
  const double unit = radicalInverse(sequence_index, kHaltonBases[dimension]);
  return cfg.rbf_center_min +
         (cfg.rbf_center_max - cfg.rbf_center_min) * unit;
}

std::uint64_t splitMix64(std::uint64_t value) {// SplitMix64 哈希函数，用于生成伪随机数
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

double deterministicNormal(std::uint64_t seed, std::size_t layer,
                           std::size_t output, std::size_t node) {
  // 固定整数哈希 + Box-Muller，得到跨运行可复现的小高斯扰动；不在实时
  // 回调中调用随机数库，也不依赖不同标准库的 normal_distribution 实现。
  const std::uint64_t key =
      seed ^ (0x632be59bd9b4e019ULL * (layer + 1U)) ^
      (0x8cb92baa5f5a5f2dULL * (output + 1U)) ^
      (0x9e3779b97f4a7c15ULL * (node + 1U));
  constexpr double kTwoTo53 = 9007199254740992.0;
  const double u1 =
      (static_cast<double>(splitMix64(key) >> 11U) + 0.5) / kTwoTo53;
  const double u2 =
      (static_cast<double>(splitMix64(key ^ 0xd1b54a32d192ed03ULL) >> 11U) +
       0.5) /
      kTwoTo53;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
}

double standardizeRbf(double value, double center, double scale,
                      const Config &cfg) {// 将输入值标准化为 RBF 网络的 [-clip, clip] 范围
  return clamp((value - center) / (std::abs(scale) + kEps),
               -cfg.rbf_standardized_clip, cfg.rbf_standardized_clip);
}

template <std::size_t Dimension>
Feature sparseRbfFeatures(const std::array<double, Dimension> &input,
                          const Config &cfg) {// 计算稀疏 RBF 特征向量，使用高斯核函数
  Feature features{};
  const double denominator =
      cfg.rbf_width * cfg.rbf_width * static_cast<double>(Dimension) + kEps;
  for (std::size_t node = 0; node < kRbfNodes; ++node) {
    double squared_distance = 0.0;
    for (std::size_t dimension = 0; dimension < Dimension; ++dimension) {
      const double delta = input[dimension] -
                           sparseCenter(node, dimension, cfg);
      squared_distance += delta * delta;
    }
    features[node] = std::exp(-squared_distance / denominator);
  }
  return features;
}

Z1Input layer1NetworkInput(const Vec6 &z1, const Vec6 &q,
                           const Desired &desired, const Vec6 &q_init,
                           double beta, double beta_dot, const Config &cfg) {// 计算 Z1 网络的输入特征向量
  Z1Input input{};
  std::size_t index = 0;
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    input[index++] = standardizeRbf(z1[joint], 0.0, cfg.rbf_z1_scale[joint], cfg);// z1 先做仿射标准化，再做高斯 RBF做输入
  }
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    input[index++] = standardizeRbf(q[joint], q_init[joint],
                                    cfg.output_bound[joint], cfg);
  }
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    input[index++] = standardizeRbf(desired.q[joint], q_init[joint],
                                    cfg.output_bound[joint], cfg);
  }
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    input[index++] = standardizeRbf(desired.dq[joint], 0.0,
                                    cfg.abort_velocity[joint], cfg);
  }
  const double beta_center = 0.5 * (cfg.barrier_b0 + cfg.barrier_bf);
  const double beta_scale =
      std::max(0.5 * std::abs(cfg.barrier_b0 - cfg.barrier_bf), kEps);
  input[index++] = standardizeRbf(beta, beta_center, beta_scale, cfg);
  input[index++] =
      standardizeRbf(beta_dot, 0.0, cfg.rbf_beta_dot_scale, cfg);
  return input;
}

Z2Input layer2NetworkInput(const Vec6 &z2, const Vec6 &q, const Vec6 &dq,
                           const Vec6 &alpha, const Vec6 &alpha_dot,
                           const Vec6 &q_init, double phi, const Config &cfg) {
  Z2Input input{};
  std::size_t index = 0;
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    input[index++] = standardizeRbf(z2[joint], 0.0,
                                    cfg.abort_velocity[joint], cfg);
  }
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    input[index++] = standardizeRbf(q[joint], q_init[joint],
                                    cfg.output_bound[joint], cfg);
  }
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    input[index++] = standardizeRbf(dq[joint], 0.0,
                                    cfg.abort_velocity[joint], cfg);
  }
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    input[index++] = standardizeRbf(alpha[joint], 0.0,
                                    cfg.alpha_limit[joint], cfg);
  }
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    input[index++] = standardizeRbf(alpha_dot[joint], 0.0,
                                    cfg.alpha_dot_limit[joint], cfg);
  }
  double phi_center = 1.0;
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    phi_center += std::abs(q_init[kControlledJoints[active]]);
  }
  input[index++] = standardizeRbf(phi, phi_center, cfg.rbf_phi_scale, cfg);
  return input;
}

double featureDot(const Feature &weights, const Feature &features) {// 计算特征向量与权重向量的点积，得到网络输出
  double result = 0.0;
  for (std::size_t i = 0; i < kRbfNodes; ++i) {
    result += weights[i] * features[i];
  }
  return result;
}

Vec6 criticOutput(const WeightMatrix &weights, const Feature &features,
                  bool enabled) {
  // 对应论文 hat J_i = hat W_ci^T S_i。这里每一行权重输出一个关节分量；
  // enable_rl=false 时直接返回 0，方便与不含 RL 的反步基线作同条件对比。
  Vec6 output{};
  if (!enabled) {
    return output;
  }
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    output[joint] = featureDot(weights[active], features);
  }
  return output;
}

Vec6 actorOutput(const WeightMatrix &weights, const Feature &features,
                  const Vec6 &state, const Vec6 &limit, const Config &cfg,
                  bool enabled) {
  // 对应论文 actor 近似项 hat W_ai^T S_i。actor_limit 是实物实验额外的
  // 输出限幅；论文理想控制律本身未包含该饱和环节。
  Vec6 output{};
  if (!enabled) {
    return output;
  }
  for (std::size_t active = 0; active < kControlledDof; ++active) {
    const std::size_t joint = kControlledJoints[active];
    output[joint] = clamp(featureDot(weights[active], features), -limit[joint],
                          limit[joint]);
  }


  // 只有 CFG 显式启用时，才删除 state^T actor<0 的反向分量；该分支是
  // 真机可选保护，不属于论文控制器。
  double state_norm2 = 0.0;
  double state_actor = 0.0;
  for (std::size_t joint = 0; joint < kDof; ++joint) {
    if (isActive(cfg, joint)) {
      state_norm2 += state[joint] * state[joint];
      state_actor += state[joint] * output[joint];
    }
  }
  if (cfg.enable_actor_direction_projection && state_norm2 > kEps &&
      state_actor < 0.0) {
    const double projection = state_actor / state_norm2;
    for (std::size_t joint = 0; joint < kDof; ++joint) {
      if (isActive(cfg, joint)) {
        output[joint] = clamp(output[joint] - projection * state[joint],
                              -limit[joint], limit[joint]);
      }
    }
  }
  return output;
}

double pseudoHuberResidualFactor(const Vec6 &bellman,
                                 const Vec6 &actor_critic_error,
                                 double epsilon, const Config &cfg) {//计算伪Huber残差因子，论文式(59)的实现
  // 与仿真 _update_weights() 和论文式(59)相同：
  // ||r_i||^2=||tilde_e_i||^2+||e_Pi||^2。只统计关节 4~6 活动通道。
  double residual_norm2 = 0.0;
  for (std::size_t joint : kControlledJoints) {
    if (isActive(cfg, joint)) {
      residual_norm2 += bellman[joint] * bellman[joint] +
                        actor_critic_error[joint] *
                            actor_critic_error[joint];
    }
  }
  return 1.0 /
         std::sqrt(1.0 + epsilon * epsilon * residual_norm2);
}

Vec6 modelInputMap(const Mat6 &selected_mass,
                   const Vec6 &virtual_acceleration, const Config &cfg) {//将模型输入映射到关节力矩
  // 论文 eq:tau_output：tau_A=M_AA(q)*v_A。selected_mass 固定保留
  // 关节 4、5、6 的行列；关节 1～3 固定产生的约束反力由外部保持机构承担。
  Vec6 torque{};
  for (std::size_t row = 0; row < kDof; ++row) {
    if (!isActive(cfg, row)) {
      continue;
    }
    for (std::size_t column = 0; column < kDof; ++column) {
      if (isActive(cfg, column)) {
        torque[row] +=
            selected_mass[row][column] * virtual_acceleration[column];
      }
    }
    torque[row] += cfg.torque_bias[row];
  }
  return torque;
}

struct LayerWeights {
  // 两套独立网络分别服务反步第 1/2 层；矩阵只有 3 行，依次映射到
  // 物理关节 4、5、6；每行包含 64 个确定性高维稀疏 RBF 中心的权重。
  // 原参考控制器没有 actor/critic 权重，这两个矩阵均为论文控制器新增状态。
  WeightMatrix actor{};
  WeightMatrix critic{};
};

struct ControllerOutput {
  // 将控制器中间量集中返回，既便于 CSV 对照论文符号，也使安全层能在发送
  // 力矩前检查误差、约束比和数值有限性。torque_raw 尚未经过真机限幅/斜坡。
  Desired desired{};
  Vec6 error{};
  Vec6 z1{};
  Vec6 z2{};
  Vec6 alpha{};
  Vec6 alpha_dot{};
  Vec6 actor1{};
  Vec6 actor2{};
  Vec6 torque_raw{};
  Vec6 mass_matrix_diag{};
  Vec6 barrier_ratio{};
  double theta_hat = 0.0;
  double sigma_g1 = 1.0;
  double sigma_g2 = 1.0;
  bool mass_matrix_valid = true;
  bool finite = true;
};

class OptimalBacksteppingController {//主要写控制器的类，前面都是运算定义
public:
  OptimalBacksteppingController(const Config &cfg, const Vec6 &q_init)
      : cfg_(cfg), q_init_(q_init), theta_hat_(cfg.theta_initial) {
    Mat6 initial_lower{};
    if (!activeCholesky(selectedMassMatrix(q_init_, cfg_), cfg_,
                        initial_lower)) {
      throw std::runtime_error("q_init 处 M_456,456 不是有限正定矩阵");
    }
    // 与仿真统一为 critic=0、actor=固定种子噪声+中心斜率。实物有 64 而
    // 仿真有 25 个节点，所以斜率乘 25/64、噪声乘 sqrt(25/64)，保持初始
    // Actor 总输出/方差不会仅因节点数增加而放大。
    const double slope_scale =
        kSimulationRbfNodes / static_cast<double>(kRbfNodes);
    const double noise_scale = std::sqrt(slope_scale);
    const auto seed = static_cast<std::uint64_t>(cfg_.initial_weight_seed);
    for (std::size_t active = 0; active < kControlledDof; ++active) {
      const std::size_t joint = kControlledJoints[active];
      for (std::size_t node = 0; node < kRbfNodes; ++node) {
        layer1_.actor[active][node] =
            cfg_.initial_weight_noise * noise_scale *
                deterministicNormal(seed, 0, active, node) +
            cfg_.initial_actor_gain1[joint] *
                sparseCenter(node, active, cfg_) * slope_scale;
        layer2_.actor[active][node] =
            cfg_.initial_weight_noise * noise_scale *
                deterministicNormal(seed, 1, active, node) +
            cfg_.initial_actor_gain2[joint] *
                sparseCenter(node, active, cfg_) * slope_scale;
      }
    }
  }

  ControllerOutput compute(double time, const Vec6 &q, const Vec6 &dq) {
    // compute() 是每 1 ms 调用一次的纯控制计算主线：
    //   参考轨迹 -> 输出约束变换 -> 第一步 alpha -> z2 -> 第二步 tau
    //   -> 鲁棒参数更新 -> actor/critic 更新 -> 数值健康检查。
    // 它不直接调用 SDK；力矩限幅和急停判断位于外层，便于区分论文控制律
    // 与真机接口保护。
    ControllerOutput out;
    out.desired = desiredTrajectory(time, q_init_, cfg_);

    // [论文输出约束] beta(t)=(b0-bf)kappa(t)+bf；固定约束时 kappa=0，
    // beta=bf。这里的 beta/beta_dot 随后共同进入 s_x 与 h_beta。
    const LinkValue kappa = decisionKappa(time, cfg_);
    const double beta =
        (cfg_.barrier_b0 - cfg_.barrier_bf) * kappa.value + cfg_.barrier_bf;
    const double beta_dot =
        (cfg_.barrier_b0 - cfg_.barrier_bf) * kappa.derivative;

    Vec6 hx{};
    Vec6 hbeta{};
    Vec6 sd_dot{};
    for (std::size_t joint = 0; joint < kDof; ++joint) {
      out.error[joint] = q[joint] - out.desired.q[joint];
      // 关闭 barrier 时退化为普通误差坐标：s_x(q)=q、H_x=I、h_beta=0，
      // 因此 z1=q-q_d。该开关用于先验证无约束反步基线。
      if (!cfg_.enable_output_barrier) {
        hx[joint] = 1.0;
        out.z1[joint] = out.error[joint];
        sd_dot[joint] = out.desired.dq[joint];
        out.barrier_ratio[joint] = 0.0;
        continue;
      }

      const double lambda =
          cfg_.output_bound[joint] *
          std::sqrt(cfg_.barrier_b0 * cfg_.barrier_b0 -
                    cfg_.barrier_bf * cfg_.barrier_bf) /
          cfg_.barrier_bf;
      // 论文变换以标量 x 为输入；真机上把 x 定义为“相对实验约束中心的
      // 关节角”。center=q_init+offset 是无模型条件下新增的工程坐标选择，
      // 这样 output_bound 表示本次实验初始位姿附近的允许运动范围。
      // lambda 由 output_bound、b0、bf 反算，使有限阶段的物理边界可直接
      // 用 rad 配置，而无需用户手算变换参数。
      const double center = q_init_[joint] + cfg_.constraint_center_offset[joint];
      const BarrierScalar actual =
          barrierTransform(q[joint] - center, dq[joint], lambda, beta,
                           beta_dot, cfg_.barrier_b0);
      const BarrierScalar desired =
          barrierTransform(out.desired.q[joint] - center,
                           out.desired.dq[joint], lambda, beta, beta_dot,
                           cfg_.barrier_b0);
      hx[joint] = actual.jacobian;
      hbeta[joint] = actual.time_term;
      sd_dot[joint] = desired.derivative;
      out.z1[joint] = actual.transformed - desired.transformed;
      // z1=s_x(q)-s_x(q_d)，对应 eq:z1_output_definition；raw zeta 单独
      // 保存用于 barrier_abort_ratio，而不是把 z1 当作物理角度误差。
      out.barrier_ratio[joint] = actual.ratio;
    }

    // [反步第 1 层] 受控三轴完整 Z1_A=[z1_A,q_A,qd_A,qd_dot_A,beta,beta_dot]。
    const Z1Input z1_network_input = layer1NetworkInput(
        out.z1, q, out.desired, q_init_, beta, beta_dot, cfg_);
    const Feature s1 = sparseRbfFeatures(z1_network_input, cfg_);
    const Vec6 hc1 = criticOutput(layer1_.critic, s1, cfg_.enable_rl);
    out.actor1 = actorOutput(layer1_.actor, s1, out.z1, cfg_.actor_limit1,
                             cfg_, cfg_.enable_rl);

    for (std::size_t joint = 0; joint < kDof; ++joint) {
      if (!isActive(cfg_, joint)) {
        continue;
      }
      // 对应论文 eq:alpha_output：
      // alpha=H_x^{-1}[-Pi1*z1-h_beta+s_dot_d-actor1]。
      // pi1=k1+varrho1；分母加 kEps 防止数值除零；
      // alpha_limit 是真机额外限幅，不属于理想连续控制律。
      const double pi1 = cfg_.k1[joint] + cfg_.varrho1;
      out.alpha[joint] =
          (-pi1 * out.z1[joint] - hbeta[joint] + sd_dot[joint] -
           out.actor1[joint]) /
          (hx[joint] + kEps);
      out.alpha[joint] = clamp(out.alpha[joint], -cfg_.alpha_limit[joint],
                               cfg_.alpha_limit[joint]);
    }

    if (!alpha_initialized_) {
      // 论文需要 alpha_dot，但实时程序不对复杂 alpha 表达式做符号求导。
      // 首周期以 q_ddot_d 初始化，之后采用差分+一阶脏微分滤波；这是原参考
      // 文件中没有、为落地 eq:z2_output/eq:tau_output 增加的工程实现。
      alpha_dot_filtered_ = out.desired.ddq;
      alpha_initialized_ = true;
    } else {
      for (std::size_t joint = 0; joint < kDof; ++joint) {
        if (!isActive(cfg_, joint)) {
          continue;
        }
        const double raw = (out.alpha[joint] - previous_alpha_[joint]) / kDt;
        alpha_dot_filtered_[joint] +=
            cfg_.alpha_dot_filter * (raw - alpha_dot_filtered_[joint]);
      }
    }
    previous_alpha_ = out.alpha;
    for (std::size_t joint = 0; joint < kDof; ++joint) {
      if (!isActive(cfg_, joint)) {
        continue;
      }
      out.alpha_dot[joint] =
          clamp(alpha_dot_filtered_[joint], -cfg_.alpha_dot_limit[joint],
                cfg_.alpha_dot_limit[joint]);
      out.z2[joint] = dq[joint] - out.alpha[joint];
      // 对应论文 eq:z2_output：z2=dot q-alpha。
    }

    double phi = 1.0;
    double dq_norm2 = 0.0;
    for (std::size_t joint = 0; joint < kDof; ++joint) {
      if (!isActive(cfg_, joint)) {
        continue;
      }
      // 对应论文 core function 的可计算包络 phi(q,dq)。与配套 Python 验证
      // 程序一致，使用绝对关节状态；约束变换采用相对中心不改变这里的未知
      // 动力学包络定义。1+L1(q)+L1(dq)+0.25||dq||^2 是本文选定实现。
      phi += std::abs(q[joint]) + std::abs(dq[joint]);
      dq_norm2 += dq[joint] * dq[joint];
    }
    phi += 0.25 * dq_norm2;

    // [反步第 2 层] 受控三轴完整 Z2_A=[z2_A,q_A,dq_A,alpha_A,alpha_dot_A,phi_A]。
    const Z2Input z2_network_input = layer2NetworkInput(
        out.z2, q, dq, out.alpha, out.alpha_dot, q_init_, phi, cfg_);
    const Feature s2 = sparseRbfFeatures(z2_network_input, cfg_);
    const Vec6 hc2 = criticOutput(layer2_.critic, s2, cfg_.enable_rl);
    out.actor2 = actorOutput(layer2_.actor, s2, out.z2, cfg_.actor_limit2,
                             cfg_, cfg_.enable_rl);

    Vec6 a1_dyn{};
    Vec6 e1{};
    Vec6 ep1{};
    Mat6 lambda1{};
    Vec6 a2_dyn{};
    Vec6 e2{};
    Vec6 ep2{};
    Mat6 lambda2{};
    Vec6 second_layer_gradient_argument{};
    Vec6 virtual_acceleration{};
    const double robust_denominator = 2.0 * cfg_.iota * cfg_.iota;

    // 每周期先算完整 URDF M(q)，再固定保留关节 4~6 的 M_AA。关节 1~3
    // 的固定角仍参与完整 M(q) 构型计算，因此改变固定姿态仍会改变 M_AA。
    const Mat6 selected_mass = selectedMassMatrix(q, cfg_);
    Mat6 mass_lower{};
    out.mass_matrix_valid = activeCholesky(selected_mass, cfg_, mass_lower);
    for (std::size_t joint = 0; joint < kDof; ++joint) {
      out.mass_matrix_diag[joint] = selected_mass[joint][joint];
    }

    for (std::size_t active = 0; active < kControlledDof; ++active) {
      const std::size_t joint = kControlledJoints[active];
      if (!isActive(cfg_, joint)) {
        continue;
      }
      // 第 1 层闭环漂移 a1=-k1*z1-actor1，用于 Bellman 残差而非直接发给
      // 机械臂。g1_inv 是对角 R1 和 H_x 的标量化组合。
      const double pi1 = cfg_.k1[joint] + cfg_.varrho1;
      const double pi2 = cfg_.k2[joint] + cfg_.varrho2;
      a1_dyn[joint] = -pi1 * out.z1[joint] - out.actor1[joint];
      const double g1_inv =
          cfg_.r1_cost[joint] / (hx[joint] * hx[joint] + kEps);
      const double j1_hat =
          2.0 * g1_inv *
          (pi1 * out.z1[joint] + hbeta[joint] - sd_dot[joint] +
           hc1[joint]);
      // j1_hat、e1、ep1、lambda1 分别对应论文 eq:J1_hat_output、
      // eq:e1_output、actor-critic 输出误差及 pseudo-Huber 基准方向所需
      // lambda。Q1/R1 在 cfg 中按轴对角实现。
      e1[joint] = cfg_.q1_cost[joint] * out.z1[joint] * out.z1[joint] +
                  cfg_.r1_cost[joint] * out.alpha[joint] * out.alpha[joint] +
                  j1_hat * a1_dyn[joint];
      ep1[joint] = out.actor1[joint] - hc1[joint];
      lambda1[joint][joint] = 2.0 * g1_inv * a1_dyn[joint];

      const double mu_s = hx[joint] * out.z1[joint];
      // mu_s=H_x^T z1，对应 eq:mu_s_output；
      // nu_r=hat_vartheta*phi^2*z2/(2*iota^2)，对应 eq:robust_output。
      const double nu_r = theta_hat_ / robust_denominator * phi * phi *
                          out.z2[joint];
      // 第 2 层最优反步“虚拟加速度” v=alpha_dot-k2*z2-mu_s-nu_r-actor2。
      // virtual_accel_limit 是无模型真机试验的额外保护。
      a2_dyn[joint] = -pi2 * out.z2[joint] - mu_s - nu_r -
                       out.actor2[joint];
      virtual_acceleration[joint] =
          clamp(out.alpha_dot[joint] + a2_dyn[joint],
                -cfg_.virtual_accel_limit[joint],
                cfg_.virtual_accel_limit[joint]);
      second_layer_gradient_argument[joint] =
          pi2 * out.z2[joint] + mu_s + nu_r - out.alpha_dot[joint] +
          hc2[joint];
    }

    // 论文 eq:tau_output：tau_A=M_AA(q)*v_A。若质量矩阵健康检查失败，
    // 本周期理论力矩直接保持 0，外层随后以 mass_matrix_valid=false 停机。
    if (out.mass_matrix_valid) {
      out.torque_raw = modelInputMap(selected_mass, virtual_acceleration, cfg_);
    }

    // 论文 G2=B R2^{-1}B^T 且 B=M_AA^{-1}，所以必须同步使用
    // G2^{-1}=M_AA^T R2 M_AA；这一步保留关节 4～6 间的非对角耦合。
    const Mat6 g2_inverse = secondLayerGInverse(selected_mass, cfg_);
    Vec6 j2_hat = mat6Vector(g2_inverse, second_layer_gradient_argument);
    for (double &value : j2_hat) {
      value *= 2.0;
    }

    for (std::size_t joint = 0; joint < kDof; ++joint) {
      if (!isActive(cfg_, joint)) {
        continue;
      }
      // 对应论文 eq:J2_hat_output 与 eq:e2_output。ep2 是 actor-critic
      // 输出误差；j2_hat 已由完整 G2^{-1} 矩阵计算。
      e2[joint] = cfg_.q2_cost[joint] * out.z2[joint] * out.z2[joint] +
                  cfg_.r2_cost[joint] * out.torque_raw[joint] *
                      out.torque_raw[joint] +
                  j2_hat[joint] * a2_dyn[joint];
      ep2[joint] = out.actor2[joint] - hc2[joint];
    }
    // Lambda_c2 按论文定义为 Bellman 残差对 critic 输出的雅可比转置：
    // [Lambda]_{k,j}=d e_j/d h_c,k=2*[G2^{-1}]_{j,k}*a2_j。
    // 对角质量近似时它自动退化为旧代码的逐轴 lambda2。
    for (std::size_t critic = 0; critic < kDof; ++critic) {
      if (!isActive(cfg_, critic)) {
        continue;
      }
      for (std::size_t residual = 0; residual < kDof; ++residual) {
        if (isActive(cfg_, residual)) {
          lambda2[critic][residual] =
              2.0 * g2_inverse[residual][critic] * a2_dyn[residual];
        }
      }
    }

    if (cfg_.enable_theta_adaptation && time >= cfg_.learning_start_time) {
      // 对应 eq:vartheta_update_output 的离散 Euler 实现。只用关节 4~6
      // 的 z2 更新 hat_vartheta，避免固定轴误差放大受控轴鲁棒项；
      // [0,theta_max] 投影是数值保护。
      double active_z2_norm2 = 0.0;
      for (std::size_t joint = 0; joint < kDof; ++joint) {
        if (isActive(cfg_, joint)) {
          active_z2_norm2 += out.z2[joint] * out.z2[joint];
        }
      }
      theta_hat_ += kDt *
                    (cfg_.gamma_theta / robust_denominator * phi * phi *
                         active_z2_norm2 -
                     cfg_.sigma_theta * theta_hat_);
      theta_hat_ = clamp(theta_hat_, 0.0, cfg_.theta_max);
    }

    if (cfg_.enable_rl && cfg_.enable_learning &&
        time >= cfg_.learning_start_time) {
      // enable_rl 控制 actor/critic 是否进入控制律，enable_learning 控制权重
      // 是否更新；两个开关均为 true 且经过延迟后才执行在线学习。
      // 第一、二层使用同一算法，但拥有独立增益与 pseudo-Huber epsilon。
      out.sigma_g1 = updateLayer(s1, e1, ep1, lambda1,
                                 cfg_.gamma_a1,
                                 cfg_.gamma_c1, cfg_.weight_gain_a1,
                                 cfg_.weight_gain_c1, cfg_.leak_a1,
                                 cfg_.leak_c1, cfg_.varrho1,
                                 cfg_.bar_upsilon_a1,
                                 cfg_.pseudo_huber_eps1, layer1_);
      out.sigma_g2 = updateLayer(s2, e2, ep2, lambda2,
                                 cfg_.gamma_a2,
                                 cfg_.gamma_c2, cfg_.weight_gain_a2,
                                 cfg_.weight_gain_c2, cfg_.leak_a2,
                                 cfg_.leak_c2, cfg_.varrho2,
                                 cfg_.bar_upsilon_a2,
                                 cfg_.pseudo_huber_eps2, layer2_);
    }

    out.theta_hat = theta_hat_;
    // 非有限数值会由外层 checkSafety() 在本周期把命令清零并结束实验。
    out.finite = finiteOutput(out) && finiteWeights(layer1_) &&
                 finiteWeights(layer2_);
    return out;
  }

private:
  double updateLayer(const Feature &features, const Vec6 &bellman,
                     const Vec6 &actor_critic_error, const Mat6 &lambda,
                     const Vec6 &gamma_a, const Vec6 &gamma_c,
                     double weight_gain_a, double weight_gain_c,
                     double leak_a, double leak_c, double varrho,
                     double bar_upsilon_a,
                     double pseudo_huber_epsilon, LayerWeights &layer) {// 更新单层网络权重
    // 论文在线学习律的 1 kHz Euler 离散实现，对应：
    //   Sbar = S/(S^T S+delta_i)                    [eq:Sbar_def]
    //   dc0   = -gamma_ci*Omega_ci*Lambda_ci*e_i   [eq:dc0_output]
    //   da0   = dc0-gamma_ai*e_pi                  [eq:da0_output]
    //   sigma = 1/sqrt(1+epsilon_gi^2*||[e_i;e_pi]||^2) [pseudo-Huber]
    //   Wdot_c/Wdot_a                              [eq:Wc/Wa_update_output]
    // 其中 features 是 S，bellman 是 e_i，actor_critic_error 是 e_pi。
    // 原参考 PPC 不含学习律；这是替换控制器后新增的核心代码之一。
    double feature_norm2 = 0.0;
    for (double value : features) {
      feature_norm2 += value * value;
    }
    const double feature_denominator = feature_norm2 + cfg_.delta_norm;
    // 后续 normalized=features/feature_denominator 即论文的 Sbar。

    Vec6 dc0{};
    Vec6 da0{};
    // 完整矩阵 Omega=(Lambda Lambda^T+delta_ci I)^-1。使用 Cholesky 解
    // 线性方程而不显式求逆；第一层 Lambda 为对角，第二层随 M_AA 耦合。
    Mat6 learning_metric{};
    Vec6 lambda_residual = mat6Vector(lambda, bellman);
    for (std::size_t row = 0; row < kDof; ++row) {
      if (!isActive(cfg_, row)) {
        continue;
      }
      for (std::size_t column = 0; column < kDof; ++column) {
        if (!isActive(cfg_, column)) {
          continue;
        }
        for (std::size_t residual = 0; residual < kDof; ++residual) {
          if (isActive(cfg_, residual)) {
            learning_metric[row][column] +=
                lambda[row][residual] * lambda[column][residual];
          }
        }
      }
      learning_metric[row][row] += cfg_.delta_ci;
    }
    Vec6 normalized_bellman_direction{};
    if (!solveActiveSpd(learning_metric, lambda_residual, cfg_,
                        normalized_bellman_direction)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    for (std::size_t active = 0; active < kControlledDof; ++active) {
      const std::size_t joint = kControlledJoints[active];
      dc0[joint] =
          -gamma_c[joint] * normalized_bellman_direction[joint];
      da0[joint] = dc0[joint] -
                   gamma_a[joint] * actor_critic_error[joint];
    }
    const double sigma_g = pseudoHuberResidualFactor(
        bellman, actor_critic_error, pseudo_huber_epsilon, cfg_);//计算伪Huber残差因子，论文式(59)的实现
    // sigma_g 属于 (0,1]：残差小时接近普通梯度，Bellman/actor 误差异常大时
    // 自动压缩本周期学习步长。它只修正学习更新，不直接缩放控制力矩。

    for (std::size_t active = 0; active < kControlledDof; ++active) {
      const std::size_t joint = kControlledJoints[active];
      // 论文式(63)中的 S*S^T*W 必须使用原始 W^T*S，而不是控制通道可能
      // 经过 actor_limit/方向投影后的值。节点循环开始前固定计算，等价于
      // 仿真的向量化同步更新。
      const double raw_actor_value =
          featureDot(layer.actor[active], features);
      // 论文 eq:Upsilon_ai_def：Upsilon_ai=(1/varrho_i+bar_upsilon_ai)I。
      const double effective_upsilon = 1.0 / varrho + bar_upsilon_a;
      for (std::size_t feature = 0; feature < kRbfNodes; ++feature) {
        const double normalized = features[feature] / feature_denominator;
        layer.critic[active][feature] +=
            kDt * weight_gain_c *
            (normalized * sigma_g * dc0[joint] -
             leak_c * layer.critic[active][feature]);
        const double actor_dissipation =
            effective_upsilon * features[feature] * raw_actor_value;
        // actor_dissipation 对应论文 actor 耗散项；leak_a/leak_c 是权重泄漏，
        // 抑制无持续激励时的权重漂移。网络只存储关节 4~6 的三行权重。
        layer.actor[active][feature] +=
            kDt * weight_gain_a *
            (normalized * sigma_g * da0[joint] - actor_dissipation -
             leak_a * layer.actor[active][feature]);
      }
    }
    return sigma_g;//返回伪Huber残差因子，作为更新信号参数使用
  }

  static bool finiteWeights(const LayerWeights &layer) {
    // 在线学习最危险的失效形式之一是权重 NaN/Inf，必须逐元素检查。
    for (std::size_t active = 0; active < kControlledDof; ++active) {
      for (std::size_t feature = 0; feature < kRbfNodes; ++feature) {
        if (!isFinite(layer.actor[active][feature]) ||
            !isFinite(layer.critic[active][feature])) {
          return false;
        }
      }
    }
    return true;
  }

  static bool finiteOutput(const ControllerOutput &output) {// 检查控制器输出中间量是否有限，避免非有限数值发给机械臂。
    if (!isFinite(output.theta_hat) || !isFinite(output.sigma_g1) ||
        !isFinite(output.sigma_g2) || !output.mass_matrix_valid) {
      return false;
    }
    const std::array<const Vec6 *, 11> vectors{
        &output.desired.q, &output.desired.dq, &output.error,
        &output.z1,        &output.z2,         &output.alpha,
        &output.alpha_dot, &output.actor1,     &output.actor2,
        &output.torque_raw, &output.mass_matrix_diag};
    for (const Vec6 *values : vectors) {
      for (double value : *values) {
        if (!isFinite(value)) {
          return false;
        }
      }
    }
    return true;
  }

  const Config &cfg_;
  Vec6 q_init_{};
  LayerWeights layer1_{};
  LayerWeights layer2_{};
  Vec6 previous_alpha_{};
  Vec6 alpha_dot_filtered_{};
  double theta_hat_ = 0.0;
  bool alpha_initialized_ = false;
};

Vec6 limitTorque(const Vec6 &raw, const Vec6 &previous, double time,
                 const Config &cfg) {// 限幅力矩命令
  // 论文 tau 之外的最后一道命令整形，顺序为：启动半余弦斜坡 -> 固定三轴
  // 掩码 -> 每周期变化率限制 -> 力矩幅值限制。修改顺序会改变实际命令，
  // 因而这里明确保持与当前验证版本一致。
  double ramp = 1.0;
  if (cfg.command_ramp_time > kEps && time < cfg.command_ramp_time) {
    ramp = 0.5 * (1.0 - std::cos(kPi * time / cfg.command_ramp_time));
  }
  Vec6 output{};
  for (std::size_t joint = 0; joint < kDof; ++joint) {
    if (!isActive(cfg, joint)) {
      // 固定三轴版本在这里保证关节 1～3 始终发送 0 Nm。注意 0 Nm 通常
      // 不是位置保持；必须由厂家位置伺服、制动器或机械约束固定这些关节。
      output[joint] = 0.0;
      continue;
    }
    const double requested = ramp * raw[joint];
    const double step = cfg.tau_rate_limit[joint] * kDt;
    const double rate_limited =
        clamp(requested, previous[joint] - step, previous[joint] + step);
    output[joint] =
        clamp(rate_limited, -cfg.tau_limit[joint], cfg.tau_limit[joint]);
  }
  return output;
}

struct SafetyStatus {
  bool abort = false;
  std::string reason;
};

SafetyStatus checkSafety(const Vec6 &dq, const ControllerOutput &control,
                         const Config &cfg, bool state_read_ok) {// 检查安全性，返回是否需要中止实验
  // 与原参考相同，实时回调内在发出本周期命令前做软件保护；本版新增网络
  // 有限性与输出 barrier_ratio 检查。检查故意遍历全部 6 轴，而非只检查
  // 6 轴，避免未驱动的关节 1～3 运动失控却被忽略。
  if (!state_read_ok) {
    return {true, "实时状态读取失败"};
  }
  if (!control.finite) {
    return {true, "控制器或神经网络出现非有限数值"};
  }
  for (std::size_t joint = 0; joint < kDof; ++joint) {
    if (cfg.inactive_joints_fixed && !isActive(cfg, joint) &&
        cfg.inactive_joint_drift_limit[joint] > kEps &&
        std::abs(control.error[joint]) >
            cfg.inactive_joint_drift_limit[joint]) {
      return {true, "固定关节 " + std::to_string(joint + 1) +
                        " 偏离 q_init，M_AA 固定关节假设失效"};
    }
    if (std::abs(control.error[joint]) > cfg.abort_error[joint]) {
      return {true, "关节 " + std::to_string(joint + 1) + " 跟踪误差超限"};
    }
    if (std::abs(dq[joint]) > cfg.abort_velocity[joint]) {
      return {true, "关节 " + std::to_string(joint + 1) + " 速度超限"};
    }
    if (cfg.enable_output_barrier &&
        std::abs(control.barrier_ratio[joint]) > cfg.barrier_abort_ratio) {
      return {true, "关节 " + std::to_string(joint + 1) + " 接近输出约束边界"};
    }
  }
  return {};
}

void writeLogHeader(std::ofstream &log) {// 写入 CSV 文件表头
  // CSV 同时记录论文中间变量与“限幅前/后”力矩，便于区分控制律本身和
  // 工程保护造成的差异。barrier_ratio 是 zeta，不是 z1。
  log << "time";
  const std::array<std::string, 11> names{
      "q", "qd", "dq", "error", "z1", "z2", "alpha", "alpha_dot",
      "tau_raw", "tau_cmd", "mass_diag"};
  for (const auto &name : names) {
    for (std::size_t joint = 0; joint < kDof; ++joint) {
      log << ',' << name << (joint + 1);
    }
  }
  for (std::size_t joint = 0; joint < kDof; ++joint) {
    log << ",barrier_ratio" << (joint + 1);
  }
  log << ",theta_hat,sigma_g1,sigma_g2,abort\n";
}

void writeLogRow(std::ofstream &log, double time, const Vec6 &q,
                 const Vec6 &dq, const ControllerOutput &control,
                 const Vec6 &tau_cmd, bool abort) {// 写入 CSV 文件一行数据
  log << std::setprecision(10) << time;
  const std::array<const Vec6 *, 11> vectors{
      &q,
      &control.desired.q,
      &dq,
      &control.error,
      &control.z1,
      &control.z2,
      &control.alpha,
      &control.alpha_dot,
      &control.torque_raw,
      &tau_cmd,
      &control.mass_matrix_diag};
  for (const Vec6 *values : vectors) {
    for (double value : *values) {
      log << ',' << value;
    }
  }
  for (double value : control.barrier_ratio) {
    log << ',' << value;
  }
  log << ',' << control.theta_hat << ',' << control.sigma_g1 << ','
      << control.sigma_g2 << ',' << (abort ? 1 : 0) << '\n';
}

void throwIfError(const std::error_code &ec, const std::string &operation) {
  if (ec) {
    throw std::runtime_error(operation + "失败: " + ec.message());
  }
}

#ifndef OPTIMAL_BACKSTEPPING_OFFLINE_TEST
bool runTorqueExperiment(rokae::xMateRobot &robot, const Config &cfg) {
  // 保留原参考项目的 xCore 实验骨架：状态订阅 -> MoveJ 准备位姿 -> SDK
  // 滤波 -> 力矩模式 -> 1 kHz 回调。主要替换发生在 controller.compute()。
  using namespace rokae;
  using namespace rokae::RtSupportedFields;

  auto rt_controller = robot.getRtMotionController().lock();
  if (!rt_controller) {
    throw std::runtime_error("无法获取实时运动控制器");
  }

  std::error_code ec;
  robot.stopReceiveRobotState();
  // 当前控制律只需 q、dq；原 PPC 中用于 Y0/动力学项的 ddq 在这里不再订阅，
  // 降低数据依赖。alpha_dot 由控制器内部脏微分得到。
  robot.startReceiveRobotState(std::chrono::milliseconds(1),
                               {jointPos_m, jointVel_m});

  if (cfg.perform_ready_move) {
    ec.clear();
    const Vec6 current_q = robot.jointPos(ec);
    throwIfError(ec, "读取 MoveJ 起点");
    std::cout << "MoveJ 到已确认的准备位姿，速度系数="
              << cfg.ready_move_speed << std::endl;
    rt_controller->MoveJ(cfg.ready_move_speed, current_q, cfg.ready_q);
  } else {
    // 排查模式默认走此分支：避免把占位 ready_q=0 当成真实目标运动。
    // 随后重新实测的 q_init 会成为静止参考和输出约束中心。
    std::cout << "perform_ready_move=false：跳过 MoveJ，使用当前姿态开始静止参考测试。"
              << std::endl;
  }

  ec.clear();
  const Vec6 q_init = robot.jointPos(ec);
  throwIfError(ec, "读取实时控制初始位姿");
  // 必须在 MoveJ 完成后读取 q_init；它同时作为轨迹基准与输出约束中心基准。
  OptimalBacksteppingController controller(cfg, q_init);

  if (!rt_controller->setFilterLimit(true, cfg.sdk_filter_cutoff_hz)) {
    throw std::runtime_error("SDK 力矩滤波参数设置失败");
  }

  std::ofstream log("optimal_backstepping_log.csv", std::ios::trunc);
  if (!log.is_open()) {
    throw std::runtime_error("无法创建 optimal_backstepping_log.csv");
  }
  writeLogHeader(log);

  rt_controller->startMove(RtControllerMode::torque);
  std::cout << "已进入 1 kHz 实时力矩模式。" << std::endl;

  double time = 0.0;
  std::size_t tick = 0;
  Vec6 previous_torque{};
  bool experiment_aborted = false;
  std::string abort_reason;
  const std::size_t log_decimation =
      static_cast<std::size_t>(std::llround(cfg.log_decimation));

  std::function<Torque(void)> callback = [&]() -> Torque {// 实时回调函数，每周期执行一次，进入cmd的接口
    // 实时路径不要加入文件解析、动态模型加载或阻塞操作。这里每周期只读取
    // 状态、算控制、限幅、安全检查、按抽取率记录，并返回六轴 Torque。
    time += kDt;
    ++tick;

    Vec6 q{};
    Vec6 dq{};
    const bool state_read_ok =
        robot.getStateData(jointPos_m, q) == 0 &&
        robot.getStateData(jointVel_m, dq) == 0;

    const ControllerOutput control = controller.compute(time, q, dq);// 调用控制器计算输出
    // compute() 的 torque_raw 是论文/名义映射输出；只有 limitTorque 后的
    // torque_cmd 才会写入 SDK。调试时应同时比较日志中的两者。
    Vec6 torque_cmd = limitTorque(control.torque_raw, previous_torque, time, cfg);
    SafetyStatus safety = checkSafety(dq, control, cfg, state_read_ok);
    if (safety.abort) {
      // 软件停机发生时本周期立即发 0 Nm，并用 setFinished() 结束实时循环。
      // 这不能替代机械急停；0 Nm 也不等于机械抱闸。
      torque_cmd.fill(0.0);
      experiment_aborted = true;
      abort_reason = safety.reason;
    }
    previous_torque = torque_cmd;

    if (tick % log_decimation == 0 || safety.abort) {
      writeLogRow(log, time, q, dq, control, torque_cmd, safety.abort);
    }
    if (tick % 500 == 0) {
      std::cout << "t=" << time << " e456=[" << control.error[3] << ", "
                << control.error[4] << ", " << control.error[5]
                << "] tau456=[" << torque_cmd[3] << ", " << torque_cmd[4]
                << ", " << torque_cmd[5] << "] theta="
                << control.theta_hat << std::endl;
      log.flush();
    }

    Torque command(kDof, 0.0);
    for (std::size_t joint = 0; joint < kDof; ++joint) {
      command.tau[joint] = torque_cmd[joint];
    }
    if (safety.abort || time >= cfg.total_time) {
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
    std::cout << "实验按设定时长结束。日志: optimal_backstepping_log.csv"
              << std::endl;
  }
  return !experiment_aborted;
}
#endif

void printUsage(const char *program) {// 打印命令行用法
  std::cout
      << "用法:\n"
      << "  " << program << " --check-config [cfg路径]\n"
      << "  " << program
      << " --run [cfg路径] [机器人IP] [本机实时网卡IP]\n\n"
      << "无参数运行只检查默认配置，不连接机械臂。真机还要求 cfg 中的\n"
      << "hardware_enable 和 ready_pose_confirmed 同时为 true。" << std::endl;
}

} // namespace

#ifndef OPTIMAL_BACKSTEPPING_OFFLINE_TEST
int main(int argc, char *argv[]) {
  // 默认或 --check-config 只做离线解析，绝不连接机械臂；只有显式 --run
  // 才可能进入下面的连接流程。这是相对原参考程序新增的第一层解锁。
  const std::string command = argc > 1 ? argv[1] : "--run";// 命令行参数 from --check-config to --run
  if (command != "--check-config" && command != "--run") {
    printUsage(argv[0]);
    return 2;
  }

  const std::string config_path = argc > 2 ? argv[2] : defaultConfigPath();
  Config cfg;
  try {
    cfg = loadConfig(config_path);
    if (command == "--run") {
      if (argc > 3) {
        cfg.robot_ip = argv[3];
      }
      if (argc > 4) {
        cfg.local_ip = argv[4];
      }
      validateConfig(cfg);
    }
    printConfigSummary(cfg, config_path);
  } catch (const std::exception &exception) {
    std::cerr << "配置检查失败: " << exception.what() << std::endl;
    return 1;
  }

  if (command == "--check-config") {
    if (argc == 1) {
      printUsage(argv[0]);
    }
    return 0;
  }

  if (!cfg.hardware_enable || !cfg.ready_pose_confirmed) {
    // 第二层解锁：即使命令行给了 --run，CFG 中也必须同时确认硬件运行与
    // 准备位姿。两项默认均为 false，避免编译后误执行即上电。
    std::cerr << "拒绝连接机械臂：hardware_enable 与 ready_pose_confirmed "
                 "必须同时为 true。"
              << std::endl;
    return 3;
  }

  std::cout << "即将连接真实机械臂 " << cfg.robot_ip << "（本机 "
            << cfg.local_ip << "）。请确认急停可用且工作区无人。" << std::endl;

  try {
    std::error_code ec;
    rokae::xMateRobot robot(cfg.robot_ip, cfg.local_ip);
    bool entered_automatic = false;
    bool entered_realtime = false;
    bool powered = false;

    auto cleanup = [&]() {
      // 保留并增强原参考的退出恢复：尽最大努力切回非实时、按配置下电、
      // 回到手动状态；清理失败只报警，不掩盖主实验异常。
      if (entered_realtime) {
        ec.clear();
        robot.setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec);
        if (ec) {
          std::cerr << "警告：切回非实时模式失败: " << ec.message()
                    << std::endl;
        }
      }
      if (powered && cfg.power_off_on_exit) {
        ec.clear();
        robot.setPowerState(false, ec);
        if (ec) {
          std::cerr << "警告：下电失败: " << ec.message() << std::endl;
        }
      }
      if (entered_automatic) {
        ec.clear();
        robot.setOperateMode(rokae::OperateMode::manual, ec);
        if (ec) {
          std::cerr << "警告：切回手动模式失败: " << ec.message()
                    << std::endl;
        }
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

      const bool success = runTorqueExperiment(robot, cfg);
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
