/**
 * @file torque_control_ppc_non_adaptive.cpp
 * @brief 实时模式 - 基于虚拟参考轨迹的预设性能控制器 (PPC) - 非自适应版本
 * 此版本移除了自适应律, K1和K2使用其理论值.
 * 多个控制器参数被向量化, 以便对每个关节进行独立调整.
 *
 * @copyright Copyright (C) 2024 ROKAE (Beijing) Technology Co., LTD. All Rights Reserved.
 * Information in this file is the intellectual property of Rokae Technology Co., Ltd,
 * And may contains trade secrets that must be stored and viewed confidentially.
 */

#include <iostream>
#include <cmath>
#include <thread>
#include "rokae/robot.h"
#include "Eigen/Geometry"
#include "../print_helper.hpp"
#include "rokae/utility.h"
#include <fstream>

using namespace rokae;

/**
 * @brief 力矩控制. 注意:
 * 1) 力矩值不要超过机型的限制条件(见手册);
 * 2) 初次运行时请手握急停开关, 避免机械臂非预期运动造成碰撞
 */
void torqueControl(xMateRobot &robot) {
  using namespace RtSupportedFields;
  auto rtCon = robot.getRtMotionController().lock();
  auto model = robot.model();
  error_code ec;

  // 拖拽位置
  std::array<double,6> q_drag = {0, M_PI/6, M_PI/3, 0, M_PI/2, 0};

  robot.stopReceiveRobotState();
  robot.startReceiveRobotState(std::chrono::milliseconds(1),
                               {jointPos_m, jointVel_m, jointAcc_c, tcpPose_m});

  // 运动到拖拽位置
  rtCon->MoveJ(0.5, robot.jointPos(ec), q_drag);

  // 控制模式为力矩控制
  rtCon->startMove(RtControllerMode::torque);

  // 文件流用于数据记录
  std::ofstream ofs1("q_ppc_non_adaptive.txt");
  std::ofstream ofs2("qerror_ppc_non_adaptive.txt");
  std::ofstream ofs3("tau_ppc_non_adaptive.txt");
  std::ofstream ofs4("yd_ppc_non_adaptive.txt");
  std::ofstream ofs5("yc_ppc_non_adaptive.txt");


  // --- 1. 控制器参数初始化 ---

  // PD控制器部分
  Eigen::Matrix<double, 6, 6> Kp = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> Kd = Eigen::Matrix<double, 6, 6>::Zero();
  Kp.diagonal() << 40,400,30,35,20,0; // 根据实际情况调整
  Kd.diagonal() << 5,5,2.5,2.5,2.5,0; // 根据实际情况调整

  Eigen::Matrix<double, 6, 6> K1 = Kd.inverse() * Kp;
  Eigen::Matrix<double, 6, 6> K2 = Kd.inverse();

  // 预设性能参数 (向量化)
  Eigen::Matrix<double, 6, 1> c1, c2, lambda1, lambda2, lambda3, l;

  // 为每个关节设置参数, 您可以在这里独立调整
  c1 << 3, 0,0,0,0,0;
  c2 << 3, 0,0,0,0,0;
  lambda1 << 0.01, 0,0,0,0,0;
  lambda2 << 0.01,0,0,0,0,0;
  lambda3 << 0.01,0,0,0,0,0;
  l << 1, 10.0, 10.0, 10.0, 10.0, 10.0;

  const double d = 0.02;
  const double bf = sqrt(d*d / (l(0) + d*d)); // bf通常是标量

  // --- 2. 状态变量和初始条件 ---
  static bool init = true;
  Eigen::Matrix<double, 6, 1> q_init;
  Eigen::Matrix<double, 6, 1> yc;
  Eigen::Matrix<double, 6, 1> alpha1_prev = Eigen::Matrix<double, 6, 1>::Zero();

  // --- 3. 控制回调函数 ---
  std::function<Torque(void)> callback = [&]{
    using namespace RtSupportedFields;
    static double time = 0;
    const double step = 0.001;
    time += step;

    // 获取机器人状态
    std::array<double, 6> q_array{}, dq_array{};
    robot.getStateData(jointPos_m, q_array);
    robot.getStateData(jointVel_m, dq_array);
    Eigen::Map<Eigen::Matrix<double, 6, 1>> qm(q_array.data());
    Eigen::Map<Eigen::Matrix<double, 6, 1>> dqm(dq_array.data());

    // 首次运行时初始化
    if (init) {
        q_init = qm;
        // 虚拟参考轨迹初始值设为期望轨迹的初始值
        yc << -0.15 - 0.1*cos(0),
              q_init(1),
              q_init(2), // 假设第3-6轴保持不动
              q_init(3),
              q_init(4),
              q_init(5);
        init = false;
    }

    // --- I. 期望轨迹 (yd, dyd, ddyd) ---
    Eigen::Matrix<double, 6, 1> yd, dyd, ddyd;
    yd <<  q_init(0)+0.3*std::cos(time),
           q_init(1),
           q_init(2), q_init(3), q_init(4), q_init(5);

    dyd << -0.3*std::sin(time),
           0,
           0, 0, 0, 0;

    ddyd << -0.3*std::cos(time),
           0,
            0, 0, 0, 0;

    // --- II. 预设性能坐标变换 (基于 yd) ---
    Eigen::Matrix<double, 6, 1> e1 = qm - yd; 
    Eigen::Matrix<double, 6, 1> e2 = dqm - dyd;

    double beta = 1.0 / ((1.0 - bf) * exp(-1.5 * time) + bf);
    double dbeta = 1.5 * (1.0 - bf) * exp(-1.5 * time) / pow(((1.0 - bf) * exp(-1.5 * time) + bf), 2);

    Eigen::Matrix<double, 6, 1> eta = e1.array() / (e1.array().square() + l.array()).sqrt();
    Eigen::Matrix<double, 6, 1> zeta = beta * eta; 
    // 防止分母为0
    for(int j=0; j<6; ++j) {
        if (std::abs(1.0 - zeta(j)*zeta(j)) < 1e-9) {
            zeta(j) = std::copysign(1.0 - 1e-9, zeta(j));
        }
    }
    Eigen::Matrix<double, 6, 1> z1 = zeta.array() / (1.0 - zeta.array().square());

    Eigen::Matrix<double, 6, 1> mu = (1.0 + zeta.array().square()) / (1.0 - zeta.array().square()).square();
    Eigen::Matrix<double, 6, 1> rho = l.array() / ((e1.array().square() + l.array()).sqrt() * (e1.array().square() + l.array()));

    Eigen::Matrix<double, 6, 1> mu1 = mu.array() * beta * rho.array();
    Eigen::Matrix<double, 6, 1> mu2 = mu.array() * dbeta * eta.array();

    for(int j=0; j<6; ++j) {
        if (std::abs(mu1(j)) < 1e-9) mu1(j) = 1e-9;
    }
    Eigen::Matrix<double, 6, 1> alpha1 = dyd.array() - mu2.array() / mu1.array() - c1.array() * z1.array() / mu1.array();

    // Step 2: 误差 z2
    Eigen::Matrix<double, 6, 1> z2 = dqm - alpha1; 

    // --- III. 内部控制器 uppc ---
    // 计算 dot_alpha1 (数值微分)
    Eigen::Matrix<double, 6, 1> dot_alpha1;
    if (time <= step) {
        dot_alpha1.setZero();
    } else {
        dot_alpha1 = (alpha1 - alpha1_prev) / step;
    }
    alpha1_prev = alpha1;

    // 定义 psi^2 用于鲁棒项
    double phi = 1.0 + dqm.norm()+ e1.norm()+e2.norm() + pow(dqm.norm(), 2);
    double psi_sq = phi * phi;

    // 计算内部控制器 uppc, Eq. (20a) [cite_start][cite: 82]
    Eigen::Matrix<double, 6, 1> uppc = -c2.array() * z2.array()
                                     - lambda1.array() * z2.array() * psi_sq
                                     - lambda2.array() * mu1.array().square() * z1.array().square() * z2.array()
                                     - lambda3.array() * z2.array() * dot_alpha1.array().square();

    // --- IV. 虚拟参考轨迹 yc ---
    // 计算 yc 的导数 dyc, Eq. (15)[cite_start], 使用真实的K1, K2 [cite: 28]
    Eigen::Matrix<double, 6, 1> dyc = K1 * (yd - yc) + dyd + K2 * uppc;

    // 更新 yc (欧拉积分)
    yc += dyc * step;

    // --- V. 最终PD控制器 u ---
    // 计算相对于虚拟轨迹的误差
    Eigen::Matrix<double, 6, 1> e_c1 = qm - yc;
    Eigen::Matrix<double, 6, 1> e_c2 = dqm - dyc;

    // 计算最终的PD控制律 u, Eq. (13) [cite_start][cite: 24]
    //Eigen::VectorXd tau_d = -Kp * e_c1 - Kd * e_c2;
    Eigen::VectorXd tau_d = uppc-Kp * e1 - Kd * e2;
    // --- VI. 数据记录 ---
    ofs1 << time << " " << qm.transpose() << std::endl;
    ofs2 << time << " " << e1.transpose() << std::endl;
    ofs3 << time << " " << tau_d.transpose() << std::endl;
    ofs4 << time << " " << yd.transpose() << std::endl;
    ofs5 << time << " " << yc.transpose() << std::endl;

    Torque cmd(6);
    Eigen::VectorXd::Map(cmd.tau.data(), 6) = tau_d;

    // 仿真结束条件
    if(time > 30.0){ // 仿真持续10秒
        ofs1.close();
        ofs2.close();
        ofs3.close();
        ofs4.close();
        ofs5.close();
        print(std::cout, "PPC (Non-Adaptive) Torque Control Finished");
        cmd.setFinished();
    }
    return cmd;
  };

  // 由于需要在callback里读取状态数据, useStateDataInLoop = true
  // 并且调用startReceiveRobotState()时, 设定的发送周期是1ms
  rtCon->setControlLoop(callback, 0, true);
  rtCon->startLoop(true);
}

/**
 * @brief main program
 */
int main() {
  try {
    std::string ip = "192.168.2.200";
    std::error_code ec;
    rokae::xMateRobot robot(ip, "192.168.2.100");
    robot.setOperateMode(rokae::OperateMode::automatic, ec);
    robot.setMotionControlMode(MotionControlMode::RtCommand, ec);
    robot.setPowerState(true, ec);
    try {
      torqueControl(robot);
    } catch (const rokae::RealtimeMotionException &e) {
      print(std::cerr, e.what());
      // 发生错误, 切换回非实时模式
      robot.setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec);
    }

    robot.setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec);
    robot.setOperateMode(rokae::OperateMode::manual, ec);

  } catch (const std::exception &e) {
    print(std::cerr, e.what());
  }
  return 0;
}