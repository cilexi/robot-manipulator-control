/**
 * @file torque_control_ppc.cpp
 * @brief 实时模式 - 基于虚拟参考轨迹的预设性能控制器 (PPC)
 * 此示例根据提供的PPC算法论文和MATLAB代码进行了修改.
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
  std::ofstream ofs1("q_ppc.txt");
  std::ofstream ofs2("qerror_ppc.txt");
  std::ofstream ofs3("tau_ppc.txt");
  std::ofstream ofs4("yd_ppc.txt");
  std::ofstream ofs5("yc_ppc.txt");


  // --- 1. 控制器参数初始化 ---
  // 预设性能部分 (PPC)
  const double c1 = 3.0;
  const double c2 = 3.0;
  const double lambda1 = 0.01;
  const double lambda2 = 0.01;
  const double lambda3 = 0.01;
  const double d = 0.005;
  const double l = 10.0;
  const double bf = sqrt(d*d / (l + d*d));
  const double sigma = 0.001;

  // PD控制器部分
  Eigen::Matrix<double, 6, 6> Kp = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> Kd = Eigen::Matrix<double, 6, 6>::Zero();
  Kp.diagonal() << 40,400,30,35,20,0; // 根据实际情况调整
  Kd.diagonal() << 15,5,2.5,2.5,2.5,0; // 根据实际情况调整

  // 自适应部分
  Eigen::Matrix<double, 6, 6> Gamma = Eigen::Matrix<double, 6, 6>::Identity() * 50.0;

  // --- 2. 状态变量和初始条件 ---
  static bool init = true;
  Eigen::Matrix<double, 6, 1> q_init;
  Eigen::Matrix<double, 6, 1> yc;

  // 自适应参数初始化
  Eigen::Matrix<double, 6, 6> K1_hat = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> K2_hat = Eigen::Matrix<double, 6, 6>::Zero();

  
  // 用于计算数值微分
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
              0.2 + 0.1*cos(0),
              q_init(2), // 假设第3-6轴保持不动
              q_init(3),
              q_init(4),
              q_init(5);
        init = false;
    }

    // --- I. 期望轨迹 (yd, dyd, ddyd) ---
    Eigen::Matrix<double, 6, 1> yd, dyd, ddyd;
    yd << -0.15 - 0.1*cos(4*time),
           0.2 + 0.1*cos(2*time),
           q_init(2), q_init(3), q_init(4), q_init(5);

    dyd << 0.4*sin(4*time),
          -0.2*sin(2*time),
           0, 0, 0, 0;

    ddyd << 1.6*cos(4*time),
           -0.4*cos(2*time),
            0, 0, 0, 0;

    // --- II. 预设性能坐标变换 (基于 yd) ---
    Eigen::Matrix<double, 6, 1> e1 = qm - yd;
    Eigen::Matrix<double, 6, 1> e2 = dqm - dyd;

    double beta = 1.0 / ((1.0 - bf) * exp(-1.5 * time) + bf);
    double dbeta = 1.5 * (1.0 - bf) * exp(-1.5 * time) / pow(((1.0 - bf) * exp(-1.5 * time) + bf), 2);

    Eigen::Matrix<double, 6, 1> eta = e1.array() / (e1.array().square() + l).sqrt();
    Eigen::Matrix<double, 6, 1> zeta = beta * eta;

    // 防止分母为0
    for(int j=0; j<6; ++j) {
        if (std::abs(1.0 - zeta(j)*zeta(j)) < 1e-9) {
            zeta(j) = std::copysign(1.0 - 1e-9, zeta(j));
        }
    }
    Eigen::Matrix<double, 6, 1> z1 = zeta.array() / (1.0 - zeta.array().square());

    Eigen::Matrix<double, 6, 1> mu = (1.0 + zeta.array().square()) / (1.0 - zeta.array().square()).square();
    Eigen::Matrix<double, 6, 1> rho = l / ((e1.array().square() + l).sqrt() * (e1.array().square() + l));

    Eigen::Matrix<double, 6, 1> mu1 = mu.array() * beta * rho.array();
    Eigen::Matrix<double, 6, 1> mu2 = mu.array() * dbeta * eta.array();

    // Step 1: 虚拟控制器 alpha1
    // 防止 mu1 中有零元素导致除法错误
     for(int j=0; j<6; ++j) {
        if (std::abs(mu1(j)) < 1e-9) {
            mu1(j) = 1e-9;
        }
    }
    Eigen::Matrix<double, 6, 1> alpha1 = dyd.array() - mu2.array() / mu1.array() - c1 * z1.array() / mu1.array();

    // Step 2: 误差 z2
    Eigen::Matrix<double, 6, 1> z2 = dqm - alpha1;

    // --- III. 内部控制器 uppc 和自适应律 ---
    // 计算 dot_alpha1 (数值微分)
    Eigen::Matrix<double, 6, 1> dot_alpha1;
    if (time <= step) {
        dot_alpha1.setZero();
    } else {
        dot_alpha1 = (alpha1 - alpha1_prev) / step;
    }
    alpha1_prev = alpha1;

    // 定义 psi^2 用于鲁棒项
    double phi = 1.0 + e2.norm() + pow(e2.norm(), 2);
    double psi_sq = phi * phi;

    // 计算内部控制器 uppc, Eq. (20a)
    Eigen::Matrix<double, 6, 1> uppc = -c2 * z2.array() - lambda1 * z2.array() * psi_sq
                                     - lambda2 * mu1.array().square() * z1.array().square() * z2.array()
                                     - lambda3 * z2.array() * dot_alpha1.array().square();

    // 计算自适应律, Eq. (20b), (20c)
    Eigen::Matrix<double, 6, 6> dot_K1_hat = -Gamma * z2 * (yd - yc).transpose() - sigma * Gamma * K1_hat;
    Eigen::Matrix<double, 6, 6> dot_K2_hat = -Gamma * z2 * uppc.transpose() - sigma * Gamma * K2_hat;

    K1_hat += dot_K1_hat * step;
    K2_hat += dot_K2_hat * step;

    // --- IV. 虚拟参考轨迹 yc ---
    // 计算 yc 的导数 dyc, Eq. (15)
    Eigen::Matrix<double, 6, 1> dyc = K1_hat * (yd - yc) + dyd + K2_hat * uppc;

    // 更新 yc (欧拉积分)
    yc += dyc * step;

    // --- V. 最终PD控制器 u ---
    // 计算相对于虚拟轨迹的误差
    Eigen::Matrix<double, 6, 1> e_c1 = qm - yc;
    Eigen::Matrix<double, 6, 1> e_c2 = dqm - dyc;

    // 计算最终的PD控制律 u, Eq. (13)
    Eigen::VectorXd tau_d = -Kp * e_c1 - Kd * e_c2;

    // --- VI. 数据记录 ---
    ofs1 << time << " " << qm.transpose() << std::endl;
    ofs2 << time << " " << e1.transpose() << std::endl;
    ofs3 << time << " " << tau_d.transpose() << std::endl;
    ofs4 << time << " " << yd.transpose() << std::endl;
    ofs5 << time << " " << yc.transpose() << std::endl;

    Torque cmd(6);
    Eigen::VectorXd::Map(cmd.tau.data(), 6) = tau_d;

    // 仿真结束条件
    if(time > 10.0){ // 仿真持续10秒
        ofs1.close();
        ofs2.close();
        ofs3.close();
        ofs4.close();
        ofs5.close();
        print(std::cout, "PPC力矩控制结束");
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