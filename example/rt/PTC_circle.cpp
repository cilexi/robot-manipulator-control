
/**
 * @file torque_control.cpp
 * @brief 实时模式 - 直接力矩控制
 * 此示例需要使用xMateModel模型库，请设置编译选项XCORE_USE_XMATE_MODEL=ON
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

// mu
double mu(double t, double Tf)
{   
    double result;
    double mu1 = Tf/(Tf-t);
    result = pow(mu1,2);
    return result;
}

double dmu(double t, double Tf)
{   
    double result;
    double mu1 = Tf/(Tf-t);
    result = 2*pow(mu1,3)/Tf;
    return result;
}


struct TrajectoryPoint {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
};

// 五次多项式插值函数
TrajectoryPoint quinticInterpolation(const Eigen::Vector3d& start, 
                                     const Eigen::Vector3d& end, 
                                     double t, double T) {
    TrajectoryPoint point;
    
    // 确保t在[0, T]范围内
    t = std::clamp(t, 0.0, T);
    
    // 计算归一化时间
    double tau = t / T;
    double tau2 = tau * tau;
    double tau3 = tau2 * tau;
    double tau4 = tau3 * tau;
    double tau5 = tau4 * tau;
    
    // 五次多项式系数
    double a0 = 0;
    double a1 = 0;
    double a2 = 0;
    double a3 = 10;
    double a4 = -15;
    double a5 = 6;
    
    // 位置计算
    point.position = start + (end - start) * (a3 * tau3 + a4 * tau4 + a5 * tau5);
    
    // 速度计算 (一阶导数)
    point.velocity = (end - start) * (3*a3*tau2 + 4*a4*tau3 + 5*a5*tau4) / T;
    
    // 加速度计算 (二阶导数)
    point.acceleration = (end - start) * (6*a3*tau + 12*a4*tau2 + 20*a5*tau3) / (T*T);
    
    return point;
}
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

  //std::array<double,6> q_drag = {0, M_PI/6,M_PI/3, 0, M_PI/2, 0 };
  std::array<double,6> q_drag = {0, M_PI/6,-M_PI/2, 0, -M_PI/3, 0 };

  // std::array<double,7> q_drag = {0,-0.3,0,M_PI/2,0,M_PI/2,0};

  robot.stopReceiveRobotState();
  robot.startReceiveRobotState(std::chrono::milliseconds(1),
                               {jointPos_m, jointVel_m, jointAcc_c, tcpPose_m});

  // 运动到拖拽位置
   rtCon->MoveJ(0.8, robot.jointPos(ec), q_drag);

  // 控制模式为力矩控制
  rtCon->startMove(RtControllerMode::torque);


  // std::array<double, 16> init_position {};

  // Utils::postureToTransArray(robot.posture(rokae::CoordinateType::flangeInBase, ec), init_position);

    std::ofstream ofs1;
    std::ofstream ofs2;
    std::ofstream ofs3;
    std::ofstream ofs4;
    std::ofstream ofs5;
    std::ofstream ofs6;
    std::ofstream ofs7;

    ofs1.open("q_pd.txt", std::ios::out | std::ios::trunc);
    ofs2.open("qerror_pd.txt", std::ios::out | std::ios::trunc);
    ofs3.open("tau_pd.txt",std::ios::out | std::ios::trunc);
    ofs4.open("yd.txt",std::ios::out | std::ios::trunc);
    ofs5.open("pose_pd.txt",std::ios::out | std::ios::trunc);
    ofs6.open("pose_d.txt",std::ios::out | std::ios::trunc);
    //ofs5.open("pose_d.txt",std::ios::out | std::ios::trunc);
    static bool init = true;
    std::array<double, 6> q_init;
    std::array<double, 6> q_out;
    std::array<double, 6> q_out_1;
    std::array<double, 6> ddq_out = {0, 0, 0, 0, 0, 0 };
    std::array<double, 16> init_position {};
    Utils::postureToTransArray(robot.posture(rokae::CoordinateType::flangeInBase, ec), init_position);


    double psi = 0;
    double phi_f=0;
    double beta = 5;
    double Tf = 8.5;
    double T = Tf-0.5;
    double theta = 0;
    double rho =0.1;
    //定义中间向量变量
     Eigen::Matrix<double, 6, 1> Kp = {200,200,45,90,40,0};
     Eigen::Matrix<double, 6, 1> Kd = {50,20,10,10,10,0};
    //Eigen::Matrix<double, 6, 1> Kp = {40,400,30,35,20,0};
    //Eigen::Matrix<double, 6, 1> Kd = {15,5,2.5,2.5,2.5,0};
    Eigen::Matrix<double, 6, 1> k = {0.35,0.6,0.6,0.05,0.125,0};
    Eigen::Matrix<double, 6, 1> lambda = {0.1,
                                          0.1,
                                          0.1,
                                          0.05,
                                          0.00000001,
                                          0};
    Eigen::Matrix<double, 6, 1> dyd;
    Eigen::Matrix<double, 6, 1> qd_ddot;
    Eigen::Matrix<double, 6, 1> ddqm;
    Eigen::Matrix<double, 6, 1> dqm_1;
    Eigen::Matrix<double, 6, 1> e1;
    Eigen::Matrix<double, 6, 1> e1_dot;
    Eigen::Matrix<double, 6, 1> error_dot;
    Eigen::Matrix<double, 6, 1> e2;
    Eigen::Matrix<double, 6, 1> omega_1;
    Eigen::Matrix<double, 6, 1> omega_2;
    Eigen::Matrix<double, 6, 1> z;

        Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(init_position.data()).transpose());
        Eigen::Vector3d start_position = initial_transform.translation();
        double radius = 0.1; // 半径 (米)
        Eigen::Vector3d center = start_position + 0*Eigen::Vector3d(radius, 0, 0); // 圆心位置
        //double angular_speed = 0.5; // 角速度 (弧度/秒)
        double circle_duration = T+1; // 完成一圈所需时间
        double angular_speed = 2*M_PI/circle_duration; // 角速度 (弧度/秒)
    std::function<Torque(void)> callback = [&]{
    using namespace RtSupportedFields;
    static double time=0;
    time += 0.001;
            // 计算当前角度 (使用五次多项式平滑启动和停止)
        double angle;
        if (time < circle_duration) {
            // 计算归一化时间 (0到1)
            double t_normalized = time / circle_duration;
            
            // 五次多项式参数
            double t2 = t_normalized * t_normalized;
            double t3 = t2 * t_normalized;
            double t4 = t3 * t_normalized;
            double t5 = t4 * t_normalized;
            
            // 五次多项式系数 (确保位置、速度、加速度连续)
            double a0 = 0;
            double a1 = 0;
            double a2 = 0;
            double a3 = 10;
            double a4 = -15;
            double a5 = 6;
            
            // 计算当前角度比例
            double angle_ratio = a3 * t3 + a4 * t4 + a5 * t5;
            
            // 计算当前角度 (0到2π)
            angle = angle_ratio * 2 * M_PI;
        } else {
            // 完成一圈后保持在终点位置
            angle = 2 * M_PI;
        }



    //Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(init_position.data()).transpose());
    std::array<double, 6> q_in{}, q{}, dq_m{}, ddq_c{};
    std::array<double, 16> pos_m {};

    if (init == true)
        {
            robot.getStateData(jointPos_m, q_in);
            q_init = q_in;
            init = false;
        }

    // 接收设置为true, 回调函数中可以直接读取
    robot.getStateData(jointPos_m, q);
    robot.getStateData(jointVel_m, dq_m);
    robot.getStateData(jointAcc_c, ddq_c);


    std::array<double, 6> gravity_array = model.getTorque(q, dq_m, ddq_c, TorqueType::gravity);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> gravity(gravity_array.data());
            // convert to Eige
    Eigen::Map<Eigen::Matrix<double, 6, 1>> qm(q.data());
    Eigen::Map<Eigen::Matrix<double, 6, 1>> dqm(dq_m.data());

    // 计算目标位置 (圆形轨迹)
    Eigen::Vector3d position_d;
    position_d.x() = center.x() + radius * cos(angle);
    position_d.y() = center.y() + radius * sin(angle);
    position_d.z() = center.z(); // Z轴保持不变
    
    // 计算目标速度 (用于前馈控制)
    Eigen::Vector3d velocity_d;
    velocity_d.x() = -radius * angular_speed * sin(angle);
    velocity_d.y() = radius * angular_speed * cos(angle);
    velocity_d.z() = 0;
    
    // 构建完整位姿 - 保持初始姿态
    static Eigen::Quaterniond initial_orientation(initial_transform.linear());
    Eigen::Affine3d target_pose = Eigen::Affine3d::Identity();
    target_pose.translation() = position_d;
    target_pose.linear() = initial_orientation.toRotationMatrix();
    
    // 转换为数组格式
    std::array<double, 16> target_pose_array;
    Eigen::Map<Eigen::Matrix4d>(target_pose_array.data()) = target_pose.matrix().transpose();
    
    // 使用逆运动学求解关节角度
    std::array<double, 6> q_target;
    bool ik_success = model.getJointPos(target_pose_array, 
                                       0, 
                                       q, // 当前关节角度作为初始猜测
                                       q_target);
    auto pos_e = model.getCartPose(q);
    //auto pos_d = model.getCartPose(q_target);
        // 将位姿转换为Eigen矩阵
    Eigen::Map<Eigen::Matrix4d> pose_map(pos_e.data());
    Eigen::Matrix4d pose_matrix = pose_map.transpose(); // 注意转置，因为API返回行优先
    
    // 提取位置信息
    Eigen::Vector3d position = pose_matrix.block<3,1>(0,3);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> q_d(q_target.data());
        Eigen::Map<Eigen::Matrix<double, 6, 1>> q_d_1(q_out_1.data());
        Eigen::Map<Eigen::Matrix<double, 6, 1>> qd_ddot(ddq_out.data());
        e1 = qm - q_d;
        if (time < 0.0012)
        {
            dyd.array() = 0;
        }
        else
        {
            dyd = (q_d - q_d_1) / 0.001;
        }
        q_out_1 = q_target;
        e2 = dqm - dyd;
        // compute control
        Eigen::VectorXd tau_d(6);

        theta = 4*rho/pow(Tf,4);
        omega_1 = mu(time,Tf)*e1;
        omega_2 = dmu(time,Tf)*e1+mu(time,Tf)*e2;
        z = omega_2+beta*omega_1;
        phi_f = 1+dqm.norm()+pow(dqm.norm(),2);
        psi = 1 + phi_f + pow(qd_ddot.norm(),2);
        //printf("pos_d -", pos_d);
        //std::cout<<time<<" q "<<qm[0]<<" "<<qm[1]<<" "<<qm[2]<<" "<<qm[3]<<" "<<qm[4]<<" "<<qm[5]<<std::endl;
        std::cout<<time<<" qd "<<q_d[0]<<" "<<q_d[1]<<" "<<q_d[2]<<" "<<q_d[3]<<" "<<q_d[4]<<" "<<q_d[5]<<std::endl;
        //std::cout<<time<<" e "<<e1[0]<<" "<<e1[1]<<" "<<e1[2]<<" "<<e1[3]<<" "<<e1[4]<<" "<<e1[5]<<std::endl;
        std::cout<<time<<" de "<<e2[0]<<" "<<e2[1]<<" "<<e2[2]<<" "<<e2[3]<<" "<<e2[4]<<" "<<e2[5]<<std::endl;
        //std::cout<<time<<" pos "<<position[0]<<" "<<position[1]<<" "<<position[2]<<std::endl;
        tau_d <<0*gravity
                +0*(-(theta)*z-(psi*psi*lambda.array()*z.array()).matrix()-(k.array()*z.array()).matrix())
                -((Kp.array()*e1.array()+Kd.array()*e2.array()).matrix());



        //std::cout<<time<<" "<<tau_d[0]<<" "<<tau_d[1]<<" "<<tau_d[2]<<" "<<tau_d[3]<<" "<<tau_d[4]<<" "<<tau_d[5]<<std::endl;
        
    Torque cmd(6);
    Eigen::VectorXd::Map(cmd.tau.data(), 6) = tau_d;

    ofs1 << time << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << " " << q[4] << " " << q[5] << " "  << std::endl;
    ofs2 << time << " " << e1[0] << " " << e1[1] << " " <<  e1[2] << " " <<  e1[3] << " " <<  e1[4] << " " <<0*e1[5] << " " << std::endl;
    ofs3 << time << " " << tau_d[0] << " " <<tau_d[1] << " " <<tau_d[2] << " " <<tau_d[3] << " " <<tau_d[4] << " " << tau_d[5] << " "  << std::endl;
    ofs4 << time << " " << q_out[0] << " " <<q_out[1] << " " <<q_out[2] << " " <<q_out[3] << " " <<q_out[4] << " " << q_out[5] << " "  << std::endl;
    ofs5 << time << " " << position[0] << " " <<position[1] << " " <<position[2]<<" "<< std::endl;
    ofs6 << time << " " << position_d[0] << " " <<position_d[1] << " " <<position_d[2]<<" "  << std::endl;
    if(time > T){
      ofs1.close();
      ofs2.close();
      ofs3.close();
      ofs4.close();
      ofs5.close();
      ofs6.close();
      ofs7.close();
      print(std::cout, "力矩控制结束");
      cmd.setFinished();
    }
    return cmd;
  };

  // 由于需要在callback里读取状态数据, 这里useStateDataInLoop = true
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
    rokae::xMateRobot robot(ip, "192.168.2.100"); // ****   xMate 7-axis
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