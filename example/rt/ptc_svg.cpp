
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
#include <sstream>   // 用于字符串流
#include <string>    // 用于字符串处理

bool loadPathFromCSV(const std::string& filename, std::vector<Eigen::Vector3d>& path_waypoints) {
    // 创建一个输入文件流
    std::ifstream file(filename);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开文件 " << filename << std::endl;
        return false;
    }

    // 清空向量，以防里面有旧数据
    path_waypoints.clear();

    std::string line;
    bool is_header = true; // 用于跳过第一行（表头）

    // 逐行读取文件
    while (std::getline(file, line)) {
        // 如果是第一行（表头），则跳过
        if (is_header) {
            is_header = false;
            continue;
        }

        // 使用字符串流来解析逗号分隔的值
        std::stringstream ss(line);
        std::string value;
        double x, y,k=1;

        // 读取第一个值（x坐标）
        if (std::getline(ss, value, ',')) {
            try {
                x = k*std::stod(value);
            } catch (const std::invalid_argument& e) {
                std::cerr << "警告: 无效的x值，跳过此行: " << line << std::endl;
                continue;
            }
        }

        // 读取第二个值（y坐标）
        if (std::getline(ss, value, ',')) {
            try {
                y = k*std::stod(value);
            } catch (const std::invalid_argument& e) {
                std::cerr << "警告: 无效的y值，跳过此行: " << line << std::endl;
                continue;
            }
        }

        // 将读取到的2D坐标存为3D坐标（Z默认为0），并添加到向量中
        path_waypoints.push_back(Eigen::Vector3d(x, y, 0.0));
    }

    file.close();

    // 检查是否成功读取到了任何点
    if (path_waypoints.empty()) {
        std::cerr << "警告: 文件为空或格式错误，没有加载到任何路径点。" << std::endl;
        return false;
    }

    return true;
}
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
    double Tf = 16;
    double T = Tf-1;
    double theta = 0;
    double rho =0.1;
    //定义中间向量变量
    Eigen::Matrix<double, 6, 1> Kp = {150,200,45,100,35,0};
    Eigen::Matrix<double, 6, 1> Kd = {30,20,10,10,5,0};
    //Eigen::Matrix<double, 6, 1> Kp = {40,400,30,35,20,0};
    //Eigen::Matrix<double, 6, 1> Kd = {15,5,2.5,2.5,2.5,0};
    Eigen::Matrix<double, 6, 1> k = {0.35,0.6,0.6,0.05,0.08,0};
    Eigen::Matrix<double, 6, 1> lambda = {0.05,
                                          0.05,
                                          0.05,
                                          0.05,
                                          0.01,
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

    std::vector<Eigen::Vector3d> path_waypoints;
    // 确保您的CSV文件与程序在同一个工作目录，或使用绝对路径
    std::string csv_filepath = "xuanwo.csv"; 

    // 调用加载函数
    if (!loadPathFromCSV(csv_filepath, path_waypoints)) {
        std::cerr << "错误: 未能从CSV加载路径点，程序终止。" << std::endl;
        return; // 加载失败，直接退出函数
    }
    
    // 检查加载后的点是否需要加上机器人的初始偏移
    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(init_position.data()).transpose());
    Eigen::Vector3d start_position_offset = initial_transform.translation();
    for(auto& point : path_waypoints) {
        point += start_position_offset;
    }

    std::cout << "成功从 " << csv_filepath << " 加载了 " << path_waypoints.size() << " 个路径点。" << std::endl;
    // ####################################################################


    // 新增：预计算路径段长度和总长度
    std::vector<double> segment_lengths;
    double total_path_length = 0;
    // 注意：这里的循环上限是 path_waypoints.size() - 1
    if (path_waypoints.size() > 1) {
        for (size_t i = 0; i < path_waypoints.size() - 1; ++i) {
            double length = (path_waypoints[i + 1] - path_waypoints[i]).norm();
            segment_lengths.push_back(length);
            total_path_length += length;
        }
    } 

    // 新增：定义绘制整个路径的总时间
    double total_duration = T+1; // 我们使用您已有的时间 T
    std::function<Torque(void)> callback = [&]{
    using namespace RtSupportedFields;
    static double time=0;
    time += 0.001;
            // 计算当前角度 (使用五次多项式平滑启动和停止)
    double angle;
    Eigen::Vector3d position_d;
    Eigen::Vector3d velocity_d = Eigen::Vector3d::Zero();
        if (time < total_duration) {
        // 1. 使用五次多项式计算沿路径的期望进程（0到1），以确保平滑的时间规划
        double t_normalized = time / total_duration;
        double t2 = t_normalized * t_normalized;
        double t3 = t2 * t_normalized;
        double t4 = t3 * t_normalized;
        double t5 = t4 * t_normalized;
        
        // 五次多项式位置: 10t^3 - 15t^4 + 6t^5
        double path_ratio = 10 * t3 - 15 * t4 + 6 * t5;
        // 五次多项式速度: (30t^2 - 60t^3 + 30t^4) / total_duration
        double path_velocity_ratio = (30 * t2 - 60 * t3 + 30 * t4) / total_duration;

        // 2. 计算沿路径的期望距离和速度
        double distance_along_path = path_ratio * total_path_length;
        double speed_along_path = path_velocity_ratio * total_path_length;

        // 3. 找到机器人当前所在的路径段
        int current_segment_index = 0;
        double accumulated_length = 0;
        for (size_t i = 0; i < segment_lengths.size(); ++i) {
            if (distance_along_path <= accumulated_length + segment_lengths[i]) {
                current_segment_index = i;
                break;
            }
            accumulated_length += segment_lengths[i];
        }
        // 4. 通过在当前段内插值来计算位置
        double distance_into_segment = distance_along_path - accumulated_length;
        const Eigen::Vector3d& start_point = path_waypoints[current_segment_index];
        const Eigen::Vector3d& end_point = path_waypoints[current_segment_index + 1];

        Eigen::Vector3d segment_vector = end_point - start_point;
        // 处理段长度为零的情况，避免除以零
        if (segment_lengths[current_segment_index] > 1e-6) {
             Eigen::Vector3d segment_direction = segment_vector.normalized();
             position_d = start_point + segment_direction * distance_into_segment;
             velocity_d = segment_direction * speed_along_path;
        } else {
            position_d = start_point;
            velocity_d.setZero();
        }
    } else {
        // 轨迹结束后，保持在最终位置
        position_d = path_waypoints.back();
        velocity_d.setZero();
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
        //std::cout<<time<<" qd "<<q_d[0]<<" "<<q_d[1]<<" "<<q_d[2]<<" "<<q_d[3]<<" "<<q_d[4]<<" "<<q_d[5]<<std::endl;
        std::cout<<time<<" e "<<e1[0]<<" "<<e1[1]<<" "<<e1[2]<<" "<<e1[3]<<" "<<e1[4]<<" "<<e1[5]<<std::endl;
        std::cout<<time<<" de "<<e2[0]<<" "<<e2[1]<<" "<<e2[2]<<" "<<e2[3]<<" "<<e2[4]<<" "<<e2[5]<<std::endl;
        //std::cout<<time<<" pos "<<position[0]<<" "<<position[1]<<" "<<position[2]<<std::endl;
        tau_d <<0*gravity
                +(-(theta)*z-(psi*psi*lambda.array()*z.array()).matrix()-(k.array()*z.array()).matrix())
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