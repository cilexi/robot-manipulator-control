
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

double beta(double t, double bf)
{
  double result;
  result = (1-bf)*exp(1.5*t)+bf;
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

  // std::array<double,7> q_drag = {0, M_PI/6,M_PI/3, 0, M_PI/2, 0 };

  std::array<double,6> q_drag = {0, M_PI/6,M_PI/3, 0, M_PI/2, 0 };

  // std::array<double,7> q_drag = {0,-0.3,0,M_PI/2,0,M_PI/2,0};

  robot.stopReceiveRobotState();
  robot.startReceiveRobotState(std::chrono::milliseconds(1),
                               {jointPos_m, jointVel_m, jointAcc_c, tcpPose_m});

  // 运动到拖拽位置
   rtCon->MoveJ(0.5, robot.jointPos(ec), q_drag);

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

    ofs1.open("q_pp.txt", std::ios::out | std::ios::trunc);
    ofs2.open("qerror_pp.txt", std::ios::out | std::ios::trunc);
    ofs3.open("tau_pp.txt",std::ios::out | std::ios::trunc);
    ofs4.open("yd.txt",std::ios::out | std::ios::trunc);
    static bool init = true;
    std::array<double, 6> q_init;
    std::array<double, 6> q_out;
    std::array<double, 6> q_out_1;
    std::array<double, 6> ddq_out = {0, 0, 0, 0, 0, 0 };

    //PD控制器参数
    double Kp0 = 80;
    double Kd0 = 30;    

    double Kp1 = 650;
    double Kd1 = 10;

    double Kp2 = 50;
    double Kd2 = 5;

    double Kp4 = 20;
    double Kd4 = 2.5;
    //定义常值参数

    double psi = 0;
    double phi_f=0;
    double beta = 5;
    double Tf = 8.2;
    double theta = 0;
    double rho =0.1;
    //定义中间向量变量
    Eigen::Matrix<double, 6, 1> Kp = {40,400,30,35,20,0};
    Eigen::Matrix<double, 6, 1> Kd = {15,5,2.5,2.5,2.5,0};
    Eigen::Matrix<double, 6, 1> Kp_e = {80,600,50,50,35,0};
    Eigen::Matrix<double, 6, 1> Kd_e = {15,5,2.5,2.5,2.5,0};
    Eigen::Matrix<double, 6, 1> k = {0.7,0.6,0.6,0.025,0.125,0};
    Eigen::Matrix<double, 6, 1> k_e = {1,1.2,0.8,0.025,0.2,0};
    Eigen::Matrix<double, 6, 1> lambda = {0.0000000000001,
                                          0.0000000000001,
                                          0.0000000000001,
                                          0.000000000001,
                                          0.0000000000001,
                                          0};
    Eigen::Matrix<double, 6, 1> lambda_e = {0.0001,
                                          0.0000000001,
                                          0.0000000001,
                                          0.000000000001,
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
    
    std::function<Torque(void)> callback = [&]{
    using namespace RtSupportedFields;
    static double time=0;
    time += 0.001;

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
    
    q_out = q_init;

        q_out[0] = q_init[0]+0.3*std::cos(time);
        ddq_out[0] = -0.3*std::cos(time);
        
        q_out[1] =q_init[1] + 0.1*std::sin(time);
        ddq_out[1] = -0.1*std::sin(time);

        q_out[2] =q_init[2]-0.1*std::cos(time);
        ddq_out[2] = 0.1*std::cos(time);

        //q_out[3] += 0.1*std::sin(time);
        //ddq_out[3] = -0.1*std::sin(time);

        q_out[4] = q_init[4]+0.2*std::cos(time);
        ddq_out[4] = -0.2*std::cos(time);

        //q_out[5] += 0.0001;
        // q_out[6] += 0.0001;

        Eigen::Map<Eigen::Matrix<double, 6, 1>> q_d(q_out.data());
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
        q_out_1 = q_out;
        e2 = dqm - dyd;


        // compute control
        Eigen::VectorXd tau_d(6);
        theta = 4*rho/pow(Tf,4);
        omega_1 = mu(time,Tf)*e1;
        omega_2 = dmu(time,Tf)*e1+mu(time,Tf)*e2;
        z = omega_2+beta*omega_1;
        phi_f = 1+dqm.norm()+pow(dqm.norm(),2);
        psi = 1 + phi_f + pow(qd_ddot.norm(),2);
        

        tau_d <<(-(theta)*z-(psi*psi*lambda.array()*z.array()).matrix()-(k.array()*z.array()).matrix())-((Kp.array()*e1.array()+Kd.array()*e2.array()).matrix());
        //tau_d <<(-(theta)*z-(psi*psi*lambda_e.array()*z.array()).matrix()-(k_e.array()*z.array()).matrix())-0*((Kp.array()*e1.array()+Kd.array()*e2.array()).matrix());

        //tau_d[0] = -(Kp0*e1[0]+Kd0*e2[0]);
        //tau_d[1] = -(Kp1*e1[1]+Kd1*e2[1]);
        //tau_d[2] <<-1*0*(k+lambda*psi*psi+theta)*z-(Kp2*e1[2]+Kd2*e2[2])+0*gravity;
        //tau_d[4] <<-1*0*(k+lambda*psi*psi+theta)*z-(Kp4*e1[4]+Kd4*e2[4])+0*gravity;
        //tau_d[1]=tau_d[2]=tau_d[3] = tau_d[4] = tau_d[5] =tau_d[6] =0;
        //tau_d[1]=tau_d[3] =0;
         //tau_d <<-(Kp*e1+Kd*e2)+gravity;
        for (size_t i = 0; i < 4; i++)
        {
          if(e1[i]<0.001){
            tau_d[i] = -1*(Kp_e[i]*e1[i]+Kd_e[i]*e2[i]);
          }
        }
        if(e1[4]<0.001){
            tau_d[4] = -1*(Kp_e[4]*e1[4]+Kd_e[4]*e2[4]);
        }
        std::cout<<time<<" "<<tau_d[0]<<" "<<tau_d[1]<<" "<<tau_d[2]<<" "<<tau_d[3]<<" "<<tau_d[4]<<" "<<tau_d[5]<<std::endl;

    Torque cmd(6);
    Eigen::VectorXd::Map(cmd.tau.data(), 6) = tau_d;

    ofs1 << time << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << " " << q[4] << " " << q[5] << " "  << std::endl;
    ofs2 << time << " " << e1[0] << " " << e1[1] << " " <<  e1[2] << " " <<  e1[3] << " " <<  e1[4] << " " <<e1[5] << " " << std::endl;
    ofs3 << time << " " << tau_d[0] << " " <<tau_d[1] << " " <<tau_d[2] << " " <<tau_d[3] << " " <<tau_d[4] << " " << tau_d[5] << " "  << std::endl;
    ofs4 << time << " " << q_out[0] << " " <<q_out[1] << " " <<q_out[2] << " " <<q_out[3] << " " <<q_out[4] << " " << q_out[5] << " "  << std::endl;
    if(time > 8.2){
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