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

  std::array<double,6> q_drag = {0, M_PI/6,-M_PI/2, 0, -M_PI/3, 0 };

  robot.stopReceiveRobotState();
  robot.startReceiveRobotState(std::chrono::milliseconds(1),
                               {jointPos_m, jointVel_m, jointAcc_c, tcpPose_m});

  rtCon->MoveJ(0.8, robot.jointPos(ec), q_drag);
  rtCon->startMove(RtControllerMode::torque);

    std::ofstream ofs1;
    std::ofstream ofs2;
    std::ofstream ofs3;
    std::ofstream ofs4;
    std::ofstream ofs5;
    std::ofstream ofs6;

    ofs1.open("q_pd.txt", std::ios::out | std::ios::trunc);
    ofs2.open("qerror_pd.txt", std::ios::out | std::ios::trunc);
    ofs3.open("tau_pd.txt",std::ios::out | std::ios::trunc);
    ofs4.open("yd.txt",std::ios::out | std::ios::trunc);
    ofs5.open("pose_pd.txt",std::ios::out | std::ios::trunc);
    ofs6.open("pose_d.txt",std::ios::out | std::ios::trunc);

    static bool init = true;
    std::array<double, 6> q_init;
    std::array<double, 6> q_out;
    std::array<double, 6> q_out_1;
    std::array<double, 16> init_position {};
    Utils::postureToTransArray(robot.posture(rokae::CoordinateType::flangeInBase, ec), init_position);

    // ... (您控制器中其他的变量定义保持不变)
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
    // ... (您其他的控制器矩阵)
    Eigen::Matrix<double, 6, 1> k = {0.35,0.6,0.6,0.05,0.125,0};
    Eigen::Matrix<double, 6, 1> lambda = {0.1, 0.1, 0.1, 0.05, 0.00000001, 0};
    Eigen::Matrix<double, 6, 1> dyd;
    Eigen::Matrix<double, 6, 1> qd_ddot;
    Eigen::Matrix<double, 6, 1> e1;
    Eigen::Matrix<double, 6, 1> e2;
    Eigen::Matrix<double, 6, 1> z;
    Eigen::Matrix<double, 6, 1> omega_1;
    Eigen::Matrix<double, 6, 1> omega_2;


    // 新增：定义小鸟轮廓的自定义路径
    // 这些点在机器人基坐标系中定义。
    // 我们获取起始位置，并在此基础上添加偏移量。
    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(init_position.data()).transpose());
    Eigen::Vector3d start_position = initial_transform.translation();

    std::vector<Eigen::Vector3d> path_waypoints;
    // 这是一个简单的形状，您可以替换为您自己小鸟的坐标
    path_waypoints.push_back(start_position + Eigen::Vector3d(0.0, 0.0, 0.0));     // 1. 起点
    path_waypoints.push_back(start_position + Eigen::Vector3d(0.1, 0.05, 0.0));    // 2. 鸟头
    path_waypoints.push_back(start_position + Eigen::Vector3d(0.05, 0.08, 0.0));   // 3. 头顶
    path_waypoints.push_back(start_position + Eigen::Vector3d(-0.1, 0.06, 0.0));   // 4. 后背
    path_waypoints.push_back(start_position + Eigen::Vector3d(-0.08, 0.0, 0.0));   // 5. 尾巴
    path_waypoints.push_back(start_position + Eigen::Vector3d(0.0, 0.0, 0.0));     // 6. 回到起点

    // 新增：预计算路径段长度和总长度
    std::vector<double> segment_lengths;
    double total_path_length = 0;
    for (size_t i = 0; i < path_waypoints.size() - 1; ++i) {
        double length = (path_waypoints[i+1] - path_waypoints[i]).norm();
        segment_lengths.push_back(length);
        total_path_length += length;
    }

    // 新增：定义绘制整个路径的总时间
    double total_duration = T; // 我们使用您已有的时间 T

    std::function<Torque(void)> callback = [&]{
    using namespace RtSupportedFields;
    static double time=0;
    time += 0.001;
    
    // ####################################################################
    // 新增：自定义路径轨迹生成
    // ####################################################################

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
    
    // ####################################################################
    // 新增轨迹生成逻辑结束
    // ####################################################################

    // ... (您读取机器人状态的代码非常完美，保持不变)
    std::array<double, 6> q{}, dq_m{}, ddq_c{};
    robot.getStateData(jointPos_m, q);
    robot.getStateData(jointVel_m, dq_m);
    robot.getStateData(jointAcc_c, ddq_c);

    std::array<double, 6> gravity_array = model.getTorque(q, dq_m, ddq_c, TorqueType::gravity);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> gravity(gravity_array.data());
    Eigen::Map<Eigen::Matrix<double, 6, 1>> qm(q.data());
    Eigen::Map<Eigen::Matrix<double, 6, 1>> dqm(dq_m.data());

    // 构建完整的目标位姿，保持初始姿态不变
    static Eigen::Quaterniond initial_orientation(initial_transform.linear());
    Eigen::Affine3d target_pose = Eigen::Affine3d::Identity();
    target_pose.translation() = position_d;
    target_pose.linear() = initial_orientation.toRotationMatrix();
    
    // 转换为数组格式以用于逆运动学求解
    std::array<double, 16> target_pose_array;
    Eigen::Map<Eigen::Matrix4d>(target_pose_array.data()) = target_pose.matrix().transpose();
    
    // 使用逆运动学求解目标关节角度
    std::array<double, 6> q_target;
    model.getJointPos(target_pose_array, 0, q, q_target);

    // ... (您用于计算 tau_d 的控制器逻辑保持不变)
    auto pos_e = model.getCartPose(q);
    Eigen::Map<Eigen::Matrix4d> pose_map(pos_e.data());
    Eigen::Matrix4d pose_matrix = pose_map.transpose();
    Eigen::Vector3d position = pose_matrix.block<3,1>(0,3);

    Eigen::Map<Eigen::Matrix<double, 6, 1>> q_d(q_target.data());
    Eigen::Map<Eigen::Matrix<double, 6, 1>> q_d_1(q_out_1.data());
    e1 = qm - q_d;
    
    if (time < 0.0012) {
        dyd.array() = 0;
    } else {
        dyd = (q_d - q_d_1) / 0.001;
    }
    q_out_1 = q_target;
    e2 = dqm - dyd;

    // PD控制器现在是起作用的部分
    Eigen::VectorXd tau_d(6);
    tau_d = -((Kp.array()*e1.array()+Kd.array()*e2.array()).matrix());
    // 如果需要，您可以重新启用您的其他控制器项
    // tau_d += gravity; // 示例：添加重力补偿

    // ... (您的数据记录和循环终止逻辑保持不变)
    Torque cmd(6);
    Eigen::VectorXd::Map(cmd.tau.data(), 6) = tau_d;

    ofs1 << time << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << " " << q[4] << " " << q[5] << " "  << std::endl;
    ofs2 << time << " " << e1[0] << " " << e1[1] << " " <<  e1[2] << " " <<  e1[3] << " " <<  e1[4] << " " <<0*e1[5] << " " << std::endl;
    ofs3 << time << " " << tau_d[0] << " " <<tau_d[1] << " " <<tau_d[2] << " " <<tau_d[3] << " " <<tau_d[4] << " " << tau_d[5] << " "  << std::endl;
    ofs4 << time << " " << q_out[0] << " " <<q_out[1] << " " <<q_out[2] << " " <<q_out[3] << " " <<q_out[4] << " " << q_out[5] << " "  << std::endl;
    ofs5 << time << " " << position[0] << " " <<position[1] << " " <<position[2]<<" "<< std::endl;
    ofs6 << time << " " << position_d[0] << " " <<position_d[1] << " " <<position_d[2]<<" "  << std::endl;

    if(time > total_duration + 0.5){ // 额外给一点时间让系统稳定下来
      ofs1.close();
      ofs2.close();
      ofs3.close();
      ofs4.close();
      ofs5.close();
      ofs6.close();
      print(std::cout, "力矩控制结束");
      cmd.setFinished();
    }
    return cmd;
  };

  rtCon->setControlLoop(callback, 0, true);
  rtCon->startLoop(true);
}