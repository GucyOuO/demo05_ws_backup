#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Vector3.h>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <std_msgs/Float32MultiArray.h>

namespace gazebo {

/*
 * QuadrotorControllerPlugin
 * -------------------------
 * 这是一个“教学版”四旋翼 Gazebo 控制插件，目标是帮助初学者把链路跑通：
 * 1) 订阅 cmd_vel（速度指令）
 * 2) 订阅 cmd_attitude（姿态指令，roll/pitch/yaw）
 * 3) 在每次仿真迭代里计算四电机推力
 * 4) 把推力施加到四个电机 link
 * 5) 发布 motor_thrust 方便你在 rqt_plot 观察
 *
 * 注意：它不是工业级飞控，只是学习用基础控制器。
 */
class QuadrotorControllerPlugin : public ModelPlugin {
 public:
  QuadrotorControllerPlugin() = default;

  ~QuadrotorControllerPlugin() override {
    // 清理 ROS 回调线程，避免 Gazebo 退出时资源悬挂。
    if (this->ros_queue_thread_.joinable()) {
      this->ros_queue_.disable();
      this->ros_queue_thread_.join();
    }
    delete this->ros_node_;
  }

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    // 1) 读取模型与关键 link
    this->model_ = std::move(model);
    this->base_link_ = this->model_->GetLink("base_link");

    // 电机 link 名称必须与 xacro 文件一致。
    this->motors_[0] = this->model_->GetLink("front_left_motor");
    this->motors_[1] = this->model_->GetLink("front_right_motor");
    this->motors_[2] = this->model_->GetLink("rear_left_motor");
    this->motors_[3] = this->model_->GetLink("rear_right_motor");

    if (!this->base_link_ || !this->motors_[0] || !this->motors_[1] || !this->motors_[2] || !this->motors_[3]) {
      gzerr << "[QuadrotorControllerPlugin] required links not found\n";
      return;
    }

    // 2) 从 SDF(即 xacro 里 <plugin>...</plugin>) 读取可调参数。
    // 没写则使用默认值。
    this->arm_length_ = sdf->HasElement("arm_length") ? sdf->Get<double>("arm_length") : 0.22;
    this->yaw_moment_coeff_ = sdf->HasElement("yaw_moment_coeff") ? sdf->Get<double>("yaw_moment_coeff") : 0.02;
    this->max_motor_thrust_ = sdf->HasElement("max_motor_thrust") ? sdf->Get<double>("max_motor_thrust") : 20.0;

    // 速度环增益（简化版）
    this->kp_xy_ = sdf->HasElement("kp_xy") ? sdf->Get<double>("kp_xy") : 2.2;
    this->kp_vz_ = sdf->HasElement("kp_vz") ? sdf->Get<double>("kp_vz") : 4.2;

    // 姿态环 / 角速度环增益（简化 PD）
    this->kp_roll_ = sdf->HasElement("kp_roll") ? sdf->Get<double>("kp_roll") : 7.0;
    this->kd_roll_ = sdf->HasElement("kd_roll") ? sdf->Get<double>("kd_roll") : 1.6;
    this->kp_pitch_ = sdf->HasElement("kp_pitch") ? sdf->Get<double>("kp_pitch") : 7.0;
    this->kd_pitch_ = sdf->HasElement("kd_pitch") ? sdf->Get<double>("kd_pitch") : 1.6;
    this->kp_yaw_ = sdf->HasElement("kp_yaw") ? sdf->Get<double>("kp_yaw") : 4.0;
    this->kd_yaw_ = sdf->HasElement("kd_yaw") ? sdf->Get<double>("kd_yaw") : 0.8;
    this->kp_yaw_rate_ = sdf->HasElement("kp_yaw_rate") ? sdf->Get<double>("kp_yaw_rate") : 1.5;

    // 3) 初始化 ROS 通信
    std::string ns = "pilot";
    if (sdf->HasElement("robot_namespace")) {
      ns = sdf->Get<std::string>("robot_namespace");
    }
    if (!ros::isInitialized()) {
      gzerr << "[QuadrotorControllerPlugin] ROS node not initialized\n";
      return;
    }
    this->ros_node_ = new ros::NodeHandle(ns);

    std::string cmd_vel_topic = "cmd_vel";
    std::string cmd_att_topic = "cmd_attitude";
    if (sdf->HasElement("cmd_vel_topic")) cmd_vel_topic = sdf->Get<std::string>("cmd_vel_topic");
    if (sdf->HasElement("cmd_att_topic")) cmd_att_topic = sdf->Get<std::string>("cmd_att_topic");

    ros::SubscribeOptions vel_so = ros::SubscribeOptions::create<geometry_msgs::Twist>(
        cmd_vel_topic, 1, boost::bind(&QuadrotorControllerPlugin::OnCmdVel, this, _1),
        ros::VoidPtr(), &this->ros_queue_);
    ros::SubscribeOptions att_so = ros::SubscribeOptions::create<geometry_msgs::Vector3>(
        cmd_att_topic, 1, boost::bind(&QuadrotorControllerPlugin::OnCmdAttitude, this, _1),
        ros::VoidPtr(), &this->ros_queue_);

    this->cmd_vel_sub_ = this->ros_node_->subscribe(vel_so);
    this->cmd_att_sub_ = this->ros_node_->subscribe(att_so);

    // 发布当前四电机推力，便于调参与观察。
    this->motor_pub_ = this->ros_node_->advertise<std_msgs::Float32MultiArray>("motor_thrust", 1);

    // 单独线程处理 ROS 回调，避免阻塞 Gazebo 主循环。
    this->ros_queue_thread_ = std::thread([this]() {
      while (this->ros_node_->ok()) {
        this->ros_queue_.callAvailable(ros::WallDuration(0.01));
      }
    });

    // 4) 绑定 Gazebo 更新回调（每个仿真步都会进 OnUpdate）
    this->update_conn_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&QuadrotorControllerPlugin::OnUpdate, this));

    gzmsg << "[QuadrotorControllerPlugin] loaded in namespace /" << ns << "\n";
  }

 private:
  // 把角度归一化到 [-pi, pi]，避免“跳变”
  static double WrapAngle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }

  void OnCmdVel(const geometry_msgs::TwistConstPtr& msg) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->cmd_vel_ = *msg;
  }

  void OnCmdAttitude(const geometry_msgs::Vector3ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->cmd_att_ = *msg;
    this->has_att_cmd_ = true;
  }

  void OnUpdate() {
    // ---------- A. 取最新指令（加锁避免读写冲突） ----------
    geometry_msgs::Twist cmd_vel;
    geometry_msgs::Vector3 cmd_att;
    bool has_att = false;
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      cmd_vel = this->cmd_vel_;
      cmd_att = this->cmd_att_;
      has_att = this->has_att_cmd_;
    }

    // ---------- B. 读取当前姿态/速度状态 ----------
    const auto pose = this->base_link_->WorldPose();
    const auto rot = pose.Rot();
    const auto euler = rot.Euler();  // roll(x), pitch(y), yaw(z)

    // 世界系速度 -> 机体系速度（控制通常在机体系更直观）
    const ignition::math::Vector3d vel_world = this->base_link_->WorldLinearVel();
    const ignition::math::Vector3d vel_body = rot.RotateVectorReverse(vel_world);
    const ignition::math::Vector3d ang_body = this->base_link_->RelativeAngularVel();

    // ---------- C. 速度误差 ----------
    const double ex = cmd_vel.linear.x - vel_body.X();
    const double ey = cmd_vel.linear.y - vel_body.Y();
    const double ez = cmd_vel.linear.z - vel_body.Z();

    const double mass = this->base_link_->GetInertial()->Mass();
    const double g = std::abs(this->model_->GetWorld()->Gravity().Z());

    // 如果没有姿态指令：roll/pitch 默认回 0，yaw 默认保持当前值。
    const double desired_roll = has_att ? cmd_att.x : 0.0;
    const double desired_pitch = has_att ? cmd_att.y : 0.0;
    const double desired_yaw = has_att ? cmd_att.z : euler.Z();

    const double roll_err = WrapAngle(desired_roll - euler.X());
    const double pitch_err = WrapAngle(desired_pitch - euler.Y());
    const double yaw_err = WrapAngle(desired_yaw - euler.Z());

    // ---------- D. 姿态控制（简化 PD） ----------
    const double tau_roll = this->kp_roll_ * roll_err - this->kd_roll_ * ang_body.X();
    const double tau_pitch = this->kp_pitch_ * pitch_err - this->kd_pitch_ * ang_body.Y();
    const double tau_yaw = this->kp_yaw_ * yaw_err +
                           this->kp_yaw_rate_ * (cmd_vel.angular.z - ang_body.Z()) -
                           this->kd_yaw_ * ang_body.Z();

    // ---------- E. 推力控制 ----------
    // collective: 总升力，至少抵消重力，再加 z 速度控制。
    const double collective = mass * (g + this->kp_vz_ * ez);

    // ---------- F. 四电机混控 ----------
    // 电机顺序: FL FR RL RR
    // 基本思想：
    // 1) collective 平均分配
    // 2) roll/pitch/yaw 力矩叠加到各电机
    std::array<double, 4> f = {
      0.25 * collective - 0.5 * tau_roll / this->arm_length_ + 0.5 * tau_pitch / this->arm_length_ + 0.25 * tau_yaw / this->yaw_moment_coeff_,
      0.25 * collective + 0.5 * tau_roll / this->arm_length_ + 0.5 * tau_pitch / this->arm_length_ - 0.25 * tau_yaw / this->yaw_moment_coeff_,
      0.25 * collective - 0.5 * tau_roll / this->arm_length_ - 0.5 * tau_pitch / this->arm_length_ - 0.25 * tau_yaw / this->yaw_moment_coeff_,
      0.25 * collective + 0.5 * tau_roll / this->arm_length_ - 0.5 * tau_pitch / this->arm_length_ + 0.25 * tau_yaw / this->yaw_moment_coeff_};

    // 推力限幅：防止出现负推力或超出电机能力。
    for (double& thrust : f) {
      thrust = std::max(0.0, std::min(this->max_motor_thrust_, thrust));
    }

    // ---------- G. 把推力施加到各电机 link ----------
    // AddRelativeForce 的方向是“电机自身坐标系”。此处 +Z 方向是向上推力。
    for (size_t i = 0; i < 4; ++i) {
      this->motors_[i]->AddRelativeForce(ignition::math::Vector3d(0, 0, f[i]));
    }

    // 给 x/y 一个简化外力补偿，让初学者更容易看到“跟随 cmd_vel”效果。
    this->base_link_->AddRelativeForce(ignition::math::Vector3d(this->kp_xy_ * ex * mass,
                                                                 this->kp_xy_ * ey * mass,
                                                                 0.0));

    // ---------- H. 发布电机推力用于可视化 ----------
    std_msgs::Float32MultiArray msg;
    msg.data.resize(4);
    for (size_t i = 0; i < 4; ++i) msg.data[i] = static_cast<float>(f[i]);
    this->motor_pub_.publish(msg);
  }

 private:
  // Gazebo 对象
  physics::ModelPtr model_;
  physics::LinkPtr base_link_;
  std::array<physics::LinkPtr, 4> motors_;
  event::ConnectionPtr update_conn_;

  // 控制参数
  double arm_length_{0.22};
  double yaw_moment_coeff_{0.02};
  double max_motor_thrust_{20.0};
  double kp_xy_{2.2};
  double kp_vz_{4.2};
  double kp_roll_{7.0};
  double kd_roll_{1.6};
  double kp_pitch_{7.0};
  double kd_pitch_{1.6};
  double kp_yaw_{4.0};
  double kd_yaw_{0.8};
  double kp_yaw_rate_{1.5};

  // ROS 通信
  ros::NodeHandle* ros_node_{nullptr};
  ros::Subscriber cmd_vel_sub_;
  ros::Subscriber cmd_att_sub_;
  ros::Publisher motor_pub_;
  ros::CallbackQueue ros_queue_;
  std::thread ros_queue_thread_;

  // 指令缓存（带锁）
  std::mutex mutex_;
  geometry_msgs::Twist cmd_vel_;
  geometry_msgs::Vector3 cmd_att_;
  bool has_att_cmd_{false};
};

GZ_REGISTER_MODEL_PLUGIN(QuadrotorControllerPlugin)

}  // namespace gazebo
