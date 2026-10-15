#include <cmath>

#include "Types.hpp"

#include "FK.hpp"
#include "IK.hpp"
#include "Jacobian.hpp"
#include "parallel_ankle.hpp"

namespace parallel_ankle {

ParallelAnkle::ParallelAnkle(int left_pitch, int left_roll, int right_pitch, int right_roll)
    : ankle_motor_ready_flag_(false) {
    // 加载 FK/FK_V_JB 使用的 Chebyshev 拟合系数。
    load_coeffs(chebyshev_config_);

    // 当前启用左右踝两组二自由度并联机构。
    //
    // 聚合初始化字段顺序必须和 ParallelPairConfig 完全一致：
    //   name,
    //   motor_1_idx, motor_2_idx,
    //   pitch_idx, roll_idx,
    //   pitch_sign, roll_sign,
    //   safety_limit。
    //
    // 这里把“真实电机索引”和“任务空间 pitch/roll 索引”分开写，
    // 目的就是避免以前 roll/pitch 互相写反的问题。
    pairs_ = {
    {
        "left_ankle",
        left_pitch,   // motor_1_idx (q1 / a)
        left_roll,    // motor_2_idx (q2 / b)
        left_pitch,   // pitch_idx
        left_roll,    // roll_idx
        1.0,  // pitch_sign
        -1.0, // roll_sign（左右腿镜像）
        0.90  // safety_limit (rad)
    },
    {
        "right_ankle",
        right_pitch,  // motor_1_idx (q1 / a)
        right_roll,   // motor_2_idx (q2 / b)
        right_pitch,  // pitch_idx
        right_roll,   // roll_idx
        1.0,  // pitch_sign
        1.0,  // roll_sign
        0.90  // safety_limit (rad)
    }
    };

    pair_ready_.assign(pairs_.size(), 0);
    last_valid_.resize(pairs_.size());
    cached_jacob_.resize(pairs_.size());

    // 安全阈值说明：
    // 任务工作空间为 Pit ±65°、Rol ±20°，经 IK 映射后的电机角包络正好等于
    // Chebyshev 拟合盒：a ∈ [-0.8892, 0.8921] rad、b ∈ [-0.8883, 0.8928] rad
    // （约 ±51°，见 coeffs.cpp 里的 a_min/a_max/b_min/b_max）。
    // 因此 safety_limit 取 0.90 rad（≈51.6°）：正常工作时电机角最大约 0.893 rad，
    // 不会误触发；一旦电机角超过拟合盒，FK 多项式外推失效，立即判定异常。
    // 拟合盒内还存在机构物理不可达的电机角组合，由 solve_state_pair 里的
    // FK->IK 往返一致性检查剔除。
    //
    // 数学库零位约定：平台水平时任务零位对应电机零位 a=b=0；
    // 电机硬件零位偏置已在 IK.cpp 内部用 MOTOR_PI_OFFSET_hi/low 扣除。
}

void ParallelAnkle::parallel_ankle_solve_state(std::vector<MotorData>& joint_data) {
    // 每帧重新评估每组机构的有效性，整体 flag 为各组的与。
    ankle_motor_ready_flag_ = true;

    for (size_t k = 0; k < pairs_.size(); ++k) {
        solve_state_pair(pairs_[k], k, joint_data);
        if (!pair_ready_[k]) {
            ankle_motor_ready_flag_ = false;
        }
    }
}

void ParallelAnkle::parallel_ankle_solve_cmd(std::vector<MotorData>& joint_data) {
    // 有效性由本帧先执行的 parallel_ankle_solve_state 给出，这里直接消费。
    for (size_t k = 0; k < pairs_.size(); ++k) {
        solve_cmd_pair(pairs_[k], k, joint_data);
    }
}

void ParallelAnkle::solve_state_pair(const ParallelPairConfig& pair,
                                     size_t pair_idx,
                                     std::vector<MotorData>& joint_data) {
    char& ready = pair_ready_[pair_idx];
    LastValidState& cache = last_valid_[pair_idx];
    ready = 0;

    // 从真实电机空间读取原始反馈（motor_idx 与 pitch/roll_idx 共用槽位，
    // 必须在写回任务空间之前读完）。
    const double q_1 = joint_data[pair.motor_1_idx].pos_;
    const double q_2 = joint_data[pair.motor_2_idx].pos_;
    const double dq_1 = joint_data[pair.motor_1_idx].vel_;
    const double dq_2 = joint_data[pair.motor_2_idx].vel_;
    const double tau_1 = joint_data[pair.motor_1_idx].tau_;
    const double tau_2 = joint_data[pair.motor_2_idx].tau_;

    // 有效性检查 1：真实电机位置必须在安全限位（Chebyshev 拟合域）内。
    bool valid = is_pair_safe(pair, joint_data) &&
                 std::isfinite(q_1) && std::isfinite(q_2) &&
                 std::isfinite(dq_1) && std::isfinite(dq_2) &&
                 std::isfinite(tau_1) && std::isfinite(tau_2);

    double q_pitch = 0.0;
    double q_roll = 0.0;
    double jacob_fk[4] = {0.0, 0.0, 0.0, 0.0};

    if (valid) {
        // 正运动学：真实电机角 q1/q2 -> 数学模型 pitch/roll。
        // Chebyshev 拟合给种子，再用精确解析 IK 做 2 次 Newton 抛光：
        // 一次调用同时产出抛光后的任务角、工作点正雅可比和电机空间往返残差。
        FKResult fk;
        FK_polished(q_1, q_2, fk, chebyshev_config_);
        q_pitch = fk.Pit;
        q_roll  = fk.Rol;
        jacob_fk[0] = fk.J[0];
        jacob_fk[1] = fk.J[1];
        jacob_fk[2] = fk.J[2];
        jacob_fk[3] = fk.J[3];

        // 有效性检查 2：解必须落在物理工作空间内，且 FK->IK 往返一致
        // （残差由抛光末轮精确 IK 直接给出）。
        // 拟合盒内有一部分电机角组合是闭链机构物理上到不了的，Chebyshev 多项式
        // 在那里外推出离谱种子（实测可达上百弧度）。FK_polished 对这种域外种子
        // 拒绝抛光并置 in_domain=false；域内不可达点 Newton 也压不下残差（流形上
        // 无解）。物理反馈落在可达流形上，in_domain=true 且抛光后往返残差约
        // 1e-8 rad；不可达点要么 in_domain=false，要么残差始终 >1 rad。双条件干净区分。
        const double roundtrip_tol = 0.05;  // rad
        if (!fk.in_domain ||
            std::fabs(fk.res_a) > roundtrip_tol ||
            std::fabs(fk.res_b) > roundtrip_tol) {
            valid = false;
        }

        // 正雅可比 J_fk = d(Pit,Rol)/d(a,b)，由 Chebyshev 拟合式解析求导
        // （上面 FK_polished 已在实测电机角处算好）。速度映射和两侧力矩映射
        // 都只需要这一个雅可比（虚功原理）：
        //   v_task = J_fk·v_motor，τ_motor = J_fk^T·τ_task，τ_task = J_fk^{-T}·τ_motor。

        // Check finite FK/Jacobian outputs before any task-space transform.
        if (!std::isfinite(q_pitch) || !std::isfinite(q_roll) ||
            !std::isfinite(fk.res_a) || !std::isfinite(fk.res_b)) {
            valid = false;
        }
        for (const double v : jacob_fk) if (!std::isfinite(v)) valid = false;

        // 有效性检查 3：雅可比不得接近奇异（工作空间内 |det| 实测最小约 1.34）。
        const double det = jacob_fk[0] * jacob_fk[3] - jacob_fk[1] * jacob_fk[2];
        if (std::fabs(det) < 1e-6) {
            valid = false;
        }
    }

    if (!valid) {
        // 解算失败：任务空间保持上一帧有效值（首帧从未有效过则输出 0），
        // 不把原始电机角或外推垃圾值写给下游控制器。
        joint_data[pair.pitch_idx].pos_ = cache.valid ? cache.pitch_pos : 0.0;
        joint_data[pair.roll_idx].pos_  = cache.valid ? cache.roll_pos  : 0.0;
        joint_data[pair.pitch_idx].vel_ = cache.valid ? cache.pitch_vel : 0.0;
        joint_data[pair.roll_idx].vel_  = cache.valid ? cache.roll_vel  : 0.0;
        joint_data[pair.pitch_idx].tau_ = cache.valid ? cache.pitch_tau : 0.0;
        joint_data[pair.roll_idx].tau_  = cache.valid ? cache.roll_tau  : 0.0;
        // ready 保持 false；solve_cmd 会把该组两台电机 ff_ 清零。
        return;
    }

    // 速度映射：v_task = J_fk · v_motor（J_fk 已在有效性检查阶段算出）。
    const double dq_pitch = jacob_fk[0] * dq_1 + jacob_fk[1] * dq_2;
    const double dq_roll = jacob_fk[2] * dq_1 + jacob_fk[3] * dq_2;

    // 力矩映射（虚功原理）：τ_task = J_fk^{-T} · τ_motor。
    // 逆雅可比 J_ik = inv(J_fk) 直接对 2x2 矩阵解析求逆，
    // 不再对 IK 做有限差分——状态侧与命令侧因此严格共用同一个雅可比。
    //   J_fk = [j0 j1; j2 j3]，inv = 1/det [ j3 -j1; -j2  j0]
    const double det = jacob_fk[0] * jacob_fk[3] - jacob_fk[1] * jacob_fk[2];
    const double inv_det = 1.0 / det;
    const double ji0 =  jacob_fk[3] * inv_det;  // da/dPit
    const double ji1 = -jacob_fk[1] * inv_det;  // da/dRol
    const double ji2 = -jacob_fk[2] * inv_det;  // db/dPit
    const double ji3 =  jacob_fk[0] * inv_det;  // db/dRol

    const double tau_pitch = ji0 * tau_1 + ji2 * tau_2;
    const double tau_roll = ji1 * tau_1 + ji3 * tau_2;

    // 写回任务空间。
    // pitch 永远写 pitch_idx，roll 永远写 roll_idx；
    // 方向差异只通过 sign 处理，不再靠交换变量名硬凑。
    joint_data[pair.pitch_idx].pos_ = pair.pitch_sign * q_pitch;
    joint_data[pair.roll_idx].pos_ = pair.roll_sign * q_roll;

    joint_data[pair.pitch_idx].vel_ = pair.pitch_sign * dq_pitch;
    joint_data[pair.roll_idx].vel_ = pair.roll_sign * dq_roll;

    joint_data[pair.pitch_idx].tau_ = pair.pitch_sign * tau_pitch;
    joint_data[pair.roll_idx].tau_ = pair.roll_sign * tau_roll;

    // 缓存本帧雅可比（数学坐标），供同周期 solve_cmd 复用，避免重复求值。
    // 该雅可比由 FK_polished 在实测电机角 (q1,q2) 处一次性算出；抛光已使
    // IK(任务位姿) ≈ (q1,q2) 到 ~1e-9 rad（远优于旧的纯拟合 1e-4），
    // 故 cmd 侧工作点与 state 侧实测电机角几乎完全重合，直接复用即可，
    // 且 state/cmd 严格共用同一雅可比。
    cached_jacob_[pair_idx] = {jacob_fk[0], jacob_fk[1], jacob_fk[2], jacob_fk[3]};

    // 缓存本帧有效结果（公开符号值），供下一帧失败时保持。
    cache.valid = true;
    cache.pitch_pos = joint_data[pair.pitch_idx].pos_;
    cache.roll_pos = joint_data[pair.roll_idx].pos_;
    cache.pitch_vel = joint_data[pair.pitch_idx].vel_;
    cache.roll_vel = joint_data[pair.roll_idx].vel_;
    cache.pitch_tau = joint_data[pair.pitch_idx].tau_;
    cache.roll_tau = joint_data[pair.roll_idx].tau_;

    ready = 1;
}

void ParallelAnkle::solve_cmd_pair(const ParallelPairConfig& pair,
                                   size_t pair_idx,
                                   std::vector<MotorData>& joint_data) {
    // 该组机构本帧无效（超安全限位或正解往返不一致）：
    // 电机前馈力矩强制清零，不输出任何基于错误状态算出的力矩。
    if (!pair_ready_[pair_idx]) {
        joint_data[pair.motor_1_idx].ff_ = 0.0;
        joint_data[pair.motor_2_idx].ff_ = 0.0;
        return;
    }

    // A disabled controller must not evaluate 0 * NaN from bad feedback.
    if (joint_data[pair.pitch_idx].kp_ == 0.0 &&
        joint_data[pair.pitch_idx].kd_ == 0.0 &&
        joint_data[pair.roll_idx].kp_ == 0.0 &&
        joint_data[pair.roll_idx].kd_ == 0.0) {
        joint_data[pair.motor_1_idx].ff_ = 0.0;
        joint_data[pair.motor_2_idx].ff_ = 0.0;
        return;
    }

    // 底层主要使用前馈力矩 ff_：
    // 用任务空间的 kp/kd/pos_des/vel_des 算出 pitch/roll 力矩，
    // 再通过雅可比转成两个真实电机的前馈力矩。
    const double tau_pitch =
        joint_data[pair.pitch_idx].kp_ *
            (joint_data[pair.pitch_idx].pos_des_ - joint_data[pair.pitch_idx].pos_) +
        joint_data[pair.pitch_idx].kd_ *
            (joint_data[pair.pitch_idx].vel_des_ - joint_data[pair.pitch_idx].vel_);

    const double tau_roll =
        joint_data[pair.roll_idx].kp_ *
            (joint_data[pair.roll_idx].pos_des_ - joint_data[pair.roll_idx].pos_) +
        joint_data[pair.roll_idx].kd_ *
            (joint_data[pair.roll_idx].vel_des_ - joint_data[pair.roll_idx].vel_);

    // 力矩也需要转到数学模型坐标系下，再交给雅可比做映射。
    const double math_tau_pitch = pair.pitch_sign * tau_pitch;
    const double math_tau_roll = pair.roll_sign * tau_roll;

    // 雅可比直接复用本帧 solve_state 由 FK_polished 在实测电机角处算出并缓存的
    // J_fk（cached_jacob_）：Newton 抛光后 cmd 的求值点 IK(任务位姿) 与 state 的
    // 电机角只差 ~1e-9 rad，无需重算，且两侧严格共用同一雅可比。
    const std::array<double, 4>& jacob_fk = cached_jacob_[pair_idx];

    // 力矩逆映射使用 J_fk^T：
    // [tau_1, tau_2]^T = J_fk^T * [tau_pitch, tau_roll]^T。
    const double tau_1 = jacob_fk[0] * math_tau_pitch + jacob_fk[2] * math_tau_roll;
    const double tau_2 = jacob_fk[1] * math_tau_pitch + jacob_fk[3] * math_tau_roll;

    joint_data[pair.motor_1_idx].ff_ = std::isfinite(tau_1) ? tau_1 : 0.0;
    joint_data[pair.motor_2_idx].ff_ = std::isfinite(tau_2) ? tau_2 : 0.0;
}

bool ParallelAnkle::is_pair_safe(const ParallelPairConfig& pair,
                                 const std::vector<MotorData>& joint_data) const {
    // 安全检查看真实电机位置，而不是任务空间 pitch/roll。
    return std::fabs(joint_data[pair.motor_1_idx].pos_) <= pair.safety_limit &&
           std::fabs(joint_data[pair.motor_2_idx].pos_) <= pair.safety_limit;
}

} // namespace parallel_ankle
