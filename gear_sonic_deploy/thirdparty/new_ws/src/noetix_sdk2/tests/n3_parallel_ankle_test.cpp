// CPU-only regression against the parallel-ankle code copied from old N3HW.
// Compile with third-party/vendor/parallel_ankle/n3/{FK,IK,Jacobian,coeffs,parallel_ankle}.cpp.
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "Types.hpp"
#include "IK.hpp"
#include "parallel_ankle.hpp"

int main() {
    // Non-canonical positions verify that N3 actuator_order can change.
    const int lp = 4, lr = 1, rp = 7, rr = 2;
    parallel_ankle::ParallelAnkle ankle(lp, lr, rp, rr);
    std::vector<MotorData> joints(12);

    const double pitch = 0.10, roll = 0.03;
    double a = 0.0, b = 0.0;
    parallel_ankle::IK(pitch, roll, a, b);
    joints[lp].pos_ = a;
    joints[lr].pos_ = b;
    joints[rp].pos_ = a;
    joints[rr].pos_ = b;
    ankle.parallel_ankle_solve_state(joints);
    if (!ankle.is_motor_ready()) {
        std::cerr << "unexpected initial invalid FK: " << a << ", " << b << '\n';
        return 1;
    }
    // Old N3HW uses negative left roll and positive right roll.
    assert(std::abs(joints[lp].pos_ - pitch) < 0.003);
    assert(std::abs(joints[lr].pos_ + roll) < 0.003);
    assert(std::abs(joints[rp].pos_ - pitch) < 0.003);
    assert(std::abs(joints[rr].pos_ - roll) < 0.003);

    joints[lp].kp_ = joints[lr].kp_ = 30.0;
    joints[rp].kp_ = joints[rr].kp_ = 30.0;
    joints[lp].pos_des_ = pitch + 0.01;
    joints[lr].pos_des_ = -roll + 0.01;
    joints[rp].pos_des_ = pitch + 0.01;
    joints[rr].pos_des_ = roll + 0.01;
    ankle.parallel_ankle_solve_cmd(joints);
    for (int id : {lp, lr, rp, rr}) {
        assert(std::isfinite(joints[id].ff_));
        assert(std::abs(joints[id].ff_) < 50.0);
    }

    // Out-of-fit-range motors must not yield an extrapolated task state or torque.
    joints[lp].pos_ = 4.0;
    ankle.parallel_ankle_solve_state(joints);
    assert(!ankle.is_motor_ready());
    ankle.parallel_ankle_solve_cmd(joints);
    assert(joints[lp].ff_ == 0.0);
    assert(joints[lr].ff_ == 0.0);
    std::cout << "N3 parallel ankle CPU regression passed\n";
}
