#pragma once
/**
 * BUMI3 离线动作采用与 G1 相同的职责划分：MotionSequence 保存单条动作，
 * MotionDataReader 负责 CSV 读取与动作集合管理。转换器已固化关节顺序和参考
 * 运动学；部署时校验契约后预加载所选清单，切换时直接选择内存对象。
 */
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace bumi3 {

/**
 * 单条动作的数据容器，不执行文件读取或动作切换。关节按策略的 21 个关节顺序
 * 保存；身体四元数为 base_link 的 wxyz 方向。人体参考按所选布局保存每帧
 * 二十四点位置或四十五维五点姿态，以及独立的根四元数。MuJoCo qpos/qvel
 * 供仿真初始化与影子使用，不代表真机反馈。
 */
struct MotionSequence {
    using Point = std::array<double, 3>;
    using Quaternion = std::array<double, 4>;

    std::string name;
    int timesteps = 0;
    bool has_robot_reference = true;

    // 返回指定帧的数据视图，底层数组由动作对象持有，不复制整段动作。
    const double* JointPositions(int frame) const {
        return joint_positions_.data() + FrameOffset(frame, 21);
    }
    const double* JointVelocities(int frame) const {
        return joint_velocities_.data() + FrameOffset(frame, 21);
    }
    const Quaternion* BodyQuaternions(int frame) const {
        return body_quaternions_.data() + FrameOffset(frame, 1);
    }
    const Point* SmplJoints(int frame) const {
        if (smpl_joints_.empty()) throw std::runtime_error("motion has no SMPL joints: " + name);
        return smpl_joints_.data() + FrameOffset(frame, 24);
    }
    const Quaternion* SmplRootQuaternions(int frame) const {
        if (smpl_root_quaternions_.empty()) throw std::runtime_error("motion has no SMPL root orientation: " + name);
        return smpl_root_quaternions_.data() + FrameOffset(frame, 1);
    }
    // 五点位置和旋转已经由原始轴角离线固化；运行时只读取四十五维视图。
    const double* SmplFivePointPose(int frame) const {
        if (smpl_five_point_pose_.empty()) throw std::runtime_error("motion has no SMPL five-point pose: " + name);
        return smpl_five_point_pose_.data() + FrameOffset(frame, 45);
    }
    const double* MujocoPositions(int frame) const {
        return mujoco_positions_.data() + FrameOffset(frame, 28);
    }
    const double* MujocoVelocities(int frame) const {
        return mujoco_velocities_.data() + FrameOffset(frame, 27);
    }

    int GetNumJoints() const { return 21; }
    int GetNumBodyQuaternions() const { return 1; }
    int GetNumSmplJoints() const { return !smpl_five_point_pose_.empty() ? 5 : smpl_joints_.empty() ? 0 : 24; }

    void print_summary() const {
        std::cout << name << ": " << timesteps << " frames, " << GetNumJoints()
                  << " joints, " << GetNumSmplJoints() << " SMPL joints" << std::endl;
    }

  private:
    friend class MotionDataReader;
    std::vector<double> joint_positions_, joint_velocities_;
    std::vector<Quaternion> body_quaternions_, smpl_root_quaternions_;
    std::vector<Point> smpl_joints_;
    std::vector<double> smpl_five_point_pose_;
    std::vector<double> mujoco_positions_, mujoco_velocities_;

    // 返回帧视图前检查边界，防止重播或未来帧索引错误造成越界读取。
    std::size_t FrameOffset(int frame, std::size_t stride) const {
        if (frame < 0 || frame >= timesteps) throw std::out_of_range("motion frame out of range: " + name);
        return std::size_t(frame) * stride;
    }
};

/**
 * 使用与 G1 同名的 motions/current_motion_index_ 和 GetMotionShared 接口。
 * 启动器提供元数据清单，保留单动作、起始动作和 Robot/SMPL 的选择行为。
 * 启动时读取全部 CSV；失败时抛出具体错误，避免运行时跳过损坏的动作。
 */
class MotionDataReader {
  public:
    std::vector<std::shared_ptr<MotionSequence>> motions;
    int current_motion_index_ = 0;

    bool ReadFromCSV(const nlohmann::json& entries, bool use_smpl,
                     uint32_t contract_id, const std::vector<std::string>& joint_names,
                     bool five_point = false) {
        if (!entries.is_array() || joint_names.size() != 21) {
            throw std::runtime_error("reference playlist requires an array and 21 policy joint names");
        }
        // 全部校验成功后再替换集合，读取失败时不留下半套动作数据。
        std::vector<std::shared_ptr<MotionSequence>> loaded;
        std::unordered_set<std::string> names;
        loaded.reserve(entries.size());
        for (const auto& entry : entries) {
            auto motion = std::make_shared<MotionSequence>();
            const auto path = entry.at("reference").get<std::string>();
            const auto metadata = ReadMetadata(path, *motion, use_smpl, contract_id, joint_names);
            if (motion->name != entry.at("name").get<std::string>() || !names.insert(motion->name).second) {
                throw std::runtime_error("reference playlist/metadata name mismatch or duplicate: " + path);
            }
            const auto directory = std::filesystem::path(path).parent_path();
            const auto& arrays = metadata.at("arrays");
            const int frames = motion->timesteps;
            motion->joint_positions_ = ReadCSV(directory, arrays.at("joint_pos"), frames, 21, &joint_names);
            motion->joint_velocities_ = ReadCSV(directory, arrays.at("joint_vel"), frames, 21, &joint_names);
            motion->body_quaternions_ = ReadCSV3D<4>(directory, arrays.at("anchor_quat"), frames, 1);
            motion->mujoco_positions_ = ReadCSV(directory, arrays.at("qpos"), frames, 28);
            motion->mujoco_velocities_ = ReadCSV(directory, arrays.at("qvel"), frames, 27);
            if (use_smpl) {
                if (metadata.value("smpl_input", std::string("full")) != (five_point ? "five_point" : "full"))
                    throw std::runtime_error("SMPL CSV layout differs from selected policy: " + path);
                if (five_point) {
                    if (metadata.value("smpl_five_point_schema", 0) != 1 ||
                        metadata.at("smpl_five_point_indices") != nlohmann::json::array({15, 20, 21, 7, 8}))
                        throw std::runtime_error("SMPL five-point schema/order mismatch: " + path);
                    motion->smpl_five_point_pose_ = ReadCSV(directory, arrays.at("smpl_five_point_pose"), frames, 45);
                } else {
                    motion->smpl_joints_ = ReadCSV3D<3>(directory, arrays.at("smpl_joints"), frames, 24);
                }
                motion->smpl_root_quaternions_ = ReadCSV3D<4>(directory, arrays.at("smpl_quat"), frames, 1);
            }
            loaded.push_back(std::move(motion));
        }
        motions.swap(loaded);
        current_motion_index_ = 0;
        std::cout << "Loaded " << motions.size() << " BUMI3 reference motions from CSV" << std::endl;
        return !motions.empty();
    }

    std::shared_ptr<const MotionSequence> GetMotion(const std::string& name) const {
        for (const auto& motion : motions) {
            if (motion->name == name) return motion;
        }
        return nullptr;
    }

    std::shared_ptr<const MotionSequence> GetMotion(std::size_t index) const {
        return index < motions.size() ? motions[index] : nullptr;
    }

    std::shared_ptr<const MotionSequence> GetMotionShared(std::size_t index) const {
        return GetMotion(index);
    }

    void PrintSummary() const {
        std::cout << "Total motions loaded: " << motions.size() << std::endl;
        for (const auto& motion : motions) motion->print_summary();
    }

  private:
    // 先检查机器人、encoder、50 Hz 帧率和训练契约，再读取数值数组。
    static nlohmann::json ReadMetadata(const std::string& path, MotionSequence& motion,
                                      bool use_smpl, uint32_t contract_id,
                                      const std::vector<std::string>& joint_names) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("reference metadata not found: " + path + "; run reference/bumi3_convert_motion.py first");
        const auto metadata = nlohmann::json::parse(input);
        if (metadata.at("version") != 1 || metadata.at("format") != "bumi3_csv_v1" || metadata.at("robot_type") != "bumi3") {
            throw std::runtime_error("unsupported reference format: " + path);
        }
        if (metadata.at("encoder") != (use_smpl ? "smpl" : "robot") ||
            metadata.at("reference_contract_id") != contract_id || metadata.at("policy_joint_names").get<std::vector<std::string>>() != joint_names) {
            throw std::runtime_error("reference/config mismatch; reconvert with --overwrite: " + path);
        }
        motion.timesteps = metadata.at("frames");
        motion.name = metadata.at("name");
        motion.has_robot_reference = metadata.at("has_robot_reference");
        const double fps = metadata.at("fps");
        if (motion.timesteps < 1 || !std::isfinite(fps) || std::abs(fps - 50) > 1e-6) {
            throw std::runtime_error("reference must have frames at 50 Hz: " + path);
        }
        return metadata;
    }

    // 首行为表头；每帧列数、总帧数、有限数值及关节列顺序必须与元数据一致。
    static std::vector<double> ReadCSV(const std::filesystem::path& directory,
                                      const nlohmann::json& field, int frames, int width,
                                      const std::vector<std::string>* joint_names = nullptr) {
        if (field.at("shape") != nlohmann::json::array({frames, width})) {
            throw std::runtime_error("reference array shape mismatch");
        }
        const auto relative = std::filesystem::path(field.at("path").get<std::string>());
        if (relative.empty() || relative.is_absolute()) throw std::runtime_error("reference CSV paths must be relative to metadata.json");
        for (const auto& part : relative) {
            if (part == "..") throw std::runtime_error("reference CSV path must not leave the motion directory");
        }
        const auto path = directory / relative;
        std::ifstream file(path);
        if (!file) throw std::runtime_error("reference CSV not found: " + path.string());
        std::string row;
        if (!std::getline(file, row)) throw std::runtime_error("reference CSV has no header: " + path.string());
        if (!row.empty() && row.back() == '\r') row.pop_back();
        std::istringstream header(row);
        std::string column;
        int columns = 0;
        while (std::getline(header, column, ',')) {
            if (joint_names && (columns >= width || column != joint_names->at(columns))) {
                throw std::runtime_error("reference CSV joint order mismatch: " + path.string());
            }
            ++columns;
        }
        if (columns != width || (!row.empty() && row.back() == ',')) throw std::runtime_error("reference CSV header width mismatch: " + path.string());
        std::vector<double> values;
        values.reserve(std::size_t(frames) * width);
        int rows = 0;
        while (std::getline(file, row)) {
            if (!row.empty() && row.back() == '\r') row.pop_back();
            if (row.empty() || row.back() == ',') throw std::runtime_error("empty reference CSV row or cell: " + path.string());
            if (rows >= frames) throw std::runtime_error("reference CSV contains extra frames: " + path.string());
            std::istringstream line(row);
            std::string cell;
            int count = 0;
            while (std::getline(line, cell, ',')) {
                std::size_t consumed = 0;
                const double value = std::stod(cell, &consumed);
                if (!std::isfinite(value) || cell.find_first_not_of(" \t\r", consumed) != std::string::npos) {
                    throw std::runtime_error("nonfinite or invalid reference CSV value: " + path.string());
                }
                values.push_back(value);
                ++count;
            }
            if (count != width) throw std::runtime_error("reference CSV row width mismatch: " + path.string());
            ++rows;
        }
        if (file.bad() || rows != frames) throw std::runtime_error("reference CSV read failure or frame count mismatch: " + path.string());
        return values;
    }

    // 将点或四元数整理成定长数组，使取帧接口与 G1 的身体/SMPL 数据形式一致。
    template<std::size_t Width>
    static std::vector<std::array<double, Width>> ReadCSV3D(const std::filesystem::path& directory,
                                                          const nlohmann::json& field,
                                                          int frames, int items_per_frame) {
        const auto values = ReadCSV(directory, field, frames, int(Width) * items_per_frame);
        std::vector<std::array<double, Width>> result(values.size() / Width);
        for (std::size_t i = 0; i < result.size(); ++i) {
            std::copy_n(values.data() + i * Width, Width, result[i].begin());
        }
        return result;
    }
};

}  // BUMI3 命名空间结束，避免与原 G1 的同名动作类型冲突。
