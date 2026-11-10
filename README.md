# ROS2 — ROSOrin Pro / Orin NX

ROSOrin Pro 小车 ROS 2 资料的模块化整理副本，仓库应保持 **Private**。

## 整理原则

- 25 个 ROS 包按功能分目录；原始源码、launch、参数、模型与资源内容没有修改。
- 998 个原始文件均在本地副本保留；逐文件 SHA-256 已与 ZIP 内容比较通过。
- 仅一份包含疑似访问密钥的配置不上传：`src/ai/large_models/large_models/config.py`。使用前从本地保留副本恢复，或自行提供等效私有配置；不要提交密钥。
- 大模型、网格、推理引擎等二进制资源通过 Git LFS 保存；保留原有重复资源，避免改动程序资源定位。
- 原工作空间根文件放在 `upstream-metadata/`，包含原 `.gitmodules`、`.gitignore`、环境脚本与命令资料，供追溯，不自动应用。

## 模块索引

| ROS 包 | 新位置 | 压缩包内原位置（src_ros2/ 下） |
|---|---|---|
| `large_models` | [ai/large_models](src/ai/large_models/) | `src/large_models/large_models` |
| `large_models_examples` | [ai/large_models_examples](src/ai/large_models_examples/) | `src/large_models_examples` |
| `app` | [applications/app](src/applications/app/) | `src/app` |
| `example` | [applications/example](src/applications/example/) | `src/example` |
| `navigation` | [autonomy/navigation](src/autonomy/navigation/) | `src/navigation` |
| `slam` | [autonomy/slam](src/autonomy/slam/) | `src/slam` |
| `sdk` | [common/sdk](src/common/sdk/) | `src/driver/sdk` |
| `rosorin_description` | [description/rosorin_description](src/description/rosorin_description/) | `src/simulations/rosorin_description` |
| `controller` | [drivers/controller](src/drivers/controller/) | `src/driver/controller` |
| `ros_robot_controller` | [drivers/ros_robot_controller](src/drivers/ros_robot_controller/) | `src/driver/ros_robot_controller` |
| `test_stand` | [experiments/test_stand](src/experiments/test_stand/) | `src/test_stand` |
| `openclaw_controller` | [integrations/openclaw_controller](src/integrations/openclaw_controller/) | `src/openclaw_controller` |
| `interfaces` | [interfaces/interfaces](src/interfaces/interfaces/) | `src/interfaces` |
| `kinematics_msgs` | [interfaces/kinematics_msgs](src/interfaces/kinematics_msgs/) | `src/driver/kinematics_msgs` |
| `large_models_msgs` | [interfaces/large_models_msgs](src/interfaces/large_models_msgs/) | `src/large_models/large_models_msgs` |
| `ros_robot_controller_msgs` | [interfaces/ros_robot_controller_msgs](src/interfaces/ros_robot_controller_msgs/) | `src/driver/ros_robot_controller_msgs` |
| `servo_controller_msgs` | [interfaces/servo_controller_msgs](src/interfaces/servo_controller_msgs/) | `src/driver/servo_controller_msgs` |
| `xf_mic_asr_offline_msgs` | [interfaces/xf_mic_asr_offline_msgs](src/interfaces/xf_mic_asr_offline_msgs/) | `src/xf_mic_asr_offline_msgs` |
| `kinematics` | [manipulation/kinematics](src/manipulation/kinematics/) | `src/driver/kinematics` |
| `servo_controller` | [manipulation/servo_controller](src/manipulation/servo_controller/) | `src/driver/servo_controller` |
| `multi` | [multi_robot/multi](src/multi_robot/multi/) | `src/multi` |
| `peripherals` | [sensors/peripherals](src/sensors/peripherals/) | `src/peripherals` |
| `xf_mic_asr_offline` | [speech/xf_mic_asr_offline](src/speech/xf_mic_asr_offline/) | `src/xf_mic_asr_offline` |
| `bringup` | [system/bringup](src/system/bringup/) | `src/bringup` |
| `calibration` | [tools/calibration](src/tools/calibration/) | `src/calibration` |

## 开始阅读

1. [架构与上传前问题分析](docs/analysis.md)
2. [原路径、新路径、文件 SHA-256 与上传状态](docs/source-manifest.csv)
3. [来源压缩包校验值](docs/provenance.json)
4. [缺失的安装入口](docs/missing-entrypoints.csv)

先读 `bringup` 的整机启动，再读 `controller` 的速度/里程计、`ros_robot_controller` 的下位机通信，以及 `servo_controller` 和 `kinematics` 的机械臂控制。

## 获取资源与构建状态

克隆后先安装 Git LFS，然后在仓库根目录运行：

```bash
git lfs install
git lfs pull
```

原环境脚本指向 ROS 2 Humble 和 Orin 的多个第三方工作空间。该副本完成了静态归类和内容一致性检查，**尚未在 Linux/Orin 上构建或运行验证**。

源码仍含 `/home/ubuntu/ros2_ws/src/...` 等旧绝对路径，目录归类后这些路径不会自动更新。原 `.typerc` 的 `need_compile=False` 分支尤其依赖旧布局。也存在未安装的嵌套资源、未声明的依赖和缺失入口等待核查项目，因此当前整理不能被当作开箱即用安装包。详见分析文档；本次按要求不修复源码。

`calibration` 引用 `test_stand`，`openclaw_controller` 引用大模型示例，不能把这些目录直接视为相互独立的可选组件。

## 来源与许可

来源为用户本地厂商资料 `src_ros2-main.zip`；没有导入原 Git 历史。保留各包原有作者、许可证声明和 LICENSE 文件。多数包许可证仍为 TODO，未授予统一新许可证；第三方 SDK、模型和网格的再分发许可待确认。私有归档不代表已获得公开发布授权。

原 `.gitmodules` 指向内网服务，作为历史资料保留在 `upstream-metadata`，当前大模型源码以普通目录纳入本仓库。
