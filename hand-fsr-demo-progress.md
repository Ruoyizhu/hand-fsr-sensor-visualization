# Hand FSR Demo - 项目进度记录

## 项目基本信息
- **项目名**：基于力传感器的手功能系统评估_毕设（hand-fsr-demo）
- **路径**：E:\基于力传感器的手功能系统评估_毕设\hand-fsr-demo
- **技术栈**：Python 3.10+、PyQt6、numpy、pyqtgraph（后续 ROS2）
- **开发环境**：Windows（主开发）+ Ubuntu 虚拟机（ROS2 集成）

## 已完成任务

### ✅ 任务 1：工程最小骨架
- 创建虚拟环境 `.venv`
- 创建 `requirements.txt`（PyQt6>=6.7、numpy>=1.26）
- 实现 `src/main.py` 空窗口（可启动、可关闭）
- 验证：环境已正常运行

### ✅ 任务 4/5：QThread + 信号机制
- 实现 `MockDataSource` 数据源类（模拟 16x16 帧）
- 实现 `Worker(QObject)` 后台线程
- 实现 `MainWindow`：连接/断开按钮、日志输出、状态标签
- 关键：UI 刷新固定 20Hz（50ms 周期），数据源可高频产出
- 线程安全：用 pyqtSignal() 传递帧，主线程缓存最新帧

### ✅ 任务 6：16x16 自适应热力图
- 实现 `HeatmapGridWidget` 自定义控件
  - **固定 16 行 × 16 列 = 256 个正方形单元**
  - **单元尺寸动态计算**：`min(可用宽度/16, 可用高度/16)`
  - **间隙可配置**（默认 2 像素）
  - **随窗口自适应**

- 模拟数据源增强：压力值范围 0-1000，五种场景循环
  - uniform：均匀分布 + 棋盘扰动
  - center：中心高、周边低（高斯衰减）
  - point：多个随机点状压力核
  - line：横纵高压带
  - impact：冲击点（极高峰值、快速衰减）
  - **每 20 帧更新一次场景**（核心需求）

- 高对比色图支持：jet / viridis / plasma
  - 手写 LUT 映射（256 级颜色）
  - 低压力 → 冷色，高压力 → 暖色

- 统计信息显示：sum / max / min / mean / UI_fps / 当前场景

## 关键代码结构

```
src/main.py
├── @dataclass Frame
│   └── timestamp, index, matrix(16x16), scenario
├── class MockDataSource
│   └── 五种压力场景生成函数
├── class Worker(QObject)
│   └── frame_received, status_changed, finished 信号
├── class HeatmapGridWidget(QWidget)
│   └── paintEvent 自定义绘制 16x16 网格
├── class MainWindow(QMainWindow)
│   └── UI 布局、线程管理、信号连接、QTimer 刷新
└── main() -> QApplication
```

## 当前代码已支持的功能
1. ✅ 后台 Worker 线程产出帧（无 UI 阻塞）
2. ✅ 主线程 QTimer 固定 20Hz 刷新（即使数据源 >50Hz）
3. ✅ 16x16 严格正方形单元、自适应尺寸
4. ✅ 可配置色图和间隙（实时生效）
5. ✅ 五种压力场景自动轮换（每 20 帧）
6. ✅ 统计量实时显示（sum/max/min/mean/fps）
7. ✅ 连接/断开安全（干净线程退出）

## 下一步任务（待做）

### 任务 7：设置页（可选增强）
- 显示单位选择（raw / g / N）
- 小数位数、刷新周期等
- 用 QSettings 持久化

### 任务 8：数据记录与回放
- CSV/NPZ 格式存储帧数据
- 回放模块（无需硬件可演示）

### 任务 9：串口接入
- `SerialByteSource` 读取硬件数据
- 帧解析器 `FrameParser`（根据协议切帧、校验、解析 256 点）
- 需获取串口协议详情（帧头、格式、端点等）

### 任务 10：ROS2 集成
- 在 Ubuntu 虚拟机上实现 ROS2 Publisher
- 保留单代码库、双环境运行方案

## 易踩坑清单
- ⚠️ PyQt6 线程规则：Worker 必须在 QThread 中运行，所有 UI 更新必须在主线程（用 signal 传递）
- ⚠️ UI 刷新频率：即使数据源快，也要用 QTimer 限制 UI 刷新，避免过度渲染
- ⚠️ 虚拟环境：务必激活 `.venv`，使用 `.venv\Scripts\python.exe` 执行，避免系统 Python 混用
- ⚠️ 热力图单元：计算单元尺寸时要用 `min()` 确保宽高相同（正方形）

## 部署与验收
| 环节 | 命令 |
|------|------|
| 虚拟环境激活 | `.\.venv\Scripts\Activate.ps1` |
| 依赖安装 | `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| 运行程序 | `.\.venv\Scripts\python.exe src\main.py` |
| 验收要点 | 点连接→看热力图变化→点断开；重复20次无崩溃 |

## 当前 requirements.txt
```
PyQt6>=6.7
numpy>=1.26
pyqtgraph>=0.13
```
（后续加 pyserial、rclpy 等）

---
**最后更新**：2026-04-05  
**下次接续点**：可继续执行任务 7（设置页）或任务 9（串口接入，需协议描述）
