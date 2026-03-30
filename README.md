# 使用分子动力学模拟离子阱系统

## Part 1: Paul Trap

本项目是一个基于 Python 的离子阱 (Ion Trap) 分子动力学模拟器，旨在模拟和可视化带电粒子（离子）在保罗阱 (Paul Trap) 中的运动、冷却及结晶过程。

项目中的默认参数为${}^{171}\text{Yb}^+$离子，具有以下物理属性（可在 `units.py` 中修改）：
- **质量**: $m = 171 \, \text{u} \approx 2.84 \times 10^{-25} \, \text{kg}$
- **电荷**: $q = +e \approx 1.60 \times 10^{-19} \, \text{C}$
- **激光冷却参数**: $\lambda\approx 369.52\,\text{nm},\ \Gamma\approx 2\pi\times19.6\,\text{MHz},\ s=1$

### 1. 物理引擎 (Physics Engine)
该项目核心采用分子动力学 (Molecular Dynamics) 方法，对离子在电磁场中的运动进行数值求解。

- **动力学方程**: 系统遵循 Langevin 动力学方程，包含保守力、耗散力和随机力。
  - **保守力**:
    - **保罗阱势场**: 模拟为各向异性的谐振子势阱，提供回复力 $F_\text{trap} = -m\omega_i^2 x_i$ ($i=x,y,z$)。
    - **库仑相互作用**: 模拟离子间的静电排斥力，计算 $N$ 体相互作用。
  - **耗散与随机力**:
    - **激光冷却**: 引入与速度成正比的阻尼力 $-\gamma_\text{laser} v$。
    - **热浴耦合**: 引入环境热阻尼 $-\gamma_\text{thermal} v$ 和对应的布朗运动随机力（涨落耗散定理）。

- **数值积分算法**: 采用 **BBK (Brünger-Brooks-Karplus)** 积分器。这是一种适用于 Langevin 动力学的改进型 Verlet 算法，具有良好的数值稳定性，能够准确模拟包含摩擦和热噪声的系统演化。
  - 算法利用半步速度预估和力场更新，确保了能量守恒（在无耗散下）和正确的正则系综采样。

- **三引擎架构 (Triple-Engine Architecture)**:
  本项目实现了三种物理计算后端，以适应不同的计算需求：
  1.  **NumPy Engine (`paul_trap.py`)**:
      - 基于 CPU 的标准实现。
      - 利用 NumPy 向量化运算，适合中小规模系统（< 100 离子）。
      - 通用性强，无额外硬件依赖。
  2.  **PyTorch Engine (`paul_trap_torch.py`)**:
      - **GPU 加速**: 自动检测 CUDA 设备。在 GPU 上运行时，能够高效处理大规模离子晶体（> 500 离子）的 N-体相互作用计算。
      - **自动回退**: 若无 GPU，自动回退到 CPU 张量计算。
  3.  **JAX Engine (`paul_trap_jax.py`)**:
      - **新功能**: 基于 JAX 张量运算与 JIT 编译实现。
      - **设备适配**: 支持 CPU / CUDA GPU / TPU（取决于已安装的 jaxlib 版本）。
      - **后端优先级**: 在支持 JAX 的入口中，后端选择顺序为 **JAX -> PyTorch -> NumPy**。

### 2. 交互界面与使用方式 (Interfaces)
项目提供了三种交互方式，分别适配不同的使用场景和物理引擎。

#### A. 标准 GUI (`gui.py`)
- **后端**: 默认使用 **NumPy** 引擎。
- **渲染**: 基于 `Matplotlib`。
- **特点**: 拥有最完整的交互控件（滑块、按钮、文本框），适合教学演示和参数探索。
- **启动**: `python gui.py`

#### B. 高性能 GUI (`gui_vispy.py`)
- **后端**: **自适应**，优先级为 **JAX -> PyTorch -> NumPy**。
- **渲染**: 基于 `VisPy` (OpenGL) 和 `PyQt5`。
- **特点**: 利用 GPU 进行物理计算和图形渲染。极高的帧率，适合实时模拟大规模离子晶体。
- **启动**: `python gui_vispy.py`

#### C. 终端交互 TUI (`tui.py`)
- **后端**: **自适应**，优先级为 **JAX -> PyTorch -> NumPy**。
- **渲染**: 基于 `curses` (终端字符界面)。
- **特点**: 轻量级，适合在无图形界面的服务器或 SSH 远程连接中使用。
- **启动**: `python tui.py`

### 3. 界面功能详解 (Interface Features)
*(以标准 GUI 为例，VisPy 版与 TUI 实现了主要核心功能)*

#### 实时数据 (Real-time Metrics)
界面实时显示系统的关键状态：
- **FPS**: 模拟的渲染帧率。
- **Time**: 物理模拟的累积时间。
- **Temp**: 离子系统的实时温度（由平均动能计算得出）。
- **Count**: 当前阱中捕获的离子总数。
- **Lindemann** (TUI): Lindemann 参数，衡量离子晶体的结构稳定性，定义为相邻离子平均距离的标准差与平均距离之比。

#### 控制面板 (Control Panel)
- **参数滑块 (Sliders)**
  - **Freq X / Y / Z**: 独立调节 $x, y, z$ 三个方向的阱频率，可观察结构相变。
  - **Laser Damp**: 调节激光冷却的阻尼因数 $\gamma_\text{laser}$。
  - **Therm Damp**: 调节热浴的耦合强度 $\gamma_\text{thermal}$。
  - **Temp Set**: 设定热浴的目标温度。
  - **Zoom**: 调整视图观察范围。
  - **Steps**: 调整每帧渲染对应的物理模拟步数 (SPF)，平衡速度与流畅度。

- **功能开关 (Switches)**
  - **Laser**: 开启/关闭激光冷却。
  - **Real Laser**: 锁定为真实的物理实验参数（基于 `units.py` 定义）。
  - **Vacuum**: 开启真空模式（即关闭热浴耦合）。
  - **Stochastic**: 开启/关闭随机噪声。

- **操作按钮 (Actions)**
  - **Add Ion**: 动态添加离子。
  - **Discard / Catch**: 清空或重置并捕获离子。
  - **Reset**: 重置系统状态。

### 4. 依赖与安装 (Requirements)
运行本项目需要安装以下 Python 包：

**基础依赖 (Basic)**
- `numpy`: 数值计算
- `matplotlib`: 标准 GUI 绘图
- `scipy`: 科学常数和物理单位定义

**高性能/高级依赖 (Advanced)**
- `jax`: JAX 引擎前端
- `jaxlib`: JAX 设备后端（CPU 或 CUDA 版本）
- `torch`: PyTorch 引擎 (支持 GPU)
- `vispy`: 高性能 OpenGL 绘图
- `PyQt5`: VisPy 的 GUI 后端框架

**安装命令**:
```bash
pip install numpy matplotlib scipy
pip install jax jaxlib
pip install torch vispy PyQt5
```

> 说明：若需 JAX 的 CUDA 加速，请按 JAX 官方文档安装与 CUDA 版本匹配的 `jaxlib` 轮子；仅执行 `pip install jax jaxlib` 通常会安装 CPU 版本。
