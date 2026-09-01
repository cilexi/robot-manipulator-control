# 第五关节顺序调参停止记录

停止时间：2026-09-01 17:10 后（Asia/Shanghai）

已按用户要求停止调参，不再启动新的机械臂实验。停止时没有发生第五关节位移超限，
因此未要求机械臂复位；搜索过程中出现的中止均由第五关节速度保护触发。

## 当前推荐参数

- `k1_5 = 112`
- `k2_5 = 187`
- `nominal_inertia5 = 0.045`
- `gamma_theta = 0.10`
- `sigma_theta = 0.10`
- `alpha_limit5 = 3.0 rad/s`
- `alpha_dot_limit5 = 3.0 rad/s^2`
- `virtual_accel_limit5 = 50.0 rad/s^2`
- `tau_limit5 = 1.0 Nm`

上述参数已写入 `example/rt/nomal_backstepping.py` 默认参数和
`example/rt/nomal_backstepping.cfg`。

## 已锁定区间

| 参数 | 推荐值 | 区间 | 宽度/下界 |
|---|---:|---:|---:|
| `k1_5` | 112 | `[112, 128)` | 14.29% |
| `k2_5` | 187 | `[187, 214)` | 14.44% |
| `nominal_inertia5` | 0.045 | `[0.045, 0.050)` | 11.11% |
| `gamma_theta` | 0.100 | `[0.100, 0.115]` | 15.00% |
| `sigma_theta` | 0.100 | `[0.100, 0.115)` | 15.00% |

## 最终代表实验

最终推荐组合的完整单周期实验目录为 `20260901_170643`：

- 累积绝对误差：`0.0068184259 rad·s`
- RMS 跟踪误差：`0.0007004991 rad`
- 峰值跟踪误差：`0.0038632793 rad`
- 峰值实际命令力矩：`0.935521 Nm`
- 虚拟加速度饱和比例：`1.6667%`
- 力矩速率限制比例：`35.6%`
- 实验未中止，保持关节漂移未超限。

该结果来自单周期 15 秒筛选实验。当前默认配置恢复为两周期、总时长 27 秒，尚未用
最终组合执行两周期复验。

## 日志与图片

- `arm5-control-gain-tuning-summary-20260901.csv`
- `arm5-nominal-inertia5-tuning-summary-20260901.csv`
- `arm5-gamma-theta-tuning-summary-20260901.csv`
- `arm5-sigma-theta-tuning-summary-20260901.csv`
- `arm5-sequential-tuning-final-summary-20260901.csv`
- `arm5-control-gain-recommended-20260901.png`
- `arm5-nominal-inertia5-recommended-20260901.png`
- `arm5-gamma-sigma-final-recommended-20260901.png`

注意：高参数边界附近出现过多次速度保护中止，尤其是 `k1_5/k2_5=128/214`、
`nominal_inertia5>=0.05`、`gamma_theta>=0.2` 以及部分低泄漏组合。后续若恢复实验，
应从当前推荐值开始，不应直接采用这些已中止参数。
