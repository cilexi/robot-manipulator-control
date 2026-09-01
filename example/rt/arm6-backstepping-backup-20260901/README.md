# 第六关节 Backstepping 实验备份

备份日期：2026-09-01（Asia/Shanghai）

本目录归档 `nomal_backstepping` 的第六关节控制实验。`M` 表示配置项
`nominal_inertia6`，`k1`、`k2` 分别表示 `k1_6`、`k2_6`。

## 目录

- `code/arm6_backstepping_tune_run.py`：第六关节调参、运行、分析和绘图脚本。
- `code/arm6_backstepping_controller.cpp`：第六关节控制程序源码。
- `code/arm6_backstepping.cfg`：归档时使用的第六关节参数配置。
- `records/`：当前磁盘中仍存在的 6 组完整实验记录。
- `summaries/`：历史调参、惯量扫描及对比汇总表。
- `SHA256SUMS`：备份包内文件校验值。

## 命名规则

实验目录：

`arm6-YYYYMMDD-HHMMSS-k1-<k1_6>-k2-<k2_6>-M-<nominal_inertia6>[-aborted]`

单次实验文件：

`arm6-k1-<k1_6>-k2-<k2_6>-M-<nominal_inertia6>-<内容类型>`

文件名中的小数点使用 `p`，例如 `M-0p135` 表示
`nominal_inertia6 = 0.135`。提前终止的实验使用 `aborted` 后缀。

## 原目录到备份目录的映射

| 原记录 | 备份名称 | 状态 |
|---|---|---|
| `20260901_113628` | `arm6-20260901-113628-k1-46-k2-40-M-0p05` | 完成 |
| `20260901_150528` | `arm6-20260901-150528-k1-62-k2-50-M-0p05` | 完成 |
| `20260901_152145` | `arm6-20260901-152145-k1-62-k2-50-M-0p0425` | 完成 |
| `20260901_152223` | `arm6-20260901-152223-k1-62-k2-50-M-0p0425-aborted` | 中止 |
| `20260901_160459` | `arm6-20260901-160459-k1-50-k2-40-M-0p135` | 完成 |
| `20260901_160639` | `arm6-20260901-160639-k1-62-k2-50-M-0p135` | 完成 |

## 汇总表命名

- `arm6-k-tuning-summary.csv`：早期 k 与自适应参数调节。
- `arm6-k-tuning-cumulative-error-summary.csv`：以累积绝对误差为指标的 k 调节。
- `arm6-M-scan-k1-50-k2-40-summary.csv`：固定 k 条件下的小范围 M 扫描。
- `arm6-M-expand-k1-50-k2-40-summary.csv`：固定 k 条件下的放大步长 M 调节。
- `arm6-M-0p120-to-0p150-k1-50-k2-40-comparison.csv`：M=0.120–0.150 对比。

部分汇总表引用的早期实验目录在归档时已不在磁盘中，因此汇总数据得到保留，
但对应的原始日志和图片不能补入本备份。现存 6 组记录均完整包含配置、CSV 日志、
指标 JSON、控制台输出和结果图片。

## 使用说明

备份代码不替换工程当前入口。若要继续复现实验，可使用原工程已构建的
`build/bin/nomal_backstepping`：

```bash
python3 example/rt/arm6-backstepping-backup-20260901/code/arm6_backstepping_tune_run.py
```

新运行结果会自动写入本备份的 `records/`，并按上述 arm6 参数规则命名。
若要单独编译备份的 C++ 源码，应将其作为 `arm6_backstepping` 构建目标加入 CMake；
当前工程 CMake 未改动，以免影响现有实验环境及后续第五关节开发。
