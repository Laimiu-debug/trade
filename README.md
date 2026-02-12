# 威科夫选股系统（TrendPicker）

基于威科夫量价分析的 A 股选股与事件回测工具，提供从数据接入、分层筛选、结构分析到回测报告导出的完整流程。

## 快速开始

### 环境要求

- Windows + PowerShell（项目内含 `run.ps1`、`build_exe.ps1`）
- Python 3.10+

### 安装依赖

```powershell
pip install -r requirements.txt
```

### 本地运行

推荐直接使用脚本（自动创建虚拟环境并安装依赖）：

```powershell
.\run.ps1
```

或手动启动：

```powershell
streamlit run app/streamlit_app.py
```

默认访问地址：

```text
http://127.0.0.1:8501
```

### 打包 EXE

```powershell
.\build_exe.ps1
```

打包产物：

```text
dist\TrendPicker.exe
```

## 核心能力

- 多源数据接入：`TDX 本地日线`、`CSV 单文件（多股票）`、`CSV 文件夹（每股一文件）`
- 分层筛选：板块/市值/上市天数 + 趋势 + 量能 + 均线 + 结构评分
- 威科夫结构识别：HH/HL/HC、阶段判定（吸筹/派发 A-E）
- 威科夫事件识别：PS、SC、AR、ST、TSO、Spring、SOS、JOC、LPS、UTAD、SOW、LPSY
- 八步序列检测：PS -> SC -> AR -> ST -> Spring -> SOS -> JOC -> LPS（支持完整度评估）
- 独立回测模块：T+1、TopK 同日限流、持仓上限、止损止盈、交易成本
- 报告导出：候选结果 CSV、回测交易 CSV、完整回测 ZIP（权益曲线/指标/参数快照）

## 使用流程

1. 选择数据来源并加载行情数据（TDX / CSV）。
2. 可选上传元数据 CSV，或启用 API 补全名称/行业/板块等字段。
3. 配置筛选参数并运行分层筛选。
4. 查看候选池、阶段池、评分与事件识别结果。
5. 查看单股 K 线与阶段/事件标注，进行结构确认。
6. 进入回测页，配置回测参数并运行。
7. 下载回测交易 CSV 或完整报告 ZIP。

## 数据格式

### 行情 CSV（必需）

至少包含下列字段：

```text
日期,开盘,最高,最低,收盘,成交量,成交额,代码,名称
```

### 元数据 CSV（可选）

示例字段：

```text
代码,名称,板块,上市天数,流通市值,行业,板块5日涨幅
```

### TDX 本地数据

支持自动定位或手动指定 `vipdoc` 目录并读取 `.day` 日线数据。

## 回测模块

回测模块与分层筛选解耦，仅复用行情数据和可选股票池。

### 回测区间模式

- `lookback_bars`：按每只股票最近 N 根 K 线回测
- `custom_dates`：使用自定义开始/结束日期

### 关键参数

| 参数 | 说明 |
|------|------|
| 单边交易成本（bps） | 买入和卖出分别收取，10 bps = 0.10% |
| 最大持仓 K 线 | 单笔交易最长持有周期 |
| 冷却 K 线 | 交易结束后等待 N 根 K 线才可再次入场 |
| 最大并发持仓 | 同一时刻最多持有的股票数量 |
| 同日候选 TopK | 同日仅保留评分前 K 个信号，0 表示不限 |
| T+1 约束 | 入场当日禁止卖出，符合 A 股规则 |

### 回测指标

| 指标 | 说明 |
|------|------|
| Win Rate | 盈利交易占比 |
| Avg Trade | 单笔平均收益率 |
| Cum Return | 累计收益率 |
| Max DD | 最大回撤 |
| Fill Rate | 成交笔数 / 候选信号笔数 |
| Skipped/MaxPos | 因组合约束跳过的信号数 |

## 威科夫事件对照

| 事件 | 英文全称 | 中文 | 侧别 |
|------|----------|------|------|
| PS | Preliminary Support | 初步支撑 | 吸筹 |
| SC | Selling Climax | 卖出高潮 | 吸筹 |
| AR | Automatic Rally | 自动反弹 | 吸筹 |
| ST | Secondary Test | 二次测试 | 吸筹 |
| TSO | Terminal Shakeout | 终极震仓 | 吸筹 |
| Spring | Spring | 弹簧/假跌破 | 吸筹 |
| SOS | Sign of Strength | 强势信号 | 吸筹 |
| JOC | Jump Over Creek | 跃过小溪 | 吸筹 |
| LPS | Last Point of Support | 最后支撑点 | 吸筹 |
| UTAD | Upthrust After Distribution | 派发后假突破 | 派发 |
| SOW | Sign of Weakness | 弱势信号 | 派发 |
| LPSY | Last Point of Supply | 最后供应点 | 派发 |

## 项目结构

```text
app/
  streamlit_app.py     # Streamlit 主应用
  run_app.py           # EXE 启动入口
  user_settings.json   # 用户设置持久化
build_exe.ps1          # 打包脚本
run.ps1                # 本地运行脚本
requirements.txt       # Python 依赖
```

## 依赖版本

- streamlit 1.31.1
- pandas 2.2.2
- numpy 1.26.4
- plotly 5.22.0
- openpyxl 3.1.5
- reportlab 4.2.2
- streamlit-aggrid 0.3.4.post3

## 常见问题

- 行情 CSV 没有 `代码` 列：请在数据中补充，或使用“CSV 文件夹模式”从文件名提取代码。
- EXE 无法启动：先用 `.\run.ps1` 验证源码运行正常，再重新打包。
- 回测无成交：检查入场/离场事件、TopK 限流、最大持仓、回测区间是否过严。
- TDX 路径识别失败：启用深度扫描，或手动输入 `vipdoc` 路径。

## 风险提示

回测结果不代表未来收益。威科夫事件属于概率型证据，不构成投资建议。
