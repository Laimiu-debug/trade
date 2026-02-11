# 威科夫选股系统 v1.0

以威科夫量价分析为核心的 A 股选股与回测平台。支持阶段扫描、事件筛选、结构评分、策略回测与报告导出。

## 功能概览

- 多源数据接入：通达信 TDX 本地日线、CSV 单文件（多股票）、CSV 文件夹（每股一文件）
- 五层漏斗筛选：板块/市值/上市天数 → 趋势 → 量能 → 均线 → 综合评分
- 威科夫结构分析：HH/HL/HC 结构识别、阶段判定（吸筹/派发 A-E）、关键事件检测
- 威科夫事件识别：PS、SC、AR、ST、TSO、Spring、SOS、JOC、LPS（吸筹侧）及 UTAD、SOW、LPSY（派发侧）
- 八步标准序列检测：PS → SC → AR → ST → Spring → SOS → JOC → LPS，支持完整度评估
- 独立回测模块：T+1 约束、可配置仓位/成本/止损止盈、优中选优排序、TopK 限流
- 回测报告导出：交易明细 CSV、完整 ZIP（权益曲线、指标、参数快照）
- K 线图可视化：叠加威科夫事件标记、阶段提示、结构信号
- 元数据增强：支持 API 补全股票名称、行业、板块、市值等信息

## 快速开始

### 环境要求

- Python 3.10+
- 依赖包见 `requirements.txt`

### 安装依赖

```powershell
pip install -r requirements.txt
```

### 本地运行

```powershell
.\run.ps1
```

或手动启动：

```powershell
streamlit run app/streamlit_app.py
```

### 打包 EXE

```powershell
.\build_exe.ps1
```

生成文件位于 `dist\TrendPicker.exe`。

## 使用流程

1. 选择数据来源（TDX 本地 / CSV 单文件 / CSV 文件夹），加载行情数据
2. 可选上传元数据 CSV 或启用 API 补全
3. 配置筛选参数（趋势窗口、量能、均线、板块等）
4. 运行分层筛选，查看结果表格
5. 点击个股查看 K 线图，确认威科夫阶段与结构
6. 进入威科夫事件回测模块，配置回测参数
7. 分析回测指标与交易明细，导出报告

## 数据格式

### 行情 CSV（必须）

至少包含：`日期`、`开盘`、`最高`、`最低`、`收盘`、`代码`

```
日期,开盘,最高,最低,收盘,成交量,成交额,代码,名称
```

### 元数据 CSV（可选）

```
代码,名称,板块,上市天数,流通市值,行业,板块5日涨幅
```

### 通达信 TDX 本地数据

程序自动定位 `vipdoc` 目录读取 `.day` 日线数据，也支持手动指定路径和深度扫描。

## 威科夫事件回测

回测模块独立于分层筛选，只复用数据和可选股票池。


### 回测区间模式

- 按最近 K 线数：使用每只股票最近 N 根 K 线
- 自定义日期区间：指定开始/结束日期

### 关键回测参数

| 参数 | 说明 |
|------|------|
| 单边交易成本(bps) | 每次买/卖各收一次，10 bps = 0.10% |
| 最大持仓K线 | 单笔交易最长持有K线数，超出自动离场 |
| 冷却K线 | 交易结束后等待 N 根K线才能再次入场 |
| 最大并发持仓 | 同一时刻最多持有的股票数量 |
| 同日候选TopK | 同日仅保留评分前 K 名信号，0 为不限 |
| T+1约束 | 入场当天禁止卖出，符合 A 股规则 |

### 回测指标

| 指标 | 说明 |
|------|------|
| Win Rate | 盈利交易占比 |
| Avg Trade | 每笔平均收益率 |
| Cum Return | 累计收益率 |
| Max DD | 最大回撤（权益峰值到谷值的最大跌幅） |
| Fill Rate | 成交笔数 / 候选信号笔数 |
| Skipped/MaxPos | 被组合约束跳过的信号数 |

## 威科夫事件说明

| 事件 | 英文全称 | 中文 | 侧 |
|------|----------|------|-----|
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

## 目录结构

```
app/
  streamlit_app.py   # 可视化主程序
  run_app.py         # EXE 启动入口
  user_settings.json # 用户设置持久化
build_exe.ps1        # 打包脚本
run.ps1              # 本地运行脚本
requirements.txt     # Python 依赖
```

## 依赖

- streamlit 1.31.1
- pandas 2.2.2
- numpy 1.26.4
- plotly 5.22.0
- openpyxl 3.1.5
- reportlab 4.2.2
- streamlit-aggrid 0.3.4

## 常见问题

- 行情 CSV 没有"代码"列：导出时包含代码列，或使用文件夹模式通过文件名提取代码
- EXE 无法启动：先用 `run.ps1` 验证正常运行，再重新打包
- 回测区间设置：在回测模块中先选"回测区间模式"，再填写对应参数
- TDX 路径找不到：尝试启用"深度扫描磁盘"或手动输入 vipdoc 路径

## 风险提示

回测收益不代表未来收益。威科夫事件是概率型证据，不是确定性买卖点，建议结合阶段、结构、量能综合判断。
