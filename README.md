# Solo Tape · US Market Direction Terminal

一个全自动更新的美股方向判断看盘看板，聚焦**趋势水位、市场情绪与信用红线**，用于在盘后快速判断市场所处阶段与风险边界。

## 核心视角

- **10 秒极简结论**：超买/超卖、市场情绪、个人动作建议三栏对照。
- **核心三图**：
  - 标普 500（SPX）趋势与 50/200 日均线、站上 50 日线占比（`$S5FI`）
  - CNN 恐慌贪婪指数仪表盘
  - 高收益债利差（HY OAS）与信用风险参考（5.0% 熊市红线）
- **2×2 深度验证**：AAII 散户情绪、CBOE VIX 期限结构、2Y/10Y 美债利率与期限利差、费城半导体（SOX）及机构/期权指标。

## 技术与自动化

- **前端展示**：HTML / Tailwind CSS / ECharts，托管于 **GitHub Pages**
- **数据更新**：Python 抓取市场与利率数据，通过 **GitHub Actions** 定时自动提交 `data.json`
- **在线看板**：`https://solo-tape.github.io/us-market-dashboard/`

> 免责声明：本仓库为个人投研与数据可视化工具记录，所有数据与图表仅供研究参考，不构成任何投资建议。
