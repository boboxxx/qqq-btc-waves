# QQQ 与 BTC 波浪进度

GitHub Actions 每 30 分钟运行一次 `code/wave_tracker.py`：从 Yahoo Finance 取日线收盘价，按 `code/waves/wave_count.json` 里的数法判断现在走到哪一浪，然后把页面发布到 GitHub Pages。

- 当前这一浪从极值反向走出 5%（BTC 12%），记为“可能已结束”。
- 收盘越过这一浪里最后一个反向摆动点才算确认；如果这一浪里没有这样的摆动点，就要越过这一浪的起点。确认后 A → B → C 自动推进。
- 状态和事件只按已收盘的日线判断。`data/waves/wave_state.json` 和 `wave_events.csv` 由 Actions 写回仓库，页面上的盘中价只做参考。
- 下载失败时这次不发布，页面保留上一版。页面顶部会显示更新时间。
- 每个大浪自动细分两级：中浪 (i)–(v) / (a)–(c)，小浪 i–v / a–c。推动浪分 5 段，调整浪分 3 段。在候选转折点里选回调最大、同时满足三条铁律的分法。页面上可以切换级别，也可以放大到近一年、近半年、近三月。

改数法（例如人工确认某个转折点）时，只改 `wave_count.json`，程序会重新建立基线，不补发事件。

本地运行：

```
pip install -r requirements.txt
python code/wave_tracker.py
```

这是对走势的事后描述，不是交易建议。
