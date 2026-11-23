# pairwise-video-stabilize

## 领域

游戏、图形与多媒体

## 主题

视频电子防抖

## 说明

时间戳对齐、滚动快门与裁剪窗口。

初始状态下 `media/video_stabilize.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
