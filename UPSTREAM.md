# 上游仓库记录

本仓库中的 5 个作业目录克隆自 Stanford CS336 官方仓库（2026-07-05），内层 `.git` 已移除。
以后如需同步上游更新，可对照以下 commit 手动 diff：

| 目录 | 上游仓库 | 克隆时 commit |
|------|---------|--------------|
| assignment1-basics | https://github.com/stanford-cs336/assignment1-basics | `a158843b20107949f1a8d7df1b05cd33b9166712` |
| assignment2-systems | https://github.com/stanford-cs336/assignment2-systems | `ca8bc81a59b70516f7ebb2da4808daade877c736` |
| assignment3-scaling | https://github.com/stanford-cs336/assignment3-scaling | `03e9372992e913061b9e78b5cfcb62ad8a87de35` |
| assignment4-data | https://github.com/stanford-cs336/assignment4-data | `0555bea66369872d912652debf10b115ca0688c8` |
| assignment5-alignment | https://github.com/stanford-cs336/assignment5-alignment | `c2734a26308710949fe13226960a1e8cece94b7e` |

同步方法示例：
```bash
git clone --depth 50 https://github.com/stanford-cs336/assignment1-basics /tmp/upstream-a1
cd /tmp/upstream-a1 && git diff a158843b..HEAD   # 查看上游改了什么
```
