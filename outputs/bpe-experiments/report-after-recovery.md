# 存储恢复后的核验

最初 200 MiB 训练的日志保留，但故障后的旧 pickle 为 0 字节。新目录 optimized-200m-recovered 中的 bpe.pkl 已重载验证。复跑训练 113.348 秒，峰值 RSS 1,068,296 KiB，10,000 词表、9,743 次合并，唯一最长项为带前导空格的 enthusiastically（17 字节）。复跑与完整语料下载重叠。此前的 87.667 秒仍是一次独立的训练耗时观测。恢复后官方 BPE 测试 3/3 通过，5 MiB、50 MiB 和 OpenWebText 32K 产物也再次重载通过。

## 保留集比较

源于同一公开训练文件中位于各训练前缀之后的完整文档。TinyStories 有 1,302 篇、1,030,570 字节，OpenWebText 有 234 篇、1,040,478 字节。统计不含文档分隔符。

| 样本训练的分词器 | TinyStories 每 token 字节数（token 数） | OpenWebText 每 token 字节数（token 数） |
|---|---:|---:|
| TinyStories 200 MiB / 10K | 4.0743（252944） | 3.1822（326968） |
| OpenWebText 16 MiB / 32K | 3.9595（260276） | 4.3066（241603） |

每套词表在各自保留集上使用的 token 都更少。词表容量、训练规模和语料来源均不同，结果未控制这些变量。复现入口为 compare.py，原始输出见 comparison.log。

WSL 总内存为 5,924 MiB。200 MiB 样本已用约 1.02 GiB RSS，当前实现仍一次性读取并切分全文。全量 TinyStories 的线性 RSS 外推约 10.8 GiB，现有资源下未启动全量。OpenWebText 只执行了端到端样本训练，未启动全量。未使用付费资源。

## 上一轮性能记录
# CS336 BPE 样本训练报告

| 实验 | 词表 / 合并 | 训练秒数 | 峰值 RSS KiB | 最长普通 token |
|---|---:|---:|---:|---|
| TinyStories 5 MiB 原版 | 10000 / 9743 | 12.16 | 207304 | `' congratulations'`，16 字节 |
| TinyStories 5 MiB 优化 | 10000 / 9743 | 2.766 | 38260 | 同上 |
| TinyStories 50 MiB 优化 | 10000 / 9743 | 20.831 | 182488 | `' enthusiastically'`，17 字节 |
| TinyStories 200 MiB 优化 | 10000 / 9743 | 87.667 | 1068312 | 同上 |
| OpenWebText 16 MiB 优化 | 32000 / 31743 | 22.204 | 216604 | 32 个连字符和 32 个下划线并列 |

50 MiB、200 MiB 和 OpenWebText 样本分别为 52428193、209714825 和 16775330 字节，取公开训练集前缀并截断至完整文档边界。超时 600 秒，地址空间上限分别为 3.5、3.5 和 2 GiB。200 MiB 与其他任务有运行重叠。旧版 RSS 来自 GNU time，优化版来自 getrusage。

原版与优化后的 5 MiB 产物重载后，词表字典和有序合并列表完全相同，最长项唯一。样本均验证连续 ID、256 个基础字节、唯一特殊 token、合并输入存在以及新增词表项等于合并字节。官方 BPE 测试优化前后均为 3/3。

候选 pair 改用堆和惰性失效，保留频次和字节对顺序；按文档段累计预分词词频。原版 cProfile 训练累计 23.530 秒，其中 find_best_pair 17.582 秒。优化后累计 6.055 秒，其中 heappop 0.209 秒，to_byte_tokens 3.230 秒。

从仓库根目录以 PYTHONPATH=. 运行 outputs/bpe-experiments/experiment.py，传入语料路径和 --output 新目录。OpenWebText 增加 --vocab-size 32000。该目录下 optimized-5m、optimized-50m、optimized-200m 和 owt-32k 分别保存 bpe.pkl，optimized.prof 是优化后的采样。原始采样见 outputs/profile-5m-10k/training.prof。

50 到 200 MiB 的时间增至 4.2 倍，RSS 增至 5.9 倍。按 200 MiB 简单线性外推至 2.23 GB 约为 15.5 分钟和 10.8 GiB RSS，实测内存增幅更快，当前 5.8 GiB WSL 不适合直接全量训练。32 GB 内存的 CPU 机器可用于进一步验证，仍须遵守 30 GB 和 30 分钟上限。OpenWebText 全量预算为 100 GB、12 小时。

TinyStories 全量下载写入失败，WSL 启动与工作区写入随后出现 I/O 错误，Windows C 盘只有约 34 MiB 剩余。两项正式训练和保留集每 token 字节数比较待完成。可在同一批保留文档上用两套分词器测量每 token 字节数。当前样本词表大小不同，比较结果会同时包含词表容量与语料差异。


## 完整训练语料核验

TinyStories 官方训练文件已下载至 data/TinyStoriesV2-GPT4-train.txt，实际大小 2,227,753,162 字节。SHA-256 与发布者记录一致，为 6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443。OpenWebText 完整训练语料未下载，32K 样本使用课程 gzip 训练文件的前缀。
