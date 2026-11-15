# TinyStories 5 MiB BPE 样本实验

输入为 tests/fixtures/tinystories_sample_5M.txt（5,242,880 字节）。目标词表 10,000，特殊 token 为 `<|endoftext|>`，运行超时 600 秒。

| 指标 | 结果 |
|---|---:|
| 词表项 | 10,000 |
| 合并次数 | 9,743 |
| 训练时间（原始运行日志） | 12.16 秒 |
| 命令墙钟时间（原始运行日志） | 12.48 秒 |
| GNU time 峰值 RSS（原始运行日志） | 207,304 KiB（202.45 MiB） |
| cProfile 训练时间（独立复跑） | 23.55 秒 |

脚本报告的最长项为 `b' congratulations'`（16 字节，含前导空格）。脚本仅显示第一个最长项，未核验并列项。原始与复跑的 pickle 文件 SHA-256 相同，均为 815E9970915DAB8DC1CA780D12ABD895E29B32D883589D35A64877346FAA41F7。

## 性能分析

| 函数 | 调用次数 | 累计时间（秒） |
|---|---:|---:|
| train_bpe | 1 | 23.530 |
| find_best_pair | 9,743 | 17.582 |
| to_byte_tokens | 1 | 3.574 |
| pre_tokenize | 6,458 | 0.600 |
| word_pairs | 98,884 | 0.533 |
| merge_word | 45,379 | 0.225 |

find_best_pair 占插桩训练时间约 74.7%，比较键调用 53,695,350 次。累计时间包含子调用，表内行不可直接相加。原始数据保存在同目录的 training.prof。

## 正式训练准备

- TinyStories 完整训练文本为 2.23 GB。讲义预算为 CPU 30 分钟和 30 GB RAM。当前实现保留全文、预分词和 pair 索引，单次 5 MiB 结果不足以判断全量能否在 5.8 GiB WSL 内运行。可按完整文档逐档扩大样本，采集墙钟、峰值 RSS 和候选扫描耗时，再选本地内存或 CPU 实例容量。[训练文件](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-train.txt)
- OpenWebText 使用课程的 owt_train.txt.gz（4.59 GB 压缩），解压后保留文档边界并按特殊 token 切分预分词。讲义预算为 32,000 词表、CPU 12 小时和 100 GB RAM。[训练文件](https://huggingface.co/datasets/stanford-cs336/owt-sample/blob/main/owt_train.txt.gz)
- 对两套词表，在同一批保留的 TinyStories 与 OpenWebText 文本上比较每 token 字节数，并记录最长 token、训练墙钟与峰值 RSS。若要观察语料本身的影响，可额外使用相同词表大小作对照。

两项正式训练均未启动。OpenWebText 训练入口仍待准备。样本 pickle 的反序列化检查和最长项并列检查仍待完成。
