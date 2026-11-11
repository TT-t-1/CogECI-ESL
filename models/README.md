# 训练权重

三个第 1 折最佳权重发布于 [GitHub Release](https://github.com/TT-t-1/CogECI-ESL/releases/tag/v1.0.0)。

在项目根目录运行 `python code/download_weights.py`，下载 E0、E1、E2 权重至对应子目录并核验 `checksums.json` 的 SHA256。随后按项目 README 运行 `code/demo.py`。

发布地址和文件名映射见 `github_release.json`。

Release 以二进制分片提供权重。下载脚本会自动下载、合并并核验完整文件的 SHA256。
