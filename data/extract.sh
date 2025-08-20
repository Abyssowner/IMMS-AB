#!/bin/bash

# 创建目标子文件夹（如果不存在）
mkdir -p extracted

# 查找所有 .gz 文件并解压
for file in *.gz; do
    # 检查文件是否存在
    if [ -f "$file" ]; then
        # 获取文件名（不含扩展名）
        filename=$(basename "$file" .gz)
        # 解压文件到子文件夹
        gunzip -c "$file" > "extracted/$filename"
        echo "已解压: $file -> extracted/$filename"
    fi
done