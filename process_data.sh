#!/bin/bash

# 获取参数总数
argc=$#

# 检查参数是否足够
if [ $argc -lt 3 ]; then
    echo "Usage: $0 task1 [task2 ...] task_config expert_data_num"
    exit 1
fi

# 1. 提取最后两个固定参数
# ${!argc} 获取最后一个参数
expert_data_num="${!argc}"

# 计算倒数第二个参数的索引
config_idx=$((argc - 1))
# 这种写法在 bash 中获取特定索引的参数比较繁琐，我们用这种方式：
args=("$@")
task_config="${args[$config_idx-1]}"

# 2. 提取任务名称 (从第1个到倒数第3个)
# unset 移除数组最后两个元素
unset 'args[$argc-1]'
unset 'args[$argc-2]'
task_names="${args[@]}"

echo "Tasks: $task_names"
echo "Config: $task_config"
echo "Num: $expert_data_num"

# 3. 调用 Python (同样建议使用 flag)
python process_data.py --task_names $task_names --task_config $task_config --expert_data_num $expert_data_num