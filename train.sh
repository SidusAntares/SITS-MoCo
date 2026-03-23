#!/usr/bin/env bash
set -euo pipefail  # 启用严格模式：出错即停
CONDA_BASE=$(conda info --base)
PYTHON_EXEC="$CONDA_BASE/envs/moco/bin/python"
# ================== 配置区（集中管理，便于修改）==================
MODEL_PATH="checkpoints/pretrained/MoCoV2_TRSF_doy/model_best.pth"
SOURCES=("france/31TCJ/2017" "france/30TXT/2017" "denmark/32VNH/2017" "austria/33UVP/2017")
SEEDS=(111 222 333)
# ================================================================

# 安全检查：确保模型文件存在
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] 模型文件不存在: $MODEL_PATH" >&2
    exit 1
fi

# 执行单个训练任务的函数
run_experiment() {
    local source_path="$1"
    local seed="$2"
    echo "--------------------------------------------------"
    echo "[INFO] 开始训练: source=$source_path, seed=$seed"
    echo "[CMD] "$PYTHON_EXEC" main_tscls.py stnet --rc --pretrained '$MODEL_PATH' --source '$source_path' --seed $seed"
    echo "--------------------------------------------------"

    # 执行命令，失败则退出
    "$PYTHON_EXEC" main_tscls.py stnet --rc --pretrained "$MODEL_PATH" --source "$source_path" --seed "$seed"
}

# 主循环：遍历所有组合
for source in "${SOURCES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "$source" "$seed"
    done
done

echo "[SUCCESS] 所有实验已完成！共 $((${#SOURCES[@]} * ${#SEEDS[@]})) 个任务。"