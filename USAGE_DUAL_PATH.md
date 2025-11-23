# 双路径增强模块使用说明

## 概述

双路径增强模块 (DualPathRefiner) 在 CLIP image encoder 输出后，通过双路径残差形式增强特征表示。该模块包含：
- **LayerNorm**：归一化输入特征
- **AdapterResidual 路径**：轻量级残差适配器，增强局部特征
- **AttentionPoolingRefiner 路径**：注意力池化精炼器，增强全局特征
- **可学习权重**：alpha1 和 alpha2 控制两条路径的贡献

## 训练模型

### 基本训练命令

```bash
python train.py --use_dual_path_refiner
```

### 完整训练示例

```bash
python train.py \
  --use_dual_path_refiner \
  --dual_path_init_alpha 0.1 \
  --apr_n_queries 1 \
  --apr_n_heads 8 \
  --apr_dropout 0.1 \
  --n_epochs 10 \
  --lr 1e-4 \
  --batch_size 64 \
  --attention_size 7 \
  --decoder_name gpt2 \
  --experiments_dir experiments/
```

### 参数说明

**双路径增强模块参数：**
- `--use_dual_path_refiner`: 启用双路径增强模块（必需）
- `--dual_path_init_alpha`: alpha1 和 alpha2 的初始值（默认 0.1）
- `--apr_n_queries`: AttentionPoolingRefiner 的查询数量（默认 1）
- `--apr_n_heads`: AttentionPoolingRefiner 的注意力头数（默认 8）
- `--apr_proj_back`: 是否投影回 token 空间（默认 True）
- `--apr_dropout`: AttentionPoolingRefiner 的 dropout 率（默认 0.1）

**其他参数（与 AR adapter 共享）：**
- `--ar_down_ratio`: AdapterResidual 的降维比例（默认 4）
- `--ar_dropout`: AdapterResidual 的 dropout 率（默认 0.1）
- `--disable_ar_gate`: 禁用 AdapterResidual 的门控机制

### 模型保存路径

使用双路径增强模块训练的模型会自动保存到带有 `_dualpath` 后缀的路径，例如：
- `experiments/rag_7M_gpt2_dualpath/`
- `experiments/norag_7M_gpt2_dualpath/`

### 参数统计

训练时会自动打印各模块的可训练参数：
```
Training a model with X trainable parameters.
AR adapter trainable params: Y
DualPathRefiner trainable params: Z
AdaptedFFM trainable params: W
```

## 推理 (Inference)

### 基本推理命令

**使用验证集：**
```bash
python infer.py --model_path experiments/rag_7M_gpt2_dualpath
```

**使用测试集：**
```bash
python infer.py --model_path experiments/rag_7M_gpt2_dualpath --infer_test
```

**指定特定检查点：**
```bash
python infer.py \
  --model_path experiments/rag_7M_gpt2_dualpath \
  --checkpoint_path checkpoint-17712 \
  --infer_test
```

### 推理参数说明

- `--model_path`: 模型保存路径（必需）
- `--checkpoint_path`: 特定检查点路径（可选，不指定则使用所有检查点）
- `--infer_test`: 使用测试集而非验证集
- `--disable_rag`: 禁用 RAG（对于非 RAG 模型）
- `--batch_size`: 批处理大小（仅对非 RAG 模型有效，默认 64）
- `--images_dir`: 图像目录路径
- `--features_path`: 缓存的图像特征 HDF5 文件路径
- `--annotations_path`: 标注 JSON 文件路径
- `--captions_path`: 检索到的描述 JSON 文件路径
- `--template_path`: 模板文件路径

### 推理输出

推理结果保存在每个检查点目录下的 `val_preds.json` 或 `test_preds.json` 文件中。

**加载模型时，会自动显示使用的增强模块：**
```
✓ 模型使用了双路径增强模块 (DualPathRefiner)
```

## 配置说明

### 模型配置

模型配置会自动保存到 `config.json` 中，包括：
- `use_dual_path_refiner`: 是否使用双路径增强模块
- `dual_path_init_alpha`: alpha1 和 alpha2 的初始值
- `apr_n_queries`: APR 查询数量
- `apr_n_heads`: APR 注意力头数
- `apr_proj_back`: APR 是否投影回 token 空间
- `apr_dropout`: APR dropout 率
- `ar_down_ratio`: AR 降维比例
- `ar_dropout`: AR dropout 率
- `ar_use_gate`: AR 是否使用门控

### 注意事项

1. **与 AR adapter 的关系**：
   - 启用 `--use_dual_path_refiner` 时，会自动禁用原有的 AR adapter
   - 双路径模块内部已经包含了 AdapterResidual，因此不需要额外的 AR adapter

2. **向后兼容性**：
   - 不指定 `--use_dual_path_refiner` 时，模型行为与之前相同
   - 可以继续使用原有的 AR adapter（通过 `--disable_ar_adapter` 禁用）

3. **模型加载**：
   - 推理时不需要指定额外参数，模型会自动从 `config.json` 读取配置
   - 如果检查点包含双路径模块，会自动加载相应权重

## 完整训练-推理流程示例

### 1. 训练模型

```bash
python train.py \
  --use_dual_path_refiner \
  --dual_path_init_alpha 0.1 \
  --n_epochs 10 \
  --lr 1e-4 \
  --batch_size 64 \
  --attention_size 7 \
  --decoder_name gpt2 \
  --experiments_dir experiments/ \
  --features_dir features/ \
  --annotations_path data/dataset_coco.json \
  --captions_path data/retrieved_caps_resnet50x64.json \
  --template_path src/template.txt \
  --k 4
```

### 2. 推理

```bash
# 对验证集进行推理
python infer.py \
  --model_path experiments/rag_7M_gpt2_dualpath \
  --images_dir data/images/val/ \
  --annotations_path data/dataset_coco.json \
  --captions_path data/retrieved_caps_resnet50x64.json \
  --template_path src/template.txt \
  --k 4

# 对测试集进行推理
python infer.py \
  --model_path experiments/rag_7M_gpt2_dualpath \
  --checkpoint_path checkpoint-17712 \
  --infer_test \
  --images_dir data/images/test/ \
  --annotations_path data/dataset_coco.json \
  --captions_path data/retrieved_caps_resnet50x64.json \
  --template_path src/template.txt \
  --k 4
```

### 3. 评估

```bash
python coco-caption/run_eval.py \
  data/annotations/captions_val2017.json \
  experiments/rag_7M_gpt2_dualpath/checkpoint-17712/val_preds.json
```

## 故障排查

### 问题：模型加载失败

**解决方案：**
1. 确保检查点目录包含 `config.json` 文件
2. 确保检查点包含 `pytorch_model.bin` 或 `model.safetensors` 文件
3. 检查 `config.json` 中是否包含 `use_dual_path_refiner` 等配置项

### 问题：权重不匹配

**解决方案：**
- 模型会自动处理权重形状不匹配的情况
- 如果出现严重不匹配，检查训练和推理时的配置是否一致

### 问题：CUDA 内存不足

**解决方案：**
- 减小 `--batch_size` 参数
- 使用 `--features_path` 预先提取的特征而不是实时处理图像

## 技术细节

### 模块结构

```
输入 x: (B, N, C)
  ↓
LayerNorm(v = LN(x))
  ↓
并行执行：
  ├─→ AdapterResidual(v) → y1
  └─→ AttentionPoolingRefiner(v) → y2
  ↓
合并: x_out = x + alpha1 * y1 + alpha2 * y2
输出: (B, N, C)
```

### 可训练参数

- `alpha1`, `alpha2`: 可学习的标量权重参数（2 个参数）
- `AdapterResidual`: 约 `2 * C^2 / down_ratio + C` 个参数
- `AttentionPoolingRefiner`: 约 `C^2 * (n_queries + 2 * n_heads + 1) + C` 个参数

总共可训练参数数量取决于配置和模型维度。


