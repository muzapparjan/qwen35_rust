# Qwen3.5-0.8B 推理规格（逐项）

这份文档是 Rust 实现的**唯一算子规格**。模型代码只按这里写；与这里不符的实现算缺陷。

## 0. 溯源与这份文档的地位

规格来源是**参考实现的源码本身**，不是任何二手描述：

| 来源 | 版本 / 大小 / SHA256 |
|---|---|
| `transformers` | 5.23.0（4.57.x 没有 `qwen3_5`） |
| `modeling_qwen3_5.py` | 87,255 B，`1a475c45d900e0bfb06b9b245257d7f3234fd247474fcbd3363b03430e504a68` |
| `configuration_qwen3_5.py` | 7,892 B，`a2a7c5c2f193dec6bbad0c5dcd2536dfba2eba14259814b9dc0438afe01969d9` |
| `model/config.json` | 3,002 B，`f9061afb89151d97c70fd78de2ce3fe4e11cdf5e853e36ebe19e16499bf5f619` |
| `model/tokenizer.json` | 12,807,982 B，`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `model/tokenizer_config.json` | 17,013 B，`ce3cbcd05aaa91d20c1605c5c46de4bacfe439c1c0efd8b27e6b5489de968f2c` |
| `model/chat_template.jinja` | 7,908 B，`306abaf09cf3a6c0d14128373a72925bda3d7de2616a5642804d9768c7d2f78c` |
| `quantized/qwen3.5-0.8b-q4.qwqt` | 494,845,446 B，`29773850A7AC53122E1544A24748701B55FA59CA63FED89CAE71B9BAABC7DFB9` |
| `torch` | 2.11.0+cu128，Python 3.11.4 |

文中 `[源:X]` 指向 `modeling_qwen3_5.py` 的行号。凡是带「实测」的结论，读数来自
`dev/probe_operators.py`（一条一次性探针，见 §12 的读数列）；凡是写「未验证」的条目，
都还没有读数，不许当成事实用。

参考实现走的是**纯 PyTorch 回退路径**（未安装 `flash-linear-attention` /
`causal-conv1d`），运行时打印四行 fallback 警告。因此 §5、§6 的数学就是实际执行的
那份代码，不是某个融合 kernel 的近似。

`_attn_implementation` 实测为 `sdpa`（文本与视觉都是）。sdpa 在 `is_causal=True`
时做因果掩码；本规格直接写「因果掩码」，不依赖后端。

## 1. 记号与数值口径

- 形状用 `[batch, seq, ...]`，除特别说明外 batch = 1。
- 权重按 PyTorch 布局：`Linear.weight` 是 `[out_features, in_features]`，
  `y[t, o] = sum_i x[t, i] * W[o, i] + b[o]`。
- CPU 后端一律 **f32** 计算；`mamba_ssm_dtype: "float32"` 只说明线性注意力的状态是
  f32，参考实现里所有张量本来就先被转成 f32 再算（`[源:451-454]`、`[源:457-463]`）。
- 归一化的平方均值、方差、注意力 softmax 都在 f32 里做（与 torch 相同）。
- 反量化用 `q * scale`（scale 是 f16 网格上的值）在 f32 里做，与 `read_tensor`
  逐位一致（M1 的判据）。

## 2. 架构常量（全部来自 `config.json`，实测与权重形状一致）

| 常量 | 值 | 备注 |
|---|---|---|
| `hidden_size` | 1024 | |
| `num_hidden_layers` | 24 | |
| `layer_types` | `[L,L,L,F]` × 6 | `L` = linear_attention，`F` = full_attention；18 + 6 |
| `intermediate_size` | 3584 | MLP |
| `hidden_act` | silu | |
| `rms_norm_eps` | 1e-6 | |
| `vocab_size` | 248320 | 实际用到的 token id 上界是 248076 |
| `tie_word_embeddings` | true | 输出头复用 `embed_tokens`（实测两者同一块存储） |
| `head_dim` | 256 | 全注意力 |
| `num_attention_heads` | 8 | q 宽 2048 |
| `num_key_value_heads` | 2 | kv 宽 512，GQA 4:1 |
| `attn_output_gate` | true | 门与 q 同一个投影，见 §6 |
| `rope_theta` | 1e7 | 文本 |
| `partial_rotary_factor` | 0.25 | 只转前 64 维 |
| `mrope_interleaved` / `mrope_section` | true / [11,11,10] | 见 §8 |
| `linear_num_key_heads` / `linear_key_head_dim` | 16 / 128 | |
| `linear_num_value_heads` / `linear_value_head_dim` | 16 / 128 | |
| `linear_conv_kernel_dim` | 4 | |
| `mamba_ssm_dtype` | float32 | 状态 f32 |
| `vision.depth` | 12 | |
| `vision.hidden_size` / `num_heads` | 768 / 12 | 视觉 head_dim 64 |
| `vision.intermediate_size` | 3072 | |
| `vision.patch_size` / `temporal_patch_size` / `spatial_merge_size` | 16 / 2 / 2 | |
| `vision.out_hidden_size` | 1024 | |
| `vision.num_position_embeddings` | 2304 | `num_grid_per_side = 48` |
| `vision.hidden_act` | `gelu_pytorch_tanh` | **merger 用的是精确 GELU**，见 §9 |
| `vision.rope_theta` | 10000 | 不在 `config.json` 里，取 `configuration_qwen3_5.py` 的默认 |
| 特殊 token | `image_pad` 248056、`video_pad` 248057、`vision_start` 248053、`vision_end` 248054、`im_start` 248045、`im_end` 248046 | |
| `generation_config.eos_token_id` | 248044 | 实测；**不是** `im_end`（248046）。`reference/run_chat.py` 的 `generate` 只在这一号或 `max_new_tokens` 停 |

文本位置 id 的三种来源（`text`/`image`/`video`）见 §8；纯文本时三行相同。

## 3. 权重容器与张量账目

容器格式以 `quantized/README.md` 为准（本文不重复规格），Rust 侧只需解析：
`"QWQT"` + `u32 version` + `u32 hdr_len` + JSON header + payload。

实测（读 header）：

| 读数 | 值 |
|---|---|
| 张量数 | 488 |
| `q4` / `bf16` / `f32` 张量数 | 210 / 242 / 36 |
| 参数总数 | 873,438,784 |
| 模型类加载的 473 个张量 | 852,985,920 参数 |
| `mtp.*` 15 个张量 | 20,452,864 参数 |
| header 字节 | 80,634（payload 起点 = 12 + 80,634 = 80,646） |

**名字映射实测**：容器里 488 个名字与 `Qwen3_5ForConditionalGeneration.state_dict()` 的
474 个键完全一致，除了

- 容器多出 `mtp.*` 15 个；
- `state_dict()` 多出 `lm_head.weight`，它与 `model.language_model.embed_tokens.weight`
  是同一块存储（tied）。

因此 Rust 侧**不需要名字翻译表**，直接按容器名字查表即可。

`bf16` 与 `f32` 张量（2.4M 参数）是：所有 norm 权重、`linear_attn.{A_log, dt_bias,
norm, conv1d, in_proj_a, in_proj_b}`、`self_attn.{q_norm, k_norm}`、视觉塔的 bias /
norm / `patch_embed.proj`。它们不量化是因为末维不能被 32 整除或参数量太小。

一个张量的读写规则：

- `q4`：`scales` 在 `offset` 处，`groups` 个 f16 后补零到 `scales_len`；`ints` 在
  `offset + scales_len` 处；元素 `i` 的字节是 `ints[i/2]`，`i` 偶数取低半字节。
  `q × scale` 就是量化器写进文件的那个 f32 值。
- `bf16` / `f32`：原始字节，行主序。

本容器里 210 个量化张量**全部是 q4、group 32**（bit 2/3/8 的布局仍要实现：容器格式允许，
且 `read_tensor` 支持）。4-bit 布局已用独立的手写解包对照 `read_tensor` 验证：
`dev/dump_dequant.py` 每次运行都会重跑这项自检，读数 `max|d| = 0.000e+00`。

### 3.1 对齐（实测，决定了 Rust 侧怎么读）

容器把**每个张量的 payload 内偏移**对齐到 64 字节，但 payload 本身从文件第 80,646
字节开始，而 80,646 不是 4 的倍数。所以：

| 用途 | 是否成立 | 后果 |
|---|---|---|
| f16 scales 按 `u16` 读 | 成立（绝对偏移恒为偶数） | 可以按 2 字节读 |
| f32 张量按 `*const f32` 零拷贝 | **不成立**（绝对偏移 ≡ 2 mod 4） | 必须按字节读、再 `from_le_bytes` |
| 整个 `blob()` 基址 64 字节对齐 | 成立（`#[repr(C, align(64))]`） | SIMD 读 blob 无问题 |

dense 张量（bf16 + f32）合计 242 + 36 个、2,397,216 + 2,592 个值，转换一次不到 10 MB，
所以「按字节存、用时转 f32」既没有性能代价，也让 crate 里不需要任何 `unsafe`。

清单里的 `offset` 一律是**相对嵌入文件**的绝对字节偏移（生成器已经把 header 长度加上）。

## 4. 公共算子

### 4.1 zero-centered RMSNorm（`Qwen3_5RMSNorm`）`[源:834-852]`

```
y = ( x / sqrt(mean(x^2, last_dim) + eps) ) * (1 + w)
```

- 平方均值在 f32 上、按最后一维；`w` 与 `x` 同形（`hidden_size` 或 `head_dim`）。
- 与 Llama 的差别就是 `(1 + w)`：`_init_weights` 把 `w` 初始化为 0（训练起点是恒等），
  但权重是训练得到的，实测 `q_norm.weight.abs().max() = 1.2266`、
  `text.norm.weight.abs().max() = 5.3125` —— `1 + w` 并不在 1 附近，不能省掉这一项。

### 4.2 Gated RMSNorm（`linear_attn.norm`）`[源:203-218]`

```
y = w * ( x / sqrt(mean(x^2, last_dim) + eps) ) * silu(gate)
```

- 按 `head_v_dim = 128` 归一化，`w` 是 128 维，**不是** `1 + w`（实测 `w` 的均值 0.9516、
  绝对值最大 1.0596，即是被训练过的实数权重）。
- `gate` 是 `z`，**没有**预先激活；`silu` 只在这里用一次。
- 顺序是「先归一化、再乘 `w`、再乘 `silu(z)`」。实测手算与模块输出差 `0.000e+00`。

### 4.3 其它

- `silu(x) = x * sigmoid(x)`。
- `softplus(x) = x` 当 `x > 20`，否则 `ln(1 + e^x)`（torch 的默认 `beta=1, threshold=20`；
  实测 `x=20.0` 返回 20.0，`x=20.001` 返回 20.000999 而不是原值）。
- `l2norm(x) = x / sqrt(sum(x^2, last_dim) + 1e-6)` `[源:279-283]`。
- `linear`：见 §1。
- `rms` 用 `1/sqrt`；torch 用 `rsqrt` 指令，两者可能差 1 ulp，容差按相对误差定。

## 5. `linear_attention`（Gated DeltaNet，18 层）

输入是 `input_layernorm(hidden)` 之后的 `[1, S, 1024]`。所有步骤按此顺序：

### 5.1 投影 `[源:539-542, 563-570]`

| 名字 | 形状 | 说明 |
|---|---|---|
| `in_proj_qkv` | `[6144, 1024]` | q、k、v 拼在一起 |
| `in_proj_z` | `[2048, 1024]` | 门的来源，不做卷积 |
| `in_proj_b` | `[16, 1024]` | beta 的 logit |
| `in_proj_a` | `[16, 1024]` | 衰减的 logit |

四个都无 bias。`z`、`b`、`a` **不经过** 5.2 的卷积。

### 5.2 因果深度卷积 `[源:255-275, 563-598]`

```
mixed = in_proj_qkv(h)            # [1, S, 6144]
mixed = transpose(mixed)          # [1, 6144, S]  —— 通道在前
mixed = conv1d(mixed, pad_left=3, kernel=4, groups=6144, bias=None)[:, :, :S]
mixed = silu(mixed)
mixed = transpose(mixed)          # [1, S, 6144]
```

- 权重 `conv1d.weight` 形状 `[6144, 1, 4]`，每个通道自己 4 个抽头，**无 bias**。
- torch 的 `conv1d` 是互相关（核不翻转），所以第 `t` 个输出是 `sum_j w[j] * x[t-3+j]`。
- 左边补 3 个 0，卷积后再截回 `S` 个 —— 结果里第 `t` 个输出只看到 `x[t-3..t]`。
- 缓存里保存的是**卷积前的原始投影**，长度 4。实测：prefill 6 个 token 再 decode 1 个之后，
  `conv_state` 与本序列 `in_proj_qkv` 输出最后 4 列相差 `1.4e-6`（两次运行核不同带来的
  浮点差）；形状 `[1, 6144, 4]`。
- decode 单 token 时的等价形式：`conv([state ‖ x_t])`，取最后一个输出，再把 state
  更新为 `[state ‖ x_t]` 的最后 4 列。

### 5.3 拆分与形状 `[源:600-612]`

`mixed` 按 `[2048, 2048, 2048]` 切成 `q, k, v`（按这个顺序，不是 qkv 交错），再
reshape 成 `q,k: [1,S,16,128]`、`v: [1,S,16,128]`。`z` 同样 reshape 成 `[1,S,16,128]`。

### 5.4 两个逐头标量 `[源:614-616]`

```
beta = sigmoid(b)                                  # [1,S,16]
g    = -exp(A_log) * softplus(a + dt_bias)         # [1,S,16]，A_log/dt_bias 是 [16]
```

实测 `g` 的最大值是 `-6.917e-10`（即恒 `<= 0`），`beta` 落在 `0.0473 .. 0.9945`。
`A_log` 实测范围 `-6.5938 .. 0.2578`，`dt_bias` 16 个值各不相同（`-13.0625 .. 8.375`）。
`softplus` 的自变量是 `a + dt_bias`，`dt_bias` 与 `A_log` 都是训练得到的参数。

### 5.5 归一化与缩放 `[源:456-463]`

```
q = l2norm(q, last_dim=128) / sqrt(128)
k = l2norm(k, last_dim=128)
```

注意顺序：**先 L2 归一化，再除 `sqrt(128)`**；`k` 不缩放。

### 5.6 递推 `[源:473-485]`

对每个 head `h`（16 个）各有一个 `[128, 128]` 的 f32 状态 `S_h`，初值 0。对 `t = 0..S-1`
按顺序：

```
S  <- S * exp(g[t, h])            # 两个轴都乘同一个标量
kv <- S^T k[t, h]                 # [128]，对 K 轴求和
δ  <- (v[t, h] - kv) * beta[t, h]
S  <- S + k[t, h] ⊗ δ             # 外积，[128, 128]
o[t, h] <- S^T q[t, h]            # [128]
```

三个顺序上的要点（都不是可选项）：

1. **衰减在 delta 更新之前**；
2. `kv` 用的是**已衰减**的状态；
3. 输出 `o_t` 用的是**更新之后**的状态（包含本 token 的写入）。

参考实现的 `torch_chunk_gated_delta_rule` 是同一个递推的分块等价形式 `[源:286-425]`，
实测两者相对差 `6.3e-8`。Rust 主路径用上面的递推；分块形式可以作为 prefill 的加速，
但必须落在同一个容差里（§13 的 M5 判据）。

### 5.7 门控归一化与输出投影 `[源:650-655]`

```
o = reshape(o, [S*16, 128])       # token 主序，head 次序不变
z = reshape(z, [S*16, 128])
y = gated_rms_norm(o, z)          # §4.2
y = reshape(y, [1, S, 2048])
out = out_proj(y)                 # [1024, 2048]
```

## 6. `full_attention`（Gated Attention，6 层）

输入同样是 `input_layernorm(hidden)` 之后的 `[1, S, 1024]`。

### 6.1 q 与门是同一个投影，且按 head 交错 `[源:780-787]`

```
qg = q_proj(h)                                  # [4096, 1024]
qg = view(qg, [1, S, 8, 512])                   # 512 = 2 * head_dim
q, gate = chunk(qg, 2, dim=-1)                  # 各 [1,S,8,256]
gate = reshape(gate, [1, S, 2048])
q = q_norm(q)                                    # zero-centered，按 256
k = k_norm(view(k_proj(h), [1,S,2,256]))         # [512, 1024] -> [1,S,2,256]
v = view(v_proj(h), [1,S,2,256])
```

**布局是 `[q_head0(256) ‖ gate_head0(256) ‖ q_head1(256) ‖ ...]`**，不是「全部 q 后跟
全部 gate」。这是 `view(..., -1, head_dim*2)` 再 `chunk` 的直接后果；写错这一处，
q 会拿到别的 head 的门，输出仍然「看起来合理」。

两个 norm 都是 §4.1 的 zero-centered RMSNorm，**在 transpose 之前、按最后一维 256**
（`[源:785-787]`）。

### 6.2 部分 RoPE 与 mRoPE 交错 `[源:134-200, 659-700]`

- 只转前 `rotary_dim = int(256 * 0.25) = 64` 维，`q[..., :64]`，后面 192 维原样传下去。
- 逆频率 32 个：`inv_freq[i] = 1 / theta^(2i/64)`，`i = 0..31`。
- 角度 `angle = pos * inv_freq`，`cos = cos(angle)`、`sin = sin(angle)`，再按 §8 交错。
- 旋转用 `rotate_half` 的配对方式：`(q[i], q[i+32])`，即
  ```
  y[i]    = q[i] * cos[i] - q[i+32] * sin[i]
  y[i+32] = q[i+32] * cos[i+32] + q[i] * sin[i+32]
  ```
  在 64 维内部拆两半，**不是**相邻两维配对。
- `cos`/`sin` 的最后一维是 64（实测 `(1, 7, 64)`）。

### 6.3 注意力 `[源:705-739, 788-812]`

```
k, v 各 2 个 head，repeat 4 次 -> 8 个 head（每 2 个 kv head 对应 4 个 q head，连续分组）
score = q @ k^T / sqrt(256)
score = score + causal_mask        # 上三角 -inf
p     = softmax(score, last_dim, dtype=f32)
o     = p @ v                       # [1,8,S,256] -> [1,S,2048]
o     = o * sigmoid(gate)           # 逐元素，gate 形状 [1,S,2048]
out   = o_proj(o)                   # [1024, 2048]
```

实测：手写这条链与模块输出相对差 `9.1e-8`（含因果掩码）。

### 6.4 KV 缓存

每层保存 `k, v`，各 `[2, S, 256]`。f32 下 6 层合计 `6 × 2 × 256 × 2 × 4 B = 24 KiB/token`
（bf16 时减半，与前期笔记里的 12 KiB 一致）。

## 7. 解码层与整模型

### 7.1 一层 `[源:867-905]`

```
r = h
h = r + mixer(input_layernorm(h))          # linear_attn 或 self_attn，按 layer_types
r = h
h = r + mlp(post_attention_layernorm(h))
```

`mlp(x) = down_proj(silu(gate_proj(x)) * up_proj(x))` `[源:828-830]`；
`gate_proj` / `up_proj` 是 `[3584, 1024]`，`down_proj` 是 `[1024, 3584]`，都无 bias。

### 7.2 整模型

```
x = embed_tokens[ids]                      # [248320, 1024]，q4
for layer in 0..24: x = layer(x)
x = rms_norm(x, text.norm.weight)          # zero-centered
logits = x @ embed_tokens^T                # 无 lm_head 权重，tied
```

- **永远不要物化 `[S, 248320]`**：生成只需要最后一个位置的 logits（一次
  `248320 × 1024` 的矩阵乘）。prefill 阶段的中间位置不需要输出投影。
- 实测 473 个张量 = 852,985,920 参数；`lm_head.weight` 与 `embed_tokens.weight` 同存储。

## 8. 位置编码（mRoPE）

### 8.1 文本

`position_ids` 是 `[4, batch, S]` 中的第 0 行是文本位置（`[源:1249-1252]`）；
纯文本时四行都是 `arange(S)`（增量生成时加 `past_seen_tokens`）。

传给 RoPE 的是后 3 行 `[3, batch, S]`（T、H、W），纯文本时三行相同。

### 8.2 交错规则 `[源:177-200]`

```
freq[t,h,w, :] = position * inv_freq        # [3,b,S,32]
freqs_thw = freq[0]                          # 以 T 为底
for dim, offset in ((1,1), (2,2)):           # H, W
    length = mrope_section[dim] * 3          # H: 33, W: 30
    idx = slice(offset, length, 3)
    freqs_thw[..., idx] = freq[dim, ..., idx]
cos/sin = cat([freqs_thw, freqs_thw], -1)    # 32 -> 64
```

也就是 32 个频率位按模 3 分配：`i % 3 == 0` 取 T（11 个），`== 1` 取 H（11 个），
`== 2` 取 W（10 个）；然后前 32 与后 32 相同（对应 `rotate_half` 的两半）。

纯文本时三行相同，所以这条交错对结果没有影响；有图时才有区别。

### 8.3 图像/视频位置（`get_rope_index`）`[源:1360-1452]`

按 `mm_token_type_ids` 把序列切成连续段：文本段 `arange(len) + current_pos`；
图像/视频段用 `Qwen3_5Model.get_vision_position_ids(current_pos, grid_thw, 1, 2)`
`[源:1308-1350]`，然后 `current_pos += max(H, W) // 2` `[源:1443]`。

注意这里有两个同名函数，作用对象不同：

- **LLM 侧**（本节，`modeling_qwen3_5.py:1308`）给出的是**合并后**的 token 位置，
  数量 `t * h/2 * w/2`：

  ```
  T = arange(t) + start
  H = arange(h/2) + start
  W = arange(w/2) + start
  position_ids = stack(meshgrid(T, H, W, indexing="ij")).reshape(3, -1)   # 光栅序
  ```

- **视觉塔内**（§9.2，`vision_utils.py:81-127`）给出的是**每个 patch** 的位置，
  数量 `t*h*w`，且是 merge 块顺序。

图像 token 在文本序列里是 `h*w/4` 个占位符，按合并后 token 的顺序一一替换。

## 9. 视觉塔（12 层）

以下全部读自源码；**尚未与参考实现对拍**（M6 的第一件事是造 dump）。

### 9.1 patch_embed `[源:954-971]`

```
pixel_values: [N, 3*2*16*16]                  # N = t*h*w 个 patch
view(-1, 3, 2, 16, 16)
Conv3d(3 -> 768, kernel=(2,16,16), stride=(2,16,16), bias)   # [768,3,2,16,16]
```

### 9.2 位置嵌入 `[源:1138-1141, 1174-1180]`

- 表 `pos_embed.weight` 是 `[2304, 768]`，按 `48 × 48` 的方形网格理解。
- 对每张图的 `(h, w)` 做**双线性**插值，`align_corners=True`，每个 patch 4 个抽头，
  权重按两个轴的外积；采样坐标 `src = index * (48 - 1) / (size - 1)`（`size==1` 时取 0），
  抽头索引 `clamp(floor(src) + {0,1}, 0, 47)`，权重 `(1 - |src - floor - offset|).clamp(min=0)`。
- patch 顺序是 **merge 块顺序**：`within = 0..h*w-1` 先解成
  `(block_row, block_col, in_row, in_col)`，再 `row = block_row*2 + in_row`、
  `col = block_col*2 + in_col`。位置 id 与插值索引都用这一套顺序。

### 9.3 视觉 RoPE `[源:73-132]`

- `inv_freq` 16 个：`1 / 10000^(2i/32)`，`i = 0..15`（`spatial_dim = head_dim/2 = 32`）。
- 每个 token 有 `(h, w)` 两个坐标，各自算 16 个角度；拼成 `[h(16), w(16), h(16), w(16)]`
  = 64，然后**整个 64 维都转**（视觉没有 partial）。
- `rotate_half` 同样在 64 维内拆两半（32/32）。

### 9.4 12 个 block `[源:1087-1113]`

```
x = x + attn(norm1(x))        # LayerNorm(768, eps=1e-6，有 weight/bias)
x = x + mlp(norm2(x))
```

- attention：`qkv = Linear(768, 2304, bias)`，`reshape(seq, 3, 12, 64).permute` 得
  `[12, seq, 64]`；q/k 转 RoPE；`score = q k^T / 8`；softmax 在 f32；`proj` 是
  `Linear(768, 768, bias)`。
- **非因果**，但按段独立：`cu_seqlens` 以每帧（`h*w`）为一段，跨图/跨帧不互相看。
- MLP：`linear_fc1` `[3072, 768]` + bias → `gelu_pytorch_tanh` → `linear_fc2` `[768, 3072]` + bias。

### 9.5 merger `[源:974-987, 1195]`

```
x = LayerNorm(768, eps=1e-6)(x)          # 逐 patch
x = view(-1, 4*768)                       # 4 个连续 patch 拼起来（2×2 块）
x = linear_fc2( GELU_exact( linear_fc1(x) ) )
# linear_fc1: [3072, 3072] + bias
# linear_fc2: [1024, 3072] + bias
```

**GELU 在这里是精确 erf 版本**（`nn.GELU()`），与 §9.4 的 tanh 近似不是同一个函数。

### 9.6 图像 token 进入文本序列（实测钉住）

chat template 在图像位置写一个 `<|image_pad|>`（248056），**展开成 `h*w/4` 个**才交给
模型；`vision_start`（248053）与 `vision_end`（248054）各一个，都是普通文本 token
（`mm_token_type_ids` 为 0）。实测：`16x16` 的网格对应 64 个占位符，落在 `vision_start`
与 `vision_end` 之间。

`mm_token_type_ids` **只有那 64 个占位符是 1**，两侧都是 0。位置按连续段给：

- 文本段：`arange(len) + current`，`current += len`；
- 图像段：`t/1`、`h/2`、`w/2` 三个 `arange` 各加 `current` 后按 `(t, h, w)` 光栅序拼成
  `[3, 64]`，然后 `current += max(h, w) / 2`（实测 16 的网格推进 8，不是 64）；
- 生成阶段的偏移（参考叫 `rope_deltas`）：`最大位置 + 1 - 序列长度`，加到
  `past + index` 上。**没有这一项，图像之后的每个 token 位置都会错**。

merger 的行按出现顺序写进占位符位置（`masked_scatter`）。

### 9.7 图像预处理（已读定，实测钉住）

配置：`size.shortest_edge = 65536`、`size.longest_edge = 16777216`、
`patch_size 16`、`merge_size 2`、`temporal_patch_size 2`、`mean/std = 0.5`、
处理器 `Qwen2VLImageProcessorFast`（本版 transformers 里没有这个类，`image_processor_type`
落到继承 `TorchvisionBackend` 的 `Qwen2VLImageProcessor`，实测走 **PIL** 的 `BICUBIC`）。

`dev/dump_vision.py` 同时 dump 原始 RGB 字节与它产出的 `pixel_values`，下面的每一条都是
拿这两者比出来的（探针在 `tmp/`，读数在 `dev/README.md`）：

1. **`smart_resize`**：`factor = patch_size * merge_size = 32`。先按最近倍数取整
   （`round(h/32)*32`）；若乘积超过 `longest_edge`，用 `beta = sqrt(h*w/max)` 后**下取整**；
   若低于 `shortest_edge`，用 `beta = sqrt(min/(h*w))` 后**上取整**。长宽比超过 200:1 报错。
   实测：`100x70 → 320x224`、`64x64 → 256x256`（两者都是被**放大**到 65536 像素以上）。
2. **重采样**：bicubic，核是 **`a = -0.5`** 的三次核（PIL 的 BICUBIC），**不是** torchvision
   `F.interpolate` 的 `a = -0.75`。采样位置 `src = (i + 0.5) * in / out - 0.5`，抽头
   `floor(src) + {-1, 0, 1, 2}`；**出界的抽头被丢弃而不是夹到边界**，保留下来的权重再归一化。
   实测：`a = -0.75` 差 4.5e-1、夹边界差 2.9e-1、丢弃+归一化差 1e-3 以内。
3. **两遍各自落成字节**：水平遍的结果**取整并夹到 `[0,255]`** 后才进垂直遍，垂直遍同样处理。
   这一条最容易被漏掉：只在整个流程末尾取整，饱和图上会差 **14 个灰度级**；加上它之后我的
   实现在 196608 个值上与 PIL **逐位相同**（0 处不同）。
4. **归一化**：`(v / 255 - 0.5) / 0.5`，一次 rescale，不会重复。
5. **patch 化**：patch 顺序是**merge 块优先** —— 索引 `i` 解成
   `in_col = i % 2`、`in_row = (i/2) % 2`、`block_col = (i/4) % blocks`、
   `block_row = i / (4 * blocks)`，再 `row = block_row*2 + in_row`、`col = block_col*2 + in_col`。
   每个 patch 向量内是 `(channel, temporal, patch_row, patch_col)`，图像只有一帧，被**复制
   两次**填满 temporal。实测方式：每次只点亮一个 16×16 源块，看 `pixel_values` 里哪几行亮。

**Rust 侧读数**（`cargo test --release --features reference-fixtures --test vision`）：

| 用例 | 预处理 `max\|d\|` | tower 相对误差 | 端到端相对误差 |
|---|---|---|---|
| `square`（64×64 → 256×256） | 5.9e-8 | 1.0e-5 | 1.1e-5 |
| `resized`（100×70 → 320×224） | 7.8e-3（正好一个灰度级） | 9.4e-6 | 3.5e-3 |

`resized` 的预处理差一个灰度级，原因是 PIL 用定点系数；端到端 3.5e-3 是这一个灰度级被塔
放大出来的（这个塔在 fixture 上把输入放大了两个数量级：block 11 的 `max|d|` 从 4 长到 1654）。
两个数都来自实测，不是估计。

**已实现并逐点对拍**（`src/vision/`）：预处理、塔、占位符展开、mRoPE 位置与生成偏移。
Rust 侧读数与判据在下一节末尾。

| 检查项 | `square`（64×64→256×256） | `resized`（100×70→320×224） |
|---|---|---|
| 预处理 `max\|d\|` | 5.9e-8 | 7.8e-3（一个灰度级） |
| 塔（喂 dump 的 patch 行）相对误差 | 1.0e-5 | 9.4e-6 |
| 整条图像 prompt 的 logits 相对误差 | 2.5e-6 | 2.5e-4 |
| 图像之后走一步的相对误差 | 1.8e-6 | 3.0e-4 |
| top-1（prompt 与之后一步） | 1919 / 2099，与参考相同 | 同左 |
| 贪心生成与参考一致的 token 数 | 前 10 个，第 11 个分岔 | （未单独量） |
| 我们渲染的 prompt 与处理器的 ids | 84 个，全等 | 90 个，全等 |


## 10. MTP（15 个张量）

**文本模型不加载它**：实测 `named_parameters` 里 `mtp.*` 数量为 0，`modeling_qwen3_5.py:918`
把 `r"^mtp.*"` 列进 `_keys_to_ignore_on_load_unexpected`，这个包也没有对应模块。它是
**投机解码的草稿头**，只在带推测的引擎里跑：模型卡 `README.md:630` 给的是 SGLang 的
`--speculative-algo NEXTN`，`README.md:661` 是 vLLM 的 `method: "qwen3_next_mtp"`。

| 张量 | 形状 |
|---|---|
| `mtp.fc.weight` | `[1024, 2048]` |
| `mtp.pre_fc_norm_embedding.weight` | `[1024]` |
| `mtp.pre_fc_norm_hidden.weight` | `[1024]` |
| `mtp.layers.0.self_attn.{q,k,v,o}_proj` | `[4096,1024]` / `[512,1024]` / `[512,1024]` / `[1024,2048]` |
| `mtp.layers.0.self_attn.{q,k}_norm.weight` | `[256]` |
| `mtp.layers.0.{input,post_attention}_layernorm.weight` | `[1024]` |
| `mtp.layers.0.mlp.{gate,up,down}_proj` | `[3584,1024]` / `[3584,1024]` / `[1024,3584]` |
| `mtp.norm.weight` | `[1024]` |

接线取自 vLLM 的 `vllm/model_executor/models/qwen3_5_mtp.py`（commit
`663daccfb9e1f39f9f9f93034bf87bc709d91c06`），它把同一批名字按 `mtp.` → `model.` 重映射后
加载：

1. 分配 `fc = ColumnParallelLinear(2 * hidden, hidden)`，`layers` 用
   `layer_type="full_attention"` 构造，另有 `norm` / `pre_fc_norm_hidden` /
   `pre_fc_norm_embedding` 三个 `Qwen3_5RMSNorm`；
2. 前向：`embed = pre_fc_norm_embedding(embed_tokens(token))`、
   `h = pre_fc_norm_hidden(hidden)`、`cat([embed, h])` → `fc` → 那一个 full-attention 层
   → `norm(h, residual)` → 主模型的 **tied 输出头**；
3. **拼接顺序是 embedding 在前**；`Qwen3_5RMSNorm` 就是 `GemmaRMSNorm`
   （`qwen3_5.py` 里 `from ...layernorm import GemmaRMSNorm as Qwen3_5RMSNorm`），即零中心
   `(1 + w)`，与主模型同一个；
4. `mtp_num_hidden_layers = 1`、`mtp_use_dedicated_embeddings = false`：草稿层只有一层，
   输入 embedding 复用主模型的表。

**喂进去的 `hidden` 是过完 final norm 的那个**：`Qwen3NextModel.forward` 末尾是
`hidden_states, _ = self.norm(hidden_states, residual)` 然后 `return hidden_states`，
草稿头吃的就是它的返回值（`Qwen3_5Model` 继承自它）。相差一个 RMSNorm 就会全盘错。

草稿层有**自己的 KV 缓存**：它是一层 full attention，位置与主模型同一套。实测如果不先用
prompt 把这份缓存灌满，草稿只看得到自己，接受率从 0.238 掉到 0.183（同一句 prompt、
`--drafts 3`）。

### 10.1 投机解码的一轮（`src/model/mtp.rs`）

一轮 = 草稿 `steps` 个 token → 主模型一次批量验证 → 保留被确认的前缀。

- 验证批的第 `i` 行是「读完 `drafts[..=i]` 之后」的状态，所以它预测的是 `drafts[i+1]`；
  第 `i+1` 个草稿被确认当且仅当该行 logits 的 argmax 等于它。
- 被拒之后，全注意力的 KV 直接回退（写地址是 `past + token`，多出来的条目会被覆写），
  但**线性注意力的递推状态不能截断**（每个 token 都在原地乘一个衰减并累加）。所以一轮
  在验证前复制 18 份递推状态（`LinearCache`，共约 18 MB），保留的前缀再走一遍。
- 输出头占全模型四分之一的乘加，所以一轮只投影它比较的行，其余行不投影。

### 10.2 实测（`cargo run --release --example speculate`）

| 读数 | 值 |
|---|---|
| 输出与贪心逐 token 相同 | 是（`--drafts 1/2/3` 全为 `true`） |
| 草稿接受率（`--drafts 3`，数字接龙 prompt） | 24/30 = 0.800，level-1 10/10 = 1.000 |
| 草稿接受率（中文默认 prompt） | 0.238（未灌 prompt 时 0.183） |
| 每轮产出 token 数 | 3.20（`--drafts 3`）/ 2.00（`--drafts 1`，接受率 16/16） |
| 加速比 | 0.78×（`--drafts 1`）、0.77×（2）、0.67×（3） |
| 一次 pass，1 token vs 4 token | 20.9 ms / 46.0 ms |
| 输出头，1 行 vs 3 行 | 5.7 ms / 29.4 ms |
| 一次草稿步 | 4.3 ms |

**在这台 CPU 上投机解码是亏的**，原因不在接受率（可预测文本上 level-1 全中），而在成本：
decode 受带宽限制（每个 token 读 495 MB），批量 4 个只省掉那一次权重读，多出来的算力
是净增；而一轮还要额外付草稿层 `steps` 次、输出头至少两次（草稿各自一次 + 保留行一次），
以及 60% 轮次上的一次重放。上面那张表里头的 3 行 29.4 ms 是最反常的一项：批量路径的
权重流式读只有单行路径的一半速率，这是内核问题，不是模型问题。


## 11. 缓存状态与 prefill/decode 一致性

| 状态 | 形状 | dtype | 数量 |
|---|---|---|---|
| 线性注意力递推状态 | `[16, 128, 128]` | f32 | 18 层（实测 `(1,16,128,128)`） |
| 线性注意力卷积状态 | `[6144, 4]`，存**卷积前**的投影 | 与计算同 | 18 层（实测 `(1,6144,4)`） |
| 全注意力 KV | `[2, S, 256]` × 2 | f32 | 6 层 |

实测 prefill（6 token）与 prefill+decode（6+1）在最后位置 logits 上
`max|d| = 1.097e-5`、相对 `6.27e-7`。差异来自 chunk 形式与递推形式的不同求和顺序，
所以 M5 的判据是**相对容差**，不是逐位相等。

## 12. 复验读数

命令：`.venv/Scripts/python.exe dev/probe_operators.py`（f32、CPU、7 个 token、层 0 与层 3）

| 读数 | 值 |
|---|---|
| 参数 | 852,985,920 |
| `lm_head` 与 `embed_tokens` 同存储 | 是 |
| `layer_types` | `[L,L,L,F] × 6` |
| `g` 最大值 | `-6.917e-10` |
| `beta` 范围 | `0.0473 .. 0.9945` |
| 手写 linear_attn vs 模块 | `max|d| 5.96e-8`，rel `1.03e-7` |
| chunk 形式 vs 递推形式 | `max|d| 7.45e-9`，rel `6.34e-8` |
| 手写 gated norm vs 模块 | `0.000e+00` |
| 手写 full_attn vs 模块 | `max|d| 2.98e-8`，rel `9.14e-8` |
| RoPE cos 形状 | `(1, 7, 64)` |
| prefill/decode logits | `max|d| 1.097e-5`，rel `6.27e-7` |
| 卷积状态 vs 原始投影（最后 4 列） | `1.43e-6` |
| 模型内 `mtp` 参数数 | 0 |

## 13. 里程碑的检查点定义

M3 的「每层 5 个检查点」在此钉死（对每个被检查的层）：

1. `input_layernorm` 的输出；
2. token mixer（`linear_attn` 或 `self_attn`）的输出，**加残差之前**；
3. 第一次残差相加之后；
4. `post_attention_layernorm` 的输出；
5. 层输出（MLP 残差之后）。

M3 额外要求线性层内部三个中间量（卷积输出、递推输出、gated norm 输出）也 dump，
用于把错误定位到 §5 的某一步；它们不进门限，只用于定位。

M4 额外要求 `text.norm` 输出与最后位置的 logits top-1。

对拍的参考必须用**同一份量化权重**：把 `.qwqt` 反量化后装进 PyTorch 模型再 dump，
否则每一层都叠加量化误差（`quantized/README.md` 实测权重相对误差 9.1%），
门限会失去意义。这一点由 `dev/dump_reference.py` 负责。

实测（2026-11-18，三个用例、24 层）：每个检查点的最大相对误差 4.3e-6，层输出
2.5e-7..1.4e-6，logits top-1 三例全中；prefill 与逐 token 喂入的差是 `0.0`（同一条
代码路径）。读数与命令在 `dev/README.md`。

## 14. 尚未钉死的条目（每条都带判据与在哪一步钉住）

| 条目 | 由谁 / 用什么读数钉住 |
|---|---|
| MTP 的数学（拼接顺序、norm 类型、embedding/head 是否共享） | 接线取自 §10 的 vLLM 源；数值对拍需要自建 `mtp` dump |
| 分块 delta rule 是否值得做 | M9：先量递推形式的 prefill 吞吐，再决定；任何分块实现必须落在 M5 的容差里 |
| int4 GEMM 的 tile 形状与累加顺序 | M9；判据是 M1/M4 的相对容差不变，且峰值内存不随 `S` 增长 |

## 15. M2 的词表事实（实测）

这些读数来自 `AutoTokenizer.from_pretrained(model/)`，M2 的判据是 ids 完全相等。

| 事实 | 值 |
|---|---|
| `tokenizer.json` 的 `model.vocab` | 248,044 项，BPE，247,587 条 merges |
| `added_tokens`（tokenizer.json） | 26 个，id 248044 .. 248069 |
| `added_tokens_decoder`（tokenizer_config.json） | **33 个**，id 248044 .. 248076（多出 7 个 audio/tts 标记） |
| 有效词表大小 | 248,077 |
| 归一化 | NFC |
| 预分词 | `Split(Regex(...), Isolated)` + `ByteLevel(add_prefix_space=false)` |
| 正则 | `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` |
| 后处理 | 无模板，不加 BOS，`add_bos_token=false` |
| `eos_token` / `pad_token` | `<|im_end|>` 248046 / `<|endoftext|>` 248044 |

ids 实测（这些同时是 `tests/golden.rs` 的用例）：

| 文本 | ids |
|---|---|
| `Hello, world!` | `[9419, 11, 1814, 0]` |
| `你好，世界！` | `[109266, 3709, 96748, 6115]` |
| `a A` | `[64, 357]`（大小写敏感） |
| `  spaces` | `[220, 12258]` |
| `\n\n` | `[271]` |
| `don't 3.14` | `[14572, 914, 220, 18, 13, 16, 19]` |
| `<|im_start|>user\nhi<|im_end|>\n` | `[248045, 846, 198, 5834, 248046, 198]` |
| `<think>x</think>` | `[248068, 87, 248069]` |
| `🧠` | `[9008, 100, 254]` |

`<|im_end|>` 是 248046，而 `generation_config.eos_token_id` 是 248044 —— 两者不同，
生成循环按前者停还是按后者停会得到不同长度的输出，必须显式选择并记录。

### 15.1 管线的顺序（实测）

```
raw text
  -> 在原始文本上切出 33 个 added token（它们的 normalized=false）
  -> 每段剩余文本做 NFC
  -> 预分词正则切 piece
  -> 每个 piece 做 byte-level BPE
```

三点后果：

1. added token 的字节**不经过**归一化，也不会被预分词拆开，所以 `<|im_end|>` 永远是
   一个 id。
2. 归一化不会跨过 added token 合并字符（两侧是分别归一化的）。
3. BPE 只在 piece 内做合并，跨 piece 不会合并。

### 15.2 词表与切分的事实（实测）

| 事实 | 读数 |
|---|---|
| merge 产物 id | **不全是** `256 + rank`：id 104,328 处多插了一个 token，之后整体 +1。编码器用 id 当优先级，因此要求 id 随 rank 严格递增（成立） |
| byte 的 id | 词表按频率排序，空格是 32 而它的 token 是 220；256 个 byte 的 id 恰是 0..=255 的一个排列 |
| `\s*[\r\n]+` | 只在空白串**以换行结尾**时匹配：`"\n\n  "` → `["\n\n", "  "]` |
| `\s+(?!\S)` | 让回一个字符：`"    return"` → `["   ", " return"]` |
| `\p{L}`/`\p{M}`/`\p{N}` | 必须用参考自己的类表：Python 的 Unicode 14 与参考在 9,485 个码点上不同 |
| 组合类 | 必须用参考测出的值：Python 与参考在 119 个 mark 上不同（方向相反） |

### 15.3 chat template 的文本子集（实测）

| 事实 | 读数 |
|---|---|
| 内容 | 一律 `trim`；多段 `content` 直接拼接后 `trim` |
| system | 只在第一条；其余位置报错 |
| assistant 的 `<think>` 块 | 仅当该条在**最后一条 user 之后**；之前的 assistant 消息里的 reasoning 被丢弃 |
| 生成提示（`enable_thinking=false`） | `<|im_start|>assistant\n<think>\n\n</think>\n\n` |
| 生成提示（`enable_thinking=true`） | `<|im_start|>assistant\n<think>\n` |
| 没有 user 消息 | 报错（`No user query found in messages.`） |

## 16. 两条实现纪律（从实测里来的）

1. **不物化 `[S, vocab]` 的 logits。** 前期实测：8192 上下文默认路径产生 7,760 MiB
   张量、峰值 +9,540 MiB；只留最后一个位置降到 1 MiB、峰值 +1,726 MiB。
2. **不把 int4 先展开成 f32 再算。** 254M 元素的 embed 表展开一次就是数 GB 临时量；
   按 tile 直接读 int4 做 GEMM，权重全程 472 MiB。
