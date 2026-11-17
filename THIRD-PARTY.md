# 第三方数据与参考材料

本仓库不复制第三方源码。下表是**这份构建实际读过的字节**：每个文件都有字节数与
SHA256，重算一遍就能确认输入没变。

## 模型发布（Apache-2.0，Qwen 团队）

来源：`D:\Projects\Models\Qwen3.5-0.8B\model\`（本地 HF 快照）。快照没有保留上游
commit id，所以文件的 SHA256 就是它的身份。

| 文件 | 字节 | SHA256 | 用途 |
|---|---:|---|---|
| `config.json` | 3,002 | `f9061afb89151d97c70fd78de2ce3fe4e11cdf5e853e36ebe19e16499bf5f619` | 架构常量 |
| `tokenizer.json` | 12,807,982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` | BPE 词表与 merges |
| `tokenizer_config.json` | 17,013 | `ce3cbcd05aaa91d20c1605c5c46de4bacfe439c1c0efd8b27e6b5489de968f2c` | 33 个 added token 的 id |
| `chat_template.jinja` | 7,908 | `306abaf09cf3a6c0d14128373a72925bda3d7de2616a5642804d9768c7d2f78c` | 对话模板 |
| `preprocessor_config.json` | 410 | `2767d5efeb29362bc8e25e8e6baa9bc862da73fdcb5eee4db4e7f23f6625b6f7` | 图像尺寸与归一化参数 |
| `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | 未在仓内重算（1.7 GB） | 量化器的输入；索引里的 `total_size` 是 1,746,882,752 |

## 嵌入的表（衍生自上面的材料）

| 文件 | 字节 | SHA256 | 内容 |
|---|---:|---|---|
| `src/weights/qwen3.5-0.8b-q4.qwqt` | 494,845,446 | `29773850a7ac53122e1544a24748701b55fa59ca63fed89cae71b9baabc7dfb9` | 权重容器，逐字节复制 |
| `src/tokenizer/data/tokenizer.bin` | 5,808,391 | `a6a72c72711a3b8f9cc9274ae4f498de31e18df600cb141ac5977a202543be0f` | 词表、merges、added tokens |
| `src/tokenizer/data/unicode.bin` | 83,104 | `90ec30d9b7d3a18689fcb0ae8a8950e714eb3480b5450d9d49e4df28e7fb1804` | 字符类、组合类、NFC 表 |

`unicode.bin` 不是从任何一个 Unicode 版本拷来的：字符类与组合类是**实测**参考
0.23.3 的行为得到的（见 `dev/gen_unicode.py`），分解对来自 UnicodeData 14.0.0 并由
参考判定哪些可合成。

容器的生成策略记在它自己的 header 里：`g32 bits=4 embed=4 vision=4 first_last=0x8
overrides=none clip_search=True`。

## 参考实现（只读，不进仓库）

| 材料 | 版本 / 字节 / SHA256 | 用途 |
|---|---|---|
| `transformers/models/qwen3_5/modeling_qwen3_5.py` | 5.23.0，87,255 B，`1a475c45d900e0bfb06b9b245257d7f3234fd247474fcbd3363b03430e504a68` | 算子规格的来源 |
| `transformers/models/qwen3_5/configuration_qwen3_5.py` | 5.23.0，7,892 B，`a2a7c5c2f193dec6bbad0c5dcd2536dfba2eba14259814b9dc0438afe01969d9` | 视觉塔的 `rope_theta` 默认值等 |
| `reference/quant/quantlib.py` | 本机快照 | 反量化的判据（`read_tensor`） |
| `reference/run_chat.py`、`reference/benchmark.py` | 本机快照 | 参考输出与基准口径 |

## 许可

模型与 Transformers 都是 Apache-2.0。本 crate 也是 Apache-2.0。
