# qwen35-rust

Qwen3.5-0.8B 的 Rust 推理库。默认构建是 **CPU 路径、零第三方依赖**（只用 `std`）；
GPU 后端在后面的里程碑里以 feature 引入。

权重是一份 QWQT 量化容器（4-bit 分组量化，494,845,446 B），**编译期嵌入**，
运行时不做任何文件 IO。

算子级的规格（每个算子的作用轴、来自参考实现的哪一行）在
[`docs/op-spec.md`](docs/op-spec.md)。先读它，再读代码。

## 状态

| # | 交付 | 状态 | 判据 / 读数 |
|---|---|---|---|
| M0 | `dev/gen_manifest.py` + 嵌入 + 编译期校验 | **已交付** | 常量全过 `const` assert；张量数 488（模型 473 + MTP 15）；首 4 字节 `"QWQT"`；`model_info.exe` = 495,116,800 B |
| M1 | int4/int8 反量化 | **已交付** | 488/488 张量、873,438,784 个值与参考 `read_tensor` 的 CRC-32 相同；6 个整张量逐元素 `max\|d\| = 0.0` |
| M2 | tokenizer | **已交付** | 语料 56/56 的 pieces 与 ids 与参考完全相等；chat template 8/8；NFC 与字符类逐码点对过参考 |
| M3 | 文本单层（`linear_attention` 与 `full_attention` 各一） | **已交付** | 层 0 与层 3 的 5 个检查点全过，最大相对误差 3.7e-6 |
| M4 | 24 层 + final norm + 输出头 | **已交付** | 24 层逐层 residual 对拍（3 个用例），logits top-1 全中 |
| M5 | prefill / decode cache 一致性 | **已交付** | 两条路径（批量分块内核与单 token 融合内核）相对差 4.0e-7，top-1 相同；批量内任意切分逐位相同 |
| M6 | 视觉塔 + 图像输入 | **已交付** | 预处理 `max\|d\|` 5.9e-8 / 一个灰度级；塔相对误差 1.0e-5 / 9.4e-6；图像 prompt 的 logits 2.5e-6 / 2.5e-4，top-1 与参考相同；图像之后走一步 1.8e-6 / 3.0e-4；贪心生成前 10 个 token 与参考逐字相同 |
| M7 | MTP 头 | 待做 | 与自建 dump 对拍（PyTorch 主类不加载它） |
| M8 | wgpu 后端 | 待做 | 与 CPU 后端同算术路径结果一致（`--features wgpu`） |
| M9 | 性能 | **已交付**：prefill 0.333 s ＝ 243 tok/s、decode 0.020 s/token（同窗口基线 0.422 s ＝ 192 tok/s、0.113 s/token，即 prefill 1.27 倍、decode 5.6 倍） | 超过 PyTorch eager 基线（两实现交替测量；这台机器有常驻负载，不同窗口的绝对值不可比） |

每个数字的出处（命令与日期）在 [`dev/README.md`](dev/README.md)：那里只写实测
读数与未采用的方案，不写代码。

## 结构

```
src/lib.rs            公开面
src/blob.rs           对齐的嵌入缓冲与 little-endian 读取
src/weights/mod.rs    嵌入的容器、张量查找、dense 张量的 f32 视图
src/weights/manifest.rs  @generated：488 个张量的名字/形状/偏移 + 架构常量
src/weights/qwen3.5-0.8b-q4.qwqt  嵌入的权重（495 MB，-text）
src/ops.rs            数值内核：反量化、f32 读行、线性层、RMSNorm、softmax、RoPE
src/parallel.rs       fork-join worker 池（std only）
src/vision/           图像预处理（缩放、patch 化）与 12 层视觉塔
src/multimodal.rs     占位符展开、图像 token 的 mRoPE 位置、把塔的输出写进文本序列
src/cpu.rs            指令集探测与 `force-scalar` 开关
src/model/mod.rs      24 层、两种 token mixer、KV 与递推缓存
src/tokenizer/        byte-level BPE、NFC、预分词（unicode.rs / split.rs）
src/chat.rs           文本对话模板
src/tokenizer/data/   嵌入的词表（5.8 MB）与 Unicode 表（83 KB）
docs/op-spec.md       算子规格
dev/                  开发脚本（git-ignored，见 dev/README.md）
tests/golden.rs       不依赖 fixture 的检查
tests/reference.rs    与参考实现对拍（`--features reference-fixtures`）
tests/text.rs         逐层与 logits 对拍、缓存一致性
examples/model_info.rs 打印容器内容，并校验真实字节
examples/tokenize.rs   渲染 prompt、编码、解码回来
examples/say_text.rs   端到端贪心生成
examples/describe_image.rs 读一张 PPM，描述它（图像输入）
examples/bench.rs      prefill 与 decode 的计时（预热 + 中位数）
tests/vision.rs       视觉塔、预处理与图像 prompt 的对拍
```

## 构建与测试

```
cargo build
cargo test
cargo run --release --example model_info
```
默认构建没有依赖。对拍参考实现的层需要 dump，它们不在仓库里：

```
cargo test --features reference-fixtures
```

没有 dump 时这些测试会失败，而不是静默通过。

`--features force-scalar` 关掉向量内核，用来端到端测量可移植路径。

## 嵌入的表

三个文件都用 `include_bytes!` 编译进去，放在 `#[repr(C, align(64))]` 的静态里（rustc 不
保证 `include_bytes!` 的对齐，而容器里的 scale 是 f16）。运行时没有文件 IO。

| 文件 | 字节 | 内容 |
|---|---:|---|
| `src/weights/qwen3.5-0.8b-q4.qwqt` | 494,845,446 | 权重容器 |
| `src/tokenizer/data/tokenizer.bin` | 5,808,391 | 词表、merges、added tokens |
| `src/tokenizer/data/unicode.bin` | 83,104 | 字符类、组合类、NFC 表 |

`manifest.rs` 里的 `offset` 是**相对嵌入文件**的绝对字节偏移，`weights::blob()[offset..]`
就是该张量的第一个字节。manifest 由容器自己的 header 生成，所以文件换一份而 manifest
没重生成时，`Aligned<FILE_LEN>` 的长度不匹配会直接让编译失败。

**不要用「只取地址」的代码量二进制大小**：那样的静态会被链接器丢掉，而 `len()` 仍然
打印 494,845,446（长度来自类型）。`examples/model_info.rs` 对真实字节做 FNV 校验和，
所以它的 495,116,800 B 才是嵌入的代价。

## 许可与第三方

Apache-2.0。权重、tokenizer、chat template 是 Qwen 团队的 Apache-2.0 作品；
本构建所用的每个文件的字节数与 SHA256 见 [`THIRD-PARTY.md`](THIRD-PARTY.md) 与
[`NOTICE`](NOTICE)。
