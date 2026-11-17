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
| M2 | tokenizer | 待做 | 与参考 tokenizer 的 ids 完全相等（含模板渲染） |
| M3 | 文本单层（`linear_attention` 与 `full_attention` 各一） | 待做 | 每层 5 个检查点对拍（定义见规格 §13） |
| M4 | 24 层 + final norm + 输出头 | 待做 | 逐层 residual 对拍；logits top-1 与参考一致 |
| M5 | prefill / decode cache 一致性 | 待做 | 一次喂完 与 逐 token 喂，相对差在规格 §12 的量级内 |
| M6 | 视觉塔 + 图像输入 | 待做 | `patch_embed` → 12 层 → `merger` 逐点对拍 |
| M7 | MTP 头 | 待做 | 与自建 dump 对拍（PyTorch 主类不加载它） |
| M8 | wgpu 后端 | 待做 | 与 CPU 后端同算术路径结果一致（`--features wgpu`） |
| M9 | 性能 | 待做 | 超过 PyTorch eager 基线 |

## 结构

```
src/lib.rs            公开面
src/weights/mod.rs    嵌入的容器、张量查找、dense 张量的 f32 视图
src/weights/manifest.rs  @generated：488 个张量的名字/形状/偏移 + 架构常量
src/weights/qwen3.5-0.8b-q4.qwqt  嵌入的权重（495 MB，-text）
docs/op-spec.md       算子规格（M1 之前的唯一依据）
dev/                  开发脚本（git-ignored，见 dev/README.md）
tests/golden.rs       不依赖 fixture 的检查
examples/model_info.rs 打印容器内容，并校验真实字节
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

## 权重嵌入

容器整个文件用 `include_bytes!` 嵌入，放在一个 `#[repr(C, align(64))]` 的静态里：
rustc 不保证 `include_bytes!` 的对齐，而容器里的 scale 是 f16。

`manifest.rs` 里的 `offset` 是**相对嵌入文件**的绝对字节偏移，`weights::blob()[offset..]`
就是该张量的第一个字节。`manifest` 由容器自己的 header 生成，所以文件换一份、manifest
没重生成时，`Aligned<FILE_LEN>` 的长度不匹配会直接让编译失败。

**不要用「只取地址」的代码量二进制大小**：那样的静态会被链接器丢掉，而 `len()` 仍然
打印 494,845,446（长度来自类型）。`examples/model_info.rs` 对真实字节做 FNV 校验和，
所以它的 495,116,800 B 才是嵌入的代价。

## 许可与第三方

Apache-2.0。权重、tokenizer、chat template 是 Qwen 团队的 Apache-2.0 作品；
本构建所用的每个文件的字节数与 SHA256 见 [`THIRD-PARTY.md`](THIRD-PARTY.md) 与
[`NOTICE`](NOTICE)。
