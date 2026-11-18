# qwen35-rust

Qwen3.5-0.8B 的 Rust 推理库。默认构建是 **CPU 路径、零第三方依赖**（只用 `std`）；
`wgpu` feature 加上设备后端，跑同一套算术。

权重是一份 QWQT 量化容器（4-bit 分组量化，494,845,446 B），**编译期嵌入**，
运行时不做任何文件 IO。容器本身是 release asset，不在仓库里：第一次构建前先跑
`scripts/fetch-weights.ps1` 把它取到 `src/weights/` 并校验。

算子级的规格（每个算子的作用轴、来自参考实现的哪一行）在
[`docs/op-spec.md`](docs/op-spec.md)。

## 能做什么

- **文本生成。** 24 层，线性注意力与全注意力按 3:1 交错；KV 缓存与递推状态由
  `model::Cache` 持有，`Model::forward` 每次推进任意多个 token。
- **投机解码。** 容器里的 15 个 `mtp.*` 张量组成草稿头，一轮里完成草稿、验证与回滚；
  输出与贪心逐 token 相同。
- **图像输入。** `vision::encode` 做预处理与 12 层视觉塔，`multimodal::expand_placeholders`
  展开占位符，`Model::forward_prompt` 按 mRoPE 位置把塔的输出写进文本序列。
- **流式输出。** `Tokenizer::decoder` 按 id 增量给文本：一个字符的字节跨几个 id 时先压住，
  只写出完整的字符。
- **两个执行后端。** 默认可移植 CPU；`wgpu` 下 `backend::Flow` 把整趟前向录成一份 pass，
  在设备上分块提交、回读一次。

## 快速开始

```
scripts\fetch-weights.ps1
cargo run --release --example say_text -- "用一句话介绍你自己。"
cargo run --release --example bench -- --prompt hello --tokens 32
```

库里最小的一段：

```rust
use qwen35::model::{Cache, Model, NoTrace};
use qwen35::tokenizer::tokenizer;

let tokenizer = tokenizer();
let model = Model::new();
let mut cache = Cache::new();
let ids = tokenizer.encode("法国的首都是");
let logits = model.forward(&ids, &mut cache, &mut NoTrace);
// logits 是最后一个位置的分布，见 examples/say_text.rs 里的取最大与打印。
```

`Model::new` 读的是编译期嵌入的表，没有文件 IO，也不需要初始化步骤。后续的单 token 调用
接着同一个 `Cache` 走，位置由缓存自己数。

## 结构

```
src/lib.rs              公开面
src/model/mod.rs        24 层、两种 token mixer、KV 与递推缓存
src/model/mtp.rs        投机解码的草稿头
src/model/record.rs     把一整趟前向录成 backend::Spec
src/backend.rs          算子集合、录制（Flow）、按活跃区间叠存储（layout）、CPU 执行器
src/ops.rs              数值内核：反量化、f32 读行、线性层、RMSNorm、softmax、RoPE
src/gpu/mod.rs          设备执行器与 WGSL 流水线
src/gpu/kernels.wgsl    算子的设备实现（WGSL）
src/weights/mod.rs      嵌入的容器、张量查找、dense 张量的 f32 视图
src/weights/manifest.rs @generated：488 个张量的名字/形状/偏移 + 架构常量
src/weights/qwen3.5-0.8b-q4.qwqt  嵌入的权重（494.8 MB，由 scripts/fetch-weights.ps1 取得）
src/tokenizer/          byte-level BPE、NFC、预分词、增量解码
src/chat.rs             文本对话模板
src/vision/             图像预处理（缩放、patch 化）与 12 层视觉塔
src/multimodal.rs       占位符展开、图像 token 的 mRoPE 位置、把塔的输出写进文本序列
src/parallel.rs         fork-join worker 池（std only）
src/cpu.rs              指令集探测与 force-scalar 开关
src/blob.rs             对齐的嵌入缓冲与 little-endian 读取

tests/golden.rs         不依赖 fixture 的检查
tests/text.rs           逐层与 logits 对拍、prefill 与 decode 一致（需 fixture）
tests/vision.rs         视觉塔、预处理与图像 prompt 的对拍（需 fixture）
tests/reference.rs      聊天模板与参考输出（需 fixture）
tests/recorded.rs       录制路径与逐算子路径逐位相等
tests/speculation.rs    投机解码不改变输出
tests/gpu.rs            设备上的反量化与矩阵-向量
tests/gpu_ops.rs        整趟前向到不了的算子、以及超出一维调度的情形
tests/gpu_model.rs      整趟前向、跨步状态、存储共享

examples/model_info.rs  打印容器内容，并校验真实字节
examples/tokenize.rs    渲染 prompt、编码、解码回来
examples/say_text.rs    端到端生成
examples/speculate.rs   投机解码与贪心的对照
examples/describe_image.rs  读一张 PPM，描述它
examples/bench.rs       prefill 与 decode 的计时
examples/gpu_bench.rs   设备后端的计时
```

## 构建、测试与 feature

```
cargo build
cargo test
cargo run --release --example model_info
```

| feature | 作用 |
|---|---|
| （默认） | CPU，零第三方依赖 |
| `wgpu` | 设备后端，同一套算术 |
| `force-scalar` | 关掉向量内核，用来量可移植路径 |
| `reference-fixtures` | 与参考实现对拍；需要 `dev/` 下的 dump，它们不在仓库里 |

没有 dump 时 `reference-fixtures` 的测试会报错，而不是静默通过。

## 指令集

向量内核按平台各写一份，语义相同、宽度不同，由 [`cpu::features`](src/cpu.rs) 选择：

| 平台 | 内核 | 宽度 |
|---|---|---|
| x86-64（Intel、AMD 同一套） | AVX2 + FMA3 | 8 × f32 |
| ARM 64 位（Android、iOS、Apple Silicon） | Advanced SIMD（NEON）+ FMA | 4 × f32 |
| 其它（含 32 位 ARM） | 标量 | 1 |

AVX2 与 FMA3 是 x86-64 的公共扩展，不是某一家的指令；NEON 在 ARMv8-A 里是必备项，所以
64 位 ARM 不需要运行时检测。两套向量内核都与可移植路径对拍：展开逐位相等，点积在求和次序
的误差内。

生成是 fork-join 负载，会把给它的核全部用满。库因此收一个 [`parallel::Budget`]，而不是
自己决定：

```rust
qwen35::parallel::set_budget(qwen35::parallel::Budget {
    threads: 4,        // 0 = 按机器核数减去 reserve
    reserve: 2,        // 留给宿主的核数
    priority: qwen35::parallel::Priority::Background,
});
```

默认就是让路的那一档：最多 16 个 worker、留 2 个核、优先级低于所有普通线程。Windows 用
`THREAD_PRIORITY_LOWEST`，Linux/Android 用 `SCHED_IDLE`，macOS 用 `PRIO_DARWIN_BG`，三者都
无需特权且可随时改回。宿主在加载关卡、开菜单时可以调 `set_priority` 让路，也可以反过来把
核收回去。

## 设备后端

`--features wgpu` 打开第二个后端，默认关闭——打开它才第一次引入依赖，默认构建仍然是零依赖。

```
cargo run --release --features wgpu --example gpu_bench
cargo test  --features wgpu --test gpu
```

设备执行器实现全部 21 个算子：`backend::Flow` 把整趟前向（哪些内核、在哪些缓冲区、用哪些
权重）录成一份 `Spec`，`Device::execute` 分块提交、回读一次。状态（KV 缓存、卷积窗口、递推
状态）留在 `DeviceState` 的设备缓冲区里，跨调用不重传。

判据按结果种类分开，因为 WGSL 里归约的结合次序与乘加是否融合由实现决定：逐元素算子与 CPU
逐位相等，归约算子在相对容差内，argmax 相同；录制路径与逐算子路径逐位相等。

## 内存

| 项 | 数字 |
|---|---:|
| 嵌入的权重与表（常驻） | 495,116,800 B |
| 设备缓冲区，512 token 的 prefill | 49 MB |
| 设备缓冲区，1536 token | 148 MB |
| 设备缓冲区，2048 token | 197.7 MB |

设备上占的是任意瞬间活着的量，不是中间值个数乘大小：`backend::layout` 把活跃区间不重叠的
slot 叠到同一块存储上，回读点名的 slot 不共享存储。

## 性能

`cargo run --release --example bench`，16 worker、`reserve: 0`、`Priority::Normal`：

| 项 | 读数 |
|---|---|
| prefill，156 token 的提示 | 0.58 s ＝ 269 tok/s |
| decode（1.6 KB 上下文） | 0.020 s/token ＝ 51 tok/s |

这台机器上还有别的常驻负载，不同时间窗的绝对值不可比；读数、条件与没有采用的方案在
[`dev/README.md`](dev/README.md)。

## 嵌入的表

三个文件都用 `include_bytes!` 编译进去，放在 `#[repr(C, align(64))]` 的静态里（rustc 不
保证 `include_bytes!` 的对齐，而容器里的 scale 是 f16）。运行时没有文件 IO。

| 文件 | 字节 | 内容 |
|---|---:|---|
| `src/weights/qwen3.5-0.8b-q4.qwqt` | 494,845,446 | 权重容器 |
| `src/tokenizer/data/tokenizer.bin` | 5,808,391 | 词表、merges、added tokens |
| `src/tokenizer/data/unicode.bin` | 83,104 | 字符类、组合类、NFC 表 |

权重容器不在仓库里，由 `scripts/fetch-weights.ps1` 从 release asset 下载，按字节数与
SHA256 校验后落到 `src/weights/`。缺它时 `build.rs` 直接让构建失败，而不是让
`include_bytes!` 报一句只带路径的错。

`manifest.rs` 里的 `offset` 是**相对嵌入文件**的绝对字节偏移，`weights::blob()[offset..]`
就是该张量的第一个字节。manifest 由容器自己的 header 生成，所以文件换一份而 manifest
没重生成时，`Aligned<FILE_LEN>` 的长度不匹配会直接让编译失败。

`examples/model_info.rs` 报的 495,116,800 B 是嵌入的代价：它对真实字节做校验和，而不是读
类型里的长度。

## 许可

本仓的代码是 **Apache-2.0**，见 [`LICENSE`](LICENSE)。

它**分发**三份衍生数据（它们随本项目分发，不是运行时读的外部文件），义务因此不止是「引用」：

| 文件 | 上游 | 与上游的关系 |
|---|---|---|
| `src/weights/qwen3.5-0.8b-q4.qwqt` | Qwen3.5-0.8B 权重（Apache-2.0，Copyright 2026 Alibaba Cloud） | `quantized/` 里那份额外 4-bit 量化产物的逐字节复制 |
| `src/tokenizer/data/tokenizer.bin` | 模型的 `tokenizer.json` / `tokenizer_config.json`（同上） | 重新编码成打包表 |
| `src/tokenizer/data/unicode.bin` | Unicode 数据（Unicode-3.0） | 由 UCD 与参考分词器的实测行为生成 |

上游的版权声明、逐文件的改动说明与 Unicode 的许可声明在 [`NOTICE`](NOTICE)；输入文件的
字节数与 SHA256 在 [`THIRD-PARTY.md`](THIRD-PARTY.md)。

**「Qwen」是 Alibaba Cloud 的商标**，本仓用它描述所运行的模型，不代表与其有关联或受其认可。
