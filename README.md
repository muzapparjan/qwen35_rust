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

## 指令集与资源上限

向量内核按平台各写一份，语义相同、宽度不同，由 [`cpu::features`](src/cpu.rs) 选择：

| 平台 | 内核 | 宽度 |
|---|---|---|
| x86-64（Intel、AMD 同一套） | AVX2 + FMA3 | 8 × f32 |
| ARM 64 位（Android、iOS、Apple Silicon） | Advanced SIMD（NEON）+ FMA | 4 × f32 |
| 其它（含 32 位 ARM） | 标量 | 1 |

AVX2 与 FMA3 是 x86-64 的公共扩展，不是某一家的指令；NEON 在 ARMv8-A 里是必备项，所以
64 位 ARM 不需要运行时检测。两个内核都由 `ops` 的单元测试与标量路径对拍（展开逐位相等，
点积在求和次序的误差内），这些测试在任何平台上构建时都会跑。

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

## GPU 后端

`--features wgpu` 打开第二个后端，默认关闭——打开它才第一次引入依赖，默认构建仍然是零依赖。

```
cargo run --release --features wgpu --example gpu_bench
cargo test  --features wgpu --test gpu
```

两个内核目前搬到了设备上：q4 展开与 q4 矩阵-向量乘。判据按结果种类分开，因为在 WGSL 里
**归约的结合次序与乘加是否融合由实现决定**：

| 内核 | 判据 | 实测（RTX 4090 / Vulkan） |
|---|---|---|
| 展开（逐元素，无归约） | 逐位相等 | 524,288 / 524,288 个值等于 CPU |
| 矩阵-向量（归约） | 相对误差 < 1e-5 | 最差 5.8e-7 |

## 一次前向只提交一次

设备后端**不接受逐算子调用**。`backend::Flow` 记录整趟前向（哪些内核、在哪些缓冲区、用哪些
权重），`Device::execute` 按 64 个算子一块边录边提交，回读一次。分块而不是一张 command buffer
是量出来的：设备在主机录后面的块时就已经开工，解码一步因此少约 2 ms，队列保证跨块的顺序。
原因是可算的：单次提交 + 回读约 0.6 ms，一次前向约 200 个算子 ⇒ 逐算子会付 120 ms 的搬运。

已搬进 `Flow` 的算子：

| 算子 | 判据 | 实测误差 |
|---|---|---|
| `Dequantize` | 逐元素，**逐位相等** | 0 |
| `Linear`（q4 矩阵-向量） | 归约，相对误差 | ≤ 5.8e-7 |
| `RmsNorm` / `GatedRmsNorm` / `LayerNorm` | 归约，相对误差 | ≤ 4.1e-7 |
| `SiluMul` / `Scale` / `Rope` | 逐元素，**逐位相等** | 0 |
| `Softmax` / `Sum` / `ArgMax` | 归约 / 精确 | ≤ 1.1e-7 |
| `DeltaRule`（GDN 递推） | 归约，相对误差 | ≤ 1.1e-7 |

一层的 MLP 作为 5 个算子的整趟 pass：

| 调用方式 | 实测 |
|---|---|
| 逐算子提交，`k_proj` 512×1024 | 0.81 ms（CPU 0.06 ms，**0.08×**） |
| 逐算子提交，`embed_tokens` 248320×1024 | 2.24 ms（CPU 6.24 ms，2.79×） |
| **整趟 5 个算子一次提交** | **0.87 ms**（CPU 0.94 ms，1.09×，五个中间缓冲全部 ≤ 1.1e-7） |

同样的 5 个算子若各自提交，按 `k_proj` 那行量到的固定开销算要约 4 ms；合成一次提交后是
0.87 ms。**这就是"一次前向一次调度"与"一算子一次调度"的差别。**

### 整趟前向已经可以录下来

`Model::forward_recorded` 把**整个模型**录成一份 `Spec`（24 层、两种注意力、MLP、输出头），
再由 `backend::execute` 一趟执行完。它与逐算子路径的判据是**逐位相等**：

```
tests/recorded.rs
  the_recorded_pass_is_the_direct_one_bit_for_bit        ok   logits 全等
  the_recorded_pass_reads_the_cache_it_is_given          ok   从已填缓存继续
  the_recorded_steps_carry_their_state                   ok   连续三步解码，token 序列相同
```

第三条是生成循环的形状：状态（KV 缓存、卷积窗口、递推状态）以 `Slot::External` 交给 pass，
`Model::forward_recorded_step` 把它交还给调用方，下一步再喂回去。如果这一步没接对，第一步
会对、第二步就发散。

### 设备侧

设备执行器实现全部 21 个算子，整趟前向（514 个算子、199 个张量、5 个 token）一次录完，
与 CPU 执行器的相对误差 ≤ 1.8e-5，argmax 相同。状态（KV 缓存、卷积窗口、递推状态）留在
`DeviceState` 的设备缓冲区里，跨调用不重传。

录制声明的是**每个中间值一个 slot**，512 token 的 prefill 因此声明 1.9 GB，而任意瞬间活着的
只有几个缓冲。`backend::layout` 按活跃区间把 slot 叠到同一块存储上，同一趟变成 12 块、49 MB；
1536 token 从“超过单缓冲上限直接报错”变成 148 MB 可跑。回读点名的 slot 不共享存储，所以
`execute_stateful_reading` 读到的一定是它自己最后写进去的值。

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
