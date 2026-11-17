# 开发笔记

每个数字旁边都写着产生它的命令与日期。重跑之前不要相信旧读数：torch、checkpoint 与
Rust 工具链都没有冻结。

`dev/` 与 `tmp/` 在 `.gitignore` 里（库的使用者选解析器；开发脚本不随 crate 发布）。
生成的 `src/weights/manifest.rs` 是入库的，`dev/gen_manifest.py` 只在本机存在。

## 环境（2026-11-18）

- `rustc 1.97.1`，Windows，`core.autocrlf = true`
- Python 3.11.4，torch 2.11.0+cu128，transformers 5.23.0（在
  `D:\Projects\Models\Qwen3.5-0.8B\.venv`）
- 容器 `qwen3.5-0.8b-q4.qwqt`，494,845,446 B

## M0：manifest 与嵌入（2026-11-18）

生成命令：

```
D:/Projects/Models/Qwen3.5-0.8B/.venv/Scripts/python.exe dev/gen_manifest.py ^
    --container src/weights/qwen3.5-0.8b-q4.qwqt ^
    --config D:/Projects/Models/Qwen3.5-0.8B/model/config.json ^
    --tokenizer-config D:/Projects/Models/Qwen3.5-0.8B/model/tokenizer_config.json ^
    --out src/weights/manifest.rs
```

| 读数 | 值 |
|---|---|
| 张量数 | 488（模型 473 + `mtp.*` 15） |
| 参数 | 873,438,784（模型 852,985,920） |
| header 字节 | 80,634，payload 起点 80,646 |
| payload 字节 | 494,764,800 |
| 层划分 | 18 个 `linear_attention`、6 个 `full_attention` |
| tokenizer 词表上界 | 248,077（33 个 added token，id 248044..248076） |

编译与测试（同一天）：

| 命令 | 读数 |
|---|---|
| `cargo build`（debug，首次） | 4.41 s |
| `cargo test`（debug） | 3.64 s 编译，18 个测试全绿 |
| `cargo run --release --example model_info` | 21.75 s（含 example 与 495 MB 链接） |
| `target/release/examples/model_info.exe` | **495,116,800 B（472.2 MiB）** |

最后一行是嵌入权重的代价，不是缺陷。它与前期读数 494,971,904 B 差 144,896 B，
差在 manifest 表、FNV 校验与 `println!` 代码上。

## 两个实测结论（改实现前先读）

### 1. 只取地址量不到二进制大小

前期实测：一份只做 `as_ptr()` 与 `len()` 的程序，静态被链接器整个丢掉，二进制
125 KB，而 `bytes.len()` 照样打印 494,845,446 —— 长度来自类型，不来自数据。
`examples/model_info.rs` 因此对真实字节做 FNV 校验和；那个 472.2 MiB 才是嵌入的代价。
这一条没有在我们这边复现：写一个故意错误的测量程序没有信息量。

### 2. payload 起点不是 4 的倍数

第一版把 f32 张量按 `&[f32]` 零拷贝切出来，测试当场报
`A_log holds 17551584000000000000000000000000000`。

原因有两层，两层都要记住：

- 容器 header 里的 `offset` 是**相对 payload** 的，不是相对文件。直接用到文件上的
  切片会读到 header 的字节。
- 即使把 header 长度加上，payload 从第 80,646 字节开始，而 `80,646 % 4 = 2`，
  f32 张量的绝对地址不在 4 字节边界上。

采用的修法：manifest 存绝对偏移；dense 张量按字节存，用 `from_le_bytes` 转 f32
（dense 一共 2.4M 个值，转换一次不到 10 MB）；f16 scales 只需要 2 字节对齐，绝对偏移
恒为偶数，所以按 `u16` 读是安全的。crate 里因此没有任何 `unsafe`。

**没有采用**的方案：在文件前面垫 2 个字节让它对齐。它依赖 `#[repr(C)]` 的字段布局，
一旦 `include_bytes!` 长度或 header 长度变化就会静默错位，而收益只是省掉 2,592 个
f32 值的转换。

## M1：反量化（2026-11-18）

生成参考读数并对比：

```
D:/Projects/Models/Qwen3.5-0.8B/.venv/Scripts/python.exe dev/dump_dequant.py
cargo test --release --features reference-fixtures --test reference -- --nocapture
```

| 读数 | 值 |
|---|---|
| 参考 dump | 488 张量、873,438,784 个值、7.7 s |
| 布局自检（手写 unpack 对照 `read_tensor`） | `max|d| = 0.000e+00` |
| Rust 全量对比 | 488/488 张量的 CRC-32 相同，8.10 s |
| 逐元素对比 | 6 个整张量 `max|d| = 0.000e+00`（q4 四个、bf16 一个、f32 一个） |
| `cargo test --features force-scalar --test golden` | 12 个测试全绿 |

### 判据用了 CRC-32，不是「看起来一样」

一个张量有上亿个值，逐元素打印没有信息量；CRC-32 把「低半字节是偶数下标」「分组不跨行」
「scale 乘在整组上」「bf16 取高 16 位」这四件事任一处的错误压成一个数，覆盖**每一个**值。
Python 侧是 `zlib.crc32`，Rust 侧在 `tests/common/mod.rs` 里按同一条多项式实现，先用
`crc32(b"123456789") == 0xCBF43926` 钉住它。

`max|d| = 0.0` 的逐元素断言放在 dump 出来的 6 个整张量上（16 到 3.7M 个值），因为那才是
「每个值的差都是 0」这句话能给出的最强形式；CRC-32 负责全部 488 个。

### 向量路径与标量路径逐位相同

反量化的每个输出是「一个精确可表示的整数 × 一个 f32」，没有求和，所以两条路径没有可以
分歧的次序。这不是「碰巧落在容差内」，是同一个算式：`tests/golden.rs` 的
`the_vector_path_and_the_portable_path_agree_bit_for_bit` 比 `to_bits()`。
AVX2 路径因此不需要单独的容差，`force-scalar` 与默认构建给出同一批字节。

### 没有采用的方案

把 dense 张量也塞进同一条解包路径（统一成一个实现）。它省一个分支，但让 2,592 个 f32
常量与 871M 个 q4 值走同一个循环。现在 dense 走 `dense_values()`，量化走
`ops::dequantize`。

## M2：tokenizer（2026-11-18，表与 BPE 内核，未完成）

生成命令：

```
D:/Projects/Models/Qwen3.5-0.8B/.venv/Scripts/python.exe dev/gen_tokenizer.py ^
    --tokenizer D:/Projects/Models/Qwen3.5-0.8B/model/tokenizer.json ^
    --tokenizer-config D:/Projects/Models/Qwen3.5-0.8B/model/tokenizer_config.json ^
    --out-data src/tokenizer/data/tokenizer.bin ^
    --out-manifest src/tokenizer/manifest.rs
```

| 读数 | 值 |
|---|---|
| base token（单个 byte 的符号） | 256 |
| merges | 247,587 |
| vocab | 248,044 |
| added token | 33（id 248044..248076） |
| blob | 5,808,391 B，sha256 `a6a72c72…cbe0f` |

### 两个被生成器当场否掉的「显然」的假设

1. **merge 产物不是 `256 + rank`。** 生成器第一版按 `len(vocab) - len(merges) = 457`
   做基址，立刻报 `merge 0 produces id 256`。真实关系是 `256 + rank`，**但**在
   rank 104072 处多插入了一个 token（id 104328），之后偏移 +1。所以每个 merge 把产物
   id 显式存下来；生成器另外断言这些 id 随 rank 严格递增——编码器用 id 当合并优先级，
   这条性质不成立时「按 id 最小」就不等于「按 rank 最小」。
2. **byte 的 id 不是 byte 本身。** 空格是 32，而它的 token 是 220。词表按频率排序，
   所以生成器导出 256 项的 byte→id 表，并断言这 256 个 id 恰好是 0..=255 的一个排列。

### 词表里存的是 byte，不是符号串

`vocab_pool` 存每个 token 对应的**原始字节**（生成时用反向 byte 映射还原），所以
`decode` 是把字节拼起来，不需要反向表；编码端也从不查字符串，因为 byte→id 表已经给了
起点。生成器对每个 token 检查「所有字符都在 byte 映射里」，否则退出。

### 已对拍的部分

| 判据 | 读数 |
|---|---|
| 语料的 pieces + ids（`dev/dump_tokenizer.py` → `dev/fixtures/tokenizer.txt`） | 56/56 相等 |
| chat template（`dev/dump_chat.py` → `dev/fixtures/chat.txt`） | 8/8 相等 |
| NFC / NFD 逐码点（生成器内部） | 1,114,112 个码点，0 处不一致 |
| 组合类排序（每个 mark × 每个哨兵，两种顺序） | 0 处不一致 |
| 字符类逐码点（参考正则 vs 我们的表） | 生成时就是参考的读数 |

### Unicode 表：两个方向都不一致，所以表是**测**出来的

Python 3.11 的 `unicodedata` 是 Unicode 14.0.0，参考的 `tokenizers` 0.23.3 不是同一个
版本，而且**两边各有一部分对方没有的东西**：

| 项 | 不一致处数 | 方向 |
|---|---|---|
| NFD / NFC 逐码点 | 1（U+11938） | 参考会分解它，Python 不会 |
| `\p{L}`/`\p{M}`/`\p{N}` 逐码点 | 9,485 | 全是 Python 认为「未分配」而参考知道是字母或 mark |
| 组合类 | 119 个 mark | Python 给了非零类，参考却把它当 starter（如 U+16FF0） |

所以没有任何一方能当真相。最终做法：

1. **字符类**：拿参考自己的预分词正则对 1,114,112 个码点逐个探测（`behavior="removed"`
   时整串被吃掉就说明匹配），得到 1,142 个区间。
2. **组合类**：用「排序探测」测量——`NFC(chr(b) + chr(a))` 以 `a` 开头 ⟹ `a` 的类更小。
   把所有 mark 用这个比较器排序，相邻且互不更小的归为一组；每组取一个两边都认的成员
   作为它的类值，两边都不认的组由相邻两组夹出唯一的合法值。starters 先用三个探针
   识别（含用 `chr(c) + U+0300` 与 `chr(c) + U+0334` 区分 0 与 230）。
3. **分解 / 合成**：分解用参考的 NFD（含 U+11938）；合成对来自 `unicodedata.decomposition`
   的**直接**分解对，再由参考的 NFC 判定是否真能合成（自动得到排除表）。

第 3 条踩过一个坑：第一版把「NFD 长度 2」当成交互对，于是 U+01D5（U 分两段合成）没有
进入合成表 —— 生成器的逐码点检查当场报了 287 处。直接用 UnicodeData 的**直接**分解
才对。

还有一条不显然的事实：**带分解的 mark 不需要组合类**。`NFC` 先分解，轮到排序时那个
mark 已经不在了，所以表里给它记一个类就是死数据。生成器只测原子 mark（2,451 个）。

### 预分词：两个分支的回溯是可见的

- `\s*[\r\n]+` 只在**空白串以换行结尾**时匹配；`"\n\n  "` 会拆成 `["\n\n", "  "]`。
- `\s+(?!\S)` 会**让回一个字符**，所以 `"    return"` 拆成 `["   ", " return"]`，
  而不是 `["    ", "return"]`。这就是缩进不会和后面的词粘成一片的原因。

### 没有采用的方案

- **从 Rust 的 `regex` / `unicode-normalization` crate 直接导表**：省掉探测，但那两个
  crate 的 Unicode 版本（18.0.0 / 16.0.0）比参考更新，恰好在「参考不认识的字符」上
  给出与参考不同的行为。版本新不等于对。
- **只信 Python 的 `unicodedata`**：见上表，9,485 个字符会被分成 `Other`，于是 CJK
  扩展区等新字母会被当成标点，切法与参考不同。

## M3–M5：文本模型（2026-11-18）

```
D:/Projects/Models/Qwen3.5-0.8B/.venv/Scripts/python.exe dev/dump_reference.py
cargo test --release --features reference-fixtures --test text -- --nocapture
```

dump 用**反量化后的 q4 权重**跑参考（`read_tensor` 把 473 个参数覆盖成容器里的值），
所以层差是算术差，不是量化误差。三个用例：`hello`（4 token）、`chinese`（4）、
`mixed`（18）。

| 读数 | 值 |
|---|---|
| 每个用例的 dump | 171 个张量 |
| 逐层对比 | 24 层 × 5 个检查点 + mixer 内部量 + final norm + logits，全部在 1e-3 以内 |
| 最大相对误差 | 4.3e-6（`chinese` 的层 19 `self_attn.out`） |
| 层输出（`layers_{i}_out`） | 2.5e-7 .. 1.4e-6 |
| logits top-1 | 3/3 与参考相同（271、198、727） |
| prefill vs 逐 token | `max|d| = 0.0`（同一条代码路径） |
| 生成 | `say_text`：28 token prefill 7.77 s，首 token 0.39 s，输出「你好！我是 Qwen3.5…」 |

参考自己两条路径差 6.3e-7（chunked vs recurrent）；我们的两条路径是同一段代码，所以是 0。

### 抓到两个错，都属于「形状看起来对」

1. **`beta` 忘了取 `sigmoid`。** 参考是 `beta = b.sigmoid()`，我直接把 `in_proj_b` 的
   输出当成 beta。前三个检查点（embed、input_layernorm、conv）全绿，`delta.out` 没有
   dump，第一个报红的是 `linear_attn.norm.out` —— 一个整体算错的层，要等到第 4 个检查点
   才看得见。**教训：没有 dump 的中间量，错误会从最近的下游检查点冒出来，看起来像是那个
   下游的问题。**
2. **KV 缓存用「当前长度」当步长。** `keys[head * total + position]` 在 `total` 从 9 变
   10 时把 head 1 的数据留在了旧位置，decode 与 prefill 差 0.119。改成每个 head 一个
   `Vec`，步长与容量无关。判据只能是「两种喂法逐位相同」，因为两边都「看起来合理」。

### 没有采用的方案

- **分块 delta rule（chunked UT 变换）**：参考 prefill 用的是它。它快得多，但与递推形式
  只在 6.3e-8 内一致 —— 那是另一条算术路径，会把层对拍的容差花在它身上。先用递推，分块
  留给 M9，届时的判据是 M5 的 `max|d|` 不退化。
- **给 dense 权重（bf16/f32）做缓存**：`in_proj_a/b` 每次调用都转 16k 个值，18 层 × 每
  token 约 600 KB 临时量。看起来该缓存，但它们合计 2.4M 参数，而热点是 871M 个 q4 值；
  先优化后者。

## M9：性能（2026-11-18）

```
cargo run --release --example bench -- --tokens 8
D:/Projects/Models/Qwen3.5-0.8B/.venv/Scripts/python.exe dev/bench_reference.py --tokens 81 --decode 8
```

两边都是 81 token 的 prompt、丢弃一次预热、取中位数。参考跑它的原生 float32 与默认线程数
（24 个），本移植用 16 个 worker。

### 先说测量本身：同一对命令，不同时间窗差 3 倍

这台机器上常驻着 `QQPCTray`、`PerfTune`、`Weixin` 与两个 VS Code（各占上万 CPU 秒）。
同一对命令在两个时间窗里量到的结果差到 3 倍：

| 时间窗 | 本移植 prefill | PyTorch eager prefill |
|---|---|---|
| 空载 | 0.336 s ＝ 241 tok/s | 0.323 s ＝ 250 tok/s |
| 有负载 | 0.333 s ＝ 243 tok/s | 0.422 s ＝ 192 tok/s |

参考实现用 torch 的 24 个线程、本移植用 16 个，而本移植的 worker 常驻自旋：负载对两者的
打击不一样。**所以只有同一次交替测量的两个数可以相减**，不同窗口的绝对值不能拼成一张表。
下面每一张表都取自一个窗口。

### 最大的单项不是内核，是递推里的访存顺序

`linear_attention` 的 `memory = Sᵀk` 原来写成「列在外、行在内」，读
`state[row * 128 + index]` 的步长是 128 个 float：每个输出元素都跨一条缓存行。改成行在外、
列在内（`ops::transposed_matvec`）之后，同一段算术遍历整个 `state` 是连续的。
**prefill 0.720 → 0.457 s，decode 0.031 → 0.027 s**，而每一层的检查点逐位不变（每个
`out[i]` 仍然是按行号递增求和）。

同批改的还有：因果卷积改成直接形式（历史窗口从缓存里拼成 4 个平面），于是 token 之间互相
独立、可以并行；`rms_norm_gated` 按行并行。

### 第二个大项：那一段串行链

改完访存顺序后出现一个反常读数：**prefill 完全不随 worker 数变化**（8 / 12 / 16 / 20 / 24
个 worker 都是 0.42–0.46 s），而纯乘加的探针在同期从 198 涨到 294 GMAC/s。这说明时间被
一段串行代码占着。它就在递推里：

```
for token { for head { ... } }   // 81 个 token 一个接一个
```

而 16 个 head 彼此完全独立（各有各的 `state`，一个 head 的 token 轴只依赖它自己）。把两个
循环换序、按 head 切开（`Pool::chunks_mut_pair`，状态与输出各自的步长不同所以要成对切），
并顺手把 6 层 attention 的 `(token, head)` 对也并行（它们只读缓存）：

| | 改前 | 改后 |
|---|---|---|
| prefill | 0.457 s | 0.347 s |
| decode | 0.027 s | 0.025 s |

### 线程数：8 是错的

串行链还在的时候，加 worker 没有收益，我据此把上限设成 8。串行链一去掉，同一份代码：

| worker | prefill | decode |
|---|---|---|
| 8 | 0.443 s | 0.030 s |
| 12 | 0.364 s | 0.021 s |
| **16** | **0.330 s** | **0.019 s** |
| 20 | 0.339 s | 0.025 s |
| 24 | 0.340 s | 0.041 s |

16 是拐点（8 个 P-core 加一半 E-core），之后 decode 反而退化。上限改成 16。
**这个 8 是「用一个有串行瓶颈的程序量出来的最优值」，它测的是瓶颈，不是内核。**

### 现在的位置

同一时间窗内交替测（`dev/bench_reference.py` 与 `examples/bench.rs`，各 3 次取中位数）：

| | 本移植（16 worker） | PyTorch eager（f32 / 24 线程） |
|---|---|---|
| prefill 81 token | 0.333 s ＝ 243 tok/s | 0.422 s ＝ 192 tok/s |
| decode | 0.020 s/token ＝ 50 tok/s | 0.113 s/token ＝ 8.8 tok/s |

空载窗口下参考是 0.323 s（250 tok/s），本移植同期是 0.336 s（241 tok/s）—— 两个窗口里都是
本移植在前或持平，但**不是「快很多」**：0.333 s 对应 209 GMAC/s，而纯乘加在 16 个 worker
的天花板是 256 GMAC/s，即已经把 f32 的算力用掉 82%。

### 再往下就不是工程问题了，是算术问题：W4A8 的实测代价

要「快很多」只能减少每条指令背后的乘法数。4-bit 权重乘 8-bit 激活可以用
`_mm256_maddubs_epi16` + `_mm256_madd_epi16` 两条指令做 32 个乘积（f32 形式要 4 条），
权重直接取存储的 0..15 nibble，最后减 `8 * sum(a)` 再乘两级 scale。

代价用参考模型量：给全部 237 个 `nn.Linear` 的输入按 32 一组的对称 int8 量化再还原
（`dev/simulate_w4a8.py`），端到端 logits：

| 用例 | `max|d|` | 相对 | top-1 |
|---|---|---|---|
| hello | 2.889e-01 | 1.77e-02 | 271 → 271 |
| chinese | 2.571e-01 | 1.68e-02 | 198 → 198 |
| mixed | 2.275e-01 | 1.27e-02 | 727 → 727 |

分组取 128 或整行只会更差（2.3e-2 / 3.4e-2）。也就是说 **W4A8 会把 logits 的相对差从
4.8e-6 抬到 1.5e-2，比层对拍的 1e-3 门大一个数量级**。因此它没有进这一版：它不是一次调优，
是把「与参考逐层一致」这个契约换成「答案仍然对、数值只到百分级」。要不要换是这个项目的
决定，不是实现的细节。

### 量化矩阵乘的三个内核

1. **单 token（decode）**：`quantised_dot_avx2`，把 q4 字节直接解成乘积累加，不物化任何
   行。物化的代价是每次前向写 871M 个 `f32`（3.5 GB），实测让 decode 从 0.031 s 变成
   0.078 s —— 单 token 没有第二个 token 来分摊这次展开。
2. **批量（prefill）**：`panel_rows`，一次展开 128 列（4 个量化组）到 L1，然后**一次调用
   覆盖整个 token 轴**。原来每个 token 单独调用一次 `dot`，128 个值的点积要付一次调用开销
   （序言、`vzeroupper`、指令集探测），那笔开销是工作量的三倍。
3. **行分块**：四个输出行共用同一批输入向量（`accumulate_slab_quad`），或两个行
   （`accumulate_slab_pair`）。实测 2 行 × 4 token 与 4 行 × 2 token 在模型上分别是
   0.693 / 0.720 s，取前者。

用真权重、长计时（每组 0.25 s、取 5 次最好值）量到的内核速率：`in_proj_qkv`
143–185、`gate_proj` 149–187、`down_proj` 215–250、`q_proj` 99–147 GMAC/s（区间是我的
多次运行之间的散布，不是同一次测量的置信区间）。

### 一个我自己搞错的探针

我先写了一个「纯乘加上限」探针：`for _ in 0..n { total += dot(&a, &b) }`，两个操作数在
循环里不变 —— 这整段被优化器外提了，量出的 27 / 144 / 205 GMAC/s 是假的。把两个切片
`black_box` 之后再量（同期同一台机器）：

| worker | GMAC/s | 每 worker |
|---|---|---|
| 1 | 30 | 30 |
| 8 | 198 | 25 |
| 16 | 256 | 16 |
| 24 | 294 | 12 |

单链的上限是 30 GMAC/s（一个 fma 要两次装载），8 个 worker 之前是线性叠加。
**所以那组 27 / 144 / 205 当时被我写进文档，是错的；以这张表为准。**

### prefill 与 decode 的判据变了，原因写在测试里

两条路径现在是两个内核（融合 vs 分块），求和次序不同，`max|d|` 从 0.0 变成 7.2e-6
（相对 4.0e-7）。`tests/text.rs` 的门因此从 `== 0.0` 改成相对 1e-6：它要抓的是缓存索引
或步长写错，那种缺陷会把差异推到百分之一的量级。

换内核之后 M5 的判据没有退化：24 层逐层对拍 worst rel 3.7e-6（layer 13 的
`linear_attn.norm.out`），logits top-1 仍是 3/3。

## M6：视觉塔（2026-11-18）

```
D:/Projects/Models/Qwen3.5-0.8B/.venv/Scripts/python.exe dev/dump_vision.py
cargo test --release --features reference-fixtures --test vision -- --nocapture
```

dump 用**反量化后的 q4 权重**跑参考，并把**原始 RGB 字节**与它产出的 `pixel_values`
一起存下来，所以预处理与塔各自可判。两个用例：`100x70`（会被放大到 320x224，缩放比
3.2）与 `64x64`（放大到 256x256，缩放比整数 4）。

| 读数 | `square` | `resized` |
|---|---|---|
| 预处理 `max|d|` | 5.9e-8 | 7.8e-3（正好一个灰度级） |
| 塔（喂 dump 的 patch 行）相对误差 | 1.0e-5 | 9.4e-6 |
| 端到端相对误差 | 1.1e-5 | 3.5e-3 |

塔的 1e-5 与文本层的 3.7e-6 同量级；端到端的 3.5e-3 全部来自预处理那一个灰度级，
被塔放大了（见下）。

### 预处理：四条规则，每条都是比出来的

1. **缩放比与取整**：`factor = 32`，`round(h/32)`，超上限下取整、低于下限上取整。
   第一版我按「`100x70` 只需轻微缩放」估，实测两个用例都是被**放大**到 65536 像素。
2. **核是 PIL 的 bicubic（`a = -0.5`），不是 torchvision 的（`a = -0.75`）**。四个候选
   一起量：`a = -0.75` 差 3.7e-1、`a = -0.75` 夹边界差 4.5e-1、`a = -0.5` 夹边界差
   2.9e-1、**直接调 PIL 差 2.9e-8**。所以核来自 PIL。
3. **边界抽头丢弃而不是夹到边缘**，保留的权重再归一化。这一条用「把单个像素点亮再看
   输出」量出来：`8 → 16` 的 `0.8667 / 0.2275` 只能由 `a = -0.5`、`src = (i+0.5)*in/out
   - 0.5` 给出；边界值 `1.0`（>1 被夹）与我的 1.1119 对得上。
4. **两遍各自落成字节**。这一条最难堪：只在整个流程末尾取整，饱和图上差 **14 个灰度级**；
   加上「水平遍取整并夹到 `[0,255]`」之后，我的实现在 196608 个值上与 PIL **逐位相同**
   （`stages=1` 差 18、`stages=2` 差 0）。

第 4 条之所以必须：处理器把中间结果放在 `uint8` 缓冲里，两遍之间就已经量化过一次。
这也解释了为什么 `resized` 还差一个灰度级 —— PIL 用定点系数，我这边是 `f64`；要逐位相同
就得复刻它的定点算术，而那是处理器的实现细节，不是模型规格。

### patch 顺序：点亮一个块量出来的

不看公式，每次只把一个 16×16 源块点亮，`pixel_values` 里哪几行亮就说明那一块落在哪：
`64x64` 的块 `(0,0)` 亮的是 `0..7` 与 `32..39`，与「merge 块优先、块内 `in_row` 慢于
`in_col`」的解码完全一致。这条同时钉住了每个 patch 向量内部的
`(channel, temporal, patch_row, patch_col)` 布局 —— 图像只有一帧，被复制两次填满 temporal。

### 一个必须分开量的事实

塔在这个 fixture 上把输入**放大两个数量级**（block 11 的 `max|d|` 从 4 长到 1654，用原始
bf16 权重也一样：1719 对 1654）。所以端到端差 3.5e-3 不是塔算错，而是预处理那一个灰度级
被放大了 —— 把塔单独喂 dump 的 patch 行，它就回到 1e-5。判据因此分成两条：塔看相对误差
1e-3，预处理看「不超过一个灰度级」。

### 图像 token 进文本序列

`dump_vision.py` 还会把**整条图像 prompt** 的 logits 存下来，所以下面这几项都在判据里：

| 检查项 | `square` | `resized` |
|---|---|---|
| 整条 prompt 的 logits 相对误差 | 2.5e-6 | 2.5e-4 |
| 图像之后走一步的相对误差 | 1.8e-6 | 3.0e-4 |
| top-1（prompt / 之后一步） | 1919 / 2099 | 同左 |
| 我们渲染的 prompt 与处理器的 ids | 84 个，全等 | 90 个，全等 |

三个容易写错的地方，都是量出来的：

1. **占位符要展开。** 模板里只有一个 `<|image_pad|>`，处理器把它展开成 `h*w/4` 个；我们
   用 `chat::render` 渲染同一段文本再展开，得到的 ids 与处理器**逐个相等**（84 / 90）。
2. **`mm_token_type_ids` 只有占位符是 1。** `vision_start` / `vision_end` 是文本（0）。我
   原以为整段视觉区间都是 1。
3. **图像之后的位置按 `max(h, w)/2` 推进，不是按 token 数**；生成阶段还要再加
   `最大位置 + 1 - 序列长度`。这一项错了，图像之后的每个 token 都会偏 —— 判据就是「图像之后
   走一步」的那 1.8e-6，而不是 prompt 自己的 logits（后者对位置偏移不敏感）。

### 贪心生成：前 10 个 token 与参考相同，第 11 个分岔

同一张 `64x64` 图、同一个问题，两边贪心：

```
参考: [1919, 2099, 369, 264, 2972, 3313, 1218, 50717, 5215, 332, 22687, 314, ...]
本仓: [1919, 2099, 369, 264, 2972, 3313, 1218, 50717, 5215, 332, 1892, 424, ...]
```

文本都是「This image is a **colorful geometric pattern** composed of」，第 11 个 token
开始不同。这是预处理那一个灰度级 + f32 噪声在十步贪心里的累积 —— 贪心每步都取最大者，
一旦两个候选接近就会翻。**结论只能写成「前十步一致」**，不能写成「输出相同」。

## 下一步

- **M7：MTP 头。** 容器里有 15 个 `mtp.*` 张量，模型类不加载它们，所以 dump 要自己搭。
- **M8：wgpu 后端**，与 CPU 后端逐点一致（`--features wgpu`）。
- **M9 的剩余空间与它的代价已经量清楚了。** f32 算术下模型跑在 209 GMAC/s，天花板 256，
  即 82%；要快很多只能换成 W4A8，代价是 logits 相对差 1.5e-2，见上面那张表。
