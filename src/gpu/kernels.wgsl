// Kernels for the GPU backend.
//
// The CPU backend's arithmetic is the specification: every kernel here computes
// the same expression over the same values. What differs is the reduction order
// and whether a multiply-add is fused, which WGSL implementations are free to
// choose, so the host compares reductions within a tolerance and elementwise
// results exactly.
//
// Two layouts, because a quantised weight is packed bytes and a data value is an
// `f32`, and a storage binding has one element type.
//
// The float layout offers five read operands, two read-write operands (the
// recurrent state and caches), the output, and the parameters. A kernel that
// needs fewer binds whatever it has to the same buffer; the host keeps those
// bindings distinct, because a buffer used with two different access modes in one
// dispatch is a validation error.

struct Parameters {
    rows: u32,
    columns: u32,
    groups: u32,
    group: u32,
    // Rows per dispatch row: a dispatch dimension is capped at 65535, and the
    // output head has more rows than that.
    grid_x: u32,
    eps: f32,
    // A kernel's second width: the rotary length, a head's width, or a stride.
    extra: u32,
    // How the host sized the dispatch: 0 elementwise, 1 one workgroup per row,
    // 2 one per head, 3 one workgroup in total. The shaders do not read it; it is
    // here so the host and the shader agree on the block's layout.
    kind: u32,
    // A grouped copy's outer stride and band offset.
    stride: u32,
    offset: u32,
    // Attention: how many positions the cache already holds.
    past: u32,
    // The convolution's tap count.
    taps: u32,
    // Attention: query heads and key/value heads.
    heads: u32,
    kv_heads: u32,
    // How many tokens the pass multiplies against a weight.
    tokens: u32,
    padding: u32,
}

fn byte_of(word: u32, index: u32) -> u32 {
    return (word >> (8u * index)) & 0xffu;
}

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

fn softplus(value: f32) -> f32 {
    // `log(1 + exp(x))`, written so a large x does not overflow the exponential.
    return max(value, 0.0) + log(1.0 + exp(-abs(value)));
}

// ---------------------------------------------------------------------------
// Quantised layout
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read> integers: array<u32>;
// One `f16` scale per group, two to a word, as the container stores them.
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> quant_input: array<f32>;
@group(0) @binding(3) var<storage, read_write> quant_output: array<f32>;
// The same buffer as `quant_input`, read four values at a time.
//
// A four-bit weight is half a byte per value, so a row of it is a handful of loads
// while the input that multiplies it is four bytes per value. Reading the input one
// float at a time makes the kernel load-instruction bound rather than bandwidth
// bound; a vector load quarters the count.
@group(0) @binding(4) var<storage, read> quant_input4: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> quant_parameters: Parameters;

// The four-bit value at a flat index of the table.
fn nibble_at(index: u32) -> f32 {
    let word = integers[(index / 2u) / 4u];
    let byte = byte_of(word, (index / 2u) % 4u);
    let stored = select(byte & 0x0fu, byte >> 4u, (index & 1u) == 1u);
    return f32(stored) - 8.0;
}

// The scale of group `index`, from the word that holds it.
fn scale_at(index: u32) -> f32 {
    let pair = unpack2x16float(scales[index / 2u]);
    return select(pair.x, pair.y, (index & 1u) == 1u);
}

// `value = (stored - 8) * scale` for four-bit values.
//
// One invocation per output element. Nothing is reduced, so the result is exact
// and matches the CPU path bit for bit.
@compute @workgroup_size(64)
fn dequantize_one(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = quant_parameters.rows * quant_parameters.columns;
    if index >= total {
        return;
    }
    let scale = scale_at(index / quant_parameters.group);
    quant_output[index] = nibble_at(index) * scale;
}

// The embedding rows the ids name, each dequantised as it is read.
//
// `quant_input` holds one id per row, `quant_output` receives `rows x columns`.
@compute @workgroup_size(64)
fn embed_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= quant_parameters.rows * quant_parameters.columns {
        return;
    }
    let row = index / quant_parameters.columns;
    let column = index % quant_parameters.columns;
    let table_row = u32(quant_input[row]);
    let flat = table_row * quant_parameters.columns + column;
    let scale = scale_at(table_row * quant_parameters.groups + column / quant_parameters.group);
    quant_output[index] = nibble_at(flat) * scale;
}

// Lanes that walk one weight row.
const THREADS: u32 = 32u;

// Warps in one matvec workgroup, which is also the number of weight rows it
// computes. `kernels.wgsl` has no other reader of this; the host mirrors it.
const WARPS: u32 = 4u;
// One accumulator per token of a chunk, for every lane of every warp.
var<workgroup> chunked: array<f32, WARPS * THREADS * TOKEN_CHUNK>;

// Tokens a single pass over a weight row serves.
//
// The weight is the expensive operand: a projection reads its whole row for every
// token, so a pass per token reads the model's weights `tokens` times. Eight tokens
// against one fetch halves that again over four, and eight is the common prefill.
// Sixteen was measured and is slower: the accumulators, the token offsets and the
// unpacked words together spill the registers, and the spill costs more than the
// weight reads it saves.
const TOKEN_CHUNK: u32 = 8u;

// `out[row] = sum over groups of scale * sum(q * input)`.
//
// One workgroup per `(weight row, chunk of tokens)`, and the row's groups are
// split across the group's threads. One thread per row leaves the device idle: a
// row is a few hundred bytes and the reduction is a serial walk.
//
// The token chunk is what makes this a matrix multiply rather than four matrix-vector
// products: every token of the chunk is multiplied against the same unpacked weight
// words, so the weights are fetched once.
@compute @workgroup_size(WARPS * THREADS)
fn quantised_matvec_one(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    // A dispatch dimension counts workgroups, and each workgroup here is one
    // `(WARPS weight rows, chunk of tokens)`.
    let workgroup_index = id.x / (WARPS * THREADS) + id.y * quant_parameters.grid_x;
    if workgroup_index >= quant_parameters.rows {
        return;
    }
    let warp = lane.x / THREADS;
    let lane_in_warp = lane.x % THREADS;
    let columns = quant_parameters.columns;
    let group = quant_parameters.group;
    let groups = quant_parameters.groups;
    let group_bytes = group / 2u;
    let weight_rows = quant_parameters.extra;
    let row_groups = (weight_rows + WARPS - 1u) / WARPS;
    let weight_row = (workgroup_index % row_groups) * WARPS + warp;
    // A trailing partial group has warps with no row. They still walk the loops, so
    // they reach every barrier with the rest, and they store nothing.
    let live = weight_row < weight_rows;
    let row = min(weight_row, weight_rows - 1u);
    let token_base = (workgroup_index / row_groups) * TOKEN_CHUNK;
    let tokens = quant_parameters.tokens;
    let row_word = row * (columns / 2u) / 4u;
    // The scales of a weight are indexed by its own row, not by the output row.
    let scale_row = row * groups;
    let first_group = lane_in_warp;

    // A chunk shorter than `TOKEN_CHUNK` repeats the last token rather than reading
    // past the input; the repeat is accumulated but never stored.
    let last_token = tokens - 1u;
    // How many of the chunk's slots are real. A decode step fills one, and the
    // guard below is uniform across the workgroup, so the other seven cost a branch
    // rather than seven times the arithmetic.
    let remaining = min(TOKEN_CHUNK, tokens - token_base);
    var row_base = array<u32, TOKEN_CHUNK>();
    for (var t = 0u; t < TOKEN_CHUNK; t = t + 1u) {
        row_base[t] = min(token_base + t, last_token) * columns;
    }
    // The accumulator is an array with a constant trip count, so the compiler
    // unrolls it and the values stay in registers.
    // One `vec4` per token: the eight products of a word become two vector
    // multiply-adds rather than eight scalar ones, which is four arithmetic
    // operations per instruction.
    var acc: array<vec4<f32>, TOKEN_CHUNK>;
    for (var t = 0u; t < TOKEN_CHUNK; t = t + 1u) {
        acc[t] = vec4<f32>(0.0);
    }

    // Consecutive lanes take consecutive groups, so one instruction covers a
    // contiguous run of the row. Handing each lane a contiguous block of several
    // groups instead was measured: it moves the same bytes with fewer lanes pulling
    // at once, and it ran the same speed, so the simple stride stays.
    for (var slot = first_group; slot < groups; slot = slot + THREADS) {
        let base = row_word + slot * group_bytes / 4u;
        let scale = scale_at(scale_row + slot);
        let value_base = slot * group;
        for (var index = 0u; index < group_bytes / 4u; index = index + 1u) {
            let word = integers[base + index];
            let at = value_base + index * 8u;
            // Unpacked once, used by every token in the chunk, and scaled here so
            // the inner loop over tokens is nothing but multiply-add.
            //
            // Scalar per lane on purpose: the same unpack written as vector shifts
            // and masks over the word measured twelve percent slower, which says
            // the compiler already schedules the scalar form and the `vec4` form
            // costs registers.
            let low = vec4<f32>(
                f32(i32(byte_of(word, 0u) & 0x0fu) - 8),
                f32(i32(byte_of(word, 0u) >> 4u) - 8),
                f32(i32(byte_of(word, 1u) & 0x0fu) - 8),
                f32(i32(byte_of(word, 1u) >> 4u) - 8),
            ) * scale;
            let high = vec4<f32>(
                f32(i32(byte_of(word, 2u) & 0x0fu) - 8),
                f32(i32(byte_of(word, 2u) >> 4u) - 8),
                f32(i32(byte_of(word, 3u) & 0x0fu) - 8),
                f32(i32(byte_of(word, 3u) >> 4u) - 8),
            ) * scale;
            for (var t = 0u; t < TOKEN_CHUNK; t = t + 1u) {
                if (t < remaining) {
                    let p = (row_base[t] + at) / 4u;
                    acc[t] = acc[t] + low * quant_input4[p] + high * quant_input4[p + 1u];
                }
            }
        }
    }
    let lane_base = warp * THREADS * TOKEN_CHUNK + lane_in_warp * TOKEN_CHUNK;
    for (var t = 0u; t < TOKEN_CHUNK; t = t + 1u) {
        let value = acc[t];
        chunked[lane_base + t] = value.x + value.y + value.z + value.w;
    }
    workgroupBarrier();

    // One pass of the tree reduces every token of every warp at once.
    var width = THREADS / 2u;
    loop {
        if width == 0u {
            break;
        }
        if lane_in_warp < width {
            let here = warp * THREADS * TOKEN_CHUNK + lane_in_warp * TOKEN_CHUNK;
            let there = here + width * TOKEN_CHUNK;
            for (var t = 0u; t < TOKEN_CHUNK; t = t + 1u) {
                chunked[here + t] = chunked[here + t] + chunked[there + t];
            }
        }
        workgroupBarrier();
        width = width / 2u;
    }
    if lane_in_warp == 0u && live {
        let destination = token_base * weight_rows + weight_row;
        let base = warp * THREADS * TOKEN_CHUNK;
        for (var t = 0u; t < remaining; t = t + 1u) {
            quant_output[destination + t * weight_rows] = chunked[base + t];
        }
    }
}

// `out[row] = scale * sum(q * input)` for one token: the decode step's shape.
//
// The kernel above carries eight tokens of accumulators because a prefill has
// eight; a decode step fills one and the other seven are dead registers the
// compiler still has to keep. This is the same arithmetic with one accumulator, so
// the sums are identical — the same lane walks the same groups in the same order —
// and the device has room for twice as many workgroups in flight.
@compute @workgroup_size(WARPS * THREADS)
fn quantised_matvec_single(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    let workgroup_index = id.x / (WARPS * THREADS) + id.y * quant_parameters.grid_x;
    if workgroup_index >= quant_parameters.rows {
        return;
    }
    let warp = lane.x / THREADS;
    let lane_in_warp = lane.x % THREADS;
    let columns = quant_parameters.columns;
    let group = quant_parameters.group;
    let groups = quant_parameters.groups;
    let group_bytes = group / 2u;
    let weight_rows = quant_parameters.extra;
    let row_groups = (weight_rows + WARPS - 1u) / WARPS;
    let weight_row = (workgroup_index % row_groups) * WARPS + warp;
    let live = weight_row < weight_rows;
    let row = min(weight_row, weight_rows - 1u);
    let row_word = row * (columns / 2u) / 4u;
    let scale_row = row * groups;

    var acc = vec4<f32>(0.0);
    for (var slot = lane_in_warp; slot < groups; slot = slot + THREADS) {
        let base = row_word + slot * group_bytes / 4u;
        let scale = scale_at(scale_row + slot);
        let value_base = slot * group;
        for (var index = 0u; index < group_bytes / 4u; index = index + 1u) {
            let word = integers[base + index];
            let at = value_base + index * 8u;
            let low = vec4<f32>(
                f32(i32(byte_of(word, 0u) & 0x0fu) - 8),
                f32(i32(byte_of(word, 0u) >> 4u) - 8),
                f32(i32(byte_of(word, 1u) & 0x0fu) - 8),
                f32(i32(byte_of(word, 1u) >> 4u) - 8),
            ) * scale;
            let high = vec4<f32>(
                f32(i32(byte_of(word, 2u) & 0x0fu) - 8),
                f32(i32(byte_of(word, 2u) >> 4u) - 8),
                f32(i32(byte_of(word, 3u) & 0x0fu) - 8),
                f32(i32(byte_of(word, 3u) >> 4u) - 8),
            ) * scale;
            let p = at / 4u;
            acc = acc + low * quant_input4[p] + high * quant_input4[p + 1u];
        }
    }
    let base = warp * THREADS + lane_in_warp;
    chunked[base] = acc.x + acc.y + acc.z + acc.w;
    workgroupBarrier();

    var width = THREADS / 2u;
    loop {
        if width == 0u {
            break;
        }
        if lane_in_warp < width {
            let here = warp * THREADS + lane_in_warp;
            chunked[here] = chunked[here] + chunked[here + width];
        }
        workgroupBarrier();
        width = width / 2u;
    }
    if lane_in_warp == 0u && live {
        quant_output[weight_row] = chunked[warp * THREADS];
    }
}

// ---------------------------------------------------------------------------
// Float layout
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> d: array<f32>;
@group(0) @binding(4) var<storage, read_write> e: array<f32>;
@group(0) @binding(5) var<storage, read> f: array<f32>;
@group(0) @binding(6) var<storage, read> g: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> parameters: Parameters;

// `out[row] = a[row % extra] · b[row / extra]`, a dense weight.
//
// The container keeps the two small projections of a linear-attention layer
// (`in_proj_a`, `in_proj_b`) dense, so they arrive as `f32` and need their own
// kernel: the quantised one reads packed bytes and a scale per group.
//
// Same one-workgroup-per-row shape as the quantised kernel, for the same reason.
@compute @workgroup_size(32)
fn dense_matvec(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    let row = id.x / 32u + id.y * parameters.grid_x;
    if row >= parameters.rows {
        return;
    }
    let columns = parameters.columns;
    let weight_rows = parameters.extra;
    let weight_row = row % weight_rows;
    let input_base = (row / weight_rows) * columns;
    let base = weight_row * columns;
    let per_thread = (columns + 31u) / 32u;
    let first = lane.x * per_thread;
    let last = min(first + per_thread, columns);
    var total = 0.0;
    for (var index = first; index < last; index = index + 1u) {
        total = total + a[base + index] * b[input_base + index];
    }
    row_sums[lane.x] = total;
    reduce_sums(lane.x);
    if lane.x == 0u {
        out[row] = row_sums[0];
    }
}

var<workgroup> row_sums: array<f32, 32>;
// One partial per lane for each of the four warps a score workgroup has.
var<workgroup> score_sums: array<f32, 128>;

// Threads a row-normalising kernel puts on one row.
//
// A decode step hands these kernels a single row, so the workgroup is the whole
// of the parallelism there is: at 32 threads one warp walked a thousand values in
// a serial loop, and the dispatch cost more than the arithmetic. At 128 the row is
// four warps, each lane reading with a stride of the workgroup, which is also the
// coalesced order.
const ROW_THREADS: u32 = 128u;
var<workgroup> wide_sums: array<f32, 128>;

// Reduce this workgroup's `ROW_THREADS` partials, leaving the sum in `wide_sums[0]`.
fn reduce_wide(lane: u32) {
    workgroupBarrier();
    var step = ROW_THREADS / 2u;
    loop {
        if step == 0u {
            break;
        }
        if lane < step {
            wide_sums[lane] = wide_sums[lane] + wide_sums[lane + step];
        }
        workgroupBarrier();
        step = step / 2u;
    }
}

// The same for a maximum, which softmax needs.
fn reduce_wide_max(lane: u32) {
    workgroupBarrier();
    var step = ROW_THREADS / 2u;
    loop {
        if step == 0u {
            break;
        }
        if lane < step {
            wide_sums[lane] = max(wide_sums[lane], wide_sums[lane + step]);
        }
        workgroupBarrier();
        step = step / 2u;
    }
}

// Reduce this workgroup's lane sums and leave the total in `row_sums[0]`.
fn reduce_sums(lane: u32) {
    workgroupBarrier();
    var step = 16u;
    loop {
        if step == 0u {
            break;
        }
        if lane < step {
            row_sums[lane] = row_sums[lane] + row_sums[lane + step];
        }
        workgroupBarrier();
        step = step / 2u;
    }
}

// `out[i] = a[i] * f[0]`.
@compute @workgroup_size(64)
fn scale_all(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= parameters.rows * parameters.columns {
        return;
    }
    out[index] = a[index] * f[0];
}

// `out[i] = a[i] + b[i]`.
@compute @workgroup_size(64)
fn add_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= parameters.rows * parameters.columns {
        return;
    }
    out[index] = a[index] + b[index];
}

// `out[i] = a[i] * sigmoid(b[i])`.
@compute @workgroup_size(64)
fn gate_mul(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= parameters.rows * parameters.columns {
        return;
    }
    out[index] = a[index] * sigmoid(b[index]);
}

// `out[i] = silu(a[i]) * b[i]`.
@compute @workgroup_size(64)
fn silu_mul_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= parameters.rows * parameters.columns {
        return;
    }
    let value = a[index];
    out[index] = (value / (1.0 + exp(-value))) * b[index];
}

// Split every `2 * extra` row of `a` into its first and second halves.
@compute @workgroup_size(64)
fn split_bands(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let width = parameters.extra;
    if index >= parameters.rows * width {
        return;
    }
    let row = index / width;
    let column = index % width;
    d[index] = a[row * 2u * width + column];
    out[index] = a[row * 2u * width + width + column];
}

// One band out of a `[row / group][groups of width][band]` block.
//
// `stride` is the outer stride and `offset` the band's first element, so
// `out[row][i] = a[(row / group) * stride + offset + (row % group) * columns + i]`.
@compute @workgroup_size(64)
fn gather_bands(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let width = parameters.columns;
    if index >= parameters.rows * width {
        return;
    }
    let row = index / width;
    let column = index % width;
    let start = (row / parameters.extra) * parameters.stride
        + parameters.offset
        + (row % parameters.extra) * width;
    out[index] = a[start + column];
}

// `out = a / rms(a) * (1 + f)` over each row of `columns`.
@compute @workgroup_size(128)
fn rms_norm_rows(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    // One workgroup owns one row, so the row comes from the group index and every
    // thread of the group works on the same row. Every thread has to reach the
    // barriers below, so the guard tests the row before any of them.
    let row = id.x / ROW_THREADS;
    if row >= parameters.rows {
        return;
    }
    let width = parameters.columns;
    let base = row * width;
    var sum = 0.0;
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        let value = a[base + index];
        sum = sum + value * value;
    }
    wide_sums[lane.x] = sum;
    reduce_wide(lane.x);
    let inverse = 1.0 / sqrt(wide_sums[0] / f32(width) + parameters.eps);
    workgroupBarrier();
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        out[base + index] = a[base + index] * inverse * (1.0 + f[index]);
    }
}

// `out = f * a / rms(a) * silu(c)` over each row of `columns`.
@compute @workgroup_size(128)
fn gated_rms_norm_rows(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    let row = id.x / ROW_THREADS;
    if row >= parameters.rows {
        return;
    }
    let width = parameters.columns;
    let base = row * width;
    var sum = 0.0;
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        let value = a[base + index];
        sum = sum + value * value;
    }
    wide_sums[lane.x] = sum;
    reduce_wide(lane.x);
    let inverse = 1.0 / sqrt(wide_sums[0] / f32(width) + parameters.eps);
    workgroupBarrier();
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        let gate = c[base + index];
        out[base + index] = f[index] * a[base + index] * inverse * (gate / (1.0 + exp(-gate)));
    }
}

// `out = (a - mean) / sqrt(variance + eps) * (1 + f) + g` over each row.
@compute @workgroup_size(128)
fn layer_norm_rows(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    let row = id.x / ROW_THREADS;
    if row >= parameters.rows {
        return;
    }
    let width = parameters.columns;
    let base = row * width;

    var sum = 0.0;
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        sum = sum + a[base + index];
    }
    wide_sums[lane.x] = sum;
    reduce_wide(lane.x);
    let mean = wide_sums[0] / f32(width);
    workgroupBarrier();

    var spread = 0.0;
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        let centred = a[base + index] - mean;
        spread = spread + centred * centred;
    }
    wide_sums[lane.x] = spread;
    reduce_wide(lane.x);
    let inverse = 1.0 / sqrt(wide_sums[0] / f32(width) + parameters.eps);
    workgroupBarrier();
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        out[base + index] = (a[base + index] - mean) * inverse * (1.0 + f[index]) + g[index];
    }
}

// `out[row] = argmax(a[row])`, one value per row.
//
// A token index is below 2^24, so it is exact as `f32`.
@compute @workgroup_size(32)
fn argmax_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x / 32u;
    if row >= parameters.rows {
        return;
    }
    let base = row * parameters.columns;
    var best = 0u;
    var value = a[base];
    for (var index = 1u; index < parameters.columns; index = index + 1u) {
        let candidate = a[base + index];
        if candidate > value {
            value = candidate;
            best = index;
        }
    }
    out[row] = f32(best);
}

// `out = exp(a - max) / sum` over each row.
@compute @workgroup_size(128)
fn softmax_rows(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    let row = id.x / ROW_THREADS;
    if row >= parameters.rows {
        return;
    }
    let width = parameters.columns;
    let base = row * width;

    var largest = -3.0e38;
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        largest = max(largest, a[base + index]);
    }
    wide_sums[lane.x] = largest;
    reduce_wide_max(lane.x);
    let top = wide_sums[0];
    workgroupBarrier();

    var total = 0.0;
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        total = total + exp(a[base + index] - top);
    }
    wide_sums[lane.x] = total;
    reduce_wide(lane.x);
    let inverse = 1.0 / wide_sums[0];
    workgroupBarrier();
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        out[base + index] = exp(a[base + index] - top) * inverse;
    }
}

// `out = a / sqrt(sum(a^2) + eps) * f[0]` over each row of `columns`.
@compute @workgroup_size(128)
fn l2_normalize_rows(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    let row = id.x / ROW_THREADS;
    if row >= parameters.rows {
        return;
    }
    let width = parameters.columns;
    let base = row * width;
    var sum = 0.0;
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        let value = a[base + index];
        sum = sum + value * value;
    }
    wide_sums[lane.x] = sum;
    reduce_wide(lane.x);
    let factor = f[0] / sqrt(wide_sums[0] + parameters.eps);
    workgroupBarrier();
    for (var index = lane.x; index < width; index = index + ROW_THREADS) {
        out[base + index] = a[base + index] * factor;
    }
}

// `rotate_half` with the pairing `(i, i + half)` over the first `extra` values of
// each row. `f` and `g` hold the angles, `extra / 2` per row.
@compute @workgroup_size(64)
fn rope_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = parameters.rows * parameters.columns;
    if index >= total {
        return;
    }
    let row = index / parameters.columns;
    let column = index % parameters.columns;
    let half = parameters.extra / 2u;
    if column >= parameters.extra {
        // Outside the rotated band the value passes through.
        out[index] = a[index];
        return;
    }
    let pair = column % half;
    let base = row * parameters.columns + pair;
    let angle = row * half + pair;
    let cosine = f[angle];
    let sine = g[angle];
    let first = a[base];
    let second = a[base + half];
    if column < half {
        out[index] = first * cosine - second * sine;
    } else {
        out[index] = second * cosine + first * sine;
    }
}

// `out[0] = sum over a[..columns]`.
@compute @workgroup_size(32)
fn sum_all(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    let length = parameters.columns;
    let per_thread = (length + 31u) / 32u;
    let first = lane.x * per_thread;
    let last = min(first + per_thread, length);
    var sum = 0.0;
    for (var index = first; index < last; index = index + 1u) {
        sum = sum + a[index];
    }
    row_sums[lane.x] = sum;
    reduce_sums(lane.x);
    if lane.x == 0u {
        out[0] = row_sums[0];
    }
}

// The causal depthwise convolution of a linear-attention layer.
//
// One invocation per `(token, channel)`: the window of token `t` reaches back
// `taps - 1` places, so the first tokens of a sequence read the carried window.
// The window is `d`, laid out `[channel][tap]`, oldest first, and this kernel does
// not touch it: the pass emits `convolve_carry` afterwards, so no invocation reads
// a window another one has already moved.
@compute @workgroup_size(64)
fn convolve_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let channels = parameters.columns;
    let tokens = parameters.rows;
    let taps = parameters.taps;
    if index >= tokens * channels {
        return;
    }
    let token = index / channels;
    let channel = index % channels;
    var sum = 0.0;
    for (var tap = 0u; tap < taps; tap = tap + 1u) {
        let back = i32(taps) - 1 - i32(tap);
        var source = 0.0;
        if i32(token) >= back {
            source = a[(u32(i32(token) - back)) * channels + channel];
        } else {
            source = d[channel * taps + u32(i32(token) + i32(tap) + 1)];
        }
        sum = sum + source * f[channel * taps + tap];
    }
    out[index] = sum / (1.0 + exp(-sum));
}

// Carry the convolution window forward: the last `taps` inputs of every channel.
//
// One invocation per channel, so no two invocations write the same window slot.
@compute @workgroup_size(64)
fn convolve_carry(@builtin(global_invocation_id) id: vec3<u32>) {
    let channel = id.x;
    let channels = parameters.columns;
    let tokens = parameters.rows;
    let taps = parameters.taps;
    if channel >= channels {
        return;
    }
    for (var slot = 0u; slot < taps; slot = slot + 1u) {
        if tokens >= taps {
            d[channel * taps + slot] = a[(tokens - taps + slot) * channels + channel];
        } else {
            let held = taps - tokens;
            if slot < held {
                d[channel * taps + slot] = d[channel * taps + tokens + slot];
            } else {
                d[channel * taps + slot] = a[(slot - held) * channels + channel];
            }
        }
    }
}

// The beta and decay of a linear-attention layer.
//
// `a` holds the raw `b` increments, `f` the `A_log` vector and `g` the `dt_bias`
// vector. One invocation per `(token, head)`.
@compute @workgroup_size(64)
fn gate_decay(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let rows = parameters.rows;
    if index >= rows {
        return;
    }
    let heads = parameters.extra;
    let head = index % heads;
    out[index] = sigmoid(a[index]);
    d[index] = exp(-(exp(f[head]) * softplus(b[index] + g[head])));
}

// Append the fresh keys and values to their caches.
//
// `a` holds the token's keys and `b` its values, `[token][head][width]`; `d` and `e`
// are the caches at `[past + token][head][width]`. Both caches are writable operands,
// so one dispatch fills both.
@compute @workgroup_size(64)
fn kv_append(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let width = parameters.columns;
    let heads = parameters.extra;
    let tokens = parameters.rows;
    if index >= tokens * heads * width {
        return;
    }
    let token = index / (heads * width);
    let rest = index % (heads * width);
    let head = rest / width;
    let column = rest % width;
    let destination = ((parameters.past + token) * heads + head) * width + column;
    d[destination] = a[index];
    e[destination] = b[index];
}

// Causal attention over a cached prefix, in three passes.
//
// `a` is the query, `d` the keys and `e` the values, both `[position][kv head][width]`.
// The query heads are grouped over the key/value heads, four to one.
//
// A score is a reduction over the head's width; the output is `width` independent
// sums over the positions, and only the first of those needs the lanes to talk to
// each other. Rather than make both pay for it:
//
// - `attention_scores` gives one warp to each score and four scores to a workgroup.
//   The lanes cover the width, so the key row is read coalesced — one thread per
//   score instead would have every lane of a warp walking a different row, which is
//   the same bytes fetched as thirty-two separate transactions.
// - `attention_softmax` normalises one row per thread: a handful of values on the
//   prefill, and still only the row's length on a long cache.
// - `attention_mix` gives one thread to every output element, where there is
//   nothing to reduce.

// Scores grouped in one workgroup, one per warp.
//
// The group's threads are what makes each dot coalesced. Four warps rather than
// eight: the reduction at the end of a dot is five workgroup barriers, and eight
// scores per group measured no better than four, so the wider group only costs
// occupancy.
const SCORES_PER_GROUP: u32 = 4u;
const SCORE_THREADS: u32 = 32u * SCORES_PER_GROUP;
var<workgroup> score_partials: array<f32, SCORE_THREADS>;

// `e` receives one row of scores per `(token, head)`, `past + tokens` long.
@compute @workgroup_size(SCORE_THREADS)
fn attention_scores(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    // A dispatch dimension counts workgroups, and a workgroup here is four scores of
    // one row.
    let group = id.x / SCORE_THREADS + id.y * parameters.grid_x;
    if group >= parameters.rows {
        return;
    }
    let warp = lane.x / 32u;
    let lane_in_warp = lane.x % 32u;
    let width = parameters.columns;
    let stride = parameters.stride;
    let row = group / parameters.groups;
    let position = (group % parameters.groups) * SCORES_PER_GROUP + warp;
    let token = row / parameters.heads;
    let head = row % parameters.heads;
    // The tail of a row is past the token's causal prefix. Every lane of the warp
    // takes this branch together, so the barriers below are not split.
    if position >= parameters.past + token + 1u {
        return;
    }
    let kv_head = head / (parameters.heads / parameters.kv_heads);
    let query = row * width;
    let other = (position * parameters.kv_heads + kv_head) * width;

    var sum = 0.0;
    for (var column = lane_in_warp; column < width; column = column + 32u) {
        sum = sum + a[query + column] * d[other + column];
    }
    // One warp's worth of the shared array, so the group's scores reduce side by side.
    let base = warp * 32u;
    score_partials[base + lane_in_warp] = sum;
    workgroupBarrier();
    var width_step = 16u;
    loop {
        if width_step == 0u {
            break;
        }
        if lane_in_warp < width_step {
            score_partials[base + lane_in_warp] = score_partials[base + lane_in_warp]
                + score_partials[base + lane_in_warp + width_step];
        }
        workgroupBarrier();
        width_step = width_step / 2u;
    }
    if lane_in_warp == 0u {
        e[row * stride + position] = score_partials[base] / sqrt(f32(width));
    }
}

// Turn each row of scores into weights: subtract the row's largest, exponentiate,
// divide by the total.
@compute @workgroup_size(64)
fn attention_softmax(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= parameters.rows {
        return;
    }
    let stride = parameters.stride;
    let visible = parameters.past + index / parameters.heads + 1u;
    let row = index * stride;

    var largest = -3.0e38;
    for (var position = 0u; position < visible; position = position + 1u) {
        largest = max(largest, e[row + position]);
    }
    var total = 0.0;
    for (var position = 0u; position < visible; position = position + 1u) {
        let value = exp(e[row + position] - largest);
        e[row + position] = value;
        total = total + value;
    }
    let inverse = 1.0 / total;
    for (var position = 0u; position < visible; position = position + 1u) {
        e[row + position] = e[row + position] * inverse;
    }
}

// `out[token][head][column] = sum over positions of weights * value`.
//
// Every column of a head is a sum over the visible positions, and at a long
// context that sum is the whole cost: one thread per output element walks the
// cache alone, so eight heads and two hundred and fifty-six columns are two
// thousand threads on a device that wants a hundred thousand. A workgroup owns one
// row and thirty-two columns, and its 256 threads split the positions eight ways;
// the partials are combined in shared memory.
const MIX_LANES: u32 = 32u;
const MIX_SPLIT: u32 = 8u;
const MIX_THREADS: u32 = MIX_LANES * MIX_SPLIT;

var<workgroup> mix_partials: array<f32, 256>;

@compute @workgroup_size(MIX_THREADS)
fn attention_mix(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    let width = parameters.columns;
    let heads = parameters.heads;
    let kv_heads = parameters.kv_heads;
    let blocks = parameters.groups;
    let workgroup_index = id.x / MIX_THREADS + id.y * parameters.grid_x;
    if workgroup_index >= parameters.rows * blocks {
        return;
    }
    let row = workgroup_index / blocks;
    let block = workgroup_index % blocks;
    let position_chunk = lane.x / MIX_LANES;
    let column_lane = lane.x % MIX_LANES;
    let column = block * MIX_LANES + column_lane;
    // A trailing block can have lanes past the last column: they reach the
    // barrier, contribute a zero partial and store nothing.
    let live = column < width;
    let token = row / heads;
    let head = row % heads;
    let visible = parameters.past + token + 1u;
    let kv_head = head / (heads / kv_heads);
    let row_weights = row * parameters.stride;
    let column_at = min(column, width - 1u);
    let per = (visible + MIX_SPLIT - 1u) / MIX_SPLIT;
    let first = position_chunk * per;
    let last = min(first + per, visible);

    var sum = 0.0;
    if live {
        for (var position = first; position < last; position = position + 1u) {
            let other = (position * kv_heads + kv_head) * width + column_at;
            sum = sum + f[row_weights + position] * e[other];
        }
    }
    mix_partials[lane.x] = sum;
    workgroupBarrier();
    if position_chunk == 0u && live {
        var total = mix_partials[column_lane];
        for (var part = 1u; part < MIX_SPLIT; part = part + 1u) {
            total = total + mix_partials[part * MIX_LANES + column_lane];
        }
        out[row * width + column] = total;
    }
}

// The Gated DeltaNet recurrence.
//
// `a`, `b`, `c` are the query, key and value bands, one row per `(token, head)`;
// `f` and `g` hold one beta and one decay per pair; `d` is the carried state,
// `heads x key_width x value_width`, read and written in place.
//
// Every step of the recurrence touches one value column of the state and nothing
// else:
//
//   memory = sum_k state[k][v] * key[k]
//   delta  = (value[v] - memory) * beta
//   state[k][v] += key[k] * delta       for every k
//   out[v] = sum_k state[k][v] * query[k]
//
// so a thread that owns a column needs neither a barrier nor the other lanes. That
// matters more than it looks: the earlier shape gave one workgroup per head, which
// is sixteen workgroups for the whole layer, each of them serialised behind a
// barrier per token.
//
// One thread per column still leaves the device idle — sixteen workgroups of 128
// threads for work that is a megabyte of state per layer — and each thread walks
// the whole key axis twice. Here a workgroup owns one head and 32 value columns,
// and its 256 threads split the key axis eight ways: sixty-four workgroups per
// layer, sixteen keys per thread and pass.
//
// A column's two sums run over the key axis, so the split needs the partials
// combined before the next step of that column: one workgroup-wide reduction each,
// with the delta passed back through shared memory.
const DELTA_VALUE_LANES: u32 = 32u;
const DELTA_KEY_SPLIT: u32 = 8u;
const DELTA_THREADS: u32 = DELTA_VALUE_LANES * DELTA_KEY_SPLIT;

var<workgroup> delta_partials: array<f32, 256>;
var<workgroup> delta_delta: array<f32, 32>;

@compute @workgroup_size(256)
fn delta_rule(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
) {
    let heads = parameters.rows;
    let key_width = parameters.groups;
    let value_width = parameters.extra;
    let tokens = parameters.columns;
    let blocks = (value_width + DELTA_VALUE_LANES - 1u) / DELTA_VALUE_LANES;
    let workgroup_index = id.x / DELTA_THREADS + id.y * parameters.grid_x;
    if workgroup_index >= heads * blocks {
        return;
    }
    let head = workgroup_index / blocks;
    let block = workgroup_index % blocks;
    let key_chunk = lane.x / DELTA_VALUE_LANES;
    let value_lane = lane.x % DELTA_VALUE_LANES;
    let value = block * DELTA_VALUE_LANES + value_lane;
    // A trailing block can have lanes past the last column. They reach every
    // barrier and contribute zero partials, but they never touch the state.
    let live = value < value_width;
    let column = min(value, value_width - 1u);
    let at = head * key_width * value_width + column;
    let chunk = (key_width + DELTA_KEY_SPLIT - 1u) / DELTA_KEY_SPLIT;
    let first_key = key_chunk * chunk;
    let last_key = min(first_key + chunk, key_width);

    for (var token = 0u; token < tokens; token = token + 1u) {
        let pair = token * heads + head;
        let decay = g[pair];
        let beta = f[pair];
        let key_row = pair * key_width;

        // Decay the column while reducing it against the key.
        var memory = 0.0;
        if live {
            for (var key = first_key; key < last_key; key = key + 1u) {
                let slot = at + key * value_width;
                let carried = d[slot] * decay;
                d[slot] = carried;
                memory = memory + carried * b[key_row + key];
            }
        }
        delta_partials[lane.x] = memory;
        workgroupBarrier();
        if key_chunk == 0u {
            var total = delta_partials[value_lane];
            for (var part = 1u; part < DELTA_KEY_SPLIT; part = part + 1u) {
                total = total + delta_partials[part * DELTA_VALUE_LANES + value_lane];
            }
            delta_delta[value_lane] = (c[pair * value_width + column] - total) * beta;
        }
        workgroupBarrier();
        let delta = delta_delta[value_lane];

        // Apply the rank-one update and project the result onto the query.
        var projected = 0.0;
        if live {
            for (var key = first_key; key < last_key; key = key + 1u) {
                let slot = at + key * value_width;
                let updated = d[slot] + b[key_row + key] * delta;
                d[slot] = updated;
                projected = projected + updated * a[key_row + key];
            }
        }
        // The reduction of the previous step read this array.
        workgroupBarrier();
        delta_partials[lane.x] = projected;
        workgroupBarrier();
        if key_chunk == 0u && live {
            var total = delta_partials[value_lane];
            for (var part = 1u; part < DELTA_KEY_SPLIT; part = part + 1u) {
                total = total + delta_partials[part * DELTA_VALUE_LANES + value_lane];
            }
            out[pair * value_width + value] = total;
        }
        // Before the next token reuses the array.
        workgroupBarrier();
    }
}
