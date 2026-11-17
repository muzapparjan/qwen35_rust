//! Checks that stand on their own: no fixtures, no reference implementation.
//!
//! The embedded container is checked here rather than in a fixture test: the
//! manifest is generated from the container's own header, so these tests are the
//! place where "the file and the code describe the same model" is asserted.

use qwen35::weights::{self, manifest};

fn spec(name: &str) -> &'static manifest::TensorSpec {
    weights::spec(name).unwrap_or_else(|| panic!("the container has no tensor {name:?}"))
}

fn shape(name: &str) -> &'static [usize] {
    spec(name).shape
}

fn expect_shape(name: &str, expected: &[usize]) {
    assert_eq!(shape(name), expected, "{name}");
}

fn expect_kind(name: &str, expected: &str) {
    assert_eq!(spec(name).kind, expected, "{name}");
}

#[test]
fn the_embedded_file_is_a_container() {
    assert_eq!(&weights::blob()[..4], b"QWQT");
    assert_eq!(weights::blob().len(), manifest::FILE_LEN);
    assert_eq!(manifest::QWQT_VERSION, 1);
}

#[test]
fn the_container_holds_every_parameter_of_the_model() {
    assert_eq!(manifest::TENSOR_COUNT, 488);
    assert_eq!(manifest::MODEL_TENSOR_COUNT, 473);
    assert_eq!(manifest::PARAM_COUNT, 873_438_784);
    assert_eq!(manifest::MODEL_PARAM_COUNT, 852_985_920);
    assert_eq!(
        manifest::TENSOR_COUNT - manifest::MODEL_TENSOR_COUNT,
        15,
        "the multi-token-prediction head is the only thing the model class skips"
    );
}

#[test]
fn the_layers_are_partitioned_by_type() {
    assert_eq!(manifest::NUM_HIDDEN_LAYERS, 24);
    assert_eq!(manifest::LINEAR_LAYERS.len(), 18);
    assert_eq!(manifest::FULL_LAYERS.len(), 6);
    for (index, layer) in manifest::LINEAR_LAYERS.iter().enumerate() {
        assert_eq!(*layer / 4, index / 3, "linear layer {layer} is out of the 3:1 pattern");
        assert_eq!(layer % 4, index % 3);
    }
    for (index, layer) in manifest::FULL_LAYERS.iter().enumerate() {
        assert_eq!(*layer, 3 + 4 * index, "full attention every fourth layer");
    }
}

#[test]
fn architecture_constants_agree_with_each_other() {
    assert_eq!(manifest::HEAD_DIM * manifest::NUM_ATTENTION_HEADS, 2048);
    assert_eq!(manifest::HEAD_DIM * manifest::NUM_KEY_VALUE_HEADS, 512);
    assert_eq!(manifest::NUM_ATTENTION_HEADS / manifest::NUM_KEY_VALUE_HEADS, 4);
    assert_eq!(manifest::ROTARY_DIM, 64);
    assert_eq!(manifest::ROTARY_DIM, 2 * manifest::MROPE_SECTION.iter().sum::<usize>());
    assert_eq!(manifest::LINEAR_KEY_DIM, 2048);
    assert_eq!(manifest::LINEAR_CONV_DIM, 6144);
    assert_eq!(manifest::LINEAR_CONV_KERNEL, 4);
    assert_eq!(manifest::LINEAR_VALUE_HEAD_DIM, 128);
    assert_eq!(manifest::VISION_DEPTH, 12);
    assert_eq!(manifest::VISION_HEAD_DIM, 64);
    assert_eq!(manifest::VISION_GRID_SIDE, 48);
    assert_eq!(manifest::VISION_MERGER_HIDDEN, 3072);
}

#[test]
fn every_text_layer_has_exactly_its_expected_tensors() {
    let hidden = manifest::HIDDEN_SIZE;
    let intermediate = manifest::INTERMEDIATE_SIZE;
    for layer in 0..manifest::NUM_HIDDEN_LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        expect_shape(&format!("{prefix}.input_layernorm.weight"), &[hidden]);
        expect_kind(&format!("{prefix}.input_layernorm.weight"), "bf16");
        expect_shape(&format!("{prefix}.post_attention_layernorm.weight"), &[hidden]);
        expect_shape(&format!("{prefix}.mlp.gate_proj.weight"), &[intermediate, hidden]);
        expect_shape(&format!("{prefix}.mlp.up_proj.weight"), &[intermediate, hidden]);
        expect_shape(&format!("{prefix}.mlp.down_proj.weight"), &[hidden, intermediate]);
        expect_kind(&format!("{prefix}.mlp.up_proj.weight"), "q4");
    }
    for layer in manifest::FULL_LAYERS {
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        expect_shape(&format!("{prefix}.q_proj.weight"), &[4096, hidden]);
        expect_shape(&format!("{prefix}.k_proj.weight"), &[512, hidden]);
        expect_shape(&format!("{prefix}.v_proj.weight"), &[512, hidden]);
        expect_shape(&format!("{prefix}.o_proj.weight"), &[hidden, 2048]);
        expect_shape(&format!("{prefix}.q_norm.weight"), &[manifest::HEAD_DIM]);
        expect_kind(&format!("{prefix}.q_norm.weight"), "bf16");
        expect_shape(&format!("{prefix}.k_norm.weight"), &[manifest::HEAD_DIM]);
    }
    for layer in manifest::LINEAR_LAYERS {
        let prefix = format!("model.language_model.layers.{layer}.linear_attn");
        expect_shape(&format!("{prefix}.A_log"), &[16]);
        expect_kind(&format!("{prefix}.A_log"), "f32");
        expect_shape(&format!("{prefix}.dt_bias"), &[16]);
        expect_kind(&format!("{prefix}.dt_bias"), "bf16");
        expect_shape(&format!("{prefix}.norm.weight"), &[manifest::LINEAR_VALUE_HEAD_DIM]);
        expect_kind(&format!("{prefix}.norm.weight"), "f32");
        expect_shape(
            &format!("{prefix}.conv1d.weight"),
            &[manifest::LINEAR_CONV_DIM, 1, manifest::LINEAR_CONV_KERNEL],
        );
        expect_shape(&format!("{prefix}.in_proj_qkv.weight"), &[6144, hidden]);
        expect_shape(&format!("{prefix}.in_proj_z.weight"), &[2048, hidden]);
        expect_shape(&format!("{prefix}.in_proj_a.weight"), &[16, hidden]);
        expect_shape(&format!("{prefix}.in_proj_b.weight"), &[16, hidden]);
        expect_shape(&format!("{prefix}.out_proj.weight"), &[hidden, 2048]);
        expect_kind(&format!("{prefix}.in_proj_qkv.weight"), "q4");
    }
}

#[test]
fn the_model_surface_has_exactly_these_tensors() {
    let hidden = manifest::HIDDEN_SIZE;
    expect_shape("model.language_model.embed_tokens.weight", &[manifest::VOCAB_SIZE, hidden]);
    expect_kind("model.language_model.embed_tokens.weight", "q4");
    expect_shape("model.language_model.norm.weight", &[hidden]);

    let vision = manifest::VISION_HIDDEN;
    let intermediate = manifest::VISION_INTERMEDIATE;
    expect_shape(
        "model.visual.patch_embed.proj.weight",
        &[vision, manifest::VISION_IN_CHANNELS, manifest::VISION_TEMPORAL_PATCH, manifest::VISION_PATCH, manifest::VISION_PATCH],
    );
    expect_shape("model.visual.patch_embed.proj.bias", &[vision]);
    expect_shape("model.visual.pos_embed.weight", &[manifest::VISION_POSITIONS, vision]);
    for block in 0..manifest::VISION_DEPTH {
        let prefix = format!("model.visual.blocks.{block}");
        expect_shape(&format!("{prefix}.attn.qkv.weight"), &[3 * vision, vision]);
        expect_shape(&format!("{prefix}.attn.qkv.bias"), &[3 * vision]);
        expect_shape(&format!("{prefix}.attn.proj.weight"), &[vision, vision]);
        expect_shape(&format!("{prefix}.mlp.linear_fc1.weight"), &[intermediate, vision]);
        expect_shape(&format!("{prefix}.mlp.linear_fc2.weight"), &[vision, intermediate]);
        expect_shape(&format!("{prefix}.norm1.weight"), &[vision]);
        expect_shape(&format!("{prefix}.norm1.bias"), &[vision]);
        expect_shape(&format!("{prefix}.norm2.weight"), &[vision]);
    }
    let merger = manifest::VISION_MERGER_HIDDEN;
    expect_shape("model.visual.merger.norm.weight", &[vision]);
    expect_shape("model.visual.merger.linear_fc1.weight", &[merger, merger]);
    expect_shape(
        "model.visual.merger.linear_fc2.weight",
        &[manifest::VISION_OUT_HIDDEN, merger],
    );
}

#[test]
fn the_multi_token_prediction_head_is_present_but_not_in_the_model() {
    let names = [
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ];
    for name in names {
        spec(name);
    }
    expect_shape("mtp.fc.weight", &[manifest::HIDDEN_SIZE, 2 * manifest::HIDDEN_SIZE]);
    expect_shape("mtp.layers.0.self_attn.q_proj.weight", &[4096, manifest::HIDDEN_SIZE]);
    expect_shape("mtp.layers.0.self_attn.k_proj.weight", &[512, manifest::HIDDEN_SIZE]);
}

#[test]
fn tensor_count_adds_up_block_by_block() {
    // 18 linear layers with 14 tensors, 6 full-attention layers with 11, the
    // embedding table and the final norm, then the vision tower: 12 blocks of
    // 12, the merger's 6, the patch projection's 2 and the position table.
    let text = 18 * 14 + 6 * 11 + 2;
    let vision = manifest::VISION_DEPTH * 12 + 6 + 2 + 1;
    assert_eq!(text, 320);
    assert_eq!(vision, 153);
    assert_eq!(text + vision, manifest::MODEL_TENSOR_COUNT);
}
