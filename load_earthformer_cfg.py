import yaml


def load_earthformer_config(datasetname: str):
    # Read the configuration parameters in the.yaml file
    if datasetname is 'cikm':
        with open("config/cikm/cfg_cikm.yaml", "r") as file:
            config = yaml.safe_load(file)
    elif datasetname is 'meteonet':
        with open("config/meteonet/cfg_meteonet.yaml", "r") as file:
            config = yaml.safe_load(file)
    elif datasetname is 'sevir':
        with open("config/sevir/cfg_sevirlr.yaml", "r") as file:
            config = yaml.safe_load(file)

    # Earthformer configuration file
    model_cfg = config['model']

    num_blocks = len(model_cfg["enc_depth"])
    enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
    dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
    dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks

    model_kwargs = {
        "input_shape": model_cfg["input_shape"],
        "target_shape": model_cfg["target_shape"],
        "base_units": model_cfg["base_units"],
        "block_units": model_cfg["block_units"],
        "scale_alpha": model_cfg["scale_alpha"],
        "enc_depth": model_cfg["enc_depth"],
        "dec_depth": model_cfg["dec_depth"],
        "enc_use_inter_ffn": model_cfg["enc_use_inter_ffn"],
        "dec_use_inter_ffn": model_cfg["dec_use_inter_ffn"],
        "dec_hierarchical_pos_embed": model_cfg["dec_hierarchical_pos_embed"],
        "downsample": model_cfg["downsample"],
        "downsample_type": model_cfg["downsample_type"],
        "enc_attn_patterns": enc_attn_patterns,
        "dec_self_attn_patterns": dec_self_attn_patterns,
        "dec_cross_attn_patterns": dec_cross_attn_patterns,
        "dec_cross_last_n_frames": model_cfg["dec_cross_last_n_frames"],
        "dec_use_first_self_attn": model_cfg["dec_use_first_self_attn"],
        "num_heads": model_cfg["num_heads"],
        "attn_drop": model_cfg["attn_drop"],
        "proj_drop": model_cfg["proj_drop"],
        "ffn_drop": model_cfg["ffn_drop"],
        "upsample_type": model_cfg["upsample_type"],
        "ffn_activation": model_cfg["ffn_activation"],
        "gated_ffn": model_cfg["gated_ffn"],
        "norm_layer": model_cfg["norm_layer"],

        "num_global_vectors": model_cfg["num_global_vectors"],
        "use_dec_self_global": model_cfg["use_dec_self_global"],
        "dec_self_update_global": model_cfg["dec_self_update_global"],
        "use_dec_cross_global": model_cfg["use_dec_cross_global"],
        "use_global_vector_ffn": model_cfg["use_global_vector_ffn"],
        "use_global_self_attn": model_cfg["use_global_self_attn"],
        "separate_global_qkv": model_cfg["separate_global_qkv"],
        "global_dim_ratio": model_cfg["global_dim_ratio"],

        "initial_downsample_type": model_cfg["initial_downsample_type"],
        "initial_downsample_activation": model_cfg["initial_downsample_activation"],
        "initial_downsample_stack_conv_num_layers": model_cfg["initial_downsample_stack_conv_num_layers"],
        "initial_downsample_stack_conv_dim_list": model_cfg["initial_downsample_stack_conv_dim_list"],
        "initial_downsample_stack_conv_downscale_list": model_cfg["initial_downsample_stack_conv_downscale_list"],
        "initial_downsample_stack_conv_num_conv_list": model_cfg["initial_downsample_stack_conv_num_conv_list"],

        "padding_type": model_cfg["padding_type"],
        "z_init_method": model_cfg["z_init_method"],
        "checkpoint_level": model_cfg["checkpoint_level"],
        "pos_embed_type": model_cfg["pos_embed_type"],
        "use_relative_pos": model_cfg["use_relative_pos"],
        "self_attn_use_final_proj": model_cfg["self_attn_use_final_proj"],

        "attn_linear_init_mode": model_cfg["attn_linear_init_mode"],
        "ffn_linear_init_mode": model_cfg["ffn_linear_init_mode"],
        "conv_init_mode": model_cfg["conv_init_mode"],
        "down_up_linear_init_mode": model_cfg["down_up_linear_init_mode"],
        "norm_init_mode": model_cfg["norm_init_mode"]
    }

    return model_kwargs