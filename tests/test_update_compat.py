"""Native CUDA operations and node loading against the installed ComfyUI."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import comfy.cli_args
import comfy.model_management
import comfy.ops
import comfy.sampler_helpers
import comfy.utils
from comfy.ldm.minimax.model import MiniMaxH3Model, time_shift_sigma
from comfy.model_patcher import ModelPatcher
from comfy.quant_ops import QuantizedTensor
from hyperflow_h3.curve import install_curve_refit
from hyperflow_h3.nodes import _apply
from hyperflow_h3.schedule import DEFAULT_SIGMAS_8STEP


def test_node_pack_registration():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("hyperflow_registration_check", root / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.NODE_CLASS_MAPPINGS) == {"ApplyHyperFlow", "ApplyHyperFlowAdvanced"}
    for node in module.NODE_CLASS_MAPPINGS.values():
        assert node.RETURN_TYPES == ("MODEL", "SIGMAS")
        assert callable(getattr(node, node.FUNCTION))
        assert "hyperflow_file" in node.INPUT_TYPES()["required"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke test")
@pytest.mark.parametrize("mode", ["merge", "bypass"])
@pytest.mark.parametrize("conditioning", ["full", "pruned", "refit"])
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "int8-convrot"])
@torch.inference_mode()
def test_native_cuda_model_load_forward_reload(tmp_path, monkeypatch, mode, conditioning, quantized):
    device = torch.device("cuda")
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: device)
    monkeypatch.setattr(comfy.cli_args.args, "disable_comfy_compiler", True)
    curve = conditioning != "full"
    torch.manual_seed(43)
    dm = MiniMaxH3Model(hidden_size=64, num_layers=1, token_refiner_num_layers=1,
                       num_attention_heads=1, attention_head_dim=64, ffn_hidden_size=128,
                       latents_dim=1, audio_latents_dim=2, text_dim=32,
                       timestep_input_dim=32, time_embed_hidden_size=48,
                       time_embed_dim=8 if curve else 24, adaln_curve_grid=1025 if curve else None,
                       rope_inv_freq_len=1, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
    tensors = {}
    with torch.no_grad():
        for name, param in dm.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1.)
            else:
                param.normal_(0., .03)
        dm.rope.inv_freq.fill_(1.)
        if curve:
            dm.adaln_t_table.normal_(0., .03)
        if quantized:
            ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
            for name in ("blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj",
                         "blocks.0.mlp.fc1", "blocks.0.mlp.fc2"):
                old = dm.get_submodule(name)
                qt = QuantizedTensor.from_float(old.weight.to(device), "TensorWiseINT8Layout",
                                                per_channel=True, convrot=True, convrot_groupsize=64)
                state = {key: value.cpu() for key, value in qt.state_dict(prefix="weight").items()}
                config = json.dumps({"format": "int8_tensorwise", "convrot": True,
                                     "convrot_groupsize": 64}).encode()
                state["comfy_quant"] = torch.tensor(list(config), dtype=torch.uint8)
                layer = ops.Linear(old.in_features, old.out_features, bias=False)
                layer.load_state_dict(state)
                comfy.utils.set_attr(dm, name, layer)
        targets = ["blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj",
                   "blocks.0.mlp.fc1", "blocks.0.mlp.fc2"]
        if not curve:
            targets += ["time_embedder.proj_in", "time_embedder.proj_out"]
        for name in targets:
            layer = dm.get_submodule(name)
            if name.endswith("qkv_proj"):
                a = torch.randn(12, layer.in_features) * .01
                b = torch.block_diag(*(torch.randn(layer.out_features // 3, 4) * .01 for _ in range(3)))
            else:
                a = torch.randn(4, layer.in_features) * .01
                b = torch.randn(layer.out_features, 4) * .01
            tensors[name + ".lora_A.weight"] = a
            tensors[name + ".lora_B.weight"] = b
            if name.startswith("time_embedder"):
                endpoint = name.replace("time_embedder", "endpoint_time_embedder")
                tensors[endpoint + ".lora_A.weight"] = a.clone()
                tensors[endpoint + ".lora_B.weight"] = b.clone()
    dm.requires_grad_(False)
    holder = torch.nn.Module()
    holder.diffusion_model = dm
    original = ModelPatcher(holder, device, torch.device("cpu"))
    path = tmp_path / "node_build.safetensors"
    save_file(tensors, path, metadata={"hyperflow": "true", "hyperflow_version": "1.0",
        "hyperflow_gate": "0.25", "lora_alpha": "4", "base_model": "MiniMaxAI/MiniMax-H3",
        "lora_rank": "4", "hyperflow_fc1_layout": "gate_value",
        "hyperflow_sigmas": json.dumps(DEFAULT_SIGMAS_8STEP)})
    patched, grid = _apply(original, str(path), 1., mode, False)
    if conditioning == "refit":
        current = (grid[:-1] * 1000. / 1000.).clamp(min=1e-6)
        data = {"t": torch.cat((1. - current, 1. - time_shift_sigma(current, 12., 3.))),
                "r": torch.cat((1. - grid[1:], 1. - time_shift_sigma(grid[1:], 12., 3.))),
                "generated": torch.randn(16, 8) * .03, "pinned": torch.randn(1025, 8) * .03}
        meta = {"gate": "0.25", "sigmas": json.dumps(DEFAULT_SIGMAS_8STEP),
                "video_shift": "12.0", "audio_shift": "3.0"}
        install_curve_refit(patched, (data, meta))
    comfy.sampler_helpers.prepare_model_patcher(patched, {}, patched.model_options)
    x = [torch.randn(1, 1, 1, 2, 4, device=device, dtype=torch.bfloat16),
         torch.randn(1, 2, 2, 1, device=device, dtype=torch.bfloat16)]
    text = torch.randn(1, 2, 32, device=device, dtype=torch.bfloat16)
    outputs = []
    with torch.inference_mode():
        for _ in range(2):
            patched.patch_model(device)
            try:
                run = []
                for sigma in grid[:-1]:
                    options = dict(patched.model_options["transformer_options"], sample_sigmas=grid.to(device))
                    out = dm(x, (sigma * 1000.).reshape(1).to(device), text, options)
                    assert [v.shape for v in out] == [v.shape for v in x]
                    assert all(torch.isfinite(v).all() for v in out)
                    run.append([v.cpu() for v in out])
                outputs.append(run)
            finally:
                patched.unpatch_model(torch.device("cpu"))
    for first, second in zip(*outputs):
        assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not original.object_patches
    assert dm.use_adaln_curves == curve
    if curve:
        assert not hasattr(dm, "time_embedder")
