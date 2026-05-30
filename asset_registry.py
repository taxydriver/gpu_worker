"""Static asset registry for ComfyUI model groups."""

from __future__ import annotations

from copy import deepcopy

ASSET_REGISTRY: dict[str, list[dict[str, str]]] = {
    "flux_stills_v1": [
        {
            "name": "flux2_vae",
            "path": "/workspace/ComfyUI/models/vae/flux2-vae.safetensors",
            "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        },
        {
            "name": "flux2_unet",
            "path": "/workspace/ComfyUI/models/diffusion_models/flux2_dev_fp8mixed.safetensors",
            "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
        },
        {
            "name": "flux2_text_encoder",
            "path": "/workspace/ComfyUI/models/text_encoders/mistral_3_small_flux2_bf16.safetensors",
            "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors",
        },
    ],
    "wan_i2v_v1": [
        {
            "name": "wan_text_encoder",
            "path": "/workspace/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        },
        {
            "name": "wan_vae",
            "path": "/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        },
        {
            "name": "wan_i2v_high_noise_unet",
            "path": "/workspace/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        },
        {
            "name": "wan_i2v_low_noise_unet",
            "path": "/workspace/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        },
        {
            "name": "wan_i2v_high_noise_lora",
            "path": "/workspace/ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        },
        {
            "name": "wan_i2v_low_noise_lora",
            "path": "/workspace/ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
        },
    ],
    "stable_audio_v1": [
        {
            "name": "stable_audio_checkpoint",
            "path": "/workspace/ComfyUI/models/checkpoints/stable-audio-open-1.0.safetensors",
            "url": "https://huggingface.co/Comfy-Org/stable-audio-open-1.0_repackaged/resolve/main/stable-audio-open-1.0.safetensors",
        },
        {
            "name": "stable_audio_t5",
            "path": "/workspace/ComfyUI/models/clip/t5-base.safetensors",
            "url": "https://huggingface.co/ComfyUI-Wiki/t5-base/resolve/main/t5-base.safetensors",
        },
    ],
    "juggernaut_stills_v1": [
        {
            "name": "juggernautxl_checkpoint",
            "path": "/workspace/ComfyUI/models/checkpoints/JuggernautXL_v9_RunDiffusionPhoto_v2.safetensors",
            "url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/a7634331b40541c153687f8b8e80bdbf2c63a0f5/JuggernautXL_v9_RunDiffusionPhoto_v2.safetensors",
        },
        {
            "name": "sdxl_clip_vision_h",
            "path": "/workspace/ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
            "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
        },
        {
            "name": "sdxl_ipadapter_face",
            "path": "/workspace/ComfyUI/models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
        },
    ],
    # LTX-2.3 Fast (distilled) — T2V/I2V with native audio. Uses ComfyUI's built-in
    # LTX-2 nodes only (no custom node). Requires a recent ComfyUI build (>= the one
    # shipping LTXAVTextEncoderLoader / the LTX-2.3 AV nodes). The distilled checkpoint
    # is all-in-one (video transformer + video VAE + audio VAE). The Gemma text encoder
    # is the ungated Comfy-Org repack, downloaded under the name the AV nodes expect.
    # Validated 2026-05-30: see filmforge_backend/docs/ltx-2-3-gpu-worker-install-2026-05-30.md.
    "ltx_i2v_v1": [
        {
            "name": "ltx2_3_distilled_checkpoint",
            "path": "/workspace/ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors",
            "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors",
        },
        {
            "name": "ltx2_3_gemma_text_encoder",
            "path": "/workspace/ComfyUI/models/text_encoders/comfy_gemma_3_12B_it.safetensors",
            "url": "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it.safetensors",
        },
    ],
}


CAPABILITY_ASSET_GROUPS: dict[str, list[str]] = {
    "flux2_stills": ["flux_stills_v1"],
    "flux_stills_v1": ["flux_stills_v1"],
    "wan_i2v": ["wan_i2v_v1"],
    "wan_i2v_v1": ["wan_i2v_v1"],
    "stable_audio": ["stable_audio_v1"],
    "stable_audio1": ["stable_audio_v1"],
    "stable_audio_v1": ["stable_audio_v1"],
    "juggernaut_stills": ["juggernaut_stills_v1"],
    "juggernaut_stills_v1": ["juggernaut_stills_v1"],
    "ltx_av": ["ltx_i2v_v1"],
    "ltx_i2v": ["ltx_i2v_v1"],
    "ltx_t2v": ["ltx_i2v_v1"],
    "ltx2_3": ["ltx_i2v_v1"],
    "ltx_i2v_v1": ["ltx_i2v_v1"],
}


def get_asset_group(name: str) -> list[dict[str, str]]:
    """Return a copy of the asset list for a known asset group."""

    assets = ASSET_REGISTRY.get(name)
    if assets is None:
        known = ", ".join(sorted(ASSET_REGISTRY))
        raise ValueError(f"Unknown asset_group={name!r}. Known groups: {known}")
    return deepcopy(assets)


def asset_groups_for_capabilities(capabilities: list[str]) -> list[str]:
    """Return downloadable asset groups implied by declared worker capabilities."""

    if not capabilities:
        return []

    groups: list[str] = []
    seen: set[str] = set()
    for capability in capabilities:
        key = str(capability).strip()
        if not key:
            continue
        for group in CAPABILITY_ASSET_GROUPS.get(key, []):
            if group in ASSET_REGISTRY and group not in seen:
                groups.append(group)
                seen.add(group)
    return groups


def asset_group_supported_by_capabilities(asset_group: str, capabilities: list[str]) -> bool:
    """Return whether a requested asset group is allowed for this worker."""

    if not capabilities:
        return asset_group in ASSET_REGISTRY
    return asset_group in set(asset_groups_for_capabilities(capabilities))
