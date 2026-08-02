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
    # LTX-2.3 Fast (distilled) — T2V/I2V with native audio. Uses ComfyUI's built-in
    # LTX-2 nodes only (no custom node). Requires a recent ComfyUI build (>= the one
    # shipping LTXAVTextEncoderLoader / the LTX-2.3 AV nodes). The distilled checkpoint
    # is all-in-one (video transformer + video VAE + audio VAE). The Gemma text encoder
    # is the ungated Comfy-Org repack, downloaded under the name the AV nodes expect.
    # Validated 2026-05-30: see Filmforge/backend/docs/discoveries/ltx-2-3-gpu-worker-install-2026-05-30.md.
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
    # Per-character identity LoRAs (FilmForge-trained, WAN 2.2 I2V). OPT-IN — gated by
    # the `character_loras` capability (see CAPABILITY_ASSET_GROUPS), which base WAN
    # deploys do NOT declare. To "also deploy the LoRAs", add it to the worker caps:
    #   WORKER_CAPABILITIES=flux2_stills,wan_i2v,ltx_i2v,character_loras
    # then the worker preloads + serves this group (and /assets/ensure allows warming
    # it). Workflows reference loras/<character>/... and chain them after the lightx2v
    # loaders at strength ~0.7. Hosted on a public HF repo (tokenless download, like
    # every other asset URL). Adding a character = upload its pair + 2 entries here.
    # See discovery identity-lora-pipeline-2026-06-21 + memory project_identity_lora_pipeline.
    "character_loras_v1": [
        {
            "name": "aigiri_young_i2v_high_noise_lora",
            "path": "/workspace/ComfyUI/models/loras/aigiri/aigiri_young_i2v_v1_high_noise.safetensors",
            "url": "https://huggingface.co/taxydriver/filmforge-loras/resolve/main/aigiri/aigiri_young_i2v_v1_high_noise.safetensors",
        },
        {
            "name": "aigiri_young_i2v_low_noise_lora",
            "path": "/workspace/ComfyUI/models/loras/aigiri/aigiri_young_i2v_v1_low_noise.safetensors",
            "url": "https://huggingface.co/taxydriver/filmforge-loras/resolve/main/aigiri/aigiri_young_i2v_v1_low_noise.safetensors",
        },
        {
            "name": "swami_i2v_high_noise_lora",
            "path": "/workspace/ComfyUI/models/loras/swami/swami_i2v_v1_high_noise.safetensors",
            "url": "https://huggingface.co/taxydriver/filmforge-loras/resolve/main/swami/swami_i2v_v1_high_noise.safetensors",
        },
        {
            "name": "swami_i2v_low_noise_lora",
            "path": "/workspace/ComfyUI/models/loras/swami/swami_i2v_v1_low_noise.safetensors",
            "url": "https://huggingface.co/taxydriver/filmforge-loras/resolve/main/swami/swami_i2v_v1_low_noise.safetensors",
        },
    ],
    # Fun-VACE module for WAN 2.2 — reference-image identity conditioning for the boys /
    # extras / one-offs (training-free, ControlNet-style). Mirrors the WAN 2.2 dual-expert
    # structure (high/low), fp8_scaled ~17.3 GB each. OPT-IN — gated by the `wan_vace`
    # capability (see CAPABILITY_ASSET_GROUPS), NOT implied by wan_i2v, so base WAN deploys
    # stay lean. The VACE workflow swaps WanImageToVideo -> WanVaceToVideo and points the
    # UNETLoaders at these modules; leads' LoRAs still chain after the lightx2v loaders, so
    # a lead (LoRA) + a boy (VACE ref) compose in the same shot. Hosted on Comfy-Org HF
    # (tokenless). See discovery vace-boys-identity-design-2026-06-21 + memory
    # project_identity_lora_pipeline.
    "wan_vace_v1": [
        {
            "name": "wan22_fun_vace_high_noise",
            "path": "/workspace/ComfyUI/models/diffusion_models/wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors",
        },
        {
            "name": "wan22_fun_vace_low_noise",
            "path": "/workspace/ComfyUI/models/diffusion_models/wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors",
        },
    ],
    # ESRGAN upscaler for the identity-quality FINISHING pass (4x then downscale to
    # net 2x → film clarity, identity-safe — no diffusion). OPT-IN via `finishing` cap.
    # The finish workflow (render.py finish_clip) loads it by filename through native
    # UpscaleModelLoader, so it just needs to be on disk. ~64MB. See discovery
    # project_wan_identity_quality.
    "finishing_v1": [
        {
            "name": "upscale_4x_ultrasharp",
            "path": "/workspace/ComfyUI/models/upscale_models/4x-UltraSharp.pth",
            "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
        },
    ],
    # Opt-in: InfiniteTalk / MultiTalk audio-driven talking shots. WAN **2.1**-based —
    # a SEPARATE weight set beside wan_i2v_v1's 2.2 stack (incl. its own lightx2v LoRA
    # and VAE), so this is never implied by wan_i2v. Runs at 25 fps vs our 16.
    # Needs Kijai's ComfyUI-WanVideoWrapper custom node too, which asset_manager cannot
    # install — hence the PROVISIONERS entry below runs *in addition to* these downloads.
    # wav2vec (TencentGameMate/chinese-wav2vec2-base) is NOT listed: the wrapper's
    # DownloadAndLoadWav2VecModel node fetches it on first run.
    # Spike-validated 2026-07-28 (A1 "Wow. It works." / A2 two-shot):
    # spikes/audio_dialogue/, docs/AUDIO_DIALOGUE_RESEARCH_2026-07-28.md.
    "infinitetalk_v1": [
        {
            "name": "wan21_i2v_base",
            "path": "/workspace/ComfyUI/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        },
        {
            "name": "infinitetalk_single",
            "path": "/workspace/ComfyUI/models/diffusion_models/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
        },
        {
            "name": "infinitetalk_multi",
            "path": "/workspace/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp16.safetensors",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp16.safetensors",
        },
        {
            "name": "wan21_vae",
            "path": "/workspace/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors",
        },
        {
            "name": "wan21_text_encoder",
            "path": "/workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors",
        },
        {
            # MUST be Comfy-Org's repackage, NOT Kijai's
            # open-clip-xlm-roberta-large-vit-huge-14_visual_fp16. The A1 graph
            # loads this through STOCK `CLIPVisionLoader`, which inspects the
            # state dict and rejects Kijai's visual-only extract outright:
            # "clip vision file is invalid and does not contain a valid vision
            # model" (ComfyUI nodes.py:1062). Verified by A1 passing on this
            # file and failing on the other, 2026-08-01.
            "name": "wan21_clip_vision",
            "path": "/workspace/ComfyUI/models/clip_vision/clip_vision_h.safetensors",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
        },
        {
            "name": "wan21_lightx2v_lora",
            "path": "/workspace/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        },
    ],
    # --- Provisioner-backed audio groups (NOT single-file ComfyUI models) ---------------
    # These run as standalone pip packages in isolated venvs (their transformers pins
    # conflict with each other and with ComfyUI), loading whole HF-repo snapshots via
    # from_pretrained — so they are NOT downloaded by asset_manager (empty file list).
    # They are installed by PROVISIONERS (see below) at deploy time, but registered here
    # as canonical groups so the worker ADVERTISES them via /health + backend registration
    # (canonicalize_groups only surfaces groups that exist in ASSET_REGISTRY) and so the
    # infra UI can offer them. OPT-IN via capability; base render deploys stay lean.
    # Validated 2026-07-15: backend/docs/discoveries/oss-dialogue-tts-validation-2026-07-15.md
    # + research/oss_audio_video_models_survey_2026-07-15.md.
    #
    # tts_dialogue_v1  -> Chatterbox Multilingual (EN/Hindi clone) + Indic Parler (Telugu/Tamil)
    #                     installed by gpu_worker/provision_tts.sh (Parler gated -> HF_TOKEN).
    "tts_dialogue_v1": [],
    # stable_audio3_v1 -> Stable Audio 3 Medium (music/score, replaces Stable Audio Open 1.0)
    #                     installed by gpu_worker/provision_sa3.sh (gated -> HF_TOKEN).
    "stable_audio3_v1": [],
}


# Provisioner-backed groups: capability declared -> deploy runs this script instead of
# (or in addition to) asset_manager file downloads. Keys are canonical ASSET_REGISTRY groups.
PROVISIONERS: dict[str, str] = {
    "tts_dialogue_v1": "gpu_worker/provision_tts.sh",
    "stable_audio3_v1": "gpu_worker/provision_sa3.sh",
    # Runs IN ADDITION to infinitetalk_v1's file downloads: installs the custom node
    # the graphs need. Weights stay with asset_manager so they cache on the data volume.
    "infinitetalk_v1": "gpu_worker/provision_infinitetalk.sh",
}


CAPABILITY_ASSET_GROUPS: dict[str, list[str]] = {
    "flux2_stills": ["flux_stills_v1"],
    "flux_stills_v1": ["flux_stills_v1"],
    "wan_i2v": ["wan_i2v_v1"],
    "wan_i2v_v1": ["wan_i2v_v1"],
    "ltx_av": ["ltx_i2v_v1"],
    "ltx_i2v": ["ltx_i2v_v1"],
    "ltx_t2v": ["ltx_i2v_v1"],
    "ltx2_3": ["ltx_i2v_v1"],
    "ltx_i2v_v1": ["ltx_i2v_v1"],
    # Opt-in: a worker only provisions/serves character LoRAs when it DECLARES this
    # capability (WORKER_CAPABILITIES=...,character_loras). Not implied by wan_i2v,
    # so base WAN deploys stay lean; deploy with it to "also deploy the LoRAs".
    "character_loras": ["character_loras_v1"],
    "character_loras_v1": ["character_loras_v1"],
    # Opt-in: Fun-VACE module for reference-image identity conditioning (the boys).
    # Declare WORKER_CAPABILITIES=...,wan_vace to provision/serve it. Not implied by wan_i2v.
    "wan_vace": ["wan_vace_v1"],
    "wan_vace_v1": ["wan_vace_v1"],
    # Opt-in: ESRGAN upscaler for the quality finishing pass.
    "finishing": ["finishing_v1"],
    "finishing_v1": ["finishing_v1"],
    # Opt-in: open-source dialogue TTS (Chatterbox + Indic Parler), provisioner-backed.
    "tts": ["tts_dialogue_v1"],
    "tts_dialogue": ["tts_dialogue_v1"],
    "chatterbox": ["tts_dialogue_v1"],
    "indic_parler": ["tts_dialogue_v1"],
    "tts_dialogue_v1": ["tts_dialogue_v1"],
    # Opt-in: Stable Audio 3 music/score (replaces Stable Audio Open 1.0), provisioner-backed.
    "stable_audio3": ["stable_audio3_v1"],
    "stable_audio_3": ["stable_audio3_v1"],
    "stable_audio3_v1": ["stable_audio3_v1"],
    # Opt-in: audio-driven talking shots (InfiniteTalk single / MultiTalk two-shot).
    # Deliberately NOT implied by wan_i2v — it is a whole second WAN 2.1 weight set.
    "infinitetalk": ["infinitetalk_v1"],
    "talking_shot": ["infinitetalk_v1"],
    "multitalk": ["infinitetalk_v1"],
    "infinitetalk_v1": ["infinitetalk_v1"],
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
