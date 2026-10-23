# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent NVFP4 candidates, scale rings and exact-capacity H200 defaults.

Candidates are measured by the collective tuner on the current device. The
portable default profiles contain configuration only, never imported scores.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, cast

from .nvfp4_policy import (
    Nvfp4ConversionLayout,
    nvfp4_shared_memory_budget,
    split_nvfp4_conversion_tactic,
    validate_nvfp4_kperm,
)
from .mxfp4_optimization import (
    hopper_mxfp4_candidates,
    normalize_mxfp4_optimization_tactic,
    resolve_mxfp4_tactic_optimizations,
)
from .nvfp4_candidate_data import (
    NVFP4_SF0_CANDIDATE_SEEDS,
    NVFP4_DEFAULT_DOMAIN,
    NVFP4_DEFAULT_TACTICS_BY_TOKENS,
    NVFP4_CURATED_SF2_STATUS,
    NVFP4_CURATED_SF2_RULES,
    NVFP4_CURATED_SF2_PROFILES,
)

# Stable cache-schema tag: single-promotion uses one FP8 weight operand.
# The nv23/nv24 suffix counts tactic fields; preserve it for cache compatibility.
NVFP4_TUNING_VERSION = "nvfp4_single_k64_nv24_curated_scale_ring_v1"


def normalize_hopper_nvfp4_tactic(tactic: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the MXFP4 core fields plus compact pull, LUT layout and scale ring."""
    core, conversion = split_nvfp4_conversion_tactic(tactic)
    return dict(normalize_mxfp4_optimization_tactic(core), **conversion)


def nvfp4_candidate_sha256(candidates: Iterable[Mapping[str, Any]]) -> str:
    values = [normalize_hopper_nvfp4_tactic(x) for x in candidates]
    return hashlib.sha256(
        json.dumps(
            {"version": NVFP4_TUNING_VERSION, "candidates": values},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def nvfp4_candidate_resource_budget(
    tactic: Mapping[str, Any],
    *,
    hidden: int,
    num_experts: int,
    kperm: int = 64,
    smem_capacity: int = 232448,
    occupancy: int = 1,
) -> dict[str, int]:
    """Exact conservative host reservation for the admitted N64/K256 shape.

    This bounded oracle covers M256 non-pingpong hybrid, BF16 combine and
    epilogue token-back only. Other shapes are not inferred. Kernel construction
    always recomputes from actual CuTe layouts using the same reservation
    function; native AB/layout validation remains required after compilation.
    """
    validate_nvfp4_kperm(kperm)
    selected = normalize_hopper_nvfp4_tactic(tactic)
    if (
        tuple(selected["mma_tiler_mnk"]) != (256, 64, 256)
        or selected["pingpong"]
        or selected["token_back_mode"] != "epi_warps"
        or selected["combine_format"] != "bf16"
        or selected["grouped_token_back"]
    ):
        raise ValueError(
            "NVFP4 resource oracle covers non-pingpong M256N64K256 epi/BF16 only"
        )
    for name, value in (("hidden", hidden), ("num_experts", num_experts)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    # TokenComm: Uint8 hidden payload, four pull mbarriers, int32 expert counts.
    # Compact reserves one hidden row per active warp; epi token-back adds no
    # standalone pull storage. The base reservation is _SmemMiscBudget=1024.
    align = lambda n, a: (n + a - 1) // a * a
    slots = selected["active_dispatch_warps"] if selected["compact_pull_buffer"] else 4
    misc = (
        1024
        + 32
        + align(num_experts * 4, 16)
        + align(
            slots * max(hidden, 2048)
            if selected["compact_pull_buffer"]
            else slots * hidden,
            128,
        )
    )
    return nvfp4_shared_memory_budget(
        smem_capacity=smem_capacity,
        occupancy=occupancy,
        ab_data_bytes=256 * 256 // 2 + 64 * 256,
        activation_sf_bytes=64 * 4 * 4,
        weight_sf_bytes=2 * (256 // 64) * 512,
        misc_bytes=misc,
        # Two epilogue WGs: each stores N64 x M64 FP8 and 4 x N64 f32 amax.
        c_bytes_total=2 * (64 * 64 + 4 * 64 * 4),
        conversion=Nvfp4ConversionLayout(
            kperm=kperm,
            lut_replicas=selected["nvfp4_lut_replicas"],
            lut_map=selected["nvfp4_lut_map"],
            sf_stages=selected["nvfp4_sf_stages"],
        ),
    )


def expand_nvfp4_sf2_candidates(
    seeds: Iterable[Mapping[str, Any]],
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    kperm: int = 64,
) -> list[dict[str, Any]]:
    """Keep unique legal sf0 seeds and append at most R1/R2 per eligible core."""
    validate_nvfp4_kperm(kperm)
    result = []
    for seed in seeds:
        selected = normalize_hopper_nvfp4_tactic(seed)
        if selected["nvfp4_sf_stages"] != 0:
            raise ValueError("scale-ring expansion requires sf0 seeds")
        core, _ = split_nvfp4_conversion_tactic(selected)
        resolve_mxfp4_tactic_optimizations(
            core,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            world_size=world_size,
        )
        if selected not in result:
            result.append(selected)
    if kperm != 64:
        return result
    for seed in tuple(result):
        if (
            tuple(seed["mma_tiler_mnk"]) != (256, 64, 256)
            or seed["pingpong"]
            or seed["token_back_mode"] != "epi_warps"
        ):
            continue
        for replicas in (1, 2):
            candidate = dict(
                seed,
                compact_pull_buffer=True,
                nvfp4_lut_replicas=replicas,
                nvfp4_lut_map="lane",
                nvfp4_sf_stages=2,
            )
            try:
                budget = nvfp4_candidate_resource_budget(
                    candidate, hidden=hidden, num_experts=num_experts, kperm=kperm
                )
            except ValueError:
                continue
            if budget["num_ab_stage"] == 4 and candidate not in result:
                result.append(candidate)
    return result


def _curated_sf2_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str,
    kperm: int,
) -> list[dict[str, Any]]:
    """Read complete scale-ring profiles without reconstructing core fields."""
    if (
        type(max_tokens) is not int
        or max_tokens not in (512, 1024, 2048)
        or (hidden, intermediate, num_experts, world_size, kperm, routing_profile)
        != (7168, 3072, 384, 4, 64, "block_permutation_v1")
    ):
        return []
    if NVFP4_CURATED_SF2_STATUS == "UNQUALIFIED":
        if any(NVFP4_CURATED_SF2_PROFILES.values()):
            raise ValueError("Unqualified sf2 table must be empty")
        return []
    if NVFP4_CURATED_SF2_STATUS != "QUALIFIED":
        raise ValueError("Unknown sf2 curation status")
    result = []
    profiles = cast(
        Iterable[dict[str, Any]], NVFP4_CURATED_SF2_PROFILES.get(max_tokens, ())
    )
    for profile in profiles:
        config = profile.get("requested_config", {})
        if (
            profile.get("qualification") != "QUALIFIED"
            or set(config) != {"kperm", "nvfp4_fc1_k256_overlap"}
            or type(config.get("kperm")) is not int
            or config.get("kperm") != 64
            or config.get("nvfp4_fc1_k256_overlap") is not True
            or profile.get("effective_overlap") is not True
            or not isinstance(profile.get("profile_id"), str)
            or not profile["profile_id"]
        ):
            raise ValueError("Curated sf2 entry lacks complete qualification/config")
        tactic = normalize_hopper_nvfp4_tactic(profile["tactic"])
        if (
            set(profile["tactic"]) != set(tactic)
            or tactic["nvfp4_sf_stages"] != 2
            or tactic["nvfp4_lut_replicas"] not in (1, 2)
            or tactic["nvfp4_lut_map"] != "lane"
            or tactic["compact_pull_buffer"] is not True
        ):
            raise ValueError(
                "Curated sf2 requires a complete R1/R2 lane/compact profile"
            )
        core, _ = split_nvfp4_conversion_tactic(tactic)
        resolve_mxfp4_tactic_optimizations(
            core,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            world_size=world_size,
        )
        if (
            nvfp4_candidate_resource_budget(
                tactic, hidden=hidden, num_experts=num_experts, kperm=kperm
            )["num_ab_stage"]
            != 4
        ):
            raise ValueError("Curated sf2 profile does not reserve four AB stages")
        if tactic in result:
            raise ValueError("Duplicate complete sf2 profile in one bucket")
        result.append(tactic)
    return result


def nvfp4_sf2_context_eligible(
    tactic: Mapping[str, Any],
    *,
    max_tokens: int,
    live_tokens: int | None,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    num_topk: int,
    gate_up_clamp: float | None,
    routing_profile: str,
    kperm: int,
    overlap: bool | None,
    device_name: str | None,
    sm_count: int | None,
) -> bool:
    """Performance eligibility only; source-level manual legality is separate."""
    if any(
        type(value) is not int
        for value in (hidden, intermediate, num_experts, world_size, num_topk, kperm)
    ):
        return False
    if (
        type(max_tokens) is not int
        or type(live_tokens) is not int
        or max_tokens not in (512, 1024, 2048)
        or live_tokens != max_tokens
        or (
            hidden,
            intermediate,
            num_experts,
            world_size,
            num_topk,
            kperm,
            routing_profile,
        )
        != (7168, 3072, 384, 4, 6, 64, "block_permutation_v1")
        or gate_up_clamp != 10
        or device_name != "NVIDIA H200"
        or type(sm_count) is not int
        or sm_count != 132
    ):
        return False
    selected = normalize_hopper_nvfp4_tactic(tactic)
    if selected not in _curated_sf2_candidates(
        max_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
        routing_profile=routing_profile,
        kperm=kperm,
    ):
        return False
    from .nvfp4_policy import resolve_nvfp4_fc1_overlap

    effective = resolve_nvfp4_fc1_overlap(
        override=overlap,
        device_name=device_name,
        sm_count=sm_count,
        num_tokens=live_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_total_experts=num_experts,
        num_topk=num_topk,
        world_size=world_size,
        mma_tiler_mnk=selected["mma_tiler_mnk"],
        cluster_shape_mnk=selected["cluster_shape_mnk"],
        fp8_scale_mode="mxfp4_hybrid",
        pingpong=selected["pingpong"],
    )
    return effective is True


def _sf0_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str,
    kperm: int,
) -> list[dict[str, Any]]:
    """Return the complete sf0 candidate set in stable order."""
    validate_nvfp4_kperm(kperm)
    seeds = list(
        hopper_mxfp4_candidates(
            max_tokens,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            world_size=world_size,
            routing_profile=routing_profile,
        )
    )
    for tactic in NVFP4_SF0_CANDIDATE_SEEDS:
        try:
            core, _ = split_nvfp4_conversion_tactic(tactic)
            resolve_mxfp4_tactic_optimizations(
                core,
                hidden=hidden,
                intermediate=intermediate,
                num_experts=num_experts,
                world_size=world_size,
            )
        except ValueError:
            continue
        seeds.append(tactic)
    result = []
    for seed in seeds:
        selected = normalize_hopper_nvfp4_tactic(seed)
        if selected not in result:
            result.append(selected)
    return result


def hopper_nvfp4_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str = "block_permutation_v1",
    kperm: int = 64,
) -> list[dict[str, Any]]:
    """Preserved sf0 union plus exact, potentially applicable curated profiles.

    Top-k, clamp, device, actual live tokens and resolved overlap are checked by
    ``filter_hopper_nvfp4_candidates`` in the autotune and cache entry points.
    Explicit exploration remains in ``expand_nvfp4_sf2_candidates``.
    """
    shape: dict[str, Any] = dict(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
        routing_profile=routing_profile,
        kperm=kperm,
    )
    return _sf0_candidates(max_tokens, **shape) + _curated_sf2_candidates(
        max_tokens, **shape
    )


def filter_hopper_nvfp4_candidates(candidates, **context) -> list[dict[str, Any]]:
    """Apply the complete live-context guard to auto/cache, preserving sf0."""
    result = []
    for candidate in candidates:
        selected = normalize_hopper_nvfp4_tactic(candidate)
        if selected["nvfp4_sf_stages"] == 0 or nvfp4_sf2_context_eligible(
            selected, **context
        ):
            result.append(selected)
    return result


def hopper_nvfp4_default_profile(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    num_topk: int,
    gate_up_clamp: float | None,
    routing_profile: str,
    kperm: int,
    device_name: str | None = None,
    sm_count: int | None = None,
    overlap: bool | None = None,
) -> dict[str, Any] | None:
    """Match a portable capacity default, returning a fresh tactic/config pair.

    Unknown hardware or an unmatched model/capacity uses the caller's existing
    heuristic. The profile's overlap is a default: an explicit caller override
    must be applied after its tactic geometry, and is not replaced here.
    """
    validate_nvfp4_kperm(kperm)
    if overlap is not None and type(overlap) is not bool:
        raise ValueError("overlap must be None or bool")
    if any(
        type(value) is not int
        for value in (
            max_tokens,
            hidden,
            intermediate,
            num_experts,
            world_size,
            num_topk,
            sm_count,
        )
    ):
        return None
    context = dict(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
        num_topk=num_topk,
        gate_up_clamp=gate_up_clamp,
        routing_profile=routing_profile,
        kperm=kperm,
        device_name=device_name,
        sm_count=sm_count,
    )
    if any(value != NVFP4_DEFAULT_DOMAIN[name] for name, value in context.items()):
        return None
    profile: dict[str, Any] | None = NVFP4_DEFAULT_TACTICS_BY_TOKENS.get(max_tokens)
    if profile is None:
        return None
    return dict(
        requested_tactic=normalize_hopper_nvfp4_tactic(profile["requested_tactic"]),
        requested_config=dict(profile["requested_config"]),
    )


def nvfp4_curated_domain_sha256(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str,
    kperm: int,
) -> str:
    """Bind exact capacity, candidate membership, scale ring and default policy."""
    candidates = hopper_nvfp4_candidates(
        max_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
        routing_profile=routing_profile,
        kperm=kperm,
    )
    payload = dict(
        version=NVFP4_TUNING_VERSION,
        max_tokens=max_tokens,
        shape=(hidden, intermediate, num_experts, world_size),
        routing_profile=routing_profile,
        kperm=kperm,
        candidates=candidates,
        default_domain=NVFP4_DEFAULT_DOMAIN,
        rules=NVFP4_CURATED_SF2_RULES,
        qualification_status=NVFP4_CURATED_SF2_STATUS,
        bucket_profiles=NVFP4_CURATED_SF2_PROFILES.get(max_tokens, ()),
        default_profile=NVFP4_DEFAULT_TACTICS_BY_TOKENS.get(max_tokens),
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hopper_nvfp4_cache_provenance_sha256(
    *,
    routing_profile: str,
    kperm: int,
    overlap: bool | None,
    max_tokens: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
) -> str:
    """Separate NV layout/overlap/default policy from other tuning namespaces."""
    validate_nvfp4_kperm(kperm)
    if overlap is not None and type(overlap) is not bool:
        raise ValueError("overlap must be None or bool")
    from .mxfp4_tuner import hopper_mxfp4_cache_provenance_sha256

    payload = dict(
        version=NVFP4_TUNING_VERSION,
        kperm=kperm,
        overlap=overlap,
        domain_sha256=nvfp4_curated_domain_sha256(
            max_tokens,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            world_size=world_size,
            routing_profile=routing_profile,
            kperm=kperm,
        ),
        geometry_provenance=hopper_mxfp4_cache_provenance_sha256(
            routing_profile=routing_profile
        ),
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


nvfp4_cache_provenance = hopper_nvfp4_cache_provenance_sha256
