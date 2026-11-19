# -*- coding: utf-8 -*-
"""Pure functions that turn one manifest job into a ComfyUI API prompt (no network, no files except the templates).

Shared by remote/remote_job.py (on Colab, and in local mode through scripts/h3local.py) and scripts/selftest_graph.py.
The templates in workflows/ are already clean API graphs (no VHS, no preview nodes): they are posted as they are,
after the parameters below are written into them. Nothing is stripped at run time.

Ported from H3 Toybox (dseditor/minimax-h3-colab, toybox/server.py): DEFAULT_CONFIG steps / loras / intensity /
extend, dims(), round32(), resolve_mentions(), alignment_line(), raw_prompt(), bypass(), build_generate(),
image_dims(), image_refs() (P-H3 branch), ph3_graph(), check_image_wiring().

A "media item" is a dict describing one file already placed in ComfyUI's input folder:
    {"kind": "image"|"video"|"audio", "name": <file name in ComfyUI input>,
     "width": int|None, "height": int|None, "has_audio": bool (videos)}
"""
import copy
import json
import random
import re
from pathlib import Path

FPS = 24
WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"
VIDEO_TEMPLATE = "h3_video_api.json"
PICTURE_TEMPLATE = "h3_image_api.json"

# fast mode (default on): fewer SelfLift steps, [total, low-resolution steps]
STEPS = {"normal": (12, 6), "fast": (8, 4)}
# LoRA node ids in h3_video_api.json. "drama" always applies; "action" only when intensity is not off,
# with strength = template strength x multiplier
LORAS_DRAMA = ("53", "400", "404")
LORAS_ACTION = ("342", "344", "343")
INTENSITY = {"low": 0.6, "mid": 1.0, "high": 1.3}
# model_set "official": Comfy-Org's own MiniMax H3 weights instead of the community merge and its turbo LoRA.
# The weights follow the task (see references/official.md): reference and text clips use the Ref2VA model,
# first/last-frame clips (I2VA／L2VA／FL2VA) the FL2VA model with its own turbo and flex LoRA.
# The MysticXXX LoRA (node 400) is bypassed, so an official-only batch never needs that file.
N_UNET, N_TURBO, N_MYSTIC, N_FLEX = "192", "404", "400", "53"
OFFICIAL = {
    "ref": {"unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "turbo": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            "flex": "pika_h3_flex_ref2va_v1_bf16.safetensors"},
    "fl": {"unet": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
           "turbo": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
           "flex": "pika_h3_flex_fl2va_v1_bf16.safetensors"},
}
FL_TASKS = ("FL2VA", "I2VA", "L2VA")
MODEL_SETS = ("default", "official")


def official_family(task_or_mode):
    """'fl' for first/last-frame clips, else 'ref' (Ref2VA and text-only clips)."""
    return "fl" if task_or_mode in FL_TASKS or task_or_mode == "frames" else "ref"


def apply_model_set(manifest):
    """A top-level "model_set" is the default for every video job that does not set its own. Returns the manifest."""
    top = manifest.get("model_set")
    if top is not None and top not in MODEL_SETS:
        raise ValueError(f"model_set 只接受 {MODEL_SETS}")
    for j in manifest.get("jobs", []):
        if j.get("type") == "video" and top and "model_set" not in j:
            j["model_set"] = top
    return manifest
# extension (H3 Motion Context): frames of the source tail pinned into the head of the new clip
EXTEND = {"context_length": "22", "audio_context_length": 24, "tail_seconds": 3.0}
# long side of a picture per size key (MiniMax-H3-Image supports up to 4K)
IMAGE_SIZES = {"1K": 1024, "1.3K": 1344, "1.6K": 1664, "2K": 2048, "2.7K": 2720, "3K": 3072, "4K": 4096}
ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3), "21:9": (21, 9)}

# node ids in h3_video_api.json
N_COND, N_SAMPLER, N_SCHED, N_BRIDGE = "265", "397", "261", "402"
N_VAE, N_AVAE, N_CLIP, N_SECONDS = "36", "37", "38", "259"
N_DECODE, N_ADECODE, N_CREATE, N_SAVE = "253", "254", "cv", "sv"
# node ids in h3_image_api.json (MiniMax-H3-Image)
N_IMG_ENC, N_IMG_NOISE, N_IMG_SAVE = "5", "7", "9"

# never allowed in a graph this skill posts: they need extra node packs or only feed a UI preview
FORBIDDEN_CLASS = ("VHS_", "Preview", "ChunkFeedForward", "ResolutionSelector")

LABEL = {"image": "Picture", "video": "Video", "audio": "Audio"}
# @圖片1 / @影片1 / @音訊1, plus @Image1 / @Video1 / @Audio1
LITERAL_RE = re.compile(r"<(Picture|Video|Audio) (\d+)>")
MENTION_RE = re.compile(r"@(圖片|图片|Image|影片|视频|Video|音訊|音频|Audio)\s?(\d+)", re.IGNORECASE)
MENTION_KIND = {"圖片": "image", "图片": "image", "image": "image", "影片": "video", "视频": "video",
                "video": "video", "音訊": "audio", "音频": "audio", "audio": "audio"}
VIDEO_MODES = {"ref": "ref", "frames": "frames", "first_last": "frames", "text": "text"}


def load_template(name, dirs=()):
    """The first `name` found in dirs, then in the skill's workflows/ folder."""
    for d in list(dirs) + [WORKFLOWS]:
        p = Path(d) / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"workflow template not found: {name}")


def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


# ---------------------------------------------------------------- sizes, seeds
def round32(x):
    return max(32, int(round(x / 32)) * 32)


def dims(aspect, resolution, auto_size=None):
    """Video size. aspect "auto" follows auto_size (first frame / extension source) and falls back to 16:9."""
    short = int(str(resolution).rstrip("pP"))
    if aspect == "auto" and auto_size and auto_size[0] and auto_size[1]:
        a, b = auto_size
    else:
        a, b = ASPECTS.get(aspect, (16, 9))
    w, h = (short * a / b, short) if a >= b else (short, short * b / a)
    return round32(w), round32(h)


def image_dims(aspect, size_key, refs):
    """Picture size from the long side; aspect "auto" follows the first reference (1:1 without one)."""
    long_side = IMAGE_SIZES.get(size_key, 1344)
    if aspect == "auto":
        a, b = (refs[0]["width"], refs[0]["height"]) if refs and refs[0].get("width") else (1, 1)
    else:
        a, b = ASPECTS.get(aspect, (1, 1))
    w, h = (long_side, long_side * b / a) if a >= b else (long_side * a / b, long_side)
    return round32(w), round32(h)


def pick_seed(seed, rng=None):
    if seed not in (None, "", -1):
        return int(seed)
    return (rng or random).randint(0, 2**48)


# ---------------------------------------------------------------- prompt text
def resolve_mentions(text, items):
    """Map @圖片N mentions (N = position among the items of that kind) to compact H3 tags.

    Only mentioned items are used, numbered by first mention. Returns (text_with_tags, chosen by kind).
    """
    if not MENTION_RE.search(text):
        # Prompt writers that follow the official H3 guide write <Picture N> / <Video N> / <Audio N> directly.
        # Read those as @-mentions in input order; otherwise every reference would be silently dropped.
        text = LITERAL_RE.sub(lambda m: "@" + {"Picture": "Image"}.get(m.group(1), m.group(1)) + m.group(2), text)
    by_kind = {"image": [], "video": [], "audio": []}
    for item in items:
        by_kind[item["kind"]].append(item)
    chosen = {"image": [], "video": [], "audio": []}
    assigned = {}

    def repl(m):
        kind = MENTION_KIND[m.group(1).lower() if m.group(1).isascii() else m.group(1)]
        n = int(m.group(2))
        pool = by_kind[kind]
        if not 1 <= n <= len(pool):
            raise ValueError(f"找不到參考素材 {m.group(0)}（inputs 裡只有 {len(pool)} 個{kind}）")
        key = (kind, n)
        if key not in assigned:
            chosen[kind].append(pool[n - 1])
            assigned[key] = len(chosen[kind])
        return f"<{LABEL[kind]} {assigned[key]}>"

    return MENTION_RE.sub(repl, text), chosen


def alignment_line(task, duration, last_shot=1):
    """The fixed first line the base guide requires for keyframe tasks."""
    end = f"{duration:.2f}"
    if task == "I2VA":
        return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    if task == "FL2VA":
        return ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
                f"0.00-second mark of the target video; Picture 2 (from Shot {last_shot}) aligns with the "
                f"{end}-second mark of the target video.")
    if task == "L2VA":
        return ("How the reference pictures align with the target video — <Picture 1> (from "
                f"[Shot {last_shot}]) aligns with the {end}-second mark of the target video.")
    return ""


def raw_prompt(text, task, duration, chosen, extending):
    """The prompt sent to H3 (Toybox's path with the rewriter off, which is what this skill always used)."""
    if task == "Ref2VA":
        if extending:
            k = len(chosen["image"])
            text += f"\n<Picture {k}> is the last frame of the previous clip; the video continues directly from it."
        return text
    head = alignment_line(task, duration)
    if not head:
        return text
    # Writers following the base guide often write this line themselves; keep only ours (its seconds are exact).
    first, _, rest = text.lstrip().partition("\n")
    if first.startswith(head[:30]):
        text = rest.lstrip("\n")
    return head + "\n\n" + text


# ---------------------------------------------------------------- video
def bypass(wf, node_id, port="model"):
    """Remove a pass-through node and wire its consumers to its input."""
    upstream = wf[node_id]["inputs"][port]
    for other in wf.values():
        for k, v in other["inputs"].items():
            if v == [node_id, 0]:
                other["inputs"][k] = upstream
    del wf[node_id]


DRIVE_TAG = "\x00DRIVE_AUDIO\x00"
DRIVE_RE = re.compile(r"<Audio 1>|@(音訊|音频|Audio)\s?1(?!\d)", re.IGNORECASE)
DRIVE_MAX_SECONDS = 15.0
DRIVE_REF_PREFIXES = ("ref_images.", "ref_videos.", "ref_video_audios.", "ref_audios.")


def drive_task(task):
    """T8 task_type for an audio-locked clip. T8 adds the recording as a reference (add_source_as_reference),
    so a text-only clip is not T2VA but Ref2VA (its only reference is the recording); T8 refuses T2VA with
    references. Key-frame modes never get here: plan_video refuses frames + drive_audio (see there)."""
    return "Ref2VA"


def plan_video(job, inputs=(), first=None, last=None, extend=None, rng=None, drive=None):
    """Everything about a video job that does not depend on the template.

    inputs: media items of `inputs` in manifest order (ref mode). first / last: media items (frames mode).
    extend: None, or {"width", "height", "has_audio", "last_frame": <image media item>} of the source video.
    drive: None, or the audio media item of `drive_audio` (with "seconds"): the recording is locked into the clip
           through T8 Audio Conditioning (lock_source) and the picture is generated to match it (lip-sync /
           singing). In the prompt that recording is <Audio 1>.
    """
    mode = VIDEO_MODES.get(job.get("mode", "ref"))
    if mode is None:
        raise ValueError(f"未知的 mode：{job.get('mode')}")
    if drive is not None:
        if drive.get("kind") != "audio":
            raise ValueError("drive_audio 必須是聲音檔（wav／mp3／flac…）")
        if extend:
            raise ValueError("drive_audio 不能和 extend 一起用（延伸影片沿用前一段的聲音）")
        if mode == "frames":  # 2026-11-20 tested: T8 then needs Hybrid, and the low-res first pass cannot carry the
            # key frame to full size (SamplerCustomAdvanced: 180 vs 405 tokens). Ref mode with the picture works.
            raise ValueError("drive_audio 不能用頭尾格（frames）模式：改用 ref 模式，把那張首格圖放進 inputs，"
                             "提示詞用 @圖片1 指它（人物和構圖會照參考圖走，嘴型跟著錄音）")
        if mode == "ref" and any(i["kind"] == "audio" for i in inputs):
            raise ValueError("用 drive_audio 時 inputs 不要再放聲音檔：錄音本身就是 <Audio 1>")
        if job.get("seconds") is None:
            if not drive.get("seconds"):
                raise ValueError("讀不到 drive_audio 的長度，請在 job 裡寫 seconds")
            job = dict(job, seconds=min(round(float(drive["seconds"]), 3), DRIVE_MAX_SECONDS))
        if float(job["seconds"]) > DRIVE_MAX_SECONDS + 0.08:
            raise ValueError(f"drive_audio 一段最長 {DRIVE_MAX_SECONDS:g} 秒；較長的歌請先切段，每段一個 job")
    duration = float(job.get("seconds", 10))
    text = job["prompt"]
    if drive is not None:  # <Audio 1> / @音訊1 is the locked recording, not one of the inputs
        text = DRIVE_RE.sub(DRIVE_TAG, text)
    chosen = {"image": [], "video": [], "audio": []}
    if mode == "ref":
        tagged, chosen = resolve_mentions(text, list(inputs))
        if extend:
            chosen["image"].append(extend["last_frame"])
        task = "Ref2VA"
    elif mode == "frames":
        if not first and not last:
            raise ValueError("頭尾格模式至少要放一張首格或尾格")
        for item in (first, last):
            if item:
                if item["kind"] != "image":
                    raise ValueError("首格與尾格必須是圖片")
                chosen["image"].append(item)
        tagged, task = text, "FL2VA" if first and last else "I2VA" if first else "L2VA"
    else:
        tagged, task = text, "T2VA"
    if extend and mode != "ref":
        raise ValueError("延伸只能用 ref 模式")

    auto_size = None
    if extend:
        auto_size = (extend.get("width"), extend.get("height"))
    elif task in ("I2VA", "FL2VA", "L2VA"):
        f0 = chosen["image"][0]
        auto_size = (f0.get("width"), f0.get("height")) if f0.get("width") else None
    width, height = dims(job.get("aspect", "auto"), job.get("resolution", "480P"), auto_size)

    total, low = STEPS["fast" if job.get("fast", True) else "normal"]
    intensity = job.get("intensity", "off")
    unused = [i for i in inputs if i not in chosen[i["kind"]]] if mode == "ref" else []
    final = raw_prompt(tagged, task, duration, chosen, bool(extend)).replace(DRIVE_TAG, "<Audio 1>")
    return {"task": task, "mode": mode, "final_prompt": final,
            "chosen": chosen, "seconds": duration, "width": width, "height": height,
            "steps": int(total), "transition_step": int(low), "seed": pick_seed(job.get("seed"), rng),
            "intensity": intensity, "multiplier": INTENSITY.get(intensity), "extend": bool(extend),
            "drive": drive["name"] if drive else None,
            "model_set": model_set(job),
            "unused_inputs": [i["name"] for i in unused]}


def model_set(job):
    ms = job.get("model_set", "default")
    if ms not in MODEL_SETS:
        raise ValueError(f"model_set 只接受 {MODEL_SETS}")
    return ms


def video_graph(template, plan, tail_name=None):
    """The template with the plan written in. tail_name: the conformed source tail in ComfyUI input (extend)."""
    wf = copy.deepcopy(template)
    width, height = plan["width"], plan["height"]
    chosen = plan["chosen"]
    if plan["task"] == "Ref2VA":
        ref = wf[N_COND]["inputs"]
        for k in [k for k in ref if k.startswith("ref_") and "." in k]:  # autogrow slots only
            del ref[k]
        for i, item in enumerate(chosen["image"]):
            wf[f"ri{i}"] = {"class_type": "LoadImage", "inputs": {"image": item["name"]}}
            ref[f"ref_images.ref_image_{i}"] = [f"ri{i}", 0]
        for i, item in enumerate(chosen["video"]):
            wf[f"rv{i}"] = {"class_type": "LoadVideo", "inputs": {"file": item["name"]}}
            wf[f"rc{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"rv{i}", 0]}}
            ref[f"ref_videos.ref_video_{i}"] = [f"rc{i}", 0]
            if item.get("has_audio") and item.get("use_audio", True):
                ref[f"ref_video_audios.ref_video_audio_{i}"] = [f"rc{i}", 1]
        for i, item in enumerate(chosen["audio"]):
            wf[f"ra{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": item["name"]}}
            ref[f"ref_audios.ref_audio_{i}"] = [f"ra{i}", 0]
        ref["prompt"] = plan["final_prompt"]
        ref["width"], ref["height"] = width, height
    else:  # T2VA / I2VA / FL2VA / L2VA share the stock image-to-video conditioning
        old = wf[N_COND]["inputs"]
        node = {"clip": [N_CLIP, 0], "vae": [N_VAE, 0], "prompt": plan["final_prompt"], "width": width,
                "height": height, "length": old["length"]}
        slots = {"I2VA": ["first_frame"], "FL2VA": ["first_frame", "last_frame"], "L2VA": ["last_frame"]}.get(plan["task"], [])
        for i, (slot, item) in enumerate(zip(slots, chosen["image"])):
            wf[f"kf{i}"] = {"class_type": "LoadImage", "inputs": {"image": item["name"]}}
            wf[f"ks{i}"] = {"class_type": "ImageScale", "inputs": {"image": [f"kf{i}", 0], "upscale_method": "lanczos",
                                                                    "width": width, "height": height, "crop": "center"}}
            node[slot] = [f"ks{i}", 0]
        wf[N_COND] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": node}

    if plan.get("drive"):  # audio lock: same references / key frames, conditioned by T8 with the recording locked
        old = wf[N_COND]["inputs"]
        keep = {k: v for k, v in old.items() if k.startswith(DRIVE_REF_PREFIXES) or k in ("first_frame", "last_frame")}
        wf["da"] = {"class_type": "LoadAudio", "inputs": {"audio": plan["drive"]}}
        wf[N_COND] = {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": {
            "clip": [N_CLIP, 0], "video_vae": [N_VAE, 0], "audio_vae": [N_AVAE, 0], "prompt": plan["final_prompt"],
            "width": width, "height": height, "length": old["length"], "task_type": drive_task(plan["task"]),
            "audio_mode": "lock_source", "audio_denoise_strength": 0.0, "add_source_as_reference": True,
            "prompt_primary_audio_ordinal": 1, "strict_prompt_tags": True, "ref_image_size": "match",
            "reference_video_policy": "official_2_to_15s", "allow_above_reference_area": True,
            "drive_audio": ["da", 0], **keep}}
        wf[N_CREATE]["inputs"]["audio"] = [N_COND, 2]  # mux_audio: the original recording, trimmed to the clip
        # Sampling = the owner's proven audio-lock route (等級一的我 / 宇宙甜點 MVs): T8 only builds the locked
        # conditioning; two plain SamplerCustomAdvanced passes do the work. SelfLiftH3Sampler rejects T8's
        # video+audio noise mask, and T8's own Avatar sampler aborts on 32 GB cards at its VRAM boundary check.
        lw, lh = drive_low_size(width, height)
        wf[N_COND]["inputs"].update(width=lw, height=lh)
        for k in ("ks0", "ks1"):
            if k in wf:
                wf[k]["inputs"].update(width=lw, height=lh)
        s = wf[N_SAMPLER]["inputs"]
        wf["dg"] = {"class_type": "BasicGuider", "inputs": {"model": s["model"], "conditioning": s["positive"]}}
        wf["dn"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": plan["seed"]}}
        wf["dsp"] = {"class_type": "SplitSigmas", "inputs": {"step": plan["transition_step"], "sigmas": s["sigmas"]}}
        wf["ds1"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["dn", 0], "guider": ["dg", 0],
                     "sampler": s["sampler"], "sigmas": ["dsp", 0], "latent_image": [N_COND, 1]}}
        wf["dsep"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["ds1", 1]}}
        wf["dup"] = {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {"latent": ["dsep", 0],
                     "model_name": s["upscaler_model"], "mode": "target dimensions", "mode.width": width,
                     "mode.height": height, "align": 32, "enable_temporal_chunking": True, "force_unload": True,
                     "device": "cuda", "precision": "fp16"}}
        wf["dcat"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["dup", 0], "audio_latent": ["dsep", 1]}}
        wf["dlock"] = {"class_type": "MiniMaxH3AudioLatentControlT8", "inputs": {"mode": "lock", "strength": 0.0,
                       "av_latent": ["dcat", 0], "source_audio": ["da", 0], "audio_vae": [N_AVAE, 0]}}
        wf[N_SAMPLER] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["dn", 0], "guider": ["dg", 0],
                         "sampler": s["sampler"], "sigmas": ["dsp", 1], "latent_image": ["dlock", 0]}}
        for nid, node in wf.items():  # decoders read the denoised output (index 1), not SelfLift's index 0
            for k, v in node["inputs"].items():
                if v == [N_SAMPLER, 0]:
                    node["inputs"][k] = [N_SAMPLER, 1]

    wf[N_SECONDS]["inputs"]["value"] = float(plan["seconds"])
    wf[N_SCHED]["inputs"]["steps"] = plan["steps"]
    if not plan.get("drive"):
        wf[N_SAMPLER]["inputs"]["transition_step"] = plan["transition_step"]
        wf[N_SAMPLER]["inputs"]["seed"] = plan["seed"]

    if plan["extend"]:  # pin the source tail into the head of the clip, then cut the pinned frames off again
        if not tail_name:
            raise ValueError("extend needs the conformed tail file")
        wf["ev"] = {"class_type": "LoadVideo", "inputs": {"file": tail_name}}
        wf["ec"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["ev", 0]}}
        wf["mc"] = {"class_type": "MiniMaxH3MotionContext", "inputs": {
            "conditioning": [N_BRIDGE, 0], "vae": [N_VAE, 0], "latent": [N_COND, 1],
            "context_length": str(EXTEND["context_length"]), "audio_context_length": int(EXTEND["audio_context_length"]),
            "context_frames": ["ec", 0], "audio_vae": [N_AVAE, 0], "context_audio": ["ec", 1]}}
        wf[N_SAMPLER]["inputs"]["positive"] = ["mc", 0]
        wf["mt"] = {"class_type": "MiniMaxH3MotionContextTrim", "inputs": {
            "images": [N_DECODE, 0], "trim_frames": ["mc", 1], "audio": [N_ADECODE, 0], "fps": float(FPS),
            "match_tail": True}}
        wf[N_CREATE]["inputs"]["images"], wf[N_CREATE]["inputs"]["audio"] = ["mt", 0], ["mt", 1]

    mult = plan["multiplier"]
    for node_id in LORAS_ACTION:
        node = wf.get(node_id)
        if node is None:
            continue
        if mult is None:  # intensity off: bypass by rewiring consumers to this LoRA's input model
            bypass(wf, node_id)
        else:
            node["inputs"]["strength_model"] = round(node["inputs"]["strength_model"] * mult, 4)
    if plan.get("model_set") == "official":
        o = OFFICIAL[official_family(plan["task"])]
        wf[N_UNET]["inputs"]["unet_name"] = o["unet"]
        wf[N_TURBO]["inputs"]["lora_name"] = o["turbo"]
        wf[N_FLEX]["inputs"]["lora_name"] = o["flex"]
        bypass(wf, N_MYSTIC)
    return wf


# ---------------------------------------------------------------- picture (P-H3-Image on the H3 model)
def plan_picture(job, inputs=(), rng=None):
    """P-H3 reads <Picture N> itself. With @ mentions only the mentioned pictures are used (in mention order);
    without any, every input picture is used in order. Picture 1 is the one edited."""
    text = job["prompt"]
    pictures = [i for i in inputs if i["kind"] == "image"]
    if MENTION_RE.search(text):
        tagged, chosen = resolve_mentions(text, pictures)
        refs = chosen["image"]
    else:
        tagged, refs = text, pictures
    if len(refs) > 16:
        raise ValueError("MiniMax-H3-Image 編輯與參考最多 16 張圖（第 1 張是要改的那張）")
    size = str(job.get("size", "2K")).upper()
    w, h = image_dims(job.get("aspect", "auto"), size, refs)
    return {"task": "Edit" if refs else "T2I", "final_prompt": tagged, "refs": refs, "width": w, "height": h,
            "seed": pick_seed(job.get("seed"), rng), "size": size}


def picture_graph(template, plan):
    g = copy.deepcopy(template)
    enc = g[N_IMG_ENC]["inputs"]
    for k in [k for k in enc if k.startswith("images.")]:
        del enc[k]
    enc["prompt"], enc["width"], enc["height"] = plan["final_prompt"], plan["width"], plan["height"]
    noise_inputs = g[N_IMG_NOISE]["inputs"]
    noise_inputs["seed" if "seed" in noise_inputs else "noise_seed"] = plan["seed"]
    for i, item in enumerate(plan["refs"], 1):  # image_1 is the picture edited, the rest are references, in order
        g[f"ir{i}"] = {"class_type": "LoadImage", "inputs": {"image": item["name"]}}
        enc[f"images.image_{i}"] = [f"ir{i}", 0]
    return g


# ---------------------------------------------------------------- TTS (32x32 H3, audio only)
# H3 is least stable in the first ~0.5 s of a clip, and a clip asked to talk from 0.00 s puts the first words exactly
# there: they get eaten, or a filler syllable ("啊／呀／然後") appears before them. Spare time does the same: the model
# fills it at the head. The fix (measured 2026-08-31: 10/15 -> 10/10 clean) is a sacrificial lead-in: a harmless
# sound at the head that we throw away.
# 2026-11-09: glued to the line ("嗯……最近…") it comes out as a hum running straight into the words, and no level
# rule can tell hum from speech (cuts landed both in the hum and in the first word). As its own line followed by a
# written pause it leaves a real silence, and the cut is that silence.
TTS_LEAD = "嗯。"
TTS_LEAD_PAUSE = " She pauses for a short silent breath, then continues: "
TTS_DENSITY = 4.6          # characters per second the line is given (denser = more head-eating)
TTS_LEAD_SECONDS = 1.0     # lead + the silence after it

D_RE = re.compile(r"<d>\s*\[([^\]]+)\]\s*(.*?)</d>", re.S)
CHINESE_TAGS = ("mandarin", "chinese", "taiwanese mandarin", "cantonese")


def tts_prompt(prompt, lead=True, convert=None):
    """Dialogue in Simplified Chinese (Traditional is misread more often) and the lead-in as its own line before the
    first one. convert: a Traditional->Simplified function (OpenCC tw2s) or None. Returns (prompt, dialogue chars)."""
    first = [True]
    chars = [0]

    def repl(m):
        tag, text = m.group(1), m.group(2)
        if tag.strip().lower() in CHINESE_TAGS and convert:
            text = convert(text)
        chars[0] += len(re.sub(r"[\s，。！？、；：,.!?…—「」『』（）()]", "", text))
        line = f"<d>[{tag}] {text}</d>"
        if lead and first[0]:
            line = f"<d>[{tag}] {TTS_LEAD}</d>{TTS_LEAD_PAUSE}" + line
        first[0] = False
        return line

    return D_RE.sub(repl, prompt), chars[0]


def tts_seconds(chars, given=None, lead=True):
    """Seconds for a TTS clip: the caller's value, or the dialogue at TTS_DENSITY plus a little air (2–15 s)."""
    if given:
        return float(given)
    return round(min(15.0, max(2.0, chars / TTS_DENSITY + 0.4 + (TTS_LEAD_SECONDS if lead else 0))), 2)


def lead_cut(db, hop_s=0.01):
    """End of the silence after the lead-in, from a level curve (dB per hop).

    The lead is the first stretch of sound lasting >= 0.2 s that starts 0.15–1.2 s in (a click or breath already there
    at 0.00 s is an onset artifact, measured 2026-11-09: 1 of 15 clips had one); the silence after it must last >= 0.08 s. The cut is 40 ms before the line resumes.
    None when there is no such pattern (the lead was eaten, or ran into the line): listen to the head."""
    if len(db) < 20:
        return None
    top = sorted(db)[int(len(db) * 0.9)]
    on = [d > top - 20 for d in db]
    runs, k = [], 0
    while k < len(on):                       # (start, end, is_on) runs
        e = k
        while e < len(on) and on[e] == on[k]:
            e += 1
        runs.append((k, e, on[k])); k = e
    for n, (s0, e0, is_on) in enumerate(runs):
        if s0 * hop_s > 1.2:
            break
        if is_on and (e0 - s0) * hop_s >= 0.2 and s0 * hop_s >= 0.15:   # a sound already there at 0.00 s is an onset artifact
            for ps, pe, p_on in runs[n + 1:]:          # the first real pause after the lead
                if not p_on and (pe - ps) * hop_s >= 0.08 and pe < len(on):
                    return max(0.0, pe * hop_s - 0.04) if pe * hop_s <= 2.0 else None
                if p_on and (pe - ps) * hop_s >= 0.5:  # ran into a long phrase without a pause: lead glued on
                    return None
            return None
    return None


# ---------------------------------------------------------------- fit file names to what this ComfyUI lists
def combo_options(spec):
    """The option list of an /object_info input spec, or None when the input is not a list choice."""
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def fit_combo_names(g, object_info):
    """Rewrite model/file names so they match this ComfyUI's own option list, in place.

    Subfolder separators differ between machines and between nodes (one lists `t8_compat/x`, another `H3\\x`), and
    an agent may "fix" a name into the other form. A name that is not in the list is matched by swapping the
    separator, then by a unique file name. Returns (changes, still_missing) as lists of strings."""
    changes, missing = [], []
    for nid, node in g.items():
        info = object_info.get(node.get("class_type"), {}).get("input", {})
        specs = {**info.get("required", {}), **info.get("optional", {})}
        for key, val in node.get("inputs", {}).items():
            opts = combo_options(specs.get(key))
            if not isinstance(val, str) or not opts or val in opts:
                continue
            want = val.replace("\\", "/")
            hit = [o for o in opts if isinstance(o, str) and o.replace("\\", "/") == want]
            if not hit:
                name = want.rsplit("/", 1)[-1]
                hit = [o for o in opts if isinstance(o, str) and o.replace("\\", "/").rsplit("/", 1)[-1] == name]
            if len(hit) == 1:
                node["inputs"][key] = hit[0]
                changes.append(f"{nid}.{key}: {val} -> {hit[0]}")
            elif any(val.endswith(ext) for ext in (".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf")):
                missing.append(f"{nid}.{key}: {val}")
    return changes, missing


# ---------------------------------------------------------------- checks (before posting, and on what really ran)
def check_graph(g):
    """Structural problems of an API graph: forbidden classes, missing class_type, dangling links."""
    problems = []
    for nid, node in g.items():
        if not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict):
            problems.append(f"node {nid}: not an API node (class_type + inputs)")
            continue
        ct = node["class_type"]
        if any(f in ct for f in FORBIDDEN_CLASS):
            problems.append(f"node {nid}: forbidden class {ct}")
        for port, v in node["inputs"].items():
            if is_link(v) and v[0] not in g:
                problems.append(f"node {nid}.{port} -> missing node {v[0]}")
    return problems


def _source(g, link, cls):
    node = g.get(link[0], {}) if is_link(link) else {}
    return node if node.get("class_type") == cls else None


def drive_low_size(width, height):
    """Audio lock samples at 2/3 size and lifts x1.5 to the target, like the owner's workflow (multiple of 32)."""
    return max(32, round(width / 1.5 / 32) * 32), max(32, round(height / 1.5 / 32) * 32)


def check_video(g, plan, tail_name=None):
    """Does this graph (built, or read back from /history) really carry the plan?"""
    problems = check_graph(g)
    if problems:
        return problems
    cond = g.get(N_COND, {})
    ci = cond.get("inputs", {})
    want_class = "MiniMaxH3ReferenceToVideo" if plan["task"] == "Ref2VA" else "MiniMaxH3ImageToVideo"
    if plan.get("drive"):
        want_class = "MiniMaxH3AudioConditioningT8"
        if ci.get("audio_mode") != "lock_source" or ci.get("task_type") != drive_task(plan["task"]):
            problems.append(f"audio lock: mode {ci.get('audio_mode')} / task {ci.get('task_type')} != lock_source / {drive_task(plan['task'])}")
        src = _source(g, ci.get("drive_audio"), "LoadAudio")
        if not src or src["inputs"].get("audio") != plan["drive"]:
            problems.append(f"audio lock: drive_audio does not load {plan['drive']}")
        if g.get(N_CREATE, {}).get("inputs", {}).get("audio") != [N_COND, 2]:
            problems.append("audio lock: CreateVideo is not muxed with the locked recording")
    if cond.get("class_type") != want_class:
        problems.append(f"{N_COND} is {cond.get('class_type')}, expected {want_class}")
    if ci.get("prompt") != plan["final_prompt"]:
        problems.append("prompt in the graph differs from the planned prompt")
    size = drive_low_size(plan["width"], plan["height"]) if plan.get("drive") else (plan["width"], plan["height"])
    if (ci.get("width"), ci.get("height")) != size:
        problems.append(f"size {ci.get('width')}x{ci.get('height')} != {size[0]}x{size[1]}")
    if plan["task"] == "Ref2VA":
        expect = {"image": ("ref_images.ref_image_", "LoadImage", "image"),
                  "video": ("ref_videos.ref_video_", "GetVideoComponents", None),
                  "audio": ("ref_audios.ref_audio_", "LoadAudio", "audio")}
        for kind, (prefix, cls, field) in expect.items():
            slots = sorted(k for k in ci if k.startswith(prefix))
            if len(slots) != len(plan["chosen"][kind]):
                problems.append(f"{kind} ref slots {len(slots)} != {len(plan['chosen'][kind])} chosen")
                continue
            for i, item in enumerate(plan["chosen"][kind]):
                src = _source(g, ci.get(f"{prefix}{i}"), cls)
                if src is None:
                    problems.append(f"{prefix}{i} is not wired to a {cls}")
                    continue
                if kind == "video":
                    src = _source(g, src["inputs"].get("video"), "LoadVideo")
                    field = "file"
                if src is None or src["inputs"].get(field) != item["name"]:
                    problems.append(f"{prefix}{i} does not load {item['name']}")
    else:
        slots = {"I2VA": ["first_frame"], "FL2VA": ["first_frame", "last_frame"], "L2VA": ["last_frame"]}.get(plan["task"], [])
        for slot in ("first_frame", "last_frame"):
            if (slot in ci) != (slot in slots):
                problems.append(f"{plan['task']}: {slot} {'missing' if slot in slots else 'should not be set'}")
        for slot, item in zip(slots, plan["chosen"]["image"]):
            scale = _source(g, ci.get(slot), "ImageScale")
            load = _source(g, scale["inputs"].get("image"), "LoadImage") if scale else None
            if not load or load["inputs"].get("image") != item["name"]:
                problems.append(f"{slot} does not load {item['name']}")
            elif (scale["inputs"].get("width"), scale["inputs"].get("height")) != size:
                problems.append(f"{slot} is not scaled to the video size")
    if g.get(N_SECONDS, {}).get("inputs", {}).get("value") != float(plan["seconds"]):
        problems.append(f"seconds node {N_SECONDS} != {plan['seconds']}")
    si = g.get(N_SAMPLER, {}).get("inputs", {})
    if plan.get("drive"):
        low = g.get("dsp", {}).get("inputs", {}).get("step")
        seed = g.get("dn", {}).get("inputs", {}).get("noise_seed")
        lock, up = g.get("dlock", {}), g.get("dup", {}).get("inputs", {})
        if g.get(N_SAMPLER, {}).get("class_type") != "SamplerCustomAdvanced" or si.get("latent_image") != ["dlock", 0]:
            problems.append("audio lock: the high-res pass does not start from the re-locked latent")
        if lock.get("class_type") != "MiniMaxH3AudioLatentControlT8" or lock["inputs"].get("mode") != "lock"                 or lock["inputs"].get("source_audio") != ["da", 0]:
            problems.append("audio lock: the recording is not locked back in after the upscale")
        if (up.get("mode.width"), up.get("mode.height")) != (plan["width"], plan["height"]):
            problems.append(f"audio lock: upscale target {up.get('mode.width')}x{up.get('mode.height')} != {plan['width']}x{plan['height']}")
        model = g.get("dg", {}).get("inputs", {}).get("model")
    else:
        low, seed, model = si.get("transition_step"), si.get("seed"), si.get("model")
    if g.get(N_SCHED, {}).get("inputs", {}).get("steps") != plan["steps"] or low != plan["transition_step"]:
        problems.append(f"steps != {plan['steps']}/{plan['transition_step']}")
    if seed != plan["seed"]:
        problems.append("seed differs")
    official = plan.get("model_set") == "official"
    if official:
        fam = official_family(plan["task"]); o = OFFICIAL[fam]
        for nid, key, what in ((N_UNET, "unet_name", "UNet"), (N_TURBO, "lora_name", "turbo LoRA"), (N_FLEX, "lora_name", "flex LoRA")):
            got = g.get(nid, {}).get("inputs", {}).get(key)
            if got != o[{"UNet": "unet", "turbo LoRA": "turbo", "flex LoRA": "flex"}[what]]:
                problems.append(f"official ({fam}): {what} is {got}, not the Comfy-Org {fam} one")
        if N_MYSTIC in g:
            problems.append("official: MysticXXX LoRA is still in the graph")
    for nid in LORAS_DRAMA:
        if nid not in g and not (official and nid == N_MYSTIC):
            problems.append(f"drama LoRA {nid} missing")
    for nid in LORAS_ACTION:
        if (nid in g) != (plan["multiplier"] is not None):
            problems.append(f"action LoRA {nid} {'missing' if plan['multiplier'] is not None else 'still in the graph with intensity off'}")
    # the model chain must still reach the UNet from the sampler
    seen, link = set(), model
    while is_link(link) and link[0] not in seen:
        seen.add(link[0])
        node = g.get(link[0], {})
        if node.get("class_type") == "UNETLoader":
            break
        link = node.get("inputs", {}).get("model")
    else:
        problems.append("sampler model chain does not reach the UNETLoader")
    mc = g.get("mc")
    if plan["extend"]:
        if not mc or si.get("positive") != ["mc", 0]:
            problems.append("extend: sampler positive is not the MotionContext")
        ev = _source(g, (mc or {}).get("inputs", {}).get("context_frames"), "GetVideoComponents")
        ev = _source(g, ev["inputs"].get("video"), "LoadVideo") if ev else None
        if tail_name and (not ev or ev["inputs"].get("file") != tail_name):
            problems.append("extend: MotionContext does not load the source tail")
        if g.get(N_CREATE, {}).get("inputs", {}).get("images") != ["mt", 0]:
            problems.append("extend: CreateVideo is not fed by MotionContextTrim")
    elif mc:
        problems.append("MotionContext present without extend")
    save = g.get(N_SAVE, {})
    if save.get("class_type") != "SaveVideo" or save.get("inputs", {}).get("video") != [N_CREATE, 0]:
        problems.append("SaveVideo is not wired to CreateVideo")
    return problems


def check_picture(g, plan):
    """Read back the graph: a LoadImage that exists but is not wired into the encoder's dynamic `images` input
    passes every other check and silently turns an edit into plain text-to-image."""
    problems = check_graph(g)
    if problems:
        return problems
    enc = g.get(N_IMG_ENC, {})
    ei = enc.get("inputs", {})
    if ei.get("prompt") != plan["final_prompt"]:
        problems.append("prompt in the graph differs from the planned prompt")
    if (ei.get("width"), ei.get("height")) != (plan["width"], plan["height"]):
        problems.append("picture size differs")
    slots = [k for k in ei if k.startswith("images.")]
    if len(slots) != len(plan["refs"]):
        problems.append(f"picture slots {len(slots)} != {len(plan['refs'])} references")
    for i, item in enumerate(plan["refs"], 1):
        src = _source(g, ei.get(f"images.image_{i}"), "LoadImage")
        if not src or src["inputs"].get("image") != item["name"]:
            problems.append(f"參考圖第 {i} 張沒有接進 {enc.get('class_type', '編碼節點')} 的 images.image_{i}")
    if g.get(N_IMG_NOISE, {}).get("inputs", {}).get("seed" if "seed" in g.get(N_IMG_NOISE, {}).get("inputs", {}) else "noise_seed") != plan["seed"]:
        problems.append("seed differs")
    return problems


def required_classes(pictures=True, dirs=()):
    """Every node class a run can post (templates + nodes added at build time), for `h3local.py check`."""
    item = lambda kind, n: {"kind": kind, "name": n, "width": 640, "height": 360, "has_audio": True}
    t = load_template(VIDEO_TEMPLATE, dirs)
    graphs = []
    for job, kw, tail in (
            ({"mode": "ref", "prompt": "@圖片1 @影片1 @音訊1", "intensity": "mid"},
             {"inputs": [item("image", "a"), item("video", "b"), item("audio", "c")],
              "extend": {"width": 640, "height": 360, "last_frame": item("image", "l")}}, "t"),
            ({"mode": "frames", "prompt": "x"}, {"first": item("image", "a"), "last": item("image", "b")}, None),
            ({"mode": "ref", "prompt": "@圖片1 <Audio 1>"},
             {"inputs": [item("image", "a")], "drive": dict(item("audio", "d"), seconds=5.0)}, None)):
        graphs.append(video_graph(t, plan_video(job, **kw), tail))
    if pictures:
        p = plan_picture({"prompt": "x"}, [item("image", "a")])
        graphs.append(picture_graph(load_template(PICTURE_TEMPLATE, dirs), p))
    return sorted({n["class_type"] for g in graphs for n in g.values()})
