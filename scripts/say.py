#!/usr/bin/env python3
"""
VoxCPM-powered `say`-style command.

Examples:

    python say.py "This is a local open-source voice."
    python say.py -v cinematic_trailer "In a world shaped by sound..."
    python say.py -v global_airport --script script.txt --output outputs/script.wav
    python say.py --list-voices
    python say.py -v noir_detective --output outputs/noir_line.wav "The case was simple until midnight."
"""

from __future__ import annotations

import argparse
import json
import os
import readline
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").exists() else SCRIPT_DIR
SRC_DIR = PROJECT_ROOT / "src"
HISTORY_PATH = PROJECT_ROOT / ".say_history"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class Voice:
    name: str
    control: str
    sample: str | None
    base_voice: str | None = None

    def prompt(self, text: str) -> str:
        return f"({self.control}){text}"


@dataclass(frozen=True)
class PerformanceSegment:
    actor: str | None
    text: str
    direction: str
    pause_before_ms: int
    pause_after_ms: int
    overlap_previous_ms: int
    start_ms: int | None
    gain_db: float
    nonverbal: str | None
    duration_ms: int | None


@dataclass(frozen=True)
class RenderedPerformanceSegment:
    segment: PerformanceSegment
    actor: str
    wav: Any


VOICES = {
    voice.name: voice
    for voice in [
        Voice(
            "booming_radio",
            "A deep adult male radio announcer with a big booming broadcast voice, rich bass, "
            "confident delivery, crisp articulation, and dramatic warmth",
            "01_booming_radio.wav",
        ),
        Voice(
            "cinematic_trailer",
            "A thunderous adult male cinematic trailer voice, very deep bass, wide dramatic presence, "
            "slow deliberate pacing, intense confidence, clean studio polish",
            "02_cinematic_trailer.wav",
        ),
        Voice(
            "luxury_brand",
            "A smooth adult male luxury brand narrator, refined baritone, intimate close-mic delivery, "
            "quiet confidence, precise diction, elegant and expensive",
            "03_luxury_brand.wav",
        ),
        Voice(
            "friendly_support",
            "A friendly adult woman customer support voice, patient, reassuring, clean diction, "
            "helpful tone, approachable without sounding scripted",
            "10_friendly_support.wav",
        ),
        Voice(
            "science_documentary",
            "A thoughtful adult woman science documentary narrator, calm curiosity, precise delivery, "
            "subtle wonder, grounded authority, smooth educational tone",
            "11_science_documentary.wav",
        ),
        Voice(
            "noir_detective",
            "A gravelly adult male noir detective narrator, low smoky tone, dry wit, unhurried pacing, "
            "cinematic shadows, restrained emotion",
            "12_noir_detective.wav",
        ),
        Voice(
            "global_airport",
            "A clear international airport announcement voice, adult woman, neutral global accent, "
            "precise consonants, calm authority, easy to understand in a noisy space",
            "15_global_airport.wav",
        ),
        Voice(
            "sleep_story",
            "A soft bedtime storyteller, quiet, slow, intimate, soothing, warm breath, gentle phrasing, "
            "peaceful late-night tone",
            "16_sleep_story.wav",
        ),
    ]
}
BUILTIN_VOICE_NAMES = frozenset(VOICES)
CUSTOM_VOICES_PATH = PROJECT_ROOT / "outputs" / "http" / "voices.json"
PREVIEW_TEXT = "This is a preview of the requested voice."


def voice_metadata(name: str, voice: Voice) -> dict[str, Any]:
    return {
        "name": name,
        "control": voice.control,
        "source": "builtin" if name in BUILTIN_VOICE_NAMES else "custom",
        "base_voice": voice.base_voice,
        "reference_sample": voice.sample,
    }


def normalize_voice_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("voice name has no usable characters.")
    return normalized


def payload_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def payload_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    field_name: str,
) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(float(str(value).strip()))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    return min(maximum, max(minimum, parsed))


def payload_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
    field_name: str,
) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    return min(maximum, max(minimum, parsed))


def build_custom_voice(payload: dict[str, Any], *, name: str = "preview") -> Voice:
    base_voice = str(payload.get("base_voice") or "").strip() or None
    if base_voice is not None:
        base_voice = normalize_voice_name(base_voice)
        if base_voice not in VOICES:
            raise ValueError(f"Unknown base_voice: {base_voice}")

    raw_control = payload.get("control") or payload.get("prompt") or payload.get("description")
    raw_changes = payload.get("changes") or payload.get("direction")
    if raw_control:
        control = normalize_text(str(raw_control))
    elif raw_changes:
        changes = normalize_text(str(raw_changes))
        if base_voice:
            control = f"{VOICES[base_voice].control}, {changes}"
        else:
            control = changes
    else:
        raise ValueError("control, prompt, description, or changes is required.")

    if not control:
        raise ValueError("voice prompt is empty.")

    sample = VOICES[base_voice].sample if base_voice else None
    return Voice(name=name, control=control, sample=sample, base_voice=base_voice)


def load_promoted_voices() -> None:
    if not CUSTOM_VOICES_PATH.exists():
        return
    try:
        data = json.loads(CUSTOM_VOICES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not load custom voices from {CUSTOM_VOICES_PATH}: {exc}", file=sys.stderr)
        return

    for item in data.get("voices", []):
        try:
            name = normalize_voice_name(str(item["name"]))
            if name in BUILTIN_VOICE_NAMES:
                continue
            voice = build_custom_voice(item, name=name)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Skipping invalid custom voice entry: {exc}", file=sys.stderr)
            continue
        VOICES[name] = voice


def save_promoted_voices() -> None:
    voices = [
        {
            "name": name,
            "control": voice.control,
            "base_voice": voice.base_voice,
        }
        for name, voice in sorted(VOICES.items())
        if name not in BUILTIN_VOICE_NAMES
    ]
    CUSTOM_VOICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_VOICES_PATH.write_text(json.dumps({"voices": voices}, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render text with VoxCPM2 and play it like /usr/bin/say.")
    parser.add_argument("text", nargs="*", help="Text to speak. Reads stdin if omitted.")
    parser.add_argument(
        "--script",
        type=Path,
        help="Read one utterance per non-empty line from a text file and render them into one WAV.",
    )
    parser.add_argument(
        "-v",
        "--voice",
        default="booming_radio",
        choices=sorted(VOICES),
        help="Voice preset to use. Default: booming_radio",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional WAV path to keep. If omitted, a temporary file is played and removed.",
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "pretrained_models" / "VoxCPM2"),
        help="Local model path or Hugging Face model id. Default: pretrained_models/VoxCPM2",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pro_voice_showcase",
        help="Directory containing the showcase WAVs used as voice references.",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        help="Override the voice reference WAV. By default, uses the selected showcase sample.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Do pure voice design from the text prompt instead of cloning the showcase sample.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help='Runtime device, for example "mps", "cpu", "cuda", or "cuda:0". Default: mps',
    )
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale. Default: 2.0",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Diffusion inference steps. Default: 10",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Maximum generation length. Default: 1024",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable torch.compile optimization and warmup. Slower first launch, faster repeated use.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Render the WAV but do not play it.",
    )
    parser.add_argument(
        "--player",
        help="Audio playback command. Defaults to afplay on macOS, then ffplay if available.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=9002,
        help="Port for the local HTTP API started by the interactive shell. Default: 9002",
    )
    parser.add_argument(
        "--no-http",
        action="store_true",
        help="Do not start the local HTTP API in interactive shell mode.",
    )
    args = parser.parse_args()
    if args.script and args.text:
        parser.error("--script cannot be combined with positional text.")
    return args


def read_text(words: list[str]) -> str:
    if words:
        return " ".join(words).strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    raise SystemExit("No text provided. Pass text as arguments or pipe text on stdin.")


def read_script(path: Path) -> list[str]:
    script_path = path.expanduser()
    try:
        lines = [
            line.strip()
            for line in script_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError as exc:
        raise SystemExit(f"Script file not found: {script_path}") from exc
    except OSError as exc:
        raise SystemExit(f"Could not read script file {script_path}: {exc}") from exc

    if not lines:
        raise SystemExit(f"Script file has no non-empty lines: {script_path}")
    return lines


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def resolve_player(player: str | None) -> list[str]:
    if player:
        return [player]

    afplay = shutil.which("afplay")
    if afplay:
        return [afplay]

    ffplay = shutil.which("ffplay")
    if ffplay:
        return [ffplay, "-nodisp", "-autoexit", "-loglevel", "error"]

    raise RuntimeError("No audio player found. Install ffmpeg or pass --player.")


def load_model(args: argparse.Namespace) -> Any:
    from voxcpm import VoxCPM

    model_path = Path(args.model).expanduser()
    model_id = str(model_path if model_path.exists() else args.model)
    print(f"Loading VoxCPM2 model: {model_id}", file=sys.stderr)
    return VoxCPM.from_pretrained(
        hf_model_id=model_id,
        load_denoiser=False,
        optimize=args.optimize,
        device=args.device,
    )


def default_reference_audio(args: argparse.Namespace, voice: Voice) -> Path | None:
    if args.no_reference or voice.sample is None:
        return None
    reference_audio = args.reference_audio or args.reference_dir / voice.sample
    reference_audio = reference_audio.expanduser()
    if not reference_audio.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {reference_audio}. "
            "Regenerate outputs/pro_voice_showcase or pass --no-reference."
        )
    return reference_audio


def build_voice_caches(args: argparse.Namespace, model: Any) -> dict[str, dict[str, Any]]:
    if args.no_reference:
        return {}

    caches: dict[str, dict[str, Any]] = {}
    for name in sorted(VOICES):
        voice = VOICES[name]
        reference_audio = default_reference_audio(args, voice)
        if reference_audio is None:
            continue

        print(f"Preloading voice reference: {name} ({reference_audio})", file=sys.stderr)
        caches[name] = model.tts_model.build_prompt_cache(reference_wav_path=str(reference_audio))

    return caches


def generate_audio(
    *,
    model: Any,
    voice: Voice,
    text: str,
    args: argparse.Namespace,
    prompt_cache: dict[str, Any] | None = None,
):
    text = normalize_text(text)
    if not text:
        raise ValueError("Text is empty.")

    target_text = voice.prompt(text)
    if prompt_cache is not None:
        wav, _, _ = model.tts_model.generate_with_prompt_cache(
            target_text=target_text,
            prompt_cache=prompt_cache,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
            max_len=args.max_len,
        )
        return wav.squeeze(0).cpu().numpy()

    reference_audio = None
    if not args.no_reference:
        reference_audio = default_reference_audio(args, voice)

    return model.generate(
        text=target_text,
        reference_wav_path=str(reference_audio) if reference_audio else None,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
    )


def generate_audio_with_features(
    *,
    model: Any,
    voice: Voice,
    text: str,
    args: argparse.Namespace,
    prompt_cache: dict[str, Any] | None = None,
    apply_voice_prompt: bool = True,
) -> tuple[Any, str, Any | None]:
    text = normalize_text(text)
    if not text:
        raise ValueError("Text is empty.")

    target_text = voice.prompt(text) if apply_voice_prompt else text
    if hasattr(model.tts_model, "generate_with_prompt_cache"):
        wav, _, audio_feat = model.tts_model.generate_with_prompt_cache(
            target_text=target_text,
            prompt_cache=prompt_cache,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
            max_len=args.max_len,
        )
        return wav.squeeze(0).cpu().numpy(), target_text, audio_feat

    return (
        generate_audio(
            model=model,
            voice=voice,
            text=text,
            args=args,
            prompt_cache=prompt_cache,
        ),
        target_text,
        None,
    )


def write_and_play(
    *,
    wav,
    sample_rate: int,
    args: argparse.Namespace,
    output_path: Path | None = None,
    fallback_name: str = "speech",
) -> Path:
    import soundfile as sf

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path = output_path
        cleanup = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="voxcpm-say-"))
        wav_path = temp_dir / f"{fallback_name}.wav"
        cleanup = True

    try:
        sf.write(str(wav_path), wav, sample_rate)
        print(f"Saved: {wav_path}", file=sys.stderr)

        if not args.no_play:
            player_cmd = resolve_player(args.player)
            subprocess.run([*player_cmd, str(wav_path)], check=True)

        return wav_path
    finally:
        if cleanup:
            try:
                wav_path.unlink(missing_ok=True)
                wav_path.parent.rmdir()
            except OSError:
                pass


@dataclass
class HttpContext:
    args: argparse.Namespace
    model: Any
    voice_caches: dict[str, dict[str, Any]]
    render_lock: threading.Lock


def http_output_path(filename: str | None = None) -> Path:
    output_dir = PROJECT_ROOT / "outputs" / "http"
    if filename:
        filename = str(filename)
        requested = Path(filename).name
        if requested != filename or requested in {"", ".", ".."}:
            raise ValueError("filename must be a plain file name, not a path.")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", requested).strip("._")
        if not safe_name:
            raise ValueError("filename has no usable characters.")
        if not safe_name.lower().endswith(".wav"):
            safe_name = f"{safe_name}.wav"
        return output_dir / safe_name

    now = time.time()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    millis = int((now % 1) * 1000)
    return output_dir / f"speech-{timestamp}-{millis:03d}.wav"


def render_http_request(context: HttpContext, payload: dict[str, Any]) -> Path:
    text = normalize_text(str(payload.get("text", "")))
    if not text:
        raise ValueError("text is required.")

    voice_name = str(payload.get("voice") or context.args.voice)
    if voice_name not in VOICES:
        raise ValueError(f"Unknown voice: {voice_name}")

    output_path = http_output_path(payload.get("filename"))
    with context.render_lock:
        print(f"HTTP rendering with voice: {voice_name}", file=sys.stderr)
        wav = generate_audio(
            model=context.model,
            voice=VOICES[voice_name],
            text=text,
            args=context.args,
            prompt_cache=voice_prompt_cache(context, VOICES[voice_name]),
        )
        return write_and_play(
            wav=wav,
            sample_rate=context.model.tts_model.sample_rate,
            args=context.args,
            output_path=output_path,
            fallback_name=voice_name,
        )


def voice_prompt_cache(context: HttpContext, voice: Voice) -> dict[str, Any] | None:
    if voice.name in context.voice_caches:
        return context.voice_caches[voice.name]
    if voice.base_voice and voice.base_voice in context.voice_caches:
        return context.voice_caches[voice.base_voice]
    return None


def performance_actor_voice(
    context: HttpContext,
    payload: dict[str, Any],
    actor: dict[str, Any] | None = None,
) -> Voice:
    if actor is None:
        raw_actor = payload.get("actor")
        actor = raw_actor if isinstance(raw_actor, dict) else {}
    voice_name = str(
        actor.get("voice")
        or payload.get("voice")
        or payload.get("base_voice")
        or context.args.voice
    )
    voice_name = normalize_voice_name(voice_name)
    if voice_name not in VOICES:
        raise ValueError(f"Unknown voice: {voice_name}")

    base_voice = VOICES[voice_name]
    actor_name = normalize_voice_name(str(actor.get("name") or payload.get("actor_name") or "performance_actor"))
    identity = normalize_text(
        str(
            actor.get("identity")
            or actor.get("description")
            or payload.get("actor_identity")
            or payload.get("identity")
            or ""
        )
    )
    style = normalize_text(str(actor.get("style") or payload.get("style") or ""))

    control_parts = [
        "Same actor identity across every rendered beat; keep the same speaker, timbre, age, accent, and microphone distance.",
        base_voice.control,
    ]
    if identity:
        control_parts.append(f"Actor identity: {identity}")
    if style:
        control_parts.append(f"Scene performance style: {style}")

    return Voice(
        name=actor_name,
        control=normalize_text(" ".join(control_parts)),
        sample=base_voice.sample,
        base_voice=voice_name,
    )


def scene_actor_voices(context: HttpContext, payload: dict[str, Any]) -> dict[str, Voice]:
    raw_actors = payload.get("actors")
    if raw_actors is None:
        actor_voice = performance_actor_voice(context, payload)
        return {actor_voice.name: actor_voice}

    actor_items: list[dict[str, Any]] = []
    if isinstance(raw_actors, list):
        for item in raw_actors:
            if not isinstance(item, dict):
                raise ValueError("actors entries must be objects.")
            actor_items.append(item)
    elif isinstance(raw_actors, dict):
        for key, value in raw_actors.items():
            if isinstance(value, str):
                actor_items.append({"name": key, "voice": value})
            elif isinstance(value, dict):
                actor_item = dict(value)
                actor_item.setdefault("name", key)
                actor_items.append(actor_item)
            else:
                raise ValueError("actors values must be objects or voice names.")
    else:
        raise ValueError("actors must be a list or object.")

    if not actor_items:
        raise ValueError("actors must not be empty.")

    actors: dict[str, Voice] = {}
    for actor_item in actor_items:
        voice = performance_actor_voice(context, payload, actor_item)
        if voice.name in actors:
            raise ValueError(f"Duplicate actor name: {voice.name}")
        actors[voice.name] = voice

    return actors


def performance_segments(payload: dict[str, Any], *, require_actor: bool = False) -> list[PerformanceSegment]:
    raw_segments = payload.get("segments")
    if raw_segments is None:
        raw_text = normalize_text(str(payload.get("text", "")))
        if not raw_text:
            raise ValueError("segments or text is required.")
        raw_segments = [{"text": item} for item in split_performance_text(raw_text)]

    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("segments must be a non-empty list.")

    default_pause_ms = payload_int(
        payload.get("default_pause_ms"),
        default=450,
        minimum=0,
        maximum=5000,
        field_name="default_pause_ms",
    )
    segments: list[PerformanceSegment] = []
    total = len(raw_segments)
    for index, raw_segment in enumerate(raw_segments):
        segment_payload: dict[str, Any]
        if isinstance(raw_segment, str):
            segment_payload = {"text": raw_segment}
        elif isinstance(raw_segment, dict):
            segment_payload = raw_segment
        else:
            raise ValueError("Each segment must be a string or object.")

        text = normalize_text(str(segment_payload.get("text", "")))
        nonverbal = normalize_text(str(segment_payload.get("nonverbal", ""))).lower() or None
        if not text and not nonverbal:
            raise ValueError(f"Segment {index + 1} text or nonverbal is required.")
        actor_name = segment_payload.get("actor") or segment_payload.get("speaker")
        normalized_actor = normalize_voice_name(str(actor_name)) if actor_name else None
        if require_actor and not normalized_actor:
            raise ValueError(f"Segment {index + 1} actor is required.")

        pause_after_default = 0 if index == total - 1 else default_pause_ms
        pause_after_raw = segment_payload.get("pause_after_ms", segment_payload.get("pause_ms"))
        pause_before_ms = payload_int(
            segment_payload.get("pause_before_ms"),
            default=0,
            minimum=0,
            maximum=5000,
            field_name=f"segments[{index}].pause_before_ms",
        )
        pause_after_ms = payload_int(
            pause_after_raw,
            default=pause_after_default,
            minimum=0,
            maximum=5000,
            field_name=f"segments[{index}].pause_after_ms",
        )
        overlap_previous_ms = payload_int(
            segment_payload.get("overlap_previous_ms"),
            default=0,
            minimum=0,
            maximum=5000,
            field_name=f"segments[{index}].overlap_previous_ms",
        )
        start_ms = None
        if segment_payload.get("start_ms") is not None:
            start_ms = payload_int(
                segment_payload.get("start_ms"),
                default=0,
                minimum=0,
                maximum=3_600_000,
                field_name=f"segments[{index}].start_ms",
            )
        gain_db = payload_float(
            segment_payload.get("gain_db"),
            default=0.0,
            minimum=-36.0,
            maximum=12.0,
            field_name=f"segments[{index}].gain_db",
        )
        duration_ms = None
        if segment_payload.get("duration_ms") is not None:
            duration_ms = payload_int(
                segment_payload.get("duration_ms"),
                default=250,
                minimum=20,
                maximum=5000,
                field_name=f"segments[{index}].duration_ms",
            )
        direction = performance_segment_direction(segment_payload)
        segments.append(
            PerformanceSegment(
                actor=normalized_actor,
                text=text,
                direction=direction,
                pause_before_ms=pause_before_ms,
                pause_after_ms=pause_after_ms,
                overlap_previous_ms=overlap_previous_ms,
                start_ms=start_ms,
                gain_db=gain_db,
                nonverbal=nonverbal,
                duration_ms=duration_ms,
            )
        )

    return segments


def split_performance_text(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def performance_segment_direction(segment: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, label in [
        ("direction", "Acting direction"),
        ("acting", "Acting direction"),
        ("emotion", "Emotion"),
        ("subtext", "Subtext"),
        ("pace", "Pace"),
        ("breath", "Breath"),
        ("volume", "Volume"),
    ]:
        value = normalize_text(str(segment.get(key, "")))
        if value:
            parts.append(f"{label}: {value}")
    return " ".join(parts)


def performance_segment_voice(actor_voice: Voice, segment: PerformanceSegment) -> Voice:
    control = actor_voice.control
    if segment.direction:
        control = (
            f"{control} Current beat only: {segment.direction} "
            "Do not change into a different person; change only this beat's emotion, pace, breath, and intention."
        )
    return Voice(
        name=actor_voice.name,
        control=normalize_text(control),
        sample=actor_voice.sample,
        base_voice=actor_voice.base_voice,
    )


def performance_prompt_mode(payload: dict[str, Any]) -> str:
    mode = normalize_text(str(payload.get("prompt_mode") or "off")).lower()
    aliases = {
        "false": "off",
        "none": "off",
        "dialogue": "off",
        "true": "beat",
        "directions": "beat",
        "acting": "beat",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"off", "voice", "beat"}:
        raise ValueError("prompt_mode must be off, voice, or beat.")
    return mode


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20)


def apply_clip_shape(wav: Any, sample_rate: int, *, gain_db: float, fade_ms: int) -> Any:
    import numpy as np

    shaped = np.asarray(wav, dtype=np.float32).copy()
    shaped *= db_to_gain(gain_db)

    fade_samples = min(len(shaped) // 2, int(sample_rate * fade_ms / 1000))
    if fade_samples > 1:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        shaped[:fade_samples] *= fade_in
        shaped[-fade_samples:] *= fade_out

    return shaped


def synthetic_nonverbal(kind: str, *, sample_rate: int, duration_ms: int | None) -> Any:
    import numpy as np

    def sine_envelope(length: int, power: float = 1.0) -> Any:
        values = np.sin(np.linspace(0.0, np.pi, length, dtype=np.float32))
        values = np.clip(values, 0.0, 1.0)
        return values**power

    normalized = kind.strip().lower().replace("-", "_")
    default_ms = {
        "breath": 360,
        "inhale": 300,
        "exhale": 420,
        "sigh": 700,
        "mouth_click": 90,
        "click": 70,
        "swallow": 220,
    }.get(normalized)
    if default_ms is None:
        raise ValueError(f"Unknown nonverbal: {kind}")

    length = max(1, int(sample_rate * (duration_ms or default_ms) / 1000))
    rng = np.random.default_rng(abs(hash((normalized, length))) % (2**32))
    t = np.arange(length, dtype=np.float32) / sample_rate

    if normalized in {"mouth_click", "click"}:
        wav = np.zeros(length, dtype=np.float32)
        burst = min(length, max(24, int(sample_rate * 0.018)))
        envelope = np.linspace(1.0, 0.0, burst, dtype=np.float32) ** 2.6
        wav[:burst] = rng.normal(0, 0.22, burst).astype(np.float32) * envelope
        tick = min(length, int(sample_rate * 0.003))
        if tick > 1:
            wav[:tick] += np.sin(np.linspace(0, np.pi * 8, tick, dtype=np.float32)) * 0.12
        return wav

    if normalized == "swallow":
        wav = np.zeros(length, dtype=np.float32)
        thump = np.sin(2 * np.pi * 95 * t) * np.exp(-t * 15.0) * 0.10
        throat = rng.normal(0, 1.0, length).astype(np.float32)
        kernel = np.ones(max(8, int(sample_rate * 0.004)), dtype=np.float32)
        kernel /= kernel.sum()
        throat = np.convolve(throat, kernel, mode="same").astype(np.float32)
        envelope = sine_envelope(length, 0.45)
        wav += thump + throat * envelope * 0.18
        click_offset = min(length - 1, int(sample_rate * 0.08))
        click = synthetic_nonverbal("mouth_click", sample_rate=sample_rate, duration_ms=70) * 0.70
        end = min(length, click_offset + len(click))
        wav[click_offset:end] += click[: end - click_offset]
        return wav

    noise = rng.normal(0, 1.0, length).astype(np.float32)
    smoothing = max(8, int(sample_rate * 0.0015))
    kernel = np.ones(smoothing, dtype=np.float32) / smoothing
    wav = np.convolve(noise, kernel, mode="same").astype(np.float32)

    if normalized in {"breath", "inhale"}:
        envelope = np.linspace(0.15, 1.0, length, dtype=np.float32)
        envelope *= np.linspace(1.0, 0.55, length, dtype=np.float32)
        level = 0.42 if normalized == "inhale" else 0.36
    elif normalized == "exhale":
        envelope = np.linspace(1.0, 0.15, length, dtype=np.float32) ** 0.85
        level = 0.44
    elif normalized == "sigh":
        envelope = sine_envelope(length, 0.35)
        voiced = np.sin(2 * np.pi * 170 * t) * np.exp(-t * 2.4) * 0.07
        return wav * envelope * 0.34 + voiced
    else:
        envelope = sine_envelope(length)
        level = 0.22

    return wav * envelope * level


def add_mouth_click(output: Any, *, sample_rate: int, start_sample: int, level: float) -> None:
    import numpy as np

    click = synthetic_nonverbal("mouth_click", sample_rate=sample_rate, duration_ms=80)
    peak = float(np.max(np.abs(click))) or 1.0
    click = (click / peak) * level
    position = max(0, start_sample - int(sample_rate * 0.055))
    end = min(len(output), position + len(click))
    if end > position:
        output[position:end] += click[: end - position]


def room_tone(sample_count: int, *, level_db: float) -> Any:
    import numpy as np

    if sample_count <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = np.random.default_rng(41)
    white = rng.normal(0, 1.0, sample_count).astype(np.float32)
    smoothing = 96
    kernel = np.ones(smoothing, dtype=np.float32) / smoothing
    low = np.convolve(white, kernel, mode="same").astype(np.float32)
    peak = float(np.max(np.abs(low))) or 1.0
    return (low / peak) * db_to_gain(level_db)


def mix_performance_segments(
    rendered_segments: list[RenderedPerformanceSegment],
    *,
    sample_rate: int,
    payload: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np

    raw_mix = payload.get("mix")
    mix = raw_mix if isinstance(raw_mix, dict) else {}
    crossfade_ms = payload_int(
        mix.get("crossfade_ms", payload.get("crossfade_ms")),
        default=18,
        minimum=0,
        maximum=250,
        field_name="mix.crossfade_ms",
    )
    room_enabled = payload_bool(mix.get("room_tone", payload.get("room_tone")))
    room_level_db = payload_float(
        mix.get("room_tone_level_db", payload.get("room_tone_level_db")),
        default=-54.0,
        minimum=-80.0,
        maximum=-24.0,
        field_name="mix.room_tone_level_db",
    )
    mouth_noises = normalize_text(str(mix.get("mouth_noises", payload.get("mouth_noises") or "off"))).lower()
    if mouth_noises not in {"off", "false", "none", "subtle", "medium"}:
        raise ValueError("mix.mouth_noises must be off, subtle, or medium.")
    mouth_probability = 0.0
    mouth_level = 0.0
    if mouth_noises == "subtle":
        mouth_probability = 0.22
        mouth_level = 0.055
    elif mouth_noises == "medium":
        mouth_probability = 0.42
        mouth_level = 0.095

    cursor_ms = 0
    timeline: list[tuple[int, int, RenderedPerformanceSegment, Any]] = []
    metadata: list[dict[str, Any]] = []
    for rendered in rendered_segments:
        segment = rendered.segment
        clip_fade_ms = min(crossfade_ms, 12) if segment.nonverbal and not segment.text else crossfade_ms
        wav = apply_clip_shape(
            rendered.wav,
            sample_rate,
            gain_db=segment.gain_db,
            fade_ms=clip_fade_ms,
        )
        duration_ms = int(round(len(wav) * 1000 / sample_rate))
        if segment.start_ms is not None:
            start_ms = segment.start_ms
        else:
            start_ms = max(0, cursor_ms + segment.pause_before_ms - segment.overlap_previous_ms)
        end_ms = start_ms + duration_ms
        cursor_ms = max(cursor_ms, end_ms) + segment.pause_after_ms
        timeline.append((start_ms, end_ms, rendered, wav))
        metadata.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
            }
        )

    total_samples = max((int(end_ms * sample_rate / 1000) for _, end_ms, _, _ in timeline), default=0)
    output = np.zeros(total_samples, dtype=np.float32)
    if room_enabled:
        output += room_tone(total_samples, level_db=room_level_db)

    rng = np.random.default_rng(29)
    for index, (start_ms, _end_ms, rendered, wav) in enumerate(timeline):
        start_sample = int(start_ms * sample_rate / 1000)
        end_sample = min(len(output), start_sample + len(wav))
        if end_sample <= start_sample:
            continue
        output[start_sample:end_sample] += wav[: end_sample - start_sample]
        if (
            mouth_probability > 0
            and rendered.segment.text
            and start_sample > 0
            and rng.random() < mouth_probability
        ):
            add_mouth_click(output, sample_rate=sample_rate, start_sample=start_sample, level=mouth_level)
        metadata[index]["start_sample"] = start_sample
        metadata[index]["end_sample"] = end_sample

    peak = float(np.max(np.abs(output))) if len(output) else 0.0
    if peak > 0.98:
        output = output * (0.98 / peak)

    return output, metadata


def render_performance(
    context: HttpContext,
    payload: dict[str, Any],
    *,
    actors: dict[str, Voice],
    segments: list[PerformanceSegment],
    default_actor_name: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    import numpy as np

    continuity = normalize_text(str(payload.get("continuity") or "rolling")).lower()
    if continuity not in {"rolling", "reference", "reset"}:
        raise ValueError("continuity must be rolling, reference, or reset.")
    prompt_mode = performance_prompt_mode(payload)

    sample_rate = context.model.tts_model.sample_rate
    rendered_segments: list[RenderedPerformanceSegment] = []
    segment_metadata: list[dict[str, Any]] = []

    with context.render_lock:
        base_caches = {
            actor_name: voice_prompt_cache(context, actor_voice)
            for actor_name, actor_voice in actors.items()
        }
        rolling_caches = dict(base_caches)

        for index, segment in enumerate(segments, start=1):
            actor_name = segment.actor or default_actor_name
            if actor_name is None:
                raise ValueError(f"Segment {index} actor is required.")
            if actor_name not in actors:
                raise ValueError(f"Segment {index} references unknown actor: {actor_name}")
            actor_voice = actors[actor_name]

            if continuity == "rolling":
                prompt_cache = rolling_caches.get(actor_name)
            elif continuity == "reference":
                prompt_cache = base_caches.get(actor_name)
            else:
                prompt_cache = None

            segment_voice = actor_voice
            if prompt_mode == "beat":
                segment_voice = performance_segment_voice(actor_voice, segment)
            print(
                f"HTTP performance segment {index}/{len(segments)} with {actor_name}",
                file=sys.stderr,
            )
            if segment.nonverbal and not segment.text:
                wav = synthetic_nonverbal(
                    segment.nonverbal,
                    sample_rate=sample_rate,
                    duration_ms=segment.duration_ms,
                )
                target_text = ""
                audio_feat = None
            else:
                wav, target_text, audio_feat = generate_audio_with_features(
                    model=context.model,
                    voice=segment_voice,
                    text=segment.text,
                    args=context.args,
                    prompt_cache=prompt_cache,
                    apply_voice_prompt=prompt_mode != "off",
                )
            rendered_segments.append(
                RenderedPerformanceSegment(
                    segment=segment,
                    actor=actor_name,
                    wav=wav,
                )
            )

            if (
                continuity == "rolling"
                and audio_feat is not None
                and hasattr(context.model.tts_model, "merge_prompt_cache")
            ):
                rolling_caches[actor_name] = context.model.tts_model.merge_prompt_cache(
                    rolling_caches.get(actor_name),
                    target_text,
                    audio_feat,
                )

            segment_metadata.append(
                {
                    "index": index,
                    "actor": actor_name,
                    "text": segment.text,
                    "direction": segment.direction,
                    "nonverbal": segment.nonverbal,
                    "pause_before_ms": segment.pause_before_ms,
                    "pause_after_ms": segment.pause_after_ms,
                    "overlap_previous_ms": segment.overlap_previous_ms,
                    "start_ms": segment.start_ms,
                    "gain_db": segment.gain_db,
                }
            )

        if not rendered_segments:
            raise ValueError("No renderable performance segments.")

        mixed_wav, timeline_metadata = mix_performance_segments(
            rendered_segments,
            sample_rate=sample_rate,
            payload=payload,
        )
        for metadata_item, timeline_item in zip(segment_metadata, timeline_metadata):
            metadata_item.update(timeline_item)

        fallback_name = default_actor_name or "scene"
        wav_path = write_and_play(
            wav=mixed_wav,
            sample_rate=sample_rate,
            args=context.args,
            output_path=http_output_path(payload.get("filename")),
            fallback_name=fallback_name,
        )

    return (
        wav_path,
        {
            "actors": {
                actor_name: voice_metadata(actor_name, actor_voice)
                for actor_name, actor_voice in actors.items()
            },
            "continuity": continuity,
            "prompt_mode": prompt_mode,
            "segments": segment_metadata,
        },
    )


def render_performance_preview(context: HttpContext, payload: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    actor_voice = performance_actor_voice(context, payload)
    return render_performance(
        context,
        payload,
        actors={actor_voice.name: actor_voice},
        segments=performance_segments(payload),
        default_actor_name=actor_voice.name,
    )


def render_scene_preview(context: HttpContext, payload: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    actors = scene_actor_voices(context, payload)
    return render_performance(
        context,
        payload,
        actors=actors,
        segments=performance_segments(payload, require_actor=len(actors) > 1),
    )


def render_custom_voice_preview(
    context: HttpContext,
    voice: Voice,
    text: str,
    filename: str | None,
) -> Path:
    text = normalize_text(text)
    if not text:
        raise ValueError("text is required.")

    output_path = http_output_path(filename)
    with context.render_lock:
        print(f"HTTP preview rendering with voice prompt: {voice.control}", file=sys.stderr)
        wav = generate_audio(
            model=context.model,
            voice=voice,
            text=text,
            args=context.args,
            prompt_cache=voice_prompt_cache(context, voice),
        )
        return write_and_play(
            wav=wav,
            sample_rate=context.model.tts_model.sample_rate,
            args=context.args,
            output_path=output_path,
            fallback_name=voice.name,
        )


def promote_http_voice(context: HttpContext, payload: dict[str, Any]) -> Voice:
    raw_name = str(payload.get("name", ""))
    name = normalize_voice_name(raw_name)
    replace = payload_bool(payload.get("replace"))
    if name in BUILTIN_VOICE_NAMES:
        raise ValueError(f"Cannot replace built-in voice: {name}")
    if name in VOICES and not replace:
        raise ValueError(f"Voice already exists: {name}. Pass replace=true to update it.")

    voice = build_custom_voice(payload, name=name)
    VOICES[name] = voice
    if voice.base_voice and voice.base_voice in context.voice_caches:
        context.voice_caches[name] = context.voice_caches[voice.base_voice]
    save_promoted_voices()
    return voice


def parse_http_payload(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw_body = handler.rfile.read(content_length) if content_length else b""
    body_text = raw_body.decode("utf-8")
    content_type = handler.headers.get("Content-Type", "")

    if "application/json" in content_type or body_text.lstrip().startswith("{"):
        try:
            payload = json.loads(body_text or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    form = parse_qs(body_text, keep_blank_values=True)
    return {key: values[-1] for key, values in form.items()}


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_http_handler(context: HttpContext) -> type[BaseHTTPRequestHandler]:
    class SayRequestHandler(BaseHTTPRequestHandler):
        server_version = "VoxCPMSayHTTP/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"HTTP {self.address_string()} - {format % args}", file=sys.stderr)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/health":
                send_json(self, 200, {"ok": True, "model_loaded": True})
                return
            if self.path == "/voices":
                send_json(
                    self,
                    200,
                    {"voices": [voice_metadata(name, VOICES[name]) for name in sorted(VOICES)]},
                )
                return
            send_json(
                self,
                404,
                {
                    "error": (
                        "Not found. Use GET /health, GET /voices, POST /say, "
                        "POST /perform/preview, POST /scene/preview, POST /voices/preview, or POST /voices/promote."
                    )
                },
            )

        def do_POST(self) -> None:
            try:
                payload = parse_http_payload(self)
                if self.path == "/say":
                    wav_path = render_http_request(context, payload)
                    send_json(self, 200, {"ok": True, "path": str(wav_path)})
                    return

                if self.path in {"/perform/preview", "/perform/say"}:
                    wav_path, performance = render_performance_preview(context, payload)
                    send_json(
                        self,
                        200,
                        {
                            "ok": True,
                            "path": str(wav_path),
                            "performance": performance,
                        },
                    )
                    return

                if self.path in {"/scene/preview", "/scene/say"}:
                    wav_path, performance = render_scene_preview(context, payload)
                    send_json(
                        self,
                        200,
                        {
                            "ok": True,
                            "path": str(wav_path),
                            "scene": performance,
                        },
                    )
                    return

                if self.path == "/voices/preview":
                    voice = build_custom_voice(payload)
                    text = str(payload.get("text") or PREVIEW_TEXT)
                    wav_path = render_custom_voice_preview(context, voice, text, payload.get("filename"))
                    send_json(
                        self,
                        200,
                        {
                            "ok": True,
                            "path": str(wav_path),
                            "voice": voice_metadata(voice.name, voice),
                        },
                    )
                    return

                if self.path == "/voices/promote":
                    voice = promote_http_voice(context, payload)
                    response: dict[str, Any] = {
                        "ok": True,
                        "voice": voice_metadata(voice.name, voice),
                        "registry": str(CUSTOM_VOICES_PATH),
                    }
                    if payload.get("text"):
                        wav_path = render_custom_voice_preview(
                            context,
                            voice,
                            str(payload["text"]),
                            payload.get("filename"),
                        )
                        response["path"] = str(wav_path)
                    send_json(self, 200, response)
                    return

                send_json(
                    self,
                    404,
                    {"error": "Not found. Use POST /say, /perform/preview, /scene/preview, /voices/preview, or /voices/promote."},
                )
            except ValueError as exc:
                send_json(self, 400, {"error": str(exc)})
                return
            except Exception as exc:
                print(f"HTTP error: {exc}", file=sys.stderr)
                send_json(self, 500, {"error": str(exc)})
                return

    return SayRequestHandler


def start_http_server(context: HttpContext, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_http_handler(context))
    thread = threading.Thread(target=server.serve_forever, name="voxcpm-say-http", daemon=True)
    thread.start()
    actual_port = server.server_address[1]
    print(f"HTTP API listening on http://127.0.0.1:{actual_port}", file=sys.stderr)
    print(
        'POST /say with JSON: {"voice":"global_airport","text":"Hello","filename":"hello.wav"}',
        file=sys.stderr,
    )
    print(
        'POST /voices/preview with JSON: {"base_voice":"global_airport","changes":"warmer and slower","text":"Preview."}',
        file=sys.stderr,
    )
    print(
        'POST /perform/preview with JSON: {"actor":{"voice":"friendly_support","identity":"naturalistic film actor"},"segments":[{"text":"Please.","direction":"whispered, afraid","pause_after_ms":700}],"prompt_mode":"off"}',
        file=sys.stderr,
    )
    print(
        'POST /scene/preview with JSON: {"actors":{"mara":"friendly_support","david":"noir_detective"},"segments":[{"actor":"mara","text":"Please."},{"actor":"david","text":"I am here."}]}',
        file=sys.stderr,
    )
    return server


def print_repl_help() -> None:
    print(
        "\nCommands:\n"
        "  /voice NAME       Change voice\n"
        "  /voice            Show current voice\n"
        "  /voices           List voices\n"
        "  /save PATH TEXT   Render TEXT and keep it at PATH\n"
        "  /help             Show this help\n"
        "  /quit             Exit\n"
        "\nType any other line to render and play it.\n"
    )


def configure_history() -> None:
    readline.set_history_length(1000)
    try:
        readline.read_history_file(HISTORY_PATH)
    except FileNotFoundError:
        pass


def save_history() -> None:
    try:
        readline.write_history_file(HISTORY_PATH)
    except OSError:
        pass


def run_shell(args: argparse.Namespace) -> None:
    model = load_model(args)
    voice_caches = build_voice_caches(args, model)
    render_lock = threading.Lock()
    http_server = None
    if not args.no_http:
        http_server = start_http_server(
            HttpContext(
                args=args,
                model=model,
                voice_caches=voice_caches,
                render_lock=render_lock,
            ),
            args.http_port,
        )
    player = resolve_player(args.player) if not args.no_play else None
    if player:
        print(f"Audio player: {' '.join(player)}", file=sys.stderr)
    configure_history()

    current_voice = VOICES[args.voice]
    print("\nVoxCPM say shell")
    print(f"Current voice: {current_voice.name}")
    print("Type /help for commands, /quit to exit.")

    try:
        while True:
            try:
                line = input(f"{current_voice.name}> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not line:
                continue

            if line in {"/quit", "/exit", "quit", "exit"}:
                return

            if line == "/help":
                print_repl_help()
                continue

            if line == "/voices":
                for name in sorted(VOICES):
                    marker = "*" if name == current_voice.name else " "
                    print(f"{marker} {name}: {VOICES[name].control}")
                continue

            if line.startswith("/voice"):
                parts = line.split(maxsplit=1)
                if len(parts) == 1:
                    print(f"Current voice: {current_voice.name}")
                    continue
                requested = parts[1].strip()
                if requested not in VOICES:
                    print(f"Unknown voice: {requested}")
                    print("Use /voices to list available voices.")
                    continue
                current_voice = VOICES[requested]
                print(f"Current voice: {current_voice.name}")
                continue

            output_path = None
            text = line
            if line.startswith("/save "):
                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    print("Usage: /save PATH TEXT")
                    continue
                output_path = Path(parts[1]).expanduser()
                text = parts[2]

            try:
                print(f"Rendering with voice: {current_voice.name}", file=sys.stderr)
                with render_lock:
                    wav = generate_audio(
                        model=model,
                        voice=current_voice,
                        text=text,
                        args=args,
                        prompt_cache=voice_caches.get(current_voice.name),
                    )
                    write_and_play(
                        wav=wav,
                        sample_rate=model.tts_model.sample_rate,
                        args=args,
                        output_path=output_path,
                        fallback_name=current_voice.name,
                    )
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
    finally:
        if http_server is not None:
            http_server.shutdown()
            http_server.server_close()
        save_history()


def run_once(args: argparse.Namespace, text: str) -> None:
    model = load_model(args)
    voice = VOICES[args.voice]
    prompt_cache = None
    if not args.no_reference:
        reference_audio = default_reference_audio(args, voice)
        if reference_audio is not None:
            print(f"Using reference audio: {reference_audio}", file=sys.stderr)
            prompt_cache = model.tts_model.build_prompt_cache(reference_wav_path=str(reference_audio))

    output_path = args.output.expanduser() if args.output else None
    print(f"Rendering with voice: {voice.name}", file=sys.stderr)
    wav = generate_audio(
        model=model,
        voice=voice,
        text=text,
        args=args,
        prompt_cache=prompt_cache,
    )
    write_and_play(
        wav=wav,
        sample_rate=model.tts_model.sample_rate,
        args=args,
        output_path=output_path,
        fallback_name=voice.name,
    )


def run_script(args: argparse.Namespace, lines: list[str]) -> None:
    import numpy as np

    model = load_model(args)
    voice = VOICES[args.voice]
    prompt_cache = None
    if not args.no_reference:
        reference_audio = default_reference_audio(args, voice)
        if reference_audio is not None:
            print(f"Using reference audio: {reference_audio}", file=sys.stderr)
            prompt_cache = model.tts_model.build_prompt_cache(reference_wav_path=str(reference_audio))

    sample_rate = model.tts_model.sample_rate
    pause = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
    chunks = []
    for index, line in enumerate(lines, start=1):
        print(f"Rendering line {index}/{len(lines)} with voice: {voice.name}", file=sys.stderr)
        wav = generate_audio(
            model=model,
            voice=voice,
            text=line,
            args=args,
            prompt_cache=prompt_cache,
        )
        chunks.append(np.asarray(wav, dtype=np.float32))
        if index != len(lines):
            chunks.append(pause)

    output_path = args.output.expanduser() if args.output else None
    write_and_play(
        wav=np.concatenate(chunks),
        sample_rate=sample_rate,
        args=args,
        output_path=output_path,
        fallback_name=f"{voice.name}_script",
    )


def main() -> None:
    load_promoted_voices()
    args = parse_args()

    if args.list_voices:
        for name in sorted(VOICES):
            print(f"{name}: {VOICES[name].control}")
        return

    if args.script:
        run_script(args, read_script(args.script))
        return

    if not args.text and sys.stdin.isatty() and args.output is None:
        run_shell(args)
        return

    text = read_text(args.text)
    run_once(args, text)


if __name__ == "__main__":
    # Keep tokenizers quiet, matching the repo's Gradio app.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
