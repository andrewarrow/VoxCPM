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
import os
import readline
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    sample: str

    def prompt(self, text: str) -> str:
        return f"({self.control}){text}"


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
    if args.no_reference:
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
        save_history()


def run_once(args: argparse.Namespace, text: str) -> None:
    model = load_model(args)
    voice = VOICES[args.voice]
    prompt_cache = None
    if not args.no_reference:
        reference_audio = default_reference_audio(args, voice)
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
