#!/usr/bin/env python3
"""
Generate a professional VoxCPM2 voice-design showcase.

Examples:

    python examples/demo_voices.py

    python examples/demo_voices.py --list-voices

    python examples/demo_voices.py \
        --model ./pretrained_models/VoxCPM2 \
        --output-dir outputs/pro_voice_showcase \
        --voices booming_radio cinematic_trailer luxury_brand

    python examples/demo_voices.py \
        --model ./pretrained_models/VoxCPM2 \
        --reference-audio examples/reference_speaker.wav \
        --voices warm_narrator

    python examples/demo_voices.py --local-files-only --no-optimize
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class VoiceSample:
    name: str
    control: str
    text: str

    @property
    def prompt(self) -> str:
        return f"({self.control}){self.text}"


VOICE_SAMPLES = [
    VoiceSample(
        name="booming_radio",
        control=(
            "A deep adult male radio announcer with a big booming broadcast voice, "
            "rich bass, confident delivery, crisp articulation, and dramatic warmth"
        ),
        text=(
            "You're listening to the VoxCPM voice gallery, broadcasting with a powerful, "
            "larger-than-life radio voice."
        ),
    ),
    VoiceSample(
        name="cinematic_trailer",
        control=(
            "A thunderous adult male cinematic trailer voice, very deep bass, wide dramatic presence, "
            "slow deliberate pacing, intense confidence, clean studio polish"
        ),
        text=(
            "In a world shaped by sound, one voice can turn a simple sentence into an event."
        ),
    ),
    VoiceSample(
        name="luxury_brand",
        control=(
            "A smooth adult male luxury brand narrator, refined baritone, intimate close-mic delivery, "
            "quiet confidence, precise diction, elegant and expensive"
        ),
        text=(
            "Designed for the moments when detail matters, this voice feels polished, calm, and quietly premium."
        ),
    ),
    VoiceSample(
        name="premium_audiobook",
        control=(
            "A mature adult woman audiobook narrator, warm textured voice, emotionally intelligent, "
            "clear characterful phrasing, gentle breath, literary and immersive"
        ),
        text=(
            "She paused at the edge of the platform, listening as the city softened into a distant hum."
        ),
    ),
    VoiceSample(
        name="news_anchor",
        control=(
            "A professional national news anchor, composed adult voice, neutral accent, measured urgency, "
            "authoritative clarity, broadcast-ready pacing"
        ),
        text=(
            "Tonight, researchers announced a new benchmark for expressive speech synthesis across many languages."
        ),
    ),
    VoiceSample(
        name="podcast_host",
        control=(
            "A charismatic adult male podcast host, conversational close-mic sound, warm smile, "
            "natural pauses, relaxed confidence, modern studio tone"
        ),
        text=(
            "Welcome back. Today we are comparing how voice design changes the feeling of the same technology."
        ),
    ),
    VoiceSample(
        name="tech_explainer",
        control=(
            "A polished adult woman technology explainer, bright but not salesy, crisp articulation, "
            "trustworthy product-demo tone, efficient and clear"
        ),
        text=(
            "With one prompt, you can move from narration to advertising, education, support, or storytelling."
        ),
    ),
    VoiceSample(
        name="calm_meditation",
        control=(
            "A serene adult woman meditation guide, soft low voice, slow pacing, long relaxed vowels, "
            "gentle reassurance, quiet room intimacy"
        ),
        text=(
            "Take a slow breath in, let your shoulders settle, and allow the sound to become a calm point of focus."
        ),
    ),
    VoiceSample(
        name="sports_promo",
        control=(
            "A high-energy adult male sports promo announcer, powerful chest voice, punchy rhythm, "
            "stadium excitement, sharp consonants, adrenaline and momentum"
        ),
        text=(
            "This is the moment before the lights come up, the crowd leans in, and everything is on the line."
        ),
    ),
    VoiceSample(
        name="friendly_support",
        control=(
            "A friendly adult woman customer support voice, patient, reassuring, clean diction, "
            "helpful tone, approachable without sounding scripted"
        ),
        text=(
            "I can walk you through it step by step, and we will make sure everything is working before we finish."
        ),
    ),
    VoiceSample(
        name="science_documentary",
        control=(
            "A thoughtful adult woman science documentary narrator, calm curiosity, precise delivery, "
            "subtle wonder, grounded authority, smooth educational tone"
        ),
        text=(
            "Inside each signal is a pattern, and inside each pattern is a clue about how machines learn to speak."
        ),
    ),
    VoiceSample(
        name="noir_detective",
        control=(
            "A gravelly adult male noir detective narrator, low smoky tone, dry wit, unhurried pacing, "
            "cinematic shadows, restrained emotion"
        ),
        text=(
            "The case looked simple at first, but simple things rarely stay that way after midnight."
        ),
    ),
    VoiceSample(
        name="animated_storyteller",
        control=(
            "An expressive adult storyteller for family animation, bright warm voice, playful timing, "
            "clear emotional shifts, theatrical but natural"
        ),
        text=(
            "The tiny inventor opened the attic door and found a machine that hummed like it had been waiting."
        ),
    ),
    VoiceSample(
        name="public_service",
        control=(
            "A serious adult male public service announcer, firm but calm, high intelligibility, "
            "official broadcast tone, steady pace, no melodrama"
        ),
        text=(
            "Please review the instructions carefully and keep this information available for everyone nearby."
        ),
    ),
    VoiceSample(
        name="global_airport",
        control=(
            "A clear international airport announcement voice, adult woman, neutral global accent, "
            "precise consonants, calm authority, easy to understand in a noisy space"
        ),
        text=(
            "Passengers for flight two zero seven may now proceed to the gate for priority boarding."
        ),
    ),
    VoiceSample(
        name="sleep_story",
        control=(
            "A soft bedtime storyteller, quiet, slow, intimate, soothing, warm breath, "
            "gentle phrasing, peaceful late-night tone"
        ),
        text=(
            "The room grew still as the last light faded, and the story began in a calm, "
            "unhurried voice."
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VoxCPM2 voice-design demo WAV files.")
    parser.add_argument(
        "--model",
        default="openbmb/VoxCPM2",
        help="Hugging Face model id or local model directory. Default: openbmb/VoxCPM2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/voice_gallery"),
        help="Directory for generated WAV files. Default: outputs/voice_gallery",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        choices=[sample.name for sample in VOICE_SAMPLES],
        help="Optional subset of voice presets to generate.",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voice preset names and exit.",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        help=(
            "Optional reference WAV for controllable voice cloning. "
            "When set, each preset controls style while cloning the reference speaker."
        ),
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
        help="Number of diffusion inference steps. Default: 10",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Maximum generation length. Default: 1024",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Runtime device, for example "cuda", "cuda:0", "mps", or "cpu". Default: auto',
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached model files.",
    )
    parser.add_argument(
        "--with-denoiser",
        action="store_true",
        help="Load the optional denoiser. Voice design does not need it by default.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable torch.compile optimization and warmup. Useful for quick compatibility checks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for repeatable generation.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "voice"


def selected_samples(names: list[str] | None) -> list[VoiceSample]:
    if not names:
        return VOICE_SAMPLES

    samples_by_name = {sample.name: sample for sample in VOICE_SAMPLES}
    return [samples_by_name[name] for name in names]


def main() -> None:
    args = parse_args()

    if args.list_voices:
        for sample in VOICE_SAMPLES:
            print(f"{sample.name}: {sample.control}")
        return

    samples = selected_samples(args.voices)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_audio = None
    if args.reference_audio:
        reference_audio = args.reference_audio.expanduser()
        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    # Keep runtime dependencies lazy so `--help` works before the package is installed.
    import soundfile as sf
    import torch

    from voxcpm import VoxCPM

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print(f"Loading model: {args.model}", file=sys.stderr)
    model = VoxCPM.from_pretrained(
        hf_model_id=args.model,
        load_denoiser=args.with_denoiser,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        optimize=not args.no_optimize,
        device=args.device,
    )

    mode = "controllable cloning" if reference_audio else "voice design"
    print(f"Generating {len(samples)} {mode} sample(s) into {args.output_dir}", file=sys.stderr)
    if reference_audio:
        print(f"Using reference audio: {reference_audio}", file=sys.stderr)

    for index, sample in enumerate(samples, start=1):
        output_path = args.output_dir / f"{index:02d}_{slugify(sample.name)}.wav"
        print(f"[{index}/{len(samples)}] {sample.name}: {sample.control}", file=sys.stderr)

        wav = model.generate(
            text=sample.prompt,
            reference_wav_path=str(reference_audio) if reference_audio else None,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
            max_len=args.max_len,
        )
        sf.write(str(output_path), wav, model.tts_model.sample_rate)

        duration = len(wav) / model.tts_model.sample_rate
        print(f"Saved {output_path} ({duration:.2f}s)", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
