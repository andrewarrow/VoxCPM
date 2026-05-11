#!/usr/bin/env python3
"""
Build a TMDB-backed Ken Burns video with VoxCPM narration.

Default output:

    outputs/ken_burns_fury_road/final.mp4

The script reads TMDB credentials using the same helper pattern as
/Users/aa/dev/tmdbapi/find_gems.py, downloads backdrop JPGs from TMDB,
renders narration with a rotating voice per sentence, and uses ffmpeg to
assemble a 3 minute 1080p video.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMDB_HELPER_DIR = Path("/Users/aa/dev/tmdbapi")
TMDB_ENV_FILE = TMDB_HELPER_DIR / ".env"
TMDB_MOVIE_BASE = "https://www.themoviedb.org/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
DEFAULT_MOVIE_ID = 76341
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ken_burns_fury_road"
DEFAULT_TARGET_SECONDS = 180.0
DEFAULT_IMAGE_COUNT = 20
VIDEO_SIZE = "1920x1080"
INTERNAL_SIZE = "3840x2160"
FPS = 60

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import say  # noqa: E402


VOICE_ROTATION = [
    "cinematic_trailer",
    "science_documentary",
    "booming_radio",
    "noir_detective",
    "luxury_brand",
    "global_airport",
    "friendly_support",
    "sleep_story",
]


NARRATION_SENTENCES = [
    "Before Mad Max: Fury Road looked inevitable, it spent years as the impossible movie George Miller refused to abandon.",
    "The idea was simple on paper: a two-hour chase across a poisoned desert, told with almost no traditional exposition.",
    "In practice, that meant hundreds of vehicles, remote locations, practical stunts, dust storms, and a director chasing precision inside chaos.",
    "Miller had already built a mythology from engines and wasteland dust, but this time the machine had to be enormous.",
    "The production finally roared into Namibia, where the landscape gave the film its brutal beauty and punished every logistical shortcut.",
    "Charlize Theron arrived as Furiosa, not as a side character, but as the emotional engine of the whole escape.",
    "Tom Hardy inherited Max from Mel Gibson, then had to play a legend who speaks mostly through damage and reaction.",
    "The stories from set became part of the movie's aura, especially the friction between Hardy and Theron during the punishing shoot.",
    "That tension was not the point of the film, but it fed the legend of a production running hot in every direction.",
    "Instead of building the action later on computers, Miller pushed for real vehicles, real rigs, and real performers in the frame.",
    "The Doof Warrior, the pole cats, the war boys, and the convoy all gave the camera something physical to believe.",
    "Producer Doug Mitchell helped keep the machine moving while the budget, schedule, and weather kept testing the whole enterprise.",
    "Editor Margaret Sixel then faced a mountain of footage and found the movie's secret weapon: absolute visual clarity at insane speed.",
    "The center-framed cutting style made every crash readable, every turn legible, and every burst of madness strangely elegant.",
    "Theron's Furiosa changed the shape of the story, turning a macho road war into a rescue mission with a wounded conscience.",
    "Hardy's Max became less a conquering hero than a haunted witness who slowly remembers how to help.",
    "When the film finally reached theaters in 2015, it felt less like a sequel than a flare fired from another era.",
    "Critics saw craft where other blockbusters were selling noise, and audiences felt the difference in their bones.",
    "The gossip made headlines, but the real scandal was artistic: a studio action film with this much discipline and nerve.",
    "Fury Road worked because Miller treated motion like music, violence like choreography, and spectacle like storytelling.",
    "Every frame seems to say that survival is not clean, heroism is not pretty, and escape is never free.",
    "That is why the movie still feels dangerous: it looks handmade, sunburned, and slightly out of control.",
    "The miracle is that the chaos was not erased in postproduction.",
    "It was organized into one of the great cinematic engines of the century.",
]


@dataclass(frozen=True)
class DownloadedImage:
    source_path: str
    local_path: str
    width: int
    height: int
    vote_average: float
    vote_count: int


@dataclass(frozen=True)
class NarrationLine:
    index: int
    voice: str
    text: str


def narration_lines() -> list[NarrationLine]:
    return [
        NarrationLine(
            index=index,
            voice=VOICE_ROTATION[(index - 1) % len(VOICE_ROTATION)],
            text=text,
        )
        for index, text in enumerate(NARRATION_SENTENCES, start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a VoxCPM/TMDB Ken Burns video.")
    parser.add_argument("--tmdb-helper-dir", type=Path, default=TMDB_HELPER_DIR)
    parser.add_argument("--env-file", type=Path, default=TMDB_ENV_FILE)
    parser.add_argument("--movie-id", type=int, default=DEFAULT_MOVIE_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-count", type=int, default=DEFAULT_IMAGE_COUNT)
    parser.add_argument("--target-seconds", type=float, default=DEFAULT_TARGET_SECONDS)
    parser.add_argument("--fps", type=int, default=FPS, help="Output frame rate. Default: 60")
    parser.add_argument(
        "--motion-renderer",
        choices=("pillow", "zoompan"),
        default="pillow",
        help="Use Pillow affine transforms for subpixel motion, or FFmpeg zoompan.",
    )
    parser.add_argument(
        "--internal-size",
        default=INTERNAL_SIZE,
        help="Oversampled motion canvas before downscaling. Default: 3840x2160",
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "pretrained_models" / "VoxCPM2"),
        help="Local VoxCPM2 model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pro_voice_showcase",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--force-video", action="store_true", help="Rebuild video segments without rerendering audio.")
    parser.add_argument("--force", action="store_true", help="Rebuild audio, image, and video artifacts.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def parse_size(value: str) -> tuple[int, int]:
    width, sep, height = value.lower().partition("x")
    if not sep:
        raise ValueError(f"Invalid size: {value}")
    return int(width), int(height)


def require_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required command not found: {name}")
    return path


def import_tmdb_helpers(helper_dir: Path) -> tuple[Any, Any, Any]:
    helper_dir = helper_dir.expanduser().resolve()
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    from find_gems import TmdbClient, find_credentials, load_env

    return TmdbClient, find_credentials, load_env


def fetch_movie_payload(args: argparse.Namespace) -> dict[str, Any]:
    TmdbClient, find_credentials, load_env = import_tmdb_helpers(args.tmdb_helper_dir)
    token, api_key = find_credentials(load_env(args.env_file.expanduser()))
    client = TmdbClient(token, api_key)
    return client.get(
        f"/movie/{args.movie_id}",
        {
            "language": "en-US",
            "append_to_response": "images,credits",
            "include_image_language": "en,null",
        },
    )


def backdrop_sort_key(item: dict[str, Any]) -> tuple[int, int, float]:
    vote_average = float(item.get("vote_average") or 0)
    vote_count = int(item.get("vote_count") or 0)
    pixels = int(item.get("width") or 0) * int(item.get("height") or 0)
    return vote_count, pixels, vote_average


def selected_backdrops(payload: dict[str, Any], image_count: int) -> list[dict[str, Any]]:
    backdrops = payload.get("images", {}).get("backdrops", [])
    usable = [
        image
        for image in backdrops
        if isinstance(image.get("file_path"), str)
        and image.get("iso_639_1") is None
        and int(image.get("width") or 0) >= 1280
        and int(image.get("height") or 0) >= 720
        and 1.6 <= float(image.get("aspect_ratio") or 0) <= 1.9
    ]
    if not usable:
        raise RuntimeError("No usable TMDB backdrops found.")
    return sorted(usable, key=backdrop_sort_key, reverse=True)[:image_count]


def average_image_hash(path: Path) -> int | None:
    try:
        with Image.open(path) as image:
            hash_image = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
            if hasattr(hash_image, "get_flattened_data"):
                pixels = list(hash_image.get_flattened_data())
            else:
                pixels = list(hash_image.getdata())
    except OSError:
        return None
    average = sum(pixels) / len(pixels)
    value = 0
    for index, pixel in enumerate(pixels):
        if pixel >= average:
            value |= 1 << index
    return value


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def download_image(image: dict[str, Any], output_path: Path, force: bool) -> DownloadedImage:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if force or not output_path.exists():
        url = f"{TMDB_IMAGE_BASE}/original{image['file_path']}"
        request = Request(url, headers={"User-Agent": "VoxCPM-TMDB-KenBurns/1.0"})
        with urlopen(request, timeout=60) as response:
            output_path.write_bytes(response.read())

    return DownloadedImage(
        source_path=str(image["file_path"]),
        local_path=str(output_path),
        width=int(image.get("width") or 0),
        height=int(image.get("height") or 0),
        vote_average=float(image.get("vote_average") or 0),
        vote_count=int(image.get("vote_count") or 0),
    )


def download_backdrops(payload: dict[str, Any], args: argparse.Namespace) -> list[DownloadedImage]:
    images_dir = args.output_dir / "images"
    downloads: list[DownloadedImage] = []
    fallback_downloads: list[DownloadedImage] = []
    image_hashes: list[int] = []
    candidates = selected_backdrops(payload, max(args.image_count * 6, args.image_count))

    for index, image in enumerate(candidates, start=1):
        if len(downloads) >= args.image_count:
            break
        suffix = Path(image["file_path"]).suffix or ".jpg"
        output_path = images_dir / f"candidate_{index:02d}_{Path(image['file_path']).stem}{suffix}"
        downloaded = download_image(image, output_path, args.force)
        image_hash = average_image_hash(output_path)
        if image_hash is not None and any(
            hamming_distance(image_hash, existing_hash) <= 7
            for existing_hash in image_hashes
        ):
            fallback_downloads.append(downloaded)
            continue
        downloads.append(downloaded)
        if image_hash is not None:
            image_hashes.append(image_hash)

    for downloaded in fallback_downloads:
        if len(downloads) >= args.image_count:
            break
        downloads.append(downloaded)

    return downloads


def write_metadata(payload: dict[str, Any], images: list[DownloadedImage], output_dir: Path) -> None:
    credits = payload.get("credits", {})
    crew = credits.get("crew", [])
    cast = credits.get("cast", [])
    metadata = {
        "movie": {
            "id": payload.get("id"),
            "title": payload.get("title"),
            "release_date": payload.get("release_date"),
            "tmdb_url": f"{TMDB_MOVIE_BASE}/{payload.get('id')}",
            "backdrop_count": len(payload.get("images", {}).get("backdrops", [])),
        },
        "director": [person["name"] for person in crew if person.get("job") == "Director"],
        "producers": [
            person["name"]
            for person in crew
            if person.get("job") in {"Producer", "Executive Producer"}
        ],
        "top_cast": [person["name"] for person in cast[:10]],
        "downloaded_images": [asdict(image) for image in images],
        "tmdb_image_base": f"{TMDB_IMAGE_BASE}/original",
        "voice_strategy": "rotating_voice_per_sentence",
        "narration_lines": [asdict(line) for line in narration_lines()],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    script_text = "\n".join(
        f"{line.index:02d} [{line.voice}] {line.text}" for line in narration_lines()
    )
    (output_dir / "script.txt").write_text(script_text + "\n", encoding="utf-8")


def render_voiceover(args: argparse.Namespace) -> Path:
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lines = narration_lines()
    chunk_paths = [
        audio_dir / f"sentence_{line.index:02d}_{line.voice}.wav"
        for line in lines
    ]

    if args.force or not all(path.exists() for path in chunk_paths):
        say_args = argparse.Namespace(
            model=args.model,
            reference_dir=args.reference_dir,
            reference_audio=None,
            no_reference=False,
            device=args.device,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
            max_len=args.max_len,
            optimize=args.optimize,
            no_play=True,
            player=None,
        )
        model = say.load_model(say_args)
        prompt_caches = say.build_voice_caches(say_args, model)

        for line, output_path in zip(lines, chunk_paths):
            if output_path.exists() and not args.force:
                continue
            voice = say.VOICES[line.voice]
            print(
                f"Rendering sentence {line.index}/{len(lines)} with {line.voice}",
                file=sys.stderr,
            )
            wav = say.generate_audio(
                model=model,
                voice=voice,
                text=line.text,
                args=say_args,
                prompt_cache=prompt_caches.get(line.voice),
            )
            say.write_and_play(
                wav=wav,
                sample_rate=model.tts_model.sample_rate,
                args=say_args,
                output_path=output_path,
                fallback_name=f"tmdb_{line.index:02d}_{line.voice}",
            )

    concat_path = audio_dir / "chunks.ffconcat"
    concat_path.write_text(
        "ffconcat version 1.0\n"
        + "".join(f"file '{path.resolve()}'\n" for path in chunk_paths),
        encoding="utf-8",
    )
    raw_path = audio_dir / "narration_raw.wav"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-f", "concat", "-safe", "0", "-i", str(concat_path), "-c:a", "pcm_s16le", str(raw_path)])

    target_path = audio_dir / "narration_180s.wav"
    factor = probe_duration(raw_path) / args.target_seconds
    filters = f"{atempo_filter(factor)},loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur=5"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(raw_path),
            "-af",
            filters,
            "-t",
            f"{args.target_seconds:.3f}",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(target_path),
        ]
    )
    return target_path


def atempo_filter(factor: float) -> str:
    if factor <= 0:
        raise ValueError("Invalid atempo factor.")
    parts: list[float] = []
    while factor > 2.0:
        parts.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        parts.append(0.5)
        factor /= 0.5
    parts.append(factor)
    return ",".join(f"atempo={part:.6f}" for part in parts)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def ease_expr(frames: int) -> str:
    t = f"(on/{max(1, frames - 1)})"
    return f"({t}*{t}*(3-2*{t}))"


def smoothstep(frame: int, frames: int) -> float:
    t = frame / max(1, frames - 1)
    return t * t * (3.0 - 2.0 * t)


def motion_path(mode: int) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    # Keep motion deliberately small; big pans read as shake in a still-image move.
    # These paths stay under roughly 20 px of pan drift.
    paths = [
        (1.018, 1.052, -7.0, 3.0, 7.0, -3.0, 3.0, 150.0, 2.0, 190.0),
        (1.052, 1.020, 6.0, 2.0, -6.0, -2.0, 2.5, 170.0, 2.0, 210.0),
        (1.016, 1.046, -4.0, -6.0, 4.0, 6.0, 2.0, 180.0, 2.5, 160.0),
        (1.048, 1.018, 5.0, -4.0, -5.0, 4.0, 2.0, 140.0, 2.0, 200.0),
        (1.020, 1.050, -8.0, -2.0, 8.0, 2.0, 2.5, 165.0, 1.5, 230.0),
        (1.050, 1.018, 3.0, 6.0, -3.0, -6.0, 1.5, 220.0, 2.5, 175.0),
    ]
    return paths[mode % len(paths)]


def zoompan_filter(mode: int, frames: int, fps: int, internal_size: str) -> str:
    internal_width, internal_height = parse_size(internal_size)
    output_width, output_height = parse_size(VIDEO_SIZE)
    ease = ease_expr(frames)
    z0, z1, x0, y0, x1, y1, wave_x, period_x, wave_y, period_y = motion_path(mode)
    z_expr = f"({z0:.4f}+({z1:.4f}-{z0:.4f})*{ease})"
    delta_x = x1 - x0
    delta_y = y1 - y0
    drift_x = (
        f"({x0:.3f}+({delta_x:.3f})*{ease}"
        f"+{wave_x:.3f}*sin(on/{period_x:.3f}))"
    )
    drift_y = (
        f"({y0:.3f}+({delta_y:.3f})*{ease}"
        f"+{wave_y:.3f}*sin(on/{period_y:.3f}))"
    )
    x_expr = f"(iw/2-(iw/zoom/2)+{drift_x})"
    y_expr = f"(ih/2-(ih/zoom/2)+{drift_y})"
    return (
        f"scale={internal_width}:{internal_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={internal_width}:{internal_height},"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={internal_size}:fps={fps},"
        f"scale={output_width}:{output_height}:flags=lanczos,"
        "setsar=1,format=yuv420p"
    )


def cover_image(path: Path, target_size: str) -> Image.Image:
    target_width, target_height = parse_size(target_size)
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = max(target_width / image.width, target_height / image.height)
        resized = image.resize(
            (math.ceil(image.width * scale), math.ceil(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    left = (resized.width - target_width) / 2
    top = (resized.height - target_height) / 2
    return resized.crop(
        (
            round(left),
            round(top),
            round(left) + target_width,
            round(top) + target_height,
        )
    )


def render_pillow_segment(
    image_path: Path,
    segment_path: Path,
    mode: int,
    frames: int,
    fps: int,
    internal_size: str,
) -> None:
    output_width, output_height = parse_size(VIDEO_SIZE)
    base = cover_image(image_path, internal_size)
    base_width, base_height = base.size
    scale_x = base_width / output_width
    scale_y = base_height / output_height
    z0, z1, x0, y0, x1, y1, wave_x, period_x, wave_y, period_y = motion_path(mode)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        VIDEO_SIZE,
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(segment_path),
    ]
    print(f"+ {' '.join(cmd)} < Pillow subpixel frames from {image_path}", file=sys.stderr)
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in range(frames):
            eased = smoothstep(frame, frames)
            zoom = z0 + (z1 - z0) * eased
            drift_x = (
                x0 + (x1 - x0) * eased + wave_x * math.sin(frame / period_x)
            ) * scale_x
            drift_y = (
                y0 + (y1 - y0) * eased + wave_y * math.sin(frame / period_y)
            ) * scale_y
            a = scale_x / zoom
            e = scale_y / zoom
            c = base_width / 2 + drift_x - a * output_width / 2
            f = base_height / 2 + drift_y - e * output_height / 2
            frame_image = base.transform(
                (output_width, output_height),
                Image.Transform.AFFINE,
                (a, 0.0, c, 0.0, e, f),
                resample=Image.Resampling.BICUBIC,
            )
            process.stdin.write(frame_image.tobytes())
    except BrokenPipeError as exc:
        raise RuntimeError("ffmpeg closed while receiving Pillow frames.") from exc
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def write_concat(path: Path, files: list[Path]) -> None:
    path.write_text(
        "ffconcat version 1.0\n"
        + "".join(f"file '{file.resolve()}'\n" for file in files),
        encoding="utf-8",
    )


def build_video(images: list[DownloadedImage], audio_path: Path, args: argparse.Namespace) -> Path:
    video_dir = args.output_dir / "video"
    segments_dir = video_dir / f"segments_{args.motion_renderer}"
    segments_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [Path(image.local_path) for image in images]
    segment_count = min(len(image_paths), args.image_count)
    segment_duration = args.target_seconds / segment_count
    frames = math.ceil(segment_duration * args.fps)
    segments: list[Path] = []

    for index, image_path in enumerate(image_paths[:segment_count], start=1):
        segment_path = segments_dir / f"segment_{index:02d}.mp4"
        segments.append(segment_path)
        if segment_path.exists() and not args.force and not args.force_video:
            continue
        if args.motion_renderer == "pillow":
            render_pillow_segment(
                image_path=image_path,
                segment_path=segment_path,
                mode=index,
                frames=frames,
                fps=args.fps,
                internal_size=args.internal_size,
            )
        else:
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-loop",
                    "1",
                    "-i",
                    str(image_path),
                    "-vf",
                    zoompan_filter(index, frames, args.fps, args.internal_size),
                    "-frames:v",
                    str(frames),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "19",
                    "-pix_fmt",
                    "yuv420p",
                    str(segment_path),
                ]
            )

    concat_path = video_dir / "segments.ffconcat"
    write_concat(concat_path, segments)
    silent_path = video_dir / "silent.mp4"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-f", "concat", "-safe", "0", "-i", str(concat_path), "-c", "copy", str(silent_path)])

    final_path = args.output_dir / "final.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(silent_path),
            "-i",
            str(audio_path),
            "-t",
            f"{args.target_seconds:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(final_path),
        ]
    )
    return final_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_command("ffmpeg")
    require_command("ffprobe")

    payload = fetch_movie_payload(args)
    images = download_backdrops(payload, args)
    write_metadata(payload, images, args.output_dir)

    title = payload.get("title") or f"movie {args.movie_id}"
    print(
        f"Selected {title}: {len(payload.get('images', {}).get('backdrops', []))} TMDB backdrops; "
        f"downloaded {len(images)}.",
        file=sys.stderr,
    )
    audio_path = render_voiceover(args)
    final_path = build_video(images, audio_path, args)

    print(f"Final video: {final_path}")
    print(f"Duration: {probe_duration(final_path):.2f}s")
    return 0


if __name__ == "__main__":
    say.os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
