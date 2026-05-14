# VoxCPM Local Say HTTP Skill

Use this skill when an agent needs to synthesize speech through the local VoxCPM `say` HTTP server.

## Requirement

The server must already be running. Start it from this repository with:

```bash
python scripts/say.py
```

The HTTP server starts only after VoxCPM has loaded the model and voice caches. Wait until the terminal prints:

```text
HTTP API listening on http://127.0.0.1:9002
```

## Health Check

Before sending speech requests, verify the server is ready:

```bash
curl http://127.0.0.1:9002/health
```

Expected response:

```json
{
  "ok": true,
  "model_loaded": true
}
```

## List Voices

```bash
curl http://127.0.0.1:9002/voices
```

Useful voice names:

- `booming_radio`
- `cinematic_trailer`
- `luxury_brand`
- `friendly_support`
- `science_documentary`
- `noir_detective`
- `global_airport`
- `sleep_story`

## Generate Speech

Send a `POST` request to `/say` with JSON.

```bash
curl -X POST http://127.0.0.1:9002/say \
  -H 'Content-Type: application/json' \
  -d '{"voice":"global_airport","text":"Flight 42 is now boarding.","filename":"boarding.wav"}'
```

Response:

```json
{
  "ok": true,
  "path": "/Users/aa/os/VoxCPM/outputs/http/boarding.wav"
}
```

The server saves the WAV under:

```text
outputs/http/
```

It also plays the WAV when rendering completes, unless the server was started with `--no-play`.

## Try A New Voice

Use `POST /voices/preview` to test a new English voice prompt without saving it as a reusable voice.

Pure voice design, with no reference voice:

```bash
curl -X POST http://127.0.0.1:9002/voices/preview \
  -H 'Content-Type: application/json' \
  -d '{"description":"A calm adult woman with a warm public radio tone, precise diction, relaxed pacing, and gentle authority.","text":"This is a preview of the custom voice.","filename":"preview_radio_warm.wav"}'
```

Custom changes anchored to an existing voice:

```bash
curl -X POST http://127.0.0.1:9002/voices/preview \
  -H 'Content-Type: application/json' \
  -d '{"base_voice":"global_airport","changes":"warmer, slower, more reassuring, with less announcement stiffness","text":"Flight 42 is now boarding.","filename":"preview_airport_warm.wav"}'
```

Response:

```json
{
  "ok": true,
  "path": "/Users/aa/os/VoxCPM/outputs/http/preview_airport_warm.wav",
  "voice": {
    "name": "preview",
    "control": "A clear international airport announcement voice..., warmer, slower...",
    "source": "custom",
    "base_voice": "global_airport",
    "reference_sample": "15_global_airport.wav"
  }
}
```

The preview WAV is saved under `outputs/http/` and played when rendering completes.

## Promote A Voice

Use `POST /voices/promote` to save a voice prompt as a named reusable voice. Promoted voices are stored in:

```text
outputs/http/voices.json
```

Promote a pure designed voice:

```bash
curl -X POST http://127.0.0.1:9002/voices/promote \
  -H 'Content-Type: application/json' \
  -d '{"name":"warm_radio","description":"A calm adult woman with a warm public radio tone, precise diction, relaxed pacing, and gentle authority."}'
```

Promote a modified version of an existing voice:

```bash
curl -X POST http://127.0.0.1:9002/voices/promote \
  -H 'Content-Type: application/json' \
  -d '{"name":"warm_airport","base_voice":"global_airport","changes":"warmer, slower, more reassuring, with less announcement stiffness"}'
```

After promotion, the voice can be reused through `/say`:

```bash
curl -X POST http://127.0.0.1:9002/say \
  -H 'Content-Type: application/json' \
  -d '{"voice":"warm_airport","text":"Flight 42 is now boarding.","filename":"warm_airport_boarding.wav"}'
```

To update an existing promoted voice, pass `replace:true`:

```bash
curl -X POST http://127.0.0.1:9002/voices/promote \
  -H 'Content-Type: application/json' \
  -d '{"name":"warm_airport","base_voice":"global_airport","changes":"warmer, slower, softer, and more reassuring","replace":true}'
```

Built-in voices cannot be replaced.

## Perform A Scene

Use `POST /scene/preview` to render screenplay-style dialogue with multiple actors. This endpoint renders each line as its own take, places the takes on a timeline, supports overlaps, and can add subtle room tone or mouth clicks in the final mix.

```bash
curl -X POST http://127.0.0.1:9002/scene/preview \
  -H 'Content-Type: application/json' \
  -d '{
    "actors": {
      "mara": "friendly_support",
      "david": "noir_detective"
    },
    "continuity": "reference",
    "prompt_mode": "off",
    "mix": {
      "room_tone": true,
      "mouth_noises": "subtle",
      "crossfade_ms": 18
    },
    "segments": [
      {"actor":"mara","text":"Please.","pause_after_ms":600},
      {"actor":"david","text":"I am here.","overlap_previous_ms":150}
    ],
    "filename":"scene_preview.wav"
  }'
```

Useful scene fields:

- `actors`: Object mapping actor names to voices, or actor objects with `voice`, `identity`, and `style`.
- `segments`: Ordered scene beats. Each beat can include `actor`, `text`, `direction`, `pause_before_ms`, `pause_after_ms`, `overlap_previous_ms`, `start_ms`, and `gain_db`.
- `prompt_mode`: Defaults to `off`, which sends only dialogue text to VoxCPM so acting directions are not spoken.
- `continuity`: `reference`, `rolling`, or `reset`. Use `reference` for stable actors; try `rolling` for tighter same-actor continuity across lines.
- `mix.room_tone`: Adds low-level room tone under the scene.
- `mix.mouth_noises`: `off`, `subtle`, or `medium`.
- `mix.crossfade_ms`: Adds tiny fades to clips to avoid hard digital cuts.
- `nonverbal`: A segment may omit `text` and use `nonverbal` instead. Supported values are `breath`, `inhale`, `exhale`, `sigh`, `mouth_click`, `click`, and `swallow`.

`POST /perform/preview` uses the same renderer for a single actor.

## Request Fields

- `text`: Required. The text to synthesize.
- `voice`: Optional. Defaults to the server's current default voice if omitted.
- `filename`: Optional. Plain output filename. If omitted, the server creates a timestamped name.

For `/voices/preview` and `/voices/promote`:

- `description`, `prompt`, or `control`: Full English voice description.
- `base_voice`: Optional existing voice to use as the timbre/style anchor.
- `changes` or `direction`: Optional custom changes to apply to `base_voice`.
- `name`: Required only for `/voices/promote`.
- `replace`: Optional boolean for updating an existing promoted voice.

`filename` must be a basename only, not a path. Valid examples:

```text
boarding.wav
boarding
line_001.wav
```

Invalid examples:

```text
../boarding.wav
outputs/http/boarding.wav
/tmp/boarding.wav
```

If `.wav` is omitted, the server adds it automatically.

## Form Data Alternative

JSON is preferred, but URL-encoded form data also works:

```bash
curl -X POST http://127.0.0.1:9002/say \
  -d 'voice=global_airport' \
  -d 'text=Flight 42 is now boarding.' \
  -d 'filename=boarding.wav'
```

## Error Handling

Bad requests return JSON with an `error` field:

```json
{
  "error": "text is required."
}
```

Common errors:

- Missing or empty `text`.
- Unknown `voice`.
- Unknown `base_voice`.
- Missing voice `description`, `prompt`, `control`, or `changes`.
- Promoting a voice without `name`.
- Promoting a voice name that already exists without `replace:true`.
- `filename` contains a directory path.
- The server is not running or the model has not finished loading.
