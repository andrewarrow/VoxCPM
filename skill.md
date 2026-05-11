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

## Request Fields

- `text`: Required. The text to synthesize.
- `voice`: Optional. Defaults to the server's current default voice if omitted.
- `filename`: Optional. Plain output filename. If omitted, the server creates a timestamped name.

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
- `filename` contains a directory path.
- The server is not running or the model has not finished loading.

