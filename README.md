# streaming_xtts
An experiment in streaming text-to-speech (TTS), interfacing with Coqui's XTTSv2 pipeline.

This package implements a streaming server + client for TTS inference.  Features / options include:

- [x] playback of generated audio on the host (using PyAudio), with sub-1-second delay once model has been warmed up
- [x] download of the generated audio as a .wav file
- [x] support for long texts through smart decomposition into a series of inference calls
- [x] lip-syncing an animated robot face (using PyLips)
- [x] support for cuda-based GPU and Apple's MPS (though with mps delays appear to be a bit higher)

The API also supports many of the original xtts knobs, e.g., *speaker*, *temperature*, etc.

## Acknowledgements

- Coqui's seminal development [https://docs.coqui.ai/en/latest/models/xtts.html](https://docs.coqui.ai/en/latest/models/xtts.html)
- IDIAP for maintaining a [fork](https://github.com/idiap/coqui-ai-TTS))
- The creators of Pylips https://github.com/interaction-lab

## Requirements

- portaudio (available on [linux](https://answers.launchpad.net/ubuntu/noble/+package/portaudio19-dev), [macos](https://formulae.brew.sh/formula/portaudio), etc.)

The lip-syncing feature additionally requires:
- [PyLips](https://github.com/witwicki/PyLips) (forked from original development by students at USC Interaction Lab)

## Quick start

On the server side (where inference is perfomed and audio optionally plays):
```shell script
uv sync
uv run streaming-xtts
```

On the client side:
```shell script
uv run streaming-tts-client -p "The rain in spain falls mainly on the plane."
```

You can also download the complete generated wave file:
```shell script
uv run streaming-tts-client -pd "Check your project directory for a wave file with the current timestamp."
```

For both the server and client script above, use **--help** to see the available optons.

## Actuating the robot face

### Locally

Use the --pylips argument when starting the server:
```shell script
uv run streaming-xtts --pylips
```

To view the animated robot face, navigate to: http://localhost:8008/face.

_Note: since this was optimized for a mobile device, zoom out in your computer's web browser window for a better viewing experience._

### On a different machine

First, serve PyLips:
```shell script
git clone https://github.com/witwicki/PyLips.git
cd Pylips
# in your favorite virtual environment
pip install .
python -m pylips.face.start --port <port,e.g.,8008>
```

Next, start the TTS server with appropriate flags:
```shell scipt
uv run streaming-xtts --pylips --pylipsserver <server_IP_or_hostname>:<port>
```

To view the animated robot face, navigate to: http://<server_IP_or_hostname>:<port>/face.

## Planned improvements
- [x] Support for Apple Silicon
- [ ] FastAPI for cleaner interface
- [ ] Emotional cues from text on animated face
- [ ] Streaming audio over HTTP (by, e.g., DASH) for accessibility on low-compute devices
- [ ] Streaming support for newer TTS models, e.g., F5
