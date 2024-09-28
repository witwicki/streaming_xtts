# streaming_xtts
An experiment in streaming text-to-speech (TTS), interfacing with Coqui's XTTSv2 pipeline.

This package implements a streaming server + client for TTS inference.  Features / options include:

- playback of generated audio on the host (using PyAudio), with sub-1-second delay once model has been warmed up
- download of the generated audio as a .wav file
- support for long texts through smart decomposition into a series of inference calls
- lip-syncing an animated robot face (using PyLips)

The API also supports many of the xtts knobs, e.g., *speaker*, *temperature*, etc.

## Acknowledgements

- Coqui's seminal development https://docs.coqui.ai/en/latest/models/xtts.html
- The creators of Pylips https://github.com/interaction-lab


## Requirements

The server-side code requires:

- Python 3.10 or greater (developed on 3.10.14)
- Coqui TTS (recommended fork: https://github.com/idiap/coqui-ai-TTS) with **xtts_v2** model
- pyaudio
- netifaces
- other miscellaneous packages (see **requirements.txt**)

The lip-syncing feature additionally requires:
- PyLips (fork: https://github.com/witwicki/PyLips, originally developed by students at USC Interaction Lab)

## Package installation

...forthcoming....

## Basic Usage

On the host:
```shell script 
./streaming_tts_server.py
```

On the client:
```shell script
./streaming_tts_client.py -pd "The rain in spain falls mainly on the plane."
```

For either of the above, use **--help** to se the available optons.

## Actuating the robot face

First, serve PyLips (could be on a different machine from the tts server):
```shell script
cd /path/to/source/of/Pylips/
python -m pylips.face.start --port 8008
```

Next, start the TTS server with *pylips* flag:
```shell scipt
./streaming_tts_server.py --pylips
``` 

To view the animated robot face, navigate to: http://pylips-server-ip-address:8008/face.


