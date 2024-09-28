import os
import re
import time
from typing import Any
from queue import Queue
from threading import Thread
import random
import torch
import torchaudio
import pyaudio, wave

import getpass

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

DATA_DIRECTORY_FOR_GENERATED_FILES = "./pylips_phrases"
TTS_CONFIG_PATH = "./xtts_config.json"
CHECKPOINT_DIRECTORY = f"/home/{getpass.getuser()}/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"

SPEAKER =  "Nova Hogarth" #"Ferran Simen"  #"Xavier Hayasaka" #"Asya Anara" # "Nova Hogarth" #"Sofia Hellen"

CHUNK_SIZE = 1024

STREAM_CHUNK_SIZE=50
OVERLAP_WAVE_LEN=2048

TEMPERATURE=0.01
SPEED=1.0


class WrongTypeError(Exception):
    def __init__(self, argument: str, expected_type: Any, actual_value=None):
        self.expected_type = expected_type
        message = f"Expected '{expected_type}' type for argument {argument}"
        super().__init__(message)

class SpeakerNotFoundError(Exception):
    def __init__(self, speaker: str):
        message = f"Speaker name '{speaker}' is invalid."
        super().__init__(message)


class StreamingTTS():
    def __init__(self, deepspeed_acceleration=False, actuate_pylips=False):
        # load TTS model
        print("\nLoading TTS model...")
        config = XttsConfig()
        config.load_json(TTS_CONFIG_PATH)
        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIRECTORY, use_deepspeed=deepspeed_acceleration)
        self._model.cuda()

        # create data directory (for wav files and visemes) if it doesn't exist
        os.makedirs(DATA_DIRECTORY_FOR_GENERATED_FILES, exist_ok = True)

        # initialize pyaudio
        print("\nInitializing playback device using pyaudio...")
        self._pyaudio = pyaudio.PyAudio()

        # attach to robot
        self._actuate_pylips = actuate_pylips
        if self._actuate_pylips:
            print("\nConnecting to robot face...")
            import animate_pylips as face
            self._robot = face.Robot()

    def streaming_wav_generation_and_playback(self, text: str, playback: bool = False, language: str = "en", 
                                            speaker: str = SPEAKER, speed: str =SPEED, 
                                            temperature: float = TEMPERATURE):

        # raise errors when called with argument types
        if not isinstance(language, str):
            raise WrongTypeError("language", str)
        if not isinstance(speed, float):
            raise WrongTypeError("speed", float)

        # random id for the this invocation
        session_id = random.randrange(1000)
        self._file_prefix = f"{DATA_DIRECTORY_FOR_GENERATED_FILES}/tts_{session_id}"
        self._base_file_prefix = os.path.basename(self._file_prefix)
        self._text = text
        self._language = language
        self._speaker = speaker
        self._speed = speed
        self._temperature = temperature
        self._playback_over_audio_device = playback

        # catch language error (based on XTTSv2 supported languages)
        if not self._language in {"ar", "cs", "de", "en", "es", "fr", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko"}:
            raise Exception(f"'{self._language}' is not a supported language.")

        # load speaker embedding    
        print(f"Loading speaker {self._speaker}...")
        try:
            self._gpt_cond_latent, self._speaker_embedding = self._model.speaker_manager.speakers[self._speaker].values()
        except KeyError:
            raise SpeakerNotFoundError(self._speaker)

        # data structure to hold waveform as it is being generated
        self._wave = []

        # separate generation/processing and playback threads, with queued playback of generated chunks
        q = Queue()
        generation_thread = Thread(target=self._generate_chunks, args=(q,))
        playback_thread = Thread(target=self._playback_chunks_using_pyaudio, args=(q,))
        generation_thread.start()
        playback_thread.start()
        playback_thread.join()
        generation_thread.join()

        return f"{self._file_prefix}.wav"

    # producer coroutine
    def _generate_chunks(self, queue):
        t0 = time.time()

        print("Starting inference...")
        chunks = self._model.inference_stream(
            self._text,
            self._language,
            self._gpt_cond_latent,
            self._speaker_embedding,
            stream_chunk_size=STREAM_CHUNK_SIZE,
            overlap_wav_len=OVERLAP_WAVE_LEN,
            temperature=self._temperature,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.85,
            do_sample=True,
            speed=self._speed,
        )
        bundle = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunck: {time.time() - t0}")
            print(f"Processing chunk {i}")
            #print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
            bundle.append(chunk)
            data = chunk.squeeze().unsqueeze(0).cpu()
            filename = f"{self._file_prefix}_{i}.wav"
            torchaudio.save(filename, data, 24000)
            wav_data = wave.open(filename,"rb")
            if self._actuate_pylips:
                self._robot.compute_and_store_visemes(filename, language='eng')
            self._wave.append([wav_data.getparams(),wav_data.readframes(wav_data.getnframes())])
            print(f"Done processing chunk {i}")
            queue.put(i)
        wav = torch.cat(bundle, dim=0)
        torchaudio.save(f"{self._file_prefix}.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)    
        queue.put(None)     
            
    # consumer coroutine
    def _playback_chunks_using_pyaudio(self, queue):
        while True:
            key = queue.get()
            if key == 0:
                # create stream
                params = self._wave[0][0]
                stream = self._pyaudio.open(format = self._pyaudio.get_format_from_width(params.sampwidth),  
                        channels = params.nchannels,  
                        rate = params.framerate,  
                        output = True)
                # play first chunk
                if self._actuate_pylips:
                    self._robot.lip_visemes(f"{self._base_file_prefix}_{key}")
                if self._playback_over_audio_device:
                    print(f"->Playing back chunk {key}")
                    stream.write(self._wave[0][1])
            elif key:
                # play current chunk
                data = self._wave[key][1]
                if self._actuate_pylips:
                    self._robot.lip_visemes(f"{self._base_file_prefix}_{key}")
                if self._playback_over_audio_device:
                    print(f"->Playing back chunk {key}")
                    stream.write(data)  
            else: # key is None:
                # terminate
                stream.stop_stream()  
                stream.close()
                break

"""
tts = StreamingTTS()

text = input("<PRESS ANY KEY TO CONTINUE>")

text = "N L Navi, at your service. Here's a route that takes us by Marina State Beach, a gem of California's coast \
    with panaramic views of the Monterey Bay marine sanctuary.  There are three E-V charging stations along the way, \
    each in close proximity of lunch options.  This route will get you to your destination with more than 50% charge \
    and is only 12 minutes extra driving time compared to the 2.5 hour fastest route.  How does THAT sound."

# break text into a list of sentences
sentences = re.split('(?<=[^A-Z].[.?]) +(?=[A-Z])', text)

for sentence in sentences:
    tts.streaming_wav_generation_and_playback(sentence)

while(True):
    text = input("Enter a sentence to be spoken: ")
    tts.streaming_wav_generation_and_playback(text)

p.terminate()

# N-L-Navi, at your service. Here's a route that takes us by Marina State Beach, a gem of California's coast with panoramic views of the Monterey Bay Marine Sanctuary.  There are three E-V charging stations along the way, each in close proximity of lunch options.  This route will get you to your destination with more than 50% charge and is only 12 minutes extra driving time compared to the 2.5-hour fastest route.  How does THAT sound.
"""
