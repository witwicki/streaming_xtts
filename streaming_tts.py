__author__ = "Stefan Witwicki"
__copyright__ = "Copyright (C) 2024 Stefan Witwicki"
__license__ = "GNU General Public License version 2"
__version__ = "0.1"

"""Module for streaming inference of Coqui's XTTSv2 model.

Functionalities:
  - can do one or both of (a) playing back audio chunk-by-chunk with < 1 second delay and (b) returning a wave file
  - optionally, can derive visemes from the audio chunks and actuate a robot face via the PyLips package

Example usage:
  tts_engine = StreamingTTS()
  tts_engine.streaming_wav_generation_and_playback("The rain in spain falls mainly in the plain", playback=True)
"""

"""License information:

    This file is part of the streaming_xtts package.
    streaming_xtts is free software: you can redistribute it and/or modify it under the terms of the 
    GNU General Public License as published by the Free Software Foundation, version 2. 
    streaming_xtts is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even 
    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public 
    License for more details. You should have received a copy of the GNU General Public License along with 
    streamging_xtts. If not, see <https://www.gnu.org/licenses/>. 
"""

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

# CONSTANTS
DATA_DIRECTORY_FOR_GENERATED_FILES = f"{os.path.dirname(os.path.realpath(__file__))}/pylips_phrases"
TTS_CONFIG_PATH = f"{os.path.dirname(os.path.realpath(__file__))}/xtts_config.json"
CHECKPOINT_DIRECTORY = f"/home/{getpass.getuser()}/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"
DEFAULT_SPEAKER =  "Nova Hogarth"
DEFAULT_SPEED=1.0
DEFAULT_TEMPERATURE=0.01
CHUNK_SIZE = 1024
STREAM_CHUNK_SIZE=50
OVERLAP_WAVE_LEN=2048


class WrongTypeError(Exception):
    """Exception to be thrown for typing errors."""
    def __init__(self, argument: str, expected_type: Any, actual_value=None):
        self.expected_type = expected_type
        message = f"Expected '{expected_type}' type for argument {argument}"
        super().__init__(message)


class SpeakerNotFoundError(Exception):
    """Exception to be thrown for invalid tts speaker references."""
    def __init__(self, speaker: str):
        message = f"Speaker name '{speaker}' is invalid."
        super().__init__(message)


class StreamingTTS():
    """Class for streaming inference of Coqui's XTTSv2 model.

    Functionalities:
    - optionally, can playing back audio chunk-by-chunk on the host with < 1 second delay
    - can return a wave file
    - optionally, can derive visemes from the audio chunks and actuate a robot face via the PyLips package

    Attributes
    ----------
    _model : Xtts
        the xtts model loaded from a local checkpoint
    _pyaudio : PyAudio
        pyaudio object for playing back audio
    _actuate_pylips : bool
        whether to use pylips to actuate a robot face
    _file_prefix : str
        prefix, including path, for output files
    _base_file_prefix : str
        prefix, excluding path
    _text : str
        the text to be spoken
    _language : str
        the language of the text, 2-letter ISO code, according to the iso_code_639_1 format
    _speaker : str
        the name of the speaker (referencing XTTSv2's pre-generated embeddings)
    _speed : float
        the speed of speech (not to be confused with tone or emotion)
    _temperature : float
        the temperature used for model inference
    _playback_over_audio_device : bool
        whether or not to playback audio on the host's default audio device
    _wave : list of lists of bytes
        the generated audio data

    Methods
    -------
    __init__(deepspeed_acceleration=False, actuate_pylips=False)
        initializes the streaming tts object
    streaming_wav_generation_and_playback(text: str, playback: bool = False, language: str = "en", 
                                            speaker: str = DEFAULT_SPEAKER, speed: str =DEFAULT_SPEED, 
                                            temperature: float = DEFAULT_TEMPERATURE)
        generates the audio data for a given text, optionally plays it back, optionally actuates pylips, and saves it to disk

    """
    def __init__(self, deepspeed_acceleration=False, actuate_pylips=False):
        """Instantiate and initialize streaming tts.

        Parameters
        ----------
        deepspeed_acceleration : bool
            whether or not to use the deepspeed package for inference acceleration
        actuate_pylips : bool
            whether or not to generate visemes and lip-sync over Pylips (if server is running)
        """
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
                                            speaker: str = DEFAULT_SPEAKER, speed: str =DEFAULT_SPEED, 
                                            temperature: float = DEFAULT_TEMPERATURE) -> str:
        """Streaming generation of speech audio, (optional) playback, and (optional) actuation of pylips.

        This is implemented using a generating/processing thread and separate playback thread and a FIFO
        for handling each chunk of generated audio.

        Parameters
        ----------
        text : str
            Text to be synthesized into speech audio.
        playback : bool
            If True, play the generated audio file using pyaudio
        language : str
            2-letter code for the language of the input text (e.g., "en" for English) in iso_code_639_1 format
        speaker : str
            the name of the speaker (referencing XTTSv2's pre-generated embeddings)
        speed : float
            the speaking rate, defauting to 1.0, where higher is faster
        temperature : float
            the temperature used for model inference

        Returns
        -------
        str
            The path to the generated wav file

        Raises
        ------
        WrongTypeError
           If language is not a string or the speed is not a float
        SpeakerNotFoundError
            If speaker is not found in XTTSv2's pre-generated embeddings
        Exception
            If the language is not found in the list of supported languages, or something else goes wrong during synthesis
        """

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
        if not self._language in {"ar", "cs", "de", "en", "es", "fr", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko", "ja"}:
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
    def _generate_chunks(self, queue: Queue):
        """Coroutine for generating chunks of audio.
        
        This runs the streaming inference, and for each returned chunk:
        - wave data is extracted and appended to self._wave[]
        - the chunk's wave data is written to disk
        - visemes are generated and written to disk
        - the index of the chunk is added to the queue

        The chunks are subsequently consumed by _playback_chunks_using_pyaudio(),
        and the full wave data is written to disk.

        Parameters
        ----------
        queue : multiprocessing.Queue
        """
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
    def _playback_chunks_using_pyaudio(self, queue: Queue):
        """Coroutine for playing back (and optionally animating) chunks of audio.

        Pop chunks off of the queue, retrieve the wave data by index, and optionally:
        - animating the Pylips robot face
        - playing back the audio chunk using pyaudio
        """
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


