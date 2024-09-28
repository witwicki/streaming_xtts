#!/usr/bin/env python

__author__ = "Stefan Witwicki"
__copyright__ = "Copyright (C) 2024 Stefan Witwicki"
__license__ = "GNU General Public License version 2"
__version__ = "0.1"

"""STREAMING TTS SERVER

This server takes text as input generates speech using Coqui's XTTSv2 model.
It can operate in one of two, or both, modes: {streaming playback on the host, generation of audio file}.
Additionally, it interfaces with PyLips for actuating a robot face, if available. 

To run:
  python streaming_tts_server.py --help
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

import argparse
import os
import json
import re
import math
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from functools import partial
import netifaces as ni
import socket
#from stream2sentence import generate_sentences
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from streaming_tts import StreamingTTS, WrongTypeError

# Constants
TTS_CHARACTER_LIMIT = 255 # the per-call character limit supported by XTTS2
SILENCE_DURATION_BETWEEN_CONSECTIVE_SPEECHES = 0.5 # seconds of silence between consecutive calls

class MyRequestHandler(BaseHTTPRequestHandler):
    """Request handling for the server.

    The intended functionality is to handle TTS directives by POST request and (optionally) to return
    the audio file as a response.  
    """
    
    def __init__(self, tts_session: StreamingTTS, *args, **kwargs):
        """
        Parameters
        ----------
        tts_session : StreamingTTS
            The StreamingTTS object to use for TTS generation.
        """
        self._tts_session = tts_session
        super().__init__(*args, **kwargs)

    def set_empty_headers(self):
        """Set and respond with an empty headers (e.g., for a successul POST request that does not return audio)."""
        try:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
        except Exception as e:
            print(f"Exception raised in HTTP response: {e}")    

    def send_error_response(self, error: str):
        """Respond with error (400).

        Parameters
        ----------
        error : str
            The error message to send in the response body.
        """
        try:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(f"<html><body><h1>Bad Request: {error}</h1></body></html>", "utf-8"))
        except Exception as e:
            print(f"Exception raised in HTTP response: {e}")

    def send_wav_file_as_response(self, wav_filename: str):
        """Send response 200 (success) with as a WAV file.

        Parameters
        ----------
        wav_filename : str
            The filename of the WAV file to send in response body.
        """
        try:
            self.send_response(200)
            self.send_header("Content-type", "audio/wav")
            self.end_headers()
            with open(wav_filename, "rb") as wav_file:
                self.wfile.write(wav_file.read())
        except Exception as e:
            print(f"Exception raised in HTTP response: {e}")

    def do_HEAD(self):
        """Empty HEAD request (for compatibility)."""
        self.set_empty_headers()

    def do_GET(self):
        """Empty GET request (for compatibility)."""
        self.set_empty_headers()
    
    def do_POST(self):
        """POST reqeust for TTS audio playback (streaming) and wave-file retrieval 
        (delivered only after playback).

        Content is expected as a utf-8 encoded JSON string, with the following fields:
        - text (string) : the text to be spoken
        - playback (bool) = True : whether or not the host should playback the audio {True, False}
        - download (bool) = True  : whether or not the host should return a wav file  {True, False}
        - split (string) = None : the method of splitting long speeches if any {None, "sentence", "intelligent"}
        - language (string) = "en" : the language of the text to be spoken in iso_code_639_1 format, lower case
        - speaker (string) = "Nova Hogarth" : the name of the xttsv2 speaker whose embeddings to use
        - speed (float) = 1.0 : the speed of the speech (independent of tone and emotion), where 2.0 is twice as fast
        - temperature = 0.01 : the temperature of the speech, where 0 is coldest

        In turn, the JSON string gets converted to a dict and passed as arguments to the TTS engine.
        """
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        content = json.loads(body.decode('utf-8'))
        # parse 'download' argument
        download_requested = False
        if "download" in content:
            # remove from dictionary so that it is not passed to the TTS engine
            download_requested = content.pop("download") 
        # parse 'speed argument
        self._speed_of_speech = 1.0
        if "speed" in content:
            self._speed_of_speech = content["speed"]
        # check for text field
        text = content.get("text", None)
        success = True
        error_string = None
        return_filename = ""
        if text:
            # invoke tts engine, recording exceptions that it raised for communication in the response  
            try:
                return_filename = self._speek_and_return_wav(**content)
            except WrongTypeError as e:
                error_string = f"WrongTypeError: {e}"
                success = False
            except Exception as e:
                error_string = f"{e}"
                success = False
        else:
            error_string = "No text field in request"
            success = False
        # send response (with or without wav file)
        if success:
            if download_requested:
                self.send_wav_file_as_response(return_filename)
            else:
                self.set_empty_headers()
        else:
            print(error_string)
            self.send_error_response(error_string)

    def _speek_and_return_wav(self, **kwargs) -> str:
        """Invoke the TTS engine and return the filename of the generated wav file.
        
        Importantly, this function also takes care of splitting the text (depending on the 'split' argument)
        into sentences or sentence-bundles in order not to exceed the maximum character limit of the TTS engine.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments that can include parameters taken by 
            StreamingTTS.streaming_wav_generation_and_playback(), as well as
            'split', a string indicating the mode for splitting sentences for 
            consecutive invokations of the TTS engine.

        Returns
        -------
        str
            The filename of the generated wav file.
        """
        text = kwargs.get("text")
        sentences = [ text ]
        split_into_sentences = None
        if "split" in kwargs:
            # remove from dictionary so that it is not passed to the TTS engine
            split_into_sentences = kwargs.pop("split")
        if split_into_sentences:
            # regex for splitting sentences (with special character 。 to support Japanese)
            sentence_splitter_j = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|。)\s+'
            sentences = re.split(sentence_splitter_j, text.replace('。','。 '))
            # TODO: compare performance with generate_sentences(content["text"]):
            if split_into_sentences == "intelligent":
                sentences = self._rebundle_sentences_intelligently_ahdhering_to_TTS_limits(sentences)
        # pass each sentence bundle (could be multiple sentenceS) into to the TTS engine
        wav_file_paths = []
        for sentence in sentences:
            print(f"Generating speech for text \"{sentence}\"...")
            kwargs["text"] = sentence
            wav_filename = self._tts_session.streaming_wav_generation_and_playback(**kwargs)
            wav_file_paths.append(wav_filename)
        # concatentate wav files and return
        return self._concatenate_wav_files(wav_file_paths)

    def _rebundle_sentences_intelligently_ahdhering_to_TTS_limits(self, sentences : list[str]) -> list[str]:
        """ Rebundle sentences intelligently to avoid exceeding TTS limits.

        The basic idea is to split on sentence boundaries and to balance the number of characters
        per bundle subject to the character limit.

        Edge case not currently handled: a single sentence that exceeds the character limit is simply passed
        running the risk of truncation in the speech mid-sentence for the corresponding TTS call.

        Parameters
        ----------
        sentences : list of str
            list of sentences to be rebundled

        Returns
        -------
        List of strings, each corresponding to a sentence bundle designed as the input to a TTS call
        """
        # calculate lengths of sentences
        sentence_lengths = [ len(s) for s in sentences ]
        total_length = sum(sentence_lengths)

        # calculate factor by which we have exceeded the TTS limit, and from that derive a target text lengh
        excess_factor_float = total_length / TTS_CHARACTER_LIMIT
        excess_factor_int = math.ceil(excess_factor_float)
        target_length = total_length / excess_factor_int
          
        # bundle sentences into texts of length >= target_length and <= TTS_CHARACTER_LIMIT
        rebundled_sentences = []
        current_bundle = ""
        running_length = 0
        sentence_index = 0
        # step thugh sentences, building bundles
        while sentence_index < len(sentences):
            sentence = sentences[sentence_index]
            bundle_is_full = False
            # consider to add sentence to bundle
            running_length_with_sentence = running_length + sentence_lengths[sentence_index]
            # if sentence fits into bundle, add it to the bundle and continue, otherwise close previous bundle
            if running_length_with_sentence <= TTS_CHARACTER_LIMIT:
                current_bundle += f" {sentence}"
                running_length = running_length_with_sentence
                sentence_index += 1
                # check whether or not we have hit the target length
                if (running_length >= target_length):
                    bundle_is_full = True
            else: # bundle is already full
                bundle_is_full = True
            # catch condition where the sentence itself exceeds the limit (to avoid infinite while loop)
            if running_length == 0 and bundle_is_full:
                print(f"WARNING: sentence exceeds {TTS_CHARACTER_LIMIT}-character limit: \"{sentence}\"")
                #TODO handle this condition more gracefully by imposing a split mid-sentence that is ideally
                # alligned with a comma ',' or dash '—' or semicolon
                current_bundle += f" {sentence}"
                running_length = running_length_with_sentence
                sentence_index += 1
            # when bundle is full, add it to the list of rebundled sentences and reset current bundle
            if bundle_is_full:
                rebundled_sentences.append(current_bundle)
                current_bundle = ""
                running_length = 0
        # process any lingering bundles
        if(current_bundle != ""):
            rebundled_sentences.append(current_bundle)

        return rebundled_sentences

    def _concatenate_wav_files(self, audio_clip_paths: list[str]) -> str:
        """Concatenate several audio files and save into one audio file,
        returning the name of that file.
        
        Parameters
        ----------
        audio_clip_paths : list of str

        Returns
        -------
        str : the name of the concatenated audio file
        """
        # edge case: no audio path
        if not audio_clip_paths:
            return None
        # edge case: only one audio path
        elif len(audio_clip_paths) == 1:
            return audio_clip_paths[0]

        # create a joined file
        data = []
        # combine the file names by starting with the tts_ prefix, then apending ids
        output_file_path = f"{os.path.dirname(audio_clip_paths[0])}/tts"
        for filename in audio_clip_paths:
            # read wave file
            w = wave.open(filename, "rb")
            wave_params = w.getparams()
            # parse ID from input file and append to name of output file
            output_file_path += re.search('(_[0-9_]+)\.wav', filename).group(1)
            # append silence
            num_silence_frames = int((SILENCE_DURATION_BETWEEN_CONSECTIVE_SPEECHES
                                      / self._speed_of_speech) * wave_params.framerate)
            silence_data = bytes(0 for i in range(num_silence_frames * wave_params.sampwidth))
            data.append([wave_params, silence_data])
            # append the audio data
            data.append([wave_params, w.readframes(w.getnframes())])
            w.close()
        output_file_path += ".wav"
        # write output file and return filename
        output = wave.open(output_file_path, "wb")
        output.setparams(data[0][0])
        for i in range(len(data)):
            output.writeframes(data[i][1])
        output.close()
        return output_file_path


def print_info_for_all_server_addresses(port: int):
    """ Print to the terminal all reachable addresses of this server.

    Parameters
    ----------
    port : int
        The port we are serving on.
    """
    print("\nServing TTS on all addresses (0.0.0.0)")
    print(f"* http://localhost:{port}")
    for interface in ni.interfaces():
        if ni.AF_INET in ni.ifaddresses(interface):
            print(f"* http://{ni.ifaddresses(interface)[ni.AF_INET][0]['addr']}:{port}")
    print(f"* http://{socket.gethostname()}:{port}")


if __name__ == "__main__":
    # parse port
    parser = argparse.ArgumentParser(description="TTS Server.")
    parser.add_argument('-p', '--port', help="(optional argument) the port to serve on (default: 8003)", type=int, default=8003)
    parser.add_argument('--deepspeed', help="(optional flag) use deepspeed package for accelerated inference", action='store_true')
    parser.add_argument('--pylips', help="(optional flag) generate visemes, and animate robot face if Pylips server is running", action='store_true')
    args = parser.parse_args()
    # instantiate and initialize streaming TTS object
    tts_session = StreamingTTS(deepspeed_acceleration=args.deepspeed, actuate_pylips=args.pylips)
    # serve
    handler = partial(MyRequestHandler, tts_session)
    httpd = HTTPServer(('0.0.0.0', args.port), handler)
    print_info_for_all_server_addresses(args.port)
    httpd.serve_forever()
