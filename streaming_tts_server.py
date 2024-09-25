import json
import re
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from functools import partial
import netifaces as ni
#from stream2sentence import generate_sentences
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from streaming_tts import StreamingTTS, WrongTypeError

PORT = 8003
TTS_CHARACTER_LIMIT = 250

''' This server takes text as input and plays back voice on the host machine.  Additionally, it serves up pylips. '''

def print_info_for_all_server_addresses():
    print("\nServing TTS on all addresses (0.0.0.0)")
    print(f"* http://localhost:{PORT}")
    for interface in ni.interfaces():
        if ni.AF_INET in ni.ifaddresses(interface):
            print(f"* http://{ni.ifaddresses(interface)[ni.AF_INET][0]['addr']}:{PORT}")


class MyRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, tts_session, *args, **kwargs):
        self._tts_session = tts_session
        super().__init__(*args, **kwargs)

    def set_empty_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()    

    def send_error_response(self, error):
        self.send_response(400)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(f"<html<body><h1>Bad Request: {error}</h1></body></html>", "utf-8"))

    def send_wav_file_as_response(self, wav_filename):
        self.send_response(200)
        self.send_header("Content-type", "audio/wav")
        self.end_headers()
        with open(wav_filename, "rb") as wav_file:
            self.wfile.write(wav_file.read())

    def do_HEAD(self):
        self.set_empty_headers()

    def do_GET(self):
        self.set_empty_headers()

    
    def do_POST(self):
        """
        TTS audio playback (streaming) and wave-file retrieval (delivery only after playback).

        Content is expected as a utf-8 encoded JSON string, with the following fields:
        - text (string) : the text to be spoken
        - playback (bool) = True : whether or not the host should playback the audio {True, False}
        - download (bool) = True  : whether or not the host should return a wav file  {True, False}
        - split (string) = None : the method of splitting long speeches if any {None, "sentence", "intelligent"}
        - language (string) = "en" : the language of the text to be spoken in iso_code_639_1 format, lower case
        - speaker (string) = "Nova Hogarth" : the name of the xttsv2 speaker whose embeddings to use
        - speed (float) = 1.0 : the speed of the speech (independent of tone and emotion), where 2.0 is twice as fast
        - temperature = 0.01 : the temperature of the speech, where 0 is coldest
        """
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        content = json.loads(body.decode('utf-8'))
        # parse 'playback' and 'download' arguments
        do_playback = False
        do_download = False
        if "playback" in content:
            do_playback = content.pop("playback")
        if "download" in content:
            do_download = content.pop("download")
        # check for text field
        text = content.get("text", None)
        success = True
        error_string = None
        return_filename = ""
        bool
        # invoke tts engine 
        if text:
            try:
                return_filename = self._speek_and_return_wav(**content)
            except WrongTypeError as e:
                error_string = f"WrongTypeError: {e}"
                success = False
            except Exception as e:
                error_string = f"Exception: {e}"
                success = False
        else:
            error_string = "No text field in request"
            success = False
        # send response (with or without wav file)
        if success:
            if do_download:
                self.send_wav_file_as_response(return_filename)
            else:
                self.set_empty_headers()
        else:
            print(error_string)
            self.send_error_response(error_string)

    def _speek_and_return_wav(self, **kwargs):
        text = kwargs.get("text")
        sentences = [ text ]
        split_into_sentences = None
        if "split" in kwargs:
            split_into_sentences = kwargs.pop("split")
        if split_into_sentences:
            # regex for splitting sentences
            sentence_splitter_j = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|。)\s+'
            sentences = re.split(sentence_splitter_j, text.replace('。','。 '))
            # TODO: compare performance with generate_sentences(content["text"]):
            if split_into_sentences == "intelligent":
                sentences = self._rebundle_sentences_intelligently_ahdhering_to_TTS_limits(sentences)
        for sentence in sentences:
            print(f"TTS saying: {sentence}")
            kwargs["text"] = sentence
            wav_filename = self._tts_session.streaming_wav_generation_and_playback(**kwargs)
            return wav_filename
        # TODO combine waveforms

        
    def _rebundle_sentences_intelligently_ahdhering_to_TTS_limits(self, sentences):
        # calculate lengths of sentences
        sentence_lengths = [ len(s) for s in sentences ]
        total_length = sum(sentence_lengths)
        print(f"sentence_lengths={sentence_lengths}")
        # calculate factor by which we have exceeded the TTS limit, and from that derive a target text lengh
        excess_factor_float = total_length / TTS_CHARACTER_LIMIT
        excess_factor_int = math.ceil(excess_factor_float)
        target_length = total_length / excess_factor_int
        print(f"excess_factor_float ={excess_factor_float}, excess_factor_int={excess_factor_int}, target_lenth={target_length}")
        
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


if __name__ == "__main__":
    tts_session = StreamingTTS()
    handler = partial(MyRequestHandler, tts_session)
    httpd = HTTPServer(('0.0.0.0', PORT), handler)
    print_info_for_all_server_addresses()
    httpd.serve_forever()
