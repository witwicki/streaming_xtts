import sys, os
import shutil
import wave
import pickle 
import time
import urllib.request
from pylips.speech import RobotFace
from pylips.speech.system_tts import IPA2VISEME
from pylips.face import ExpressionPresets as ep # default, angry, disgust, fear, happy, sad, surprise
from allosaurus.app import read_recognizer

PYLIPS_IP = "localhost"
PYLIPS_PORT = "8008"

def check_http_server(ip_address, port):
    try:
        # Attempt to connect to the IP address and port
        connection = urllib.request.urlopen("http://{}:{}".format(ip_address, port), timeout=1)
        return True
    except Exception as e:
        # If there's an exception, then the HTTP server is probably not running
        print("Error connecting to PyLips Server:", e)
        return False


class Robot():
    def __init__(self):
        # if server is not running, nullify robot
        if not check_http_server(PYLIPS_IP, PYLIPS_PORT):
            self._robot = None
        else:
            self._robot = RobotFace(server_ip=f'http://{PYLIPS_IP}:{PYLIPS_PORT}')

    def lip_sync(self, audio_file_path, language='eng', emotion=ep.default):

        # get audio file key (name without path or extension)
        audio_file_name = os.path.basename(audio_file_path)
        audio_file_key = os.path.splitext(audio_file_name)[0]

        # prep pylips_phrases directory
        output_dir=os.path.join(os.getcwd(),'./pylips_phrases')
        os.makedirs(output_dir, exist_ok=True)
        data_file_path = os.path.join(output_dir,audio_file_name)
        if not os.path.exists(data_file_path):
            shutil.copy2(audio_file_path, data_file_path)

        # check if robot is available, else return
        if self._robot is None:
            return

        # get sample rate and length
        with wave.open(audio_file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            sample_length = wf.getnframes()

        # extract phonemes
        phoneme_model = read_recognizer()
        phonemes = phoneme_model.recognize(audio_file_path, timestamp=True, lang_id=language)

        # derive visemes
        times = [i.split(' ')[0] for i in phonemes.split('\n')]
        visemes = [IPA2VISEME[i.split(' ')[-1]] for i in phonemes.split('\n')]
        times.append(sample_length/sample_rate + 0.2)
        visemes.append('IDLE')

        # create pickle
        pickle.dump((times, visemes), open(f'{output_dir}/{audio_file_key}.pkl', 'wb'))

        # animate lips
        self._robot.say_file(audio_file_key)


    def compute_and_store_visemes(self, audio_file_path, language='eng', emotion=ep.default):

        # get audio file key (name without path or extension)
        audio_file_name = os.path.basename(audio_file_path)
        audio_file_key = os.path.splitext(audio_file_name)[0]

        # prep pylips_phrases directory
        output_dir=os.path.join(os.getcwd(),'./pylips_phrases')
        os.makedirs(output_dir, exist_ok=True)
        data_file_path = os.path.join(output_dir,audio_file_name)
        if not os.path.exists(data_file_path):
            shutil.copy2(audio_file_path, data_file_path)

        # get sample rate and length
        with wave.open(audio_file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            sample_length = wf.getnframes()

        # extract phonemes
        phoneme_model = read_recognizer()
        phonemes = phoneme_model.recognize(audio_file_path, timestamp=True, lang_id=language)

        # derive visemes
        times = [i.split(' ')[0] for i in phonemes.split('\n')]
        visemes = [IPA2VISEME[i.split(' ')[-1]] for i in phonemes.split('\n')]
        times.append(sample_length/sample_rate + 0.2)
        visemes.append('IDLE')

        # create pickle
        pickle.dump((times, visemes), open(f'{output_dir}/{audio_file_key}.pkl', 'wb'))

    def lip_visemes(self, audio_file_key):
        if self._robot:
            self._robot.lip_visemes(audio_file_key)


