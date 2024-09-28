__author__ = "Stefan Witwicki"
__copyright__ = "Copyright (C) 2024 Stefan Witwicki"
__license__ = "GNU General Public License version 2"
__version__ = "0.1"

"""Module for animating a robot face using the PyLips package.

Functionalities included:
  - lip-syncing with playback from input wav file
  - generation of visemes from wav file
  - animating lips according generated visemes (animating without playing any audio)
  - changing facial expression for above cases to a one of the preset states 
    {default, angry, disgust, fear, happy, sad, surprise}

Example usage:
  face = Robot()
  face.compute_and_store_visemes("/path/to/audio_file_with_key.wav")
  face.lip_visemes("/path/to/audio_file_with_key)
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

import sys, os
import shutil
import wave
import pickle 
import time
import urllib.request
from pylips.speech import RobotFace
from pylips.speech.system_tts import IPA2VISEME
from pylips.face import ExpressionPresets as ep 
from allosaurus.app import read_recognizer

# constants
PYLIPS_IP = "localhost"
PYLIPS_PORT = "8008"

def check_http_server(ip_address: str, port: int) -> bool:
    """Check for running PyLips Server
    
    Parameters
    ----------
    ip_address : str
        IP address of the server.
    port : int
        Port number of the server.

    Returns
    -------
    bool
        True if server is running, False otherwise.
    """
    try:
        # Attempt to connect to the IP address and port
        address = f"http://{ip_address}:{port}/face"
        print(address)
        connection = urllib.request.urlopen(address, timeout=1)
        return True
    except Exception as e:
        # If there's an exception, then the HTTP server is probably not running
        print(f"Error connecting to PyLips Server @ {address}:", e)
        return False


class Robot():
    """Class for animating PyLips robot face.

    Attributes
    ----------
    _robot : pylips.speech.RobotFace
        An instance of a RobotFace object from pylips.
    
    Methods
    -------
    lip_sync(audio_file, language='eng', emotion=ep.default)
        Lip syncs and plays back the audio file in the given language, with emotion
    compute_and_store_visemes(audio_file_path, language='eng')
        Computes viseme for an audio file in the given language and stores it in a pickle file
    lip_visemes(audio_file_key, emotion=ep.default)
        Animates lips according to visemes stored in pickle file with prefix 'audio_file_key'
    """
    def __init__(self):
        # if server is not running, nullify robot
        if not check_http_server(PYLIPS_IP, PYLIPS_PORT):
            self._robot = None
        else:
            self._robot = RobotFace(server_ip=f'http://{PYLIPS_IP}:{PYLIPS_PORT}')

    def lip_sync(self, audio_file_path: str, language: str = 'eng', emotion: dict[str,float] = ep.default):
        """Lip syncs and plays back the audio file in the given language, with emotion.

        Parameters
        ----------
        audio_file_path : str
            Path to an audio file
        language : str
            Language of the audio file
        emotion : dict[str, float], optional
            Expresssion parameters for the robot face (see pylips.face.ExpressionPresets for examples)
        """

        # check if robot is available, else return
        if self._robot is None:
            return

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

        # animate lips
        self._robot.express(emotion, 0.25)
        self._robot.say_file(audio_file_key)

    def compute_and_store_visemes(self, audio_file_path: str, language: str = 'eng'):
        """Compute and store visemes for a given audio file in a given language. 
        
        Parameters
        ----------
        audio_file_path : str
            Path to the audio file.
        language : str
            Language of the audio file (default is 'eng').
        """

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

    def lip_visemes(self, audio_file_key: str, emotion: dict[str, float] = ep.default):
        """Animate lips according to visemes and emotional parameters.

        Parameters
        ----------
        audio_file_key : str
            File key (without extension) where visemes pickle is stored
        emotion : dict[str, float], optional
            Expresssion parameters for the robot face (see pylips.face.ExpressionPresets for examples)
        """
        if self._robot:
            self._robot.express(emotion, 0.25)
            self._robot.lip_visemes(audio_file_key)


