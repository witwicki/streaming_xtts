#!/usr/bin/env python

import re
from datetime import datetime
import requests # pip install requests
import argparse
import json

"""
Test Client for TTS server.

Current API:
- text (string) : the text to be spoken
- playback (bool) = False : whether or not the host should playback the audio {True, False}
- download (bool) = False  : whether or not the host should return a wav file  {True, False}
- split (string) = None : the method of splitting long speeches if any {None, "sentence", "intelligent"}
- language (string) = "en" : the language of the text to be spoken in iso_code_639_1 format, lower case
- speaker (string) = "Nova Hogarth" : the name of the xttsv2 speaker whose embeddings to use
- speed (float) = 1.0 : the speed of the speech (independent of tone and emotion), where 2.0 is twice as fast
- temperature = 0.01 : the temperature of the speech, where 0 is coldest

"""

# Parse Arguments
parser = argparse.ArgumentParser(description="TTS Client.")
parser.add_argument('text', help="the text to be spoken")
parser.add_argument('-p', '--playback', help="(streaming) playback of the audio from the host", action='store_true')
parser.add_argument('-d', '--download', help="Download the audio file upon completion", action='store_true')
parser.add_argument('--split', help="the method of splitting long speeches if any {None, \"sentence\", \"intelligent\"}", default="intelligent")
parser.add_argument('--language', help="the language of the text to be spoken in iso_code_639_1 format, lower case")
parser.add_argument('--speaker', help="the name of the xttsv2 speaker whose embeddings to use")
parser.add_argument('--speed', help="the speed of the speech (independent of tone and emotion), where 2.0 is twice as fast", type=float)
parser.add_argument('--temperature', help="the temperature of the speech, where 0.0 is coldest", type=float)
args = parser.parse_args()

# Check that one of the playback or download flags is set
if not (args.playback or args.download):
    parser.error("You must specify at least one of 'playback' or 'download' flags")

# Create dictionary of arguments
arg_list = list(arg for arg in vars(args))
arg_dict = {}
for arg in arg_list:
    if getattr(args, arg):
        arg_dict[arg] = getattr(args, arg)

# Convert to json string for HTTP POST
#text = args.text
#data = f'{{"text": "{text}", "split": "intelligent", "speed": 1.0}}'
#print(f"request: {data}\n")
data = json.dumps(arg_dict)
print(f"request: {data}\n")

# HTTP POST
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}
response = requests.post('http://localhost:8003', headers=headers, data=data)
print(f"raw response: {response}")
print(f"headers: {response.headers}")

# If not successful, print error message
if response.status_code == 400:
    # decode bytestring and remove HTML tags
    error_string = re.sub('<[^<]+?>', '', response.content.decode('utf8'))
    print(error_string)
# If successful, parse response
elif response.status_code == 200:
    # html response
    if response.headers['Content-Type'] == "text/html":
        print(response.text)
    elif response.headers['Content-Type'] == "audio/wav":
        filename = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".wav"
        with open(filename, 'wb') as file:
            for chunk in response.iter_content():
                file.write(chunk)
            print(f"...wrote ./{filename}")
