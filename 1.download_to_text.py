# Given any URLs or file type, download and convert to text

import os, re
from pathlib import Path
from langchain.document_loaders import YoutubeLoader
from PyPDF2 import PdfFileReader # for pdfs
import docx2txt # for word docs
from bs4 import BeautifulSoup # for html
import requests
from urllib.parse import urlparse
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import torch
import sys
import numpy as np
import warnings
from pathlib import Path
import whisper
import string


# Supporting functions

# Check if URL is a youtube URL
def is_youtube_url(url):
  youtube_regex = (
    r'(https?://)?(www\.)?'
    '(youtube|youtu|youtube-nocookie)\.(com|be)/'
    '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    
  youtube_regex_match = re.match(youtube_regex, url)
  if youtube_regex_match:
    return True
  return False

def get_transcript(url):
    transcript=""
    go_ahead = False
    if "watch?v=" in url:
        video = YouTube(url)
        go_ahead = True
    elif "youtu.be" in url:
        v_id = url.split("/")[-1]
        video = YouTube("https://youtube.com/watch?v="+v_id)
        go_ahead = True
    else:
        print("Invalid YouTube URL")
        transcript = "Error"
    if(go_ahead):
        try:
            srt = YouTubeTranscriptApi.get_transcript(video.video_id)
            for item in srt:
                transcript = transcript+(item['text'])+" "
        except Exception as e:
            print ("Error loading transcript - it doesn't exist or is not in English. Try another video.")
    
    return transcript

# Check if URL is an Apple podcast URL
def is_apple_podcast_url(url):
  apple_podcast_regex = r'(https?://)?(www\.)?podcasts\.apple\.com/.+/podcast/.+'
  match = re.match(apple_podcast_regex, url)

  if match:
    return True
  return False

def find_audio_url(html: str) -> str:
    # Find all .mp3 and .m4a URLs in the HTML content
    audio_urls = re.findall(r'https://[^\s^"]+(?:\.mp3|\.m4a)', html)

    # If there's at least one URL, return the first one
    if audio_urls:
        pattern = r'=https?://[^\s^"]+(?:\.mp3|\.m4a)'
        result = re.findall(pattern, audio_urls[-1])
        if result:
          return result[-1][1:]
        else:
          return audio_urls[-1]

    # Otherwise, return None
    return None

def get_file_extension(url: str) -> str:
    # Parse the URL to get the path
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Extract the file extension using os.path.splitext
    _, file_extension = os.path.splitext(path)

    print("url", url, path, file_extension)
    # Return the file extension
    return file_extension

def download_apple_podcast(url: str, folder: str):
    output_folder = folder
    response = requests.get(url)
    if response.status_code != 200:
        print(
            f"Error: Unable to fetch the podcast page. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    audio_url = find_audio_url(response.text)

    if not audio_url:
        print("Error: Unable to find the podcast audio url.")
        return

    episode_title = soup.find('span', {'class': 'product-header__title'})

    if not episode_title:
        print("Error: Unable to find the podcast title.")
        return

# Remove or replace invalid characters for Windows file names
    episode_title = episode_title.text.strip().replace('/', '-').replace('\\', '-').replace(':', '-').replace('*', '-').replace('?', '-').replace('"', '-').replace('<', '-').replace('>', '-').replace('|', '-')

    # MAC LINE 
    # episode_title = episode_title.text.strip().replace('/', '-')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, f"{episode_title}{get_file_extension(audio_url)}")

    with requests.get(audio_url, stream=True) as r:
        r.raise_for_status()
        with open(output_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    if not output_file:
        print("Error: Unable to download podcast.")
    else:
        print(f"Downloaded podcast episode '{episode_title}' to '{output_folder}'")


def transcribe(filepath, title):
    device = torch.device('cuda')
    print('Using device:', device, file=sys.stderr)

    # Model selection
    model = 'medium'
    whisper_model = whisper.load_model(model)

    # Parameters
    language = "Auto detection"
    verbose = 'Live transcription'
    output_format = 'txt'
    task = 'transcribe'
    temperature = 0.15
    temperature_increment_on_fallback = 0.2
    best_of = 5
    beam_size = 8
    patience = 1.0
    length_penalty = -0.05
    suppress_tokens = "-1"
    initial_prompt = ""
    condition_on_previous_text = True
    fp16 = True
    compression_ratio_threshold = 2.4
    logprob_threshold = -1.0
    no_speech_threshold = 0.6

    verbose_lut = {
        'Live transcription': True,
        'Progress bar': False,
        'None': None
    }

    args = dict(
        language=(None if language == "Auto detection" else language),
        verbose=verbose_lut[verbose],
        task=task,
        temperature=temperature,
        temperature_increment_on_fallback=temperature_increment_on_fallback,
        best_of=best_of,
        beam_size=beam_size,
        patience=patience,
        length_penalty=(length_penalty if length_penalty >= 0.0 else None),
        suppress_tokens=suppress_tokens,
        initial_prompt=(None if not initial_prompt else initial_prompt),
        condition_on_previous_text=condition_on_previous_text,
        fp16=fp16,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold
    )

    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    if model.endswith(".en") and args["language"] not in {"en", "English"}:
        warnings.warn(f"{model} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    audio_path_local = Path(filepath).resolve()
    print("audio local path:", audio_path_local)

    args.pop('temperature_increment_on_fallback', None)
    transcription = whisper.transcribe(
        whisper_model,
        str(audio_path_local),
        **args,
    )
    if not os.path.exists("Full-Texts"):
        os.makedirs("Full-Texts")
    # Save output as a text file
    with open(os.path.join("Full-Texts", f"{title}.txt"), "w",encoding='utf-8') as f:
        f.write(transcription["text"])

    print(f"Transcript file created: {title}.txt")



def replace_special_chars(text):
  chars_to_replace = ["\\", "/", "?", "*", '"', "<", ">", "|", "%",":"] + list(string.whitespace)

  for char in chars_to_replace:
    text = text.replace(char, "_")

  return text

# =============================================================

# Main functions

with open('0.sources.txt') as f:
    urls = f.read().splitlines()

# Download all the URLs into text files
for url in urls: 
    if is_youtube_url(url):
        # download youtube video info
        try:
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=True,
                language=["en", "id"],
                translation="en",
            )
            title = replace_special_chars(loader.load()[0].metadata["title"])
            content = loader.load()[0].page_content
            content 
            # backup method to get transcript
            # content = get_transcript(url)
            with open(os.path.join("Full-Texts", f"{title}.txt"), "w",encoding='utf-8') as f:
                f.write(str(content))
            # with open("Full-Texts/" + str(title) + '.txt', 'w') as f:
            #     f.write(content)
        except Exception as e:
            print (e)
            continue
    
    if is_apple_podcast_url(url):
        download_apple_podcast(url, "Downloads") 

        podcast_files = os.listdir("Downloads")
        # instantiate pipeline with bfloat16 and enable batching
        #pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)

        for file in podcast_files:
            if file.endswith(".mp3"):
                podcast_name = os.path.splitext(file)[0]
                # Trying to use Whisper JAX
                # audio_file = os.path.join('Downloads', file)
                # outputs = pipeline(audio_file,  task="transcribe", return_timestamps=True)
                # text = outputs["text"]
                transcribe("Downloads/" + file, podcast_name)





# # print ("\n\nContent: "+content)

# files = 'Downloads'

# for file_path in Path(files).iterdir():
#     if file_path.suffix == '.pdf':
#         pdf_reader = PdfFileReader(str(file_path))
#         text = pdf_reader.getPage(0).extractText()
#     elif file_path.suffix == '.docx':
#         text = docx2txt.process(str(file_path)) 
#     elif file_path.suffix == '.html':
#         with open(file_path) as f:
#             soup = BeautifulSoup(f, 'html.parser')
#             text = soup.get_text()

#     # Save text to output file
#     with open(str(file_path) + '.txt', 'w') as f:
#         f.write(text)

