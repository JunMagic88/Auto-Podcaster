# Given any URLs or file type, download and convert to text

import os, re
from pathlib import Path
from langchain.document_loaders import YoutubeLoader
from PyPDF2 import PdfFileReader # for pdfs
import docx2txt # for word docs
from bs4 import BeautifulSoup # for html
import requests
from urllib.parse import urlparse


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



# =============================================================

# Main functions

with open('0.sources.txt') as f:
    urls = f.read().splitlines()

for url in urls: 
    if is_youtube_url(url):
        # download youtube video info
        try:
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=True,
                language=["en", "id"],
                translation="en",
            )
            title = loader.load()[0].metadata["title"]
            content = loader.load()[0].page_content
            with open("Full-Texts/" + str(title) + '.txt', 'w') as f:
                f.write(content)
        except Exception as e:
            continue
    
    if is_apple_podcast_url(url):
        download_apple_podcast(url, "Downloads") 

    # if is_apple_podcast(url):
    #     print ("transcribe apple podcast")









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

