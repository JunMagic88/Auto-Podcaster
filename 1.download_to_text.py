# Given any URLs or file type, download and convert to text

import os, re
from pathlib import Path
from langchain.document_loaders import YoutubeLoader
from PyPDF2 import PdfFileReader # for pdfs
import docx2txt # for word docs
from bs4 import BeautifulSoup # for html

# Supporting functions
def is_youtube_url(url):
  youtube_regex = (
    r'(https?://)?(www\.)?'
    '(youtube|youtu|youtube-nocookie)\.(com|be)/'
    '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

  youtube_regex_match = re.match(youtube_regex, url)
  if youtube_regex_match:
    return True
  return False


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
            with open("Downloads/" + str(title) + '.txt', 'w') as f:
                f.write(content)
        except Exception as e:
            continue










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

