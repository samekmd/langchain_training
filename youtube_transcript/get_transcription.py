from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import re 

def get_video_id(url):
    # Regular expression to extract the video ID from the URL
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

def get_transcription(url):
    try:
        video_id = get_video_id(url)
        result = YouTubeTranscriptApi().fetch(video_id, languages=["pt", "pt-BR", "en"])
        formatter = TextFormatter()
        transcricao = formatter.format_transcript(result)
        transcricao = transcricao.replace('\n', ' ')
        return transcricao
    except Exception as e:
        print(f"Error fetching transcription: {e}")
        return ""

def save_transcription_to_file(transcricao, filename):
    # Change the way the file is written, use paragraph breaks
    with open(filename, 'w', encoding='utf-8') as file:
        paragraphs = transcricao.split('. ')
        for paragraph in paragraphs:
            file.write(paragraph.strip() + '.\n\n')

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=tXM3Ifd6_T8"  # Substitua pela URL do vídeo desejado
    transcricao = get_transcription(url)
    save_transcription_to_file(transcricao, "transcricao.txt")