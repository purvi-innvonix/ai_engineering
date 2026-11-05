import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import YoutubeLoader

def get_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL."""
    return YoutubeLoader.extract_video_id(url)

from youtube_transcript_api import YouTubeTranscriptApi

def load_youtube_transcript(url: str):
    """
    Load transcript documents for a given YouTube URL.
    Automatically detects available languages (e.g. Hindi, English, etc.)
    and loads the best matching transcript.
    Returns a list of Document objects.
    """
    video_id = get_video_id(url)
    print(f"\n[+] Extracted Video ID: {video_id}")

    # Inspect all available languages for this video
    try:
        available_transcripts = YouTubeTranscriptApi().list(video_id)
        available_languages = [t.language_code for t in available_transcripts]
        print(f"[+] Available Transcript Languages: {available_languages}")
    except Exception as e:
        print(f"[-] Note on language detection: {e}")
        available_languages = []

    # Priority order: English first, Hindi next, then any other available language
    preferred_order = ["en", "en-US", "hi", "hi-IN"]
    language_list = preferred_order + [lang for lang in available_languages if lang not in preferred_order]

    print(f"[+] Loading transcript via YoutubeLoader with language priority: {language_list[:4]}...")
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,
        language=language_list
    )
    docs = loader.load()
    return docs

def load_universal_youtube_transcript(url: str, custom_preferred_languages: list = None):
    """
    Universal Multi-Language Loader.
    Dynamically discovers ALL available languages on the video (Telugu, Tamil, Hindi,
    English, Spanish, etc.) and loads the transcript in the original language of the video.
    
    Optionally accepts a list of preferred language codes to try first.
    """
    video_id = get_video_id(url)
    print(f"\n[+] Extracted Video ID: {video_id}")

    # 1. Discover all languages available on YouTube for this video
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        available_languages = [t.language_code for t in transcript_list]
        print(f"[+] Discovered Available Languages on YouTube: {available_languages}")
    except Exception as e:
        print(f"[-] Could not fetch available languages list: {e}")
        available_languages = []

    # 2. Setup priority: Custom -> Common Indian & Global Languages -> Whatever is on the video
    default_priority = [
        "en", "en-US", "en-GB",   # English
        "hi", "hi-IN",           # Hindi
        "te", "te-IN",           # Telugu
        "ta", "ta-IN",           # Tamil
        "kn", "kn-IN",           # Kannada
        "mr", "mr-IN",           # Marathi
        "bn", "bn-IN",           # Bengali
        "gu", "gu-IN",           # Gujarati
        "es", "fr", "de", "ja"   # Global languages
    ]
    
    preferred = custom_preferred_languages or default_priority
    # Build complete list with priority order while removing duplicates
    languages_to_try = []
    for lang in preferred + available_languages:
        if lang not in languages_to_try:
            languages_to_try.append(lang)

    print(f"[+] Universal Loader attempting languages in order: {languages_to_try[:6]}...")
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,
        language=languages_to_try
    )
    docs = loader.load()
    
    # Tag each document with detected video metadata
    for doc in docs:
        doc.metadata["available_languages"] = available_languages
        doc.metadata["video_id"] = video_id
        
    return docs

# Alias for backward compatibility if called as load_transcript
load_transcript = load_youtube_transcript

def main():
    default_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    if len(sys.argv) > 1:
        url = sys.argv[1].strip()
    else:
        user_input = input(f"Enter YouTube video URL [Press Enter for default: {default_url}]: ").strip()
        url = user_input if user_input else default_url

    print(f"\nTarget URL: {url}")
    
    try:
        documents = load_youtube_transcript(url)
        
        if not documents:
            print("[-] No transcript documents found for this video.")
            return

        doc = documents[0]
        print(f"[+] Successfully loaded {len(documents)} document(s).")
        
        print("\n--- Document Metadata ---")
        print(doc.metadata)
        
        print("\n--- Document Content Statistics ---")
        print(f"Total Character Count: {len(doc.page_content)}")
        
        print("\n--- First 1,000 Characters of Transcript ---")
        print(doc.page_content)
        print("...\n-------------------------------------------")
        
    except Exception as e:
        print(f"[-] Error loading transcript: {e}")

if __name__ == "__main__":
    main()
