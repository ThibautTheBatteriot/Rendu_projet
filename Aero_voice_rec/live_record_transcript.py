import sys

# Vérifier qu'un argument a été passé
# Check that an argument was passed
if len(sys.argv) < 2:
    print("Usage: python live_record_transcript.py <pls_link>")
    sys.exit(1)

# Récupérer le paramètre passé
# Retrieve the passed parameter
pls_link = sys.argv[1]
print(f"Parameter received: {pls_link}")

####### <INPUT> ######
#stream_url = 'http://d.liveatc.net/kden1_1'
stream_url = pls_link
####### </INPUT> ######


stream_name = stream_url.split('/')[-1]
mp3_output_folder = "./templates/recordings/" + stream_name + "/"


print("""#-0-###"""+stream_name+"""###   Loading : Imports  #####""")

import pyaudio, ffmpeg, wave
import datetime, time
import os, keyboard, csv #filemanagment
import threading, queue
import subprocess, json
import numpy as np
from pydub import AudioSegment
import noisereduce as nr


if not os.path.exists(mp3_output_folder):
    os.makedirs(mp3_output_folder)

csv_file_name =  stream_name + '.csv'
csv_output_folder = "./templates/transcripts/"
csv_file_path = os.path.join(csv_output_folder, csv_file_name)

if not os.path.exists(csv_output_folder):
    os.makedirs(csv_output_folder)
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'a') as f:
        pass  # Create an empty file



print("""#-1-###"""+stream_name+"""###   LOADING pipes  #####""")
pipes = {}

from transformers import pipeline
pipes["transcribe"] = {}
pipes["transcribe"]["scy0208/whisper-aviation-base"] = pipeline("automatic-speech-recognition", model="scy0208/whisper-aviation-base")
     
print("""#-2-###"""+stream_name+"""###   LOADING functions  #####""")


def get_stream_info(stream_url):
    # Appel de ffprobe pour obtenir les détails du flux
    # Call ffprobe to get stream details
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-print_format', 'json',
        '-show_streams',
        '-select_streams', 'a',  # sélectionne seulement les streams audio
                                # select only audio streams
        stream_url
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Failed to fetch stream info")
        return None

    # Decoupe le JSON
    # Parse the JSON output
    try:
        info = json.loads(result.stdout)
        audio_streams = info.get('streams', [])
        if not audio_streams:
            print("No audio stream found")
            return None

        # Supposons que le premier flux audio est celui que nous voulons
        # Assume the first audio stream is the one we want
        audio_info = audio_streams[0]
        return {
            'rate': int(audio_info['sample_rate']),
            'channels': int(audio_info['channels']),
            'format': audio_info['codec_name'],
            'sample_format': audio_info['sample_fmt']
        }
    except json.JSONDecodeError:
        print("Failed to decode JSON from ffprobe output")
        return None
						
						
def save_chunk(data, mp3_output_folder=mp3_output_folder):
    """Sauvegarde les données audio dans un fichier horodaté et convertit en MP3."""

    timestamp       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wav_filename    = timestamp + ".wav"
    mp3_filename    = timestamp + ".mp3"
    mp3_NR_filename = timestamp + "_NR.mp3"
    wav_path        = os.path.join(mp3_output_folder, wav_filename)
    mp3_path        = os.path.join(mp3_output_folder, mp3_filename)
    mp3_NR_path     = os.path.join(mp3_output_folder, mp3_NR_filename)
    
    # Sauvegarde au fornat WAV
    # Save in WAV format
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format_audio))
        wf.setframerate(rate)
        wf.writeframes(b''.join(data))
    
    # Conversion en MP3 avec ffmpeg
    # Conversion to MP3 with ffmpeg
    #ffmpeg.input(wav_path).output(mp3_path).run(overwrite_output=True) #mode VERBOSE
    try:
        # Adjusted ffmpeg command
        ffmpeg.input(wav_path).output(mp3_path).run(overwrite_output=True, capture_stderr=True, capture_stdout=True)
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")
        raise
    os.remove(wav_path)  
    # Supprimer le fichier WAV après conversion
    # delete the WAV file afterward

    #audio, samples = getAudioAndSamples(mp3_path)
    audio = getAudioAndSamples(mp3_path)
    make_reduced_noise_version(audio, mp3_NR_path)

    return mp3_path, mp3_NR_path

def make_reduced_noise_version(audio, filepath):
    samples = np.array(audio.get_array_of_samples()) 
    # conversion de l'Audiosegment pydub vers un array numpy                
    # Convert pydub AudioSegment to numpy array
    reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate) 

    save_np_array_to_mp3(reduced_noise, audio, filepath)

    return reduced_noise

def save_np_array_to_mp3(np_array, audio, filepath):
    # Conversion inverse array numpy > pydub audio segment
    # Convert the numpy array back to pydub audio segment
    processed_audio_NR = AudioSegment(
        data=np_array.tobytes(), 
        sample_width=audio.sample_width, 
        frame_rate=audio.frame_rate, 
        channels=1
    )
    processed_audio_NR.export(filepath, format="mp3")

def getAudioAndSamples(audio_file_path) :
    audio   = AudioSegment.from_file(audio_file_path, format="mp3")
    #samples = np.array(audio.get_array_of_samples())

    return audio#, samples

def Save_and_transcribe(buffer) :
    mp3_file_path, mp3_NR_file_path  = save_chunk(buffer)
    audio_to_transcripts_queue.put([mp3_file_path, mp3_NR_file_path])


print("""#-3-###"""+stream_name+"""###   LOADING Threads  #####""")

# Fonction pour lire le flux
# Function to read the stream
def Thread_fetch_live_audio():
    process = (
        ffmpeg
        .input(stream_url, re=None)
        .output('pipe:', format='wav')
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.DEVNULL)  # Redirect stderr to dev null
    )
    try:
        while True:
            data = process.stdout.read(chunk_size)
            if not data:
                break
            audio_chunk_queue.put(data)
    finally:
        # Signal de fin
        # Sending final/terminal signal
        audio_chunk_queue.put(None)  
        process.kill()

# Fonction pour traiter l'audio
# Function to process audio
def Thread_process_audio():

    audio_buffer = []
    is_silent_since  = None
    already_recorded = False
    nb_records = 0

    while True:
        data = audio_chunk_queue.get()
        if data is None:
            break

        data_debuff = np.frombuffer(data, dtype=np.int16)
        data_sum    = np.sum(np.abs(data_debuff))

        if (data_sum<10000) : ### SILENCE
            #print(f"\rSILENCE", end="")
            if already_recorded :
                continue
            if is_silent_since is None:
                is_silent_since = datetime.datetime.now()
            elif not already_recorded :
                silence_length = (datetime.datetime.now() - is_silent_since).total_seconds()
                if silence_length >= silence_threshold :

                    Save_and_transcribe(audio_buffer)

                    nb_records += 1
                    #print(f"{nb_records} records")
                    audio_buffer = []
                    already_recorded = True

        else :                ### SPEAKING
            #print(f"\rSPEAKING", end="")
            audio_buffer.append(data)
            already_recorded = False
            is_silent_since = None

    if audio_buffer:
        Save_and_transcribe(audio_buffer)

# Fonction pour transcrire audio > text
#function to transcribe audio > text
def Thread_transcript_audio():

    while True:
        file_paths = audio_to_transcripts_queue.get()
        if file_paths is None:
            break

        transcription1 = pipes["transcribe"]["scy0208/whisper-aviation-base"](file_paths[0])['text']

        #ajout potentiel de la transcription de la version noise_reduced du fichier
        #potential add of the transcription of the noise_reduced version of the file
        #transcription1NR = pipes["transcribe"]["scy0208/whisper-aviation-base"](file_paths[1])['text']
        
        timestamp_file_name = os.path.splitext(file_paths[0].split('/')[-1])[0]
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp_file_name, file_paths[0], transcription1])

        print(transcription1)


print("""#-4-###"""+stream_name+"""###   INIT settings  #####""")



# Initialisation de PyAudio
p = pyaudio.PyAudio()

#Creation d'un lien vers /dev/null pour eviter lque le programme cherche a lire l'audio si le serveur ne dispose pas de carte audio
#lui permet de process le flux audio sans chercher a le lire
#Create a link to /dev/null to prevent the program from trying to play the audio if the server has no audio card.
#allows it to process the audio stream without trying to play it back
dev_null = open(os.devnull, 'w')

# Configuration initiale
# Initial configuration
stream_info = get_stream_info(stream_url)

chunk_size = 256
format_audio = pyaudio.paInt16
channels = stream_info['channels']
rate = stream_info['rate']
silence_threshold = 0.1  # En secondes

#Queue des segments audio transmis
#Queue of tranmitted audio chunks
audio_chunk_queue          = queue.Queue()

#Queue des fichiers a transcrire
#Queue of files to transcribes
audio_to_transcripts_queue = queue.Queue()


print("""#-5-###"""+stream_name+"""###   STARTING Threads  #####""")

# Declaration des / of threads
fetch_thread     = threading.Thread(target=Thread_fetch_live_audio)
process_thread   = threading.Thread(target=Thread_process_audio)
transcript_thread = threading.Thread(target=Thread_transcript_audio)

# Démarrage/Starting  threads
fetch_thread.start()
process_thread.start()
transcript_thread.start()


print("""#-6-###"""+stream_name+"""###   WAITING Threads ends #####""")

# Attendre que les threads finissent
# Waiting threads end

fetch_thread.join()
process_thread.join()
transcript_thread.join()

dev_null.close()


print("""#-7-###"""+stream_name+"""###   FINISH ####""")
