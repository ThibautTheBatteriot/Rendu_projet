import subprocess, os, signal, time
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from shutil import copyfile

app = Flask(__name__, static_folder='templates')
socketio = SocketIO(app)

import os
#print("Chemin d'accès du répertoire de travail actuel:", os.getcwd())  # Chemin d'accès du répertoire de travail actuel / Current working directory path

airports = pd.read_csv('Data/airports_updated.csv')

# Sélection des lignes où being_transcripted est égal à 1 / Select rows where being_transcripted is equal to 1
transcripted_airports = airports[airports['being_transcripted'] == 1]

# Affichage des résultats / Display results
print(transcripted_airports)

processes = {}
# Déterminez le chemin de l'interpréteur Python de l'environnement virtuel / Determine the path to the Python interpreter of the virtual environment
python_interpreter = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.venv', 'Scripts', 'python.exe')  # Windows

for index, row in transcripted_airports.iterrows():
    pls_link = row['PLS_link']
    print("##################### OPENING " + pls_link)
    # Lancer le script avec subprocess et passer pls_link comme argument / Launch the script with subprocess and pass pls_link as argument
    process = subprocess.Popen([python_interpreter, 'live_record_transcript.py', pls_link])
    processes[pls_link] = process

print(processes)

# Fonction pour terminer tous les sous-processus / Function to terminate all subprocesses
def terminate_processes():
    for pls_link, process in processes.items():
        print(f"Terminating process for {pls_link}")
        os.kill(process.pid, signal.SIGTERM)
        del processes[pls_link]

# Gestionnaire de signal pour la terminaison / Signal handler for termination
def signal_handler(sig, frame):
    print("Signal received, terminating processes...")
    terminate_processes()
    os._exit(0)

# Enregistrement des gestionnaires de signaux / Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/')
def index():
    # Lire les données depuis le CSV / Read data from CSV

    #print(airports.to_dict(orient='records'))
    #Convertir les données en liste de dictionnaires pour faciliter leur utilisation dans le template / Convert data to list of dictionaries for easier use in the template
    print("getting [index.html]")
    return render_template('index.html', airports=airports.to_dict(orient='records'))

@app.route('/media_player')
def media_player():

    airport_name = request.args.get('name', 'Default Airport')
    pls_link     = request.args.get('pls_link', 'Default Link')

    print("getting [media_player.html] with :\nairport_name = " + airport_name + "\nPLS_link = " + pls_link)
    return render_template('media_player.html', airport_name=airport_name, pls_link=pls_link)

@app.route('/transcripts')
def transcripts():

    airport_name = request.args.get('name', 'Default Airport')
    pls_link     = request.args.get('pls_link', 'Default Link')

    print("getting [transcribe.html] with :\nairport_name = " + airport_name + "\nPLS_link = " + pls_link)
    return render_template('transcripts.html', airport_name=airport_name, pls_link=pls_link)

@app.route('/manage_current_transcribes_url')
def manage_current_transcribes_url():

    to_do = request.args.get('to_do', '')
    manage_current_transcribes(to_do)

    return jsonify(message="Function called successfully\n"+ to_do + "\n"), 200

def manage_current_transcribes(to_do):

    pls_link = to_do.split('._.')[1]
    command = "add" if to_do.startswith('start_transcribe._.') else "remove"

    # Mettre à jour le DataFrame et le fichier CSV / Update the DataFrame and CSV file
    update_being_transcribed(pls_link, command)

def update_being_transcribed(pls_link, command):
    print(pls_link)

    # Identifier la ligne à mettre à jour / Identify the row to update
    row_index = airports[airports['PLS_link'] == pls_link].index

    if not row_index.empty:
        if command == "add":
            airports.loc[row_index, 'being_transcripted'] = 1

            process = subprocess.Popen([python_interpreter, 'live_record_transcript.py', pls_link])
            processes[pls_link] = process
        else:
            airports.loc[row_index, 'being_transcripted'] = 0

            os.kill(processes[pls_link].pid, signal.SIGTERM)
            del processes[pls_link]

        # Écrire les modifications dans le fichier CSV / Write changes to the CSV file
        airports.to_csv('Data/airports_updated.csv', index=False)
    else:
        print(f"No matching row found for PLS_link: {pls_link}")

if __name__ == '__main__':
    app.run(debug=True)
