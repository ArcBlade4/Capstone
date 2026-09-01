import PySimpleGUI as sg
import data_processing as dp
import threading 


running = False
last_size = 0
index = 0


# Define the layout of the GUI
layout = [
    # Define the title of the window
    [sg.Text('sEMG Reader', font=('Helvetica', 16), justification='center')],
    # Define file selection input
    [sg.Text('Select a file:'), sg.Input(key='-FILE-', enable_events=True), sg.FileBrowse()],
    # Define a button to start reading the file
    [sg.Button('Read File', key='-READ-', disabled=True)],
    # Define mode selection radio buttons
    [sg.Text('Select Mode:'), sg.Radio('Mode 1', 'MODE', key='-MODE1-', default=True), sg.Radio('Mode 2', 'MODE', key='-MODE2-'), sg.Radio('Mode 3', 'MODE', key='-MODE3-')]
]

# Create the window
window = sg.Window('sEMG Reader', layout)

# Event loop to process events and get the values of the inputs
while True:
    
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == '-FILE-':
        # Enable the Read File button when a file is selected
        window['-READ-'].update(disabled=False)
        file_path = values['-FILE-']
    elif event == '-READ-':
        # Handle the file reading logic here
        file_path = values['-FILE-']
        mode = 'Mode 1' if values['-MODE1-'] else 'Mode 2' if values['-MODE2-'] else 'Mode 3'
        dp.init(file_path, mode)
        dp.monitor_file_size(file_path, last_size, index)
        threading.Thread(target=dp.monitor_file_size, args=(file_path, last_size, index))

    
        

