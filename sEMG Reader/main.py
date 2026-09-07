import serial
import time


PORT = input("Enter the serial port: ")
BAUDRATE = 115200
START_TIME = time.perf_counter()

def time_since_start():
    return time.perf_counter() - START_TIME

try: 
    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
        print(f"Connected to {PORT} at {BAUDRATE} baud.")
        

        time.sleep(2) 
        
        ser.reset_input_buffer()

        while True:
            if ser.in_waiting > 0: 
                raw_data = ser.readline()

                try: 
                    decoded_data = raw_data.decode('utf-8').strip()
                    split_data = decoded_data.split(' ')
                    for i in range(len(split_data)):
                        split_data[i] = int(split_data[i])
                        split_data[i] = split_data[i] / 100.0
                        split_data[i] = round(split_data[i], 0)

                    ser.reset_input_buffer()

                                    
                    print(f"Received: {split_data} at {time_since_start():.2f} seconds since start.")

                except UnicodeDecodeError:
                    print("Received data could not be decoded.")
                    continue

               
                
except serial.SerialException as e:
    print(f"Error opening or using serial port: {e}")
except KeyboardInterrupt:
    print("\nProgram stopped by user.")


