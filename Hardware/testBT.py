import socket

IP_ADDRESS = "192.168.4.1" # Replace with the IP address of your ESP8266 in AP mode
PORT = 80

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((IP_ADDRESS, PORT))

while True:
    message = client.recv(1024).decode().strip()
    print("Recieved:",message)
    response = input("enter response") + "\n"
    client.send(response.encode())

