from runpod/base:0.4.0-cuda11.8.0
workdir /app
copy docker/setup.sh /setup.sh
add src .
cmd python3.11 -u handler.py
