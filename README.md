# thunder-cat

## Installation
1. Install Python 3.11.*.
2. Install linux video utiliy devices: `sudo apt install v4l-utils`.
3. Create a virtual environment and `pip install -r requirements.txt`. Note you can save the current dependencies using `pip freeze > requirements.txt`.
4. Add the source code folder to `PYTHONPATH` (i.e `export PYTHONPATH="${PYTHONPATH}:${PWD}/src"`)
5. Run `./generate_data_folders.sh` to create the folders to contain sound, video and logging data.
6. (Optional) run `pre-commit install` to enable pre-commit hooks.

## Running

### Streaming to another device
1. Determine the host IP address (e.g `ifconfig`) and a free port and assign them to Env variables `HOST_IP` and `HOST_PORT` on the device side.
2. Run `python ./src/main.py` on the device side.
3. On the host side, receive the video with a suitable UDP streaming tool. E.g `ffplay -fflags nobuffer udp://<HOST_IP>:<HOST_PORT>?pkt_size=1316`

## Other

### Logging
The logging of the code can be controlled with the `LOG_LEVEL` env var. For example `export LOG_LEVEL="DEBUG"`.s
