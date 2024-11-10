# thunder-cat

## Installation
1. Install Python 3.11.*.
2. Install linux video utiliy devices: `sudo apt install v4l-utils`.
3. Create a virtual environment and `pip install -r requirements.txt`. Note you can save the current dependencies using `pip freeze > requirements.txt`.
4. Add the source code folder to `PYTHONPATH` (i.e `export PYTHONPATH="${PYTHONPATH}:${PWD}/src"`)
5. (Optional) run `pre-commit install` to enable pre-commit hooks

## How to stream the camera to your host device via OpenCV
1. Determine the host IP address (e.g `ifconfig`) and a free port and assign them to Env variables `HOST_IP` and `HOST_PORT`  on the device side.
2. Run `python ./stream_camera` on the device side.
3. On the host side, receive the video with a suitable UDP streaming tool. E.g `ffplay -fflags nobuffer udp://<HOST_IP>:<HOST_PORT>?pkt_size=1316`
