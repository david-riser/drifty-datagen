# README
python3 -m venv env
pip3 install -r requirements.txt
prefect backend server
docker build -t generator_image
docker run -v $PWD/data:/data generator_image