FROM python:3.7
WORKDIR /

# Build python environment
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Get and run python code
COPY *.py ./
COPY config.yml . 
CMD ["python3", "app.py", "--config=config.yml"]
