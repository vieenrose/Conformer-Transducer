# Must use a Cuda version 11+
FROM nvcr.io/nvidia/nemo:22.11

WORKDIR /

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
ADD requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
