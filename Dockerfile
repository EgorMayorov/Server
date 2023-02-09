FROM docker.io/library/python:3.8-slim
COPY src /root/src
RUN chown -R root:root /root/src
WORKDIR /root/src
RUN pip3 install --no-cache-dir -r requirements.txt
ENV SECRET_KEY key
ENV FLASK_APP run.py
RUN chmod +x run.py
CMD ["python3", "run.py"]
