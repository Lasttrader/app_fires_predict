FROM ubuntu:latest
LABEL Julia Kotova, Grigpriy Sokolov, Michail Zaytcev. BMTSU Moscow 2021
RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev build-essential
COPY . /build
WORKDIR /build
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]
