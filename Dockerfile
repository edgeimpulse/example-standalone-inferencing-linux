FROM ubuntu:20.04

WORKDIR /app

RUN apt update && apt install -y wget curl unzip build-essential
