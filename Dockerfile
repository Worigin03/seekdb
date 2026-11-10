# MiniVectorDB - build and runtime image.
#
#   docker build -t minivectordb .
#   docker run --rm -i -v /tmp/mvdb-data:/data minivectordb
#
# The program exchanges JSONL with the judge over stdin/stdout and logs to stderr.

FROM ubuntu:24.04 AS build

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      g++ ca-certificates rapidjson-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY mini-vectordb/ /build/mini-vectordb/
# Compile directly instead of going through make: the repository .gitignore
# ignores files named Makefile, so a Makefile cannot be relied upon being
# present in a checkout of this branch.
RUN cd /build/mini-vectordb \
 && g++ -O3 -std=c++17 -funroll-loops -pthread -static-libstdc++ -static-libgcc \
      -o minivectordb src/*.cpp \
 && strip minivectordb

FROM ubuntu:24.04

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libstdc++6 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /data \
 && chmod 777 /data

COPY --from=build /build/mini-vectordb/minivectordb /usr/local/bin/minivectordb

WORKDIR /
ENTRYPOINT ["/usr/local/bin/minivectordb", "--data-dir=/data", "--jsonl"]
