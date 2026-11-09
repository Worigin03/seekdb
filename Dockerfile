# MiniVectorDB - build and runtime image.
#
#   docker build -t minivectordb .
#   docker run --rm -i -v /tmp/mvdb-data:/data minivectordb
#
# The program reads SQL from stdin, writes results/status to stdout and logs to
# stderr. It never opens a network port.

FROM ubuntu:24.04 AS build

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      g++ make ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY mini-vectordb/ /build/mini-vectordb/
RUN cd /build/mini-vectordb \
 && make -j"$(nproc)" \
      CXXFLAGS="-O3 -std=c++17 -funroll-loops -static-libstdc++ -static-libgcc" \
 && strip minivectordb

FROM ubuntu:24.04

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libstdc++6 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /data

COPY --from=build /build/mini-vectordb/minivectordb /usr/local/bin/minivectordb

WORKDIR /
ENTRYPOINT ["/usr/local/bin/minivectordb", "--data-dir=/data"]
