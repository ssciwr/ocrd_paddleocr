ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://github.com/ssciwr/ocrd_paddleocr/issues" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/ssciwr/ocrd_paddleocr" \
    org.label-schema.build-date=$BUILD_DATE

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONIOENCODING utf8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PIP pip3

# avoid HOME/.local/share (hard to predict USER here)
# so let XDG_DATA_HOME coincide with fixed system location
# (can still be overridden by derived stages)
ENV XDG_DATA_HOME /usr/local/share
# avoid the need for an extra volume for persistent resource user db
# (i.e. XDG_CONFIG_HOME/ocrd/resources.yml)
ENV XDG_CONFIG_HOME /usr/local/share/ocrd-resources

WORKDIR /build-ocrd
COPY . .
COPY ocrd-tool.json .
# prepackage ocrd-tool.json as ocrd-all-tool.json
RUN ocrd ocrd-tool ocrd-tool.json dump-tools > $(dirname $(ocrd bashlib filename))/ocrd-all-tool.json
# prepackage ocrd-all-module-dir.json
RUN ocrd ocrd-tool ocrd-tool.json dump-module-dirs > $(dirname $(ocrd bashlib filename))/ocrd-all-module-dir.json

# install everything
RUN pip install paddlepaddle==3.2.0
RUN pip install .

WORKDIR /data
VOLUME /data
