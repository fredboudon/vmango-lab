FROM jvail/plantgl-jupyter:1.0.5

ARG NB_USER=jovyan
ARG NB_UID=1000

COPY . /home/${NB_USER}
USER root
SHELL ["/bin/bash", "-c"]
RUN conda env update -f /home/${NB_USER}/environment.yml && \
    chown -R ${NB_UID} /home/${NB_USER} && \
    python /home/${NB_USER}/setup.py develop
WORKDIR /home/${NB_USER}

USER ${NB_USER}
