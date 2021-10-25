IMAGE_NAME = forecast
CONTAINER_NAME = forecast

DOCKER_REGISTRY = registry.gitlab.com/Pierre-Aurelien/forecast
TAG = latest
TOKEN = not set
USERNAME = not set

ifneq ($(TAG), latest)
	IMAGE_TAG = $(TAG)
else
	IMAGE_TAG = latest
endif

ifeq ($(PULL), true)
	PULL = $(PULL)
else
	PULL = false
endif

ifeq ($(DOCKER_VERSION), 19.03)
	DOCKER_GPU_COMMAND = --gpus all
else
	DOCKER_GPU_COMMAND = --runtime nvidia
endif

login:
ifneq ($(TOKEN), not set)
  	ifneq (${USERNAME}, not set)
		docker login $(DOCKER_REGISTRY) -u ${USERNAME} -p ${TOKEN}
  	endif
else
	docker login $(DOCKER_REGISTRY)
endif

build:
ifeq ($(PULL),true)
	docker pull $(DOCKER_REGISTRY):$(IMAGE_TAG)
	docker image tag $(DOCKER_REGISTRY):$(IMAGE_TAG) $(IMAGE_NAME):latest
else
	docker build -t $(IMAGE_NAME) --build-arg host_gid=$$(id -g) --build-arg host_uid=$$(id -u) -f Dockerfile.local .
endif

run-local:
	docker run -it -d -e MACHINE_ID=`hostname` --name $(CONTAINER_NAME) $(DOCKER_GPU_COMMAND) -v ${PWD}:/home/app/forecast --restart always $(IMAGE_NAME):latest

run-no-gpu:
	docker run -it -d -e MACHINE_ID=`hostname` --name $(CONTAINER_NAME)  -v ${PWD}:/home/app/forecast --restart always $(IMAGE_NAME):latest

bash:
	docker exec -it $(CONTAINER_NAME) bash -c "pip install -e . && /bin/bash"

bash-arpeggio:
	docker exec -it $(ARPEGGIO_CONTAINER_NAME) bash -c "pip install -e . && /bin/bash"

remove:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)


setup-local:
	make remove || true
	make build run-local

setup-no-gpu:
	make remove || true
	make build run-no-gpu
