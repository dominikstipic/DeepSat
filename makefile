PIPELINE_REPOSITORY="repository"
DATASET_DIR="data/AerialImageDataset"

build: requirements dvc structure

dvc:
	dvc pull

structure:
	mkdir -p $(PIPELINE_REPOSITORY)
	touch config.json

requirements: 
	pip install -r requirements.txt

clean:
	rm -r $(PIPELINE_REPOSITORY)
	rm -r $(DATASET_DIR)
