PIPELINE_REPOSITORY="repository"
DATASET_DIR="data/AerialImageDataset"

build: requirements dvc structure

dvc:
	dvc remote modify drive gdrive_use_service_account true
	dvc remote modify drive --local gdrive_service_account_json_file_path .dvc/service.json
	dvc pull

structure:
	mkdir -p $(PIPELINE_REPOSITORY)

requirements: 
	pip install -r requirements.txt

clean:
	rm -r $(PIPELINE_REPOSITORY)
	rm -r $(DATASET_DIR)
