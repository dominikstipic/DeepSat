PIPELINE_REPOSITORY="repository"
DATASET_DIR="data"
REPORT_DIR="reports"

build: requirements structure

auth:
	dvc remote modify drive gdrive_use_service_account true
	dvc remote modify drive --local gdrive_service_account_json_file_path .dvc/service.json

pull: auth
	dvc pull

push: auth
	dvc push

structure:
	mkdir -p $(PIPELINE_REPOSITORY)
	mkdir -p $(DATASET_DIR)
	mkdir -p $(REPORT_DIR)

test: 
	pytest

requirements: 
	pip install -r requirements.txt

clean:
	rm -r $(PIPELINE_REPOSITORY)
	rm -r $(DATASET_DIR)
	rm -r $(REPORT_DIR)
