# DeepSat
###### Description
The aim of the DeepSat project is to create a system that can semantically segment the houses from the satellite images. The DeepSat is the implementation of ML pipeline which is composed of several stages. Each pipeline stage generates output, which is input to another stage and generates artefacts, which are used for system monitoring purposes. The stages are configured inside `config.json` file. 

###### Dataset
The dataset which is being used as a part of this project is the Inria Aerial Image Labeling Dataset. 
This dataset contains aerial images of 5 cities: Austin, Chicago, Kitsap County, Western Tyrol, Vienna. It is composed of 180 RGB images and accompanying semantic masks of the size 5000x5000. The link to the dataset is provided below: 
https://project.inria.fr/aerialimagelabeling/

The dataset for this project is stored on Google Drive repository and it is versioned by `dvc` data versioning system. The `dvc` enable us to keep track of dataset evolution through time and avoids storing dataset in GitHub repository. It does this by maintaining an index file called `data.dvc` in the GitHub repository and stores real data on a remote data storage, such as Google Drive. 

###### Usage
To start using DeepSat system you need to do the following steps:
* Put Google Cloud credentials in the `service.json` file and move it to `.dvc` directory. This step avoids click-based user authentification and enables full process automation.  
* Build your project by typing `make build`. This step instals all the project requirements and creates the project structure
* Type `make pull`. This step authenticates the client application by using `service.json` credentials and pulls the current version of the dataset into `data` directory
* Run the pipeline by typing `python main.py`. The `main.py` script can also be called by providing the following arguments:
    *  `--do_report`: Generates the report of the pipeline run and stores it into `reports` directory.
    * `--do_email`: emails the receiver by sending him a previously generated report. To correctly configure emailing part of the system you must fill `email.json` file. 
    * `--do_version`: versions this run with git versioning system. 

