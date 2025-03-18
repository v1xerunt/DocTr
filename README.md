# DocTr

Source code for paper DocTr: Optimizing Clinical Trial Site Selection using Open Payments and Patient Encounter Data

## Requirements

* Install python, pytorch and RecBole. We use Python 3.7.6, Pytorch 1.12.1.
* We use the Clinical-Trial-Parser to parse trial criteria from https://github.com/facebookresearch/Clinical-Trial-Parser/tree/main.
* If you plan to use GPU computation, install CUDA
* The composite similarity metric need to be manually added to the ```RecBole/evaluator/metrics.py```. The metrics calculation function is in ```utils.py/com_sim```.

## Data resources

All data should be downloaded in the ```data``` folder.

**Public external data**
- The CMS Open payments data https://www.cms.gov/priorities/key-initiatives/open-payments/data/dataset-downloads. We use the ```OP_DTL_RSRCH_PGYRXXXX_P01202023.csv``` from 2017-2021.
- US State level zipcode mapping file ```uszips.csv``` from https://github.com/akinniyi/US-Zip-Codes-With-City-State/tree/master
- Trial XML files from https://clinicaltrials.gov/

We have provided processed data in the ```data``` folder. They can be read using ```pickle.read``` Some key files are:

- ```npi2trial.pkl```: The linked relationship between NPI and NCTID.
- ```npi_info_dict.pkl```: The clinician information extracted from CMS data, including location information and other public information.
- ```payment_dict.pkl```: The processed CMS dataset. Recording the payment record from each trial identified by NCTID to each clinician or teaching hospital identified by NPI.
- ```ie_extracted_clinical_trials.tsv```: The processed trial criteria using the Clinical-Trial-Parser.

## Notebook files

### 01 - Data preprocessing
- ```01_A_process_payment_data.ipynb```: Extract the clinical trial and clinician relationship from the OpenPayment data.
- ```01_B_process_trial_info.ipynb```: Parse clinical trial information from trial XML documents.
- ```01_C_process_trial_criteria_embd.ipynb```: Generate the trial criteria embeddings using ClinicalBERT.
- ```01_D_process_trial_summary_embd.ipynb```: Generate the trial summary embeddings using ClinicalBERT.
- ```01_E_process_claims_data.ipynb```: Process the ICD codes in the claims data.
- ```01_F_process_clinician_info.ipynb```: Extract the clinician information from the CMS data.
- ```01_G_process_geo_data.ipynb```: Extract demographics information (e.g., racial and ethnicity distributions) from the regional data.

### 02 - Data linkage
- ```02_A_gen_trial_npi_relation.ipynb```: Link and filter trials and clinicians information.
- ```02_B_get_trial_phase.ipynb```: Get trial phase and condition information.
- ```02_C_get_stat.ipynb```: Get basic data statistics of the dataset we built.
### 03 - Builiding recommendation dataset
- ```03_A_gen_atom_file.ipynb```: Build the atomic dataset under regular setting for recommendation model training, based on the requirement of the RecBole package.
- ```03_A_gen_zeroshot_atom_file.ipynb```: Build the atomic dataset under temporal setting for recommendation model training, based on the requirement of the RecBole package.

### 04 - Model training and evaluations
- ```04_train_doctr.ipynb```: We build and evaluate the proposed DocTr model.

### 05 - Additional analysis
- ```05_A_get_competing_trial.ipynb```: We extract the competing trials from the trial relationships.
- ```05_B_fairness_analysis.ipynb```: We run the genetic algorithm to improve the fairness of the recommendation results, and report the results. The genetic algorithm is in ```genetic.py```.
