# FEAT Fairness Methodology: Predictive Underwriting Assessment

This repository provides Jupyter notebooks and supporting code for the
Predictive Underwriting case study assessment in the FEAT Principles Assessment Case
Studies Document (Veritas Document 4). Please see Section 2 of that document for the Predictive Underwriting
case study assessment itself.

This code should be considered an *alpha* pre-release version.
It comes with **absolutely no warranty**.
It is not intended for use in production,
or for assessing high risk AIDA systems under the methodology.

This work was undertaken as part of the Veritas initiative commissioned by the
Monetary Authority of Singapore, whose goal is to accelerate the adoption of
responsible Artificial Intelligence and Data Analytics (AIDA) in the financial
services industry.

## Contents

The key files and folders are the following:

- `data_exploration/`: Exploration of the dataset
   - **exploration_fairness.ipynb**
- `group_fairness/`: Evaluation of the group fairness
   - **Group_Fairness_Metrics_postproc_gender.ipynb**:Calculation of group fairness metrics and mitigation trade-offs between fairness metrics and model performance with a post-processing approach
   - **Group_Fairness_Metrics_postproc_race.ipynb**:Calculation of group fairness metrics and mitigation trade-offs between fairness metrics and model performance with a post-processing approach
   - **Personal_Attribute_Analysis.ipynb**: Justification of utilizing personal attributes (Leave one out method)
- `puw_modelling/`: ML models
   - **puw-model_baseline.ipynb**: ML models excluding the personal attributes
   - **puw-model_all_variables.ipynb**: ML models including the personal attributes
- `utils/`: Python code providing functionality to support the above notebooks


## License

Written by Sankarshan Mridha (Swiss Re) and Laura Alvarez (Accenture) as an extension to Phase 1 Credit Scoring Use Case code https://github.com/veritas-project/phase1/tree/main/credit_scoring 

Contact email: Veritas@mas.gov.sg


Copyright © 2021 Monetary Authority of Singapore

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License
