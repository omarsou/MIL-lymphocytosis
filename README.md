# MIL-lymphocytosis
Multi Instance Learning : Classification of Lymphocytosis from Blood Cells

## Topic : Diagnosis of lymphocytosis 
Lymphocytosis is a common finding, which can be either a reaction to infection, acute stress, and so on (termed reactive),
or the manifestation of a lymphoproliferative disorder --a type of cancer of the lymphocytes (termed tumoral).
In existing clinical practice, diagnosis as either reactive or tumoral relies on visual microscopic
examination of the blood cells together with the integration of clinical attributes such as age and 
lymphocyte count. Taking into consideration the visual assessment based on clinical attributes together
with texture and size of the lymphocytes in the blood smear, a diagnosis of the subtype of lymphoid 
malignancy is performed. On the positive side, such practice is fast and affordable. It suffers however
from poor reproducibility. Additional clinical tests are required, with flow cytometry being the gold
standard to definitively affirm the malignant nature of the lymphocytes. However, this analysis is 
relatively expensive and time-consuming, and therefore cannot be performed for every patient in 
practice. Therefore, the development of automatic and accurate processes could lead toa better way 
to determine which patient should be referred for flow cytometry analysis, augmenting and assisting 
the assessment of the clinicians.
### Classification task with Multi Instance Learning
To build a dataset for this problem, blood smears and patient attributes were collected from 204 patients
from the routine hematology laboratory of the Lyon SudUniversity Hospital. The samples were anonymized 
as required by the General Data Protection Regulation, keeping basic demographic information, age and sex, 
intact. The inclusion criteria were (a) a lymphocyte count above 4×10^9/L, and (b) absence of opposition 
to the research. The blood smears were automatically produced by a Sysmex automat tool, and the nucleated 
cells were automatically photographed with a DM-96 device. In particular, you will have access to 142 
subjects with 44 reactive and 98 malignant cases for training and 42 subjects for testing.

## Repository 
### dataloader_framework
In this folder, you will find some notebooks to deal with multi instance learning. More details can be found
in the folder's readme file.