import numpy as np
import pandas as pd
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import zscore

def get_data(filePath, labelEncode=True, hotEncode=False, skewness=False, standardize = False):

	diab_df = pd.read_csv('diabetic_data.csv') 
	print('Original data shape {}'.format(diab_df.shape))
	# Create target column
	diab_df.readmitted = diab_df.readmitted.apply(lambda x: 'Yes' if x in ['<30'] else 'No')

	# Missing data
	print('Process Missing data')
	# Weight had more than 90% of missing data. Payer code had about 40% of missingness, and it was not considered relavent to results.
	# All citoglipton and examide are 'No'
	diab_df.drop(['weight','payer_code','citoglipton', 'examide'],1,inplace=True)
	diab_df.medical_specialty.replace('?','Missing',inplace=True)
	diab_df.race.replace('?','Missing',inplace=True)
	diab_df.diag_1.replace('?','Missing',inplace=True)
	diab_df.diag_2.replace('?','Missing',inplace=True)
	diab_df.diag_3.replace('?','Missing',inplace=True)

	# Group diagnosis codes
	# Diagnosis 1
	diab_df['diag_1_group'] = diab_df['diag_1']
	diab_df.loc[diab_df['diag_1'].str.contains('V'), ['diag_1_group']] = 1000
	diab_df.loc[diab_df['diag_1'].str.contains('E'), ['diag_1_group']] = 1000
	diab_df.loc[diab_df['diag_1'].str.contains('250'), ['diag_1_group']] = 2500
	diab_df.diag_1_group.replace('Missing',-1,inplace=True)
	diab_df.diag_1_group = diab_df.diag_1_group.astype(float)
	diab_df.diag_1_group[((diab_df.diag_1_group>=390) & (diab_df.diag_1_group<460)) | (diab_df.diag_1_group==785)] = 1001
	diab_df.diag_1_group[((diab_df.diag_1_group>=460) & (diab_df.diag_1_group<520)) | (diab_df.diag_1_group==786)] = 1002
	diab_df.diag_1_group[((diab_df.diag_1_group>=520) & (diab_df.diag_1_group<580)) | (diab_df.diag_1_group==787)] = 1003
	diab_df.diag_1_group[((diab_df.diag_1_group>=800) & (diab_df.diag_1_group<1000))] = 1005
	diab_df.diag_1_group[((diab_df.diag_1_group>=710) & (diab_df.diag_1_group<740))] = 1006
	diab_df.diag_1_group[((diab_df.diag_1_group>=580) & (diab_df.diag_1_group<630)) | (diab_df.diag_1_group==788)] = 1007
	diab_df.diag_1_group[((diab_df.diag_1_group>=140) & (diab_df.diag_1_group<240))] = 1008
	diab_df.diag_1_group[((diab_df.diag_1_group>=0) & (diab_df.diag_1_group<1000))] = 1000
	diab_df.diag_1_group.replace(1001,'Circulatory',inplace=True)
	diab_df.diag_1_group.replace(1002,'Respiratory',inplace=True)
	diab_df.diag_1_group.replace(1003,'Digestive',inplace=True)
	diab_df.diag_1_group.replace(2500,'Digestive',inplace=True)
	diab_df.diag_1_group.replace(1005,'Injury',inplace=True)
	diab_df.diag_1_group.replace(1006,'Musculoskeletal',inplace=True)
	diab_df.diag_1_group.replace(1007,'Genitourinary',inplace=True)
	diab_df.diag_1_group.replace(1008,'Neoplasms',inplace=True)
	diab_df.diag_1_group.replace(1000,'Other',inplace=True)
	diab_df.diag_1_group.replace(-1,'Missing',inplace=True)
	# Diagnosis 2
	diab_df['diag_2_group'] = diab_df['diag_2']
	diab_df.loc[diab_df['diag_2'].str.contains('V'), ['diag_2_group']] = 1000
	diab_df.loc[diab_df['diag_2'].str.contains('E'), ['diag_2_group']] = 1000
	diab_df.loc[diab_df['diag_2'].str.contains('250'), ['diag_2_group']] = 2500
	diab_df.diag_2_group.replace('Missing',-1,inplace=True)
	diab_df.diag_2_group = diab_df.diag_2_group.astype(float)
	diab_df.diag_2_group[((diab_df.diag_2_group>=390) & (diab_df.diag_2_group<460)) | (diab_df.diag_2_group==785)] = 1001
	diab_df.diag_2_group[((diab_df.diag_2_group>=460) & (diab_df.diag_2_group<520)) | (diab_df.diag_2_group==786)] = 1002
	diab_df.diag_2_group[((diab_df.diag_2_group>=520) & (diab_df.diag_2_group<580)) | (diab_df.diag_2_group==787)] = 1003
	diab_df.diag_2_group[((diab_df.diag_2_group>=800) & (diab_df.diag_2_group<1000))] = 1005
	diab_df.diag_2_group[((diab_df.diag_2_group>=710) & (diab_df.diag_2_group<740))] = 1006
	diab_df.diag_2_group[((diab_df.diag_2_group>=580) & (diab_df.diag_2_group<630)) | (diab_df.diag_2_group==788)] = 1007
	diab_df.diag_2_group[((diab_df.diag_2_group>=140) & (diab_df.diag_2_group<240))] = 1008
	diab_df.diag_2_group[((diab_df.diag_2_group>=0) & (diab_df.diag_2_group<1000))] = 1000
	diab_df.diag_2_group.replace(1001,'Circulatory',inplace=True)
	diab_df.diag_2_group.replace(1002,'Respiratory',inplace=True)
	diab_df.diag_2_group.replace(1003,'Digestive',inplace=True)
	diab_df.diag_2_group.replace(2500,'Digestive',inplace=True)
	diab_df.diag_2_group.replace(1005,'Injury',inplace=True)
	diab_df.diag_2_group.replace(1006,'Musculoskeletal',inplace=True)
	diab_df.diag_2_group.replace(1007,'Genitourinary',inplace=True)
	diab_df.diag_2_group.replace(1008,'Neoplasms',inplace=True)
	diab_df.diag_2_group.replace(1000,'Other',inplace=True)
	diab_df.diag_2_group.replace(-1,'Missing',inplace=True)
	# Diagnosis 3
	diab_df['diag_3_group'] = diab_df['diag_3']
	diab_df.loc[diab_df['diag_3'].str.contains('V'), ['diag_3_group']] = 1000
	diab_df.loc[diab_df['diag_3'].str.contains('E'), ['diag_3_group']] = 1000
	diab_df.loc[diab_df['diag_3'].str.contains('250'), ['diag_3_group']] = 2500
	diab_df.diag_3_group.replace('Missing',-1,inplace=True)
	diab_df.diag_3_group = diab_df.diag_3_group.astype(float)
	diab_df.diag_3_group[((diab_df.diag_3_group>=390) & (diab_df.diag_3_group<460)) | (diab_df.diag_3_group==785)] = 1001
	diab_df.diag_3_group[((diab_df.diag_3_group>=460) & (diab_df.diag_3_group<520)) | (diab_df.diag_3_group==786)] = 1002
	diab_df.diag_3_group[((diab_df.diag_3_group>=520) & (diab_df.diag_3_group<580)) | (diab_df.diag_3_group==787)] = 1003
	diab_df.diag_3_group[((diab_df.diag_3_group>=800) & (diab_df.diag_3_group<1000))] = 1005
	diab_df.diag_3_group[((diab_df.diag_3_group>=710) & (diab_df.diag_3_group<740))] = 1006
	diab_df.diag_3_group[((diab_df.diag_3_group>=580) & (diab_df.diag_3_group<630)) | (diab_df.diag_3_group==788)] = 1007
	diab_df.diag_3_group[((diab_df.diag_3_group>=140) & (diab_df.diag_3_group<240))] = 1008
	diab_df.diag_3_group[((diab_df.diag_3_group>=0) & (diab_df.diag_3_group<1000))] = 1000
	diab_df.diag_3_group.replace(1001,'Circulatory',inplace=True)
	diab_df.diag_3_group.replace(1002,'Respiratory',inplace=True)
	diab_df.diag_3_group.replace(1003,'Digestive',inplace=True)
	diab_df.diag_3_group.replace(2500,'Digestive',inplace=True)
	diab_df.diag_3_group.replace(1005,'Injury',inplace=True)
	diab_df.diag_3_group.replace(1006,'Musculoskeletal',inplace=True)
	diab_df.diag_3_group.replace(1007,'Genitourinary',inplace=True)
	diab_df.diag_3_group.replace(1008,'Neoplasms',inplace=True)
	diab_df.diag_3_group.replace(1000,'Other',inplace=True)
	diab_df.diag_3_group.replace(-1,'Missing',inplace=True)

	diab_df.drop(['diag_1','diag_2','diag_3'],1,inplace=True)

	# Simplify some features
	# diab_df['max_glu_serum'].replace('>300','>200',inplace=True)
	# diab_df['A1Cresult'].replace('>8','>7',inplace=True)

	# Delete multipule encounters
	print('Delete multipule encounters')
	temp_df = diab_df.groupby('patient_nbr')['encounter_id'].min().reset_index()
	temp_df = pd.merge(temp_df,diab_df.drop('patient_nbr',1),'left',left_on='encounter_id',right_on='encounter_id')
	temp_df = temp_df[~temp_df['discharge_disposition_id'].isin([11,13,14,19,20,21])]
	temp_df.drop('patient_nbr',1,inplace=True)
	temp_df.drop('encounter_id',1,inplace=True)

	# Transform nominal columns to string type
	print('Transform features')
	temp_df.admission_type_id = temp_df.admission_type_id.astype(str)
	temp_df.discharge_disposition_id = temp_df.discharge_disposition_id.astype(str)
	temp_df.admission_source_id = temp_df.admission_source_id.astype(str)

	# Check outliers
	num_cols = temp_df.dtypes[temp_df.dtypes != "object"].index
	z = np.abs(zscore(temp_df[num_cols]))
	row, col = np.where(z > 4)
	df = pd.DataFrame({"row": row, "col": col})
	rows_count = df.groupby(['row']).count()

	outliers = rows_count[rows_count.col > 2].index
	# There are three rows have more than 2 features that have z-score higher than 4

	# Reduce skewness
	if skewness:
		print('Reduce skewness')
		num_cols = temp_df.dtypes[temp_df.dtypes != "object"].index
		skewed_cols = temp_df[num_cols].apply(lambda x: skew(x))
		skewed_cols = skewed_cols[abs(skewed_cols) > 0.75]
		skewed_features = skewed_cols.index

		for feat in skewed_features:
		    temp_df[feat] = boxcox1p(temp_df[feat], boxcox_normmax(temp_df[feat]+1))

	# Standardize numeric columns
	if standardize:
		print('Standardize numeric columns')
		scaler = StandardScaler()
		temp_df[num_cols] = scaler.fit_transform(temp_df[num_cols])

	# Get target column
	Y = temp_df['readmitted'].apply(lambda x: 1 if x =='Yes' else 0)
	temp_df.drop('readmitted',1,inplace=True)

	# Dummify
	if hotEncode:
		print('Conduce one-hot encoding')
		cate_col = temp_df.dtypes[temp_df.dtypes == object].index
		dummies_drop = [i + '_'+ temp_df[i].value_counts().index[0] for i in cate_col]
		temp_df = pd.get_dummies(temp_df)
		temp_df.drop(dummies_drop,axis=1,inplace=True)
		
	# LabelEncoder
	elif labelEncode:
		print('Conduce label encoding')
		cate_col = temp_df.dtypes[temp_df.dtypes == object].index
		# process columns, apply LabelEncoder to categorical features
		for i in cate_col:
		    lbl = LabelEncoder() 
		    lbl.fit(list(temp_df[i].values)) 
		    temp_df[i] = lbl.transform(list(temp_df[i].values))


	print('Data shape after preprocessing: {}'.format(temp_df.shape))
	return temp_df,Y







