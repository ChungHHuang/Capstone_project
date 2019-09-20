import numpy as np
import pandas as pd
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import zscore
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def bayesian_optimization(dataset, function, parameters):
	#Bayesian optimization
	X_train, y_train, X_test, y_test = dataset
	n_iterations = 5
	gp_params = {"alpha": 1e-4}

	BO = BayesianOptimization(function, parameters)
	BO.maximize(n_iter=n_iterations, **gp_params)

	return BO.max  

def rfc_optimization(X_train,y_train,cv_splits):
	def function(n_estimators, max_depth, min_samples_split):
		return cross_val_score(
			   RandomForestClassifier(
				   n_estimators=int(max(n_estimators,0)),                                                               
				   max_depth=int(max(max_depth,1)),
				   min_samples_split=int(max(min_samples_split,2)), 
				   n_jobs=-1, 
				   random_state=42,   
				   class_weight="balanced"),  
			   X=X_train, 
			   y=y_train, 
			   cv=cv_splits,
			   scoring="roc_auc",
			   n_jobs=-1).mean()

	parameters = {"n_estimators": (10, 1000),
				  "max_depth": (1, 150),
				  "min_samples_split": (2, 10)}
	
	return function, parameters  

def rfc_train(X_train, y_train, X_test, y_test, function, parameters):
	dataset = (X_train, y_train, X_test, y_test)
	cv_splits = 4
	
	best_solution = bayesian_optimization(dataset, function, parameters)      
	params = best_solution["params"]

	model = RandomForestClassifier(
			 n_estimators=int(max(params["n_estimators"], 0)),
			 max_depth=int(max(params["max_depth"], 1)),
			 min_samples_split=int(max(params["min_samples_split"], 2)), 
			 n_jobs=-1, 
			 random_state=42,   
			 class_weight="balanced")

	model.fit(X_train, y_train)
	
	return model 

def xgb_optimization(X_train,y_train,cv_splits, eval_set):
	def function(eta, gamma, max_depth):
			return cross_val_score(
				   xgb.XGBClassifier(
					   objective="binary:logistic",
					   learning_rate=max(eta, 0),
					   gamma=max(gamma, 0),
					   max_depth=int(max_depth),                                               
					   seed=42,
					   nthread=-1,
					   scale_pos_weight = len(y_train[y_train == 0])/
										  len(y_train[y_train == 1])),  
				   X=X_train, 
				   y=y_train, 
				   cv=cv_splits,
				   scoring="roc_auc",
				   fit_params={
						"early_stopping_rounds": 10, 
						"eval_metric": "auc", 
						"eval_set": eval_set},
				   n_jobs=-1).mean()

	parameters = {"eta": (0.001, 0.4),
				  "gamma": (0, 20),
				  "max_depth": (1, 2000)}
	
	return function, parameters

def quan_tab(model,x,y,cut=10):
	'''
	Obtain metrics table
	model: The machine learning model
	x: Input dataframe
	y: Target variable
	cut: Number of bucket
	'''
	# Get probability from model
	readmit_prob = [i[1] for i in model.predict_proba(x)]
	quantile_ = pd.qcut(readmit_prob,cut,labels=np.arange(1,cut+1))
	temp_dict = {'quantile':quantile_,'prob':readmit_prob,'Readmitted':y}
	# Create dataframe
	quan_table = pd.DataFrame(temp_dict).sort_values('quantile')
	temp1 = quan_table.groupby('quantile')['prob'].agg(['count','min','max']).reset_index()
	temp2 = quan_table.groupby(['quantile','Readmitted']).count().reset_index()
	temp2 = temp2[temp2.Readmitted==1]
	quan_table = pd.merge(temp2, temp1, how='inner',on ='quantile').drop('Readmitted',1).\
					rename({'prob': 'readmitted','count':'amount'}, axis=1)
	# Reorder columns
	quan_table = quan_table[quan_table.columns[[0,2,1,3,4]]]
	quan_table['pct_readmitted'] = quan_table.readmitted/quan_table.amount
	quan_table['lift'] = quan_table.pct_readmitted.apply(lambda x:10*x/quan_table.pct_readmitted.sum())
	return quan_table

def get_data(filePath, labelEncode=True, dummy=False, skewness=False, standardize = False, diag_group = False):

	# Read csv file
	diab_df = pd.read_csv(filePath) 
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


	if diag_group:
		# Group diagnosis codes
		# There are more than 700 unique diagnosis codes, so we formed them into 9 groups
		# The groups that covered less 3.5% of encounters were grouped as “Other”
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
	if dummy:
		print('Dummify variables and drop the most frequent category')
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







