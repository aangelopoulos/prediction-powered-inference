from folktables import ACSDataSource, ACSEmployment
import pdb

if __name__ == "__main__":
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)	
    pdb.set_trace()
    print("CRT!")
