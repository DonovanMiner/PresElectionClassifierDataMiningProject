from math import floor, sqrt
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PatchCollection



def growth(year, total_time, start_pop):
    pred_vals = []

    for val in range(len(year)):
        pop = start_pop * (2**(year[val]/total_time))
        pred_vals.append(pop)

    return pred_vals


def decay(year, total_time, start_pop):
    pred_vals = []

    for val in range(len(year)):
        pop = start_pop * (0.5**(year[val]/total_time))
        pred_vals.append(pop)

    return pred_vals



def main():
   



    #df_Master = pd.read_excel("PredictorsMasterDoc.xlsx")
    df_reduced = pd.read_excel("PredictorsReduced.xlsx")
    #df_Master = df_Master.to_numpy()
    df_reduced = df_reduced.to_numpy()

    

    #labels_master = np.array(df_Master[0:, 24:25], dtype=int)
    #data_master = np.array(df_Master[0:, 1:24], dtype=float)

    labels_reduced = np.array(df_reduced[0:, 12:13], dtype=int)
    data_reduced = np.array(df_reduced[0:, 1:12], dtype=float)
    ##print(labels_reduced)
    ##print(data_reduced)


    
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=8)

    X_train, X_test, y_train, y_test = train_test_split(data_reduced, labels_reduced, random_state=0)
    forest = RandomForestClassifier(n_estimators=100, criterion='entropy', min_impurity_decrease=0.01, max_depth=8).fit(X_train, np.ravel(y_train))
    print(np.mean(cross_val_score(forest, data_reduced, np.ravel(labels_reduced), cv=9)))


   
    class_prob = forest.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, class_prob)
    plt.show()

    


    #tree = DecisionTreeClassifier(criterion='entropy')
    #for depth in range(20):
    #    forest = RandomForestClassifier(n_estimators=100, criterion='entropy', min_impurity_decrease=0.01, max_depth=depth+1)

    #    scores = []
    #    for test in range(30):
    #        res = np.mean(cross_val_score(forest, data_reduced, np.ravel(labels_reduced), cv=9))
    #        #print(res)
    #        scores.append(res)
    #    avg_acc = np.mean(scores)
    #    print(f'Average Accuracy for Depth {depth+1}: {avg_acc}')


    #sfs = SequentialFeatureSelector(estimator=forest, cv=9)
    #sfs.fit(data_master, np.ravel(labels_master))
    #print('forward sfs support:')
    #print(sfs.get_support())


        

    #labels_train = np.array(labels[51:], dtype=int)
    #data_train = np.array(data[51:, 0:], dtype=float)
    #labels_test = np.array(labels[0:51], dtype=int)
    #data_test = np.array(data[0:51, 0:], dtype=float)
    #print(labels_train)
    #print(data_train)
    #print(labels_test)
    #print(data_test)


    #forward sequential fefature selection
    


    #decision tree CV feature selection
    #for feat in range(len(data_master[0])):
    #    curr_feat = data_master[0:, feat:feat+1]
    #    #print(feat_train)

    #    print(f'Feature {feat} Score: {np.mean(cross_val_score(tree, curr_feat, labels_master, cv=9))}')

    #    tree.fit(feat_train, labels_train)
    #    print(tree.score(feat_test, labels_test))
        









    #df_fill = pd.read_csv("ReligionStats.csv")

    #df_fill = df_fill.to_numpy()

    #df_fill = np.array(df_fill[0:, 1:7], dtype=float)
    #print(df_fill)

    #filled_results = np.zeros((1, 36), float)
    #for i in range(len(df_fill)):
        
    #    curr = df_fill[i]
    #    mean = np.mean(curr)
    #    std = np.std(curr)
    #    samples = np.random.normal(mean, std, 36)
    #    filled_results = np.insert(filled_results, i, samples, axis=0)

    #SAMPLES = pd.DataFrame(data=filled_results)
    #SAMPLES.to_excel("RandomFillValues.xlsx")
    


    
 
    
    
    #df_race = df_race.to_numpy()
    #race_years = np.array(df_race[0, len(df_race[0]):0:-1], dtype=int)
    #white = np.array(df_race[1:52, len(df_race[0]):0:-1], dtype=int)
    #hisp = np.array(df_race[53:104, len(df_race[0]):0:-1], dtype=int)
    ##print(race_years)
    ##print(white)
    ##print(hisp)

    #pred_years = np.arange(1988, 2024, 1,dtype=int)
    #white_res = np.zeros((1, 36), dtype=float)
    #hisp_res = np.zeros((1, 36), dtype=float)
    #whiteRs = np.zeros((1, 2), dtype=float)
    #hispRs = np.zeros((1, 2), dtype=float)
  

    #for state in range(len(white)):

    #    curr_state = white[state]
    #    fit = np.polyfit(race_years, curr_state, 1)
    #    p = np.poly1d(fit)
    #    pred_white = p(pred_years)
    #    R2 = r2_score(curr_state, p(race_years))

    #    print(f'R2/R: {R2} {np.sqrt(R2)}')
    #    white_res = np.insert(white_res, state, pred_white, axis=0)
    #    whiteRs = np.insert(whiteRs, state, [R2, np.sqrt(R2)], axis=0)


    #    #plt.plot(race_years, curr_state, 'o', label='Census Data')
    #    #plt.plot(pred_years, pred_white, '-', label='Regression')
    #    #plt.xlabel('Years')
    #    #plt.ylabel('White Population (in thousands)')
    #    #plt.title(f'White Population in State {state}')
    #    #plt.legend(loc='upper left')
    #    #plt.show()



    #for state in range(len(hisp)):

    #    curr_state = hisp[state]
    #    fit = np.polyfit(race_years, curr_state, 1)
    #    p = np.poly1d(fit)
    #    pred_hisp = p(pred_years)
    #    R2 = r2_score(curr_state, p(race_years))

    #    print(f'R2/R: {R2} {np.sqrt(R2)}')
    #    hisp_res = np.insert(hisp_res, state, pred_hisp, axis=0)
    #    hispRs = np.insert(hispRs, state, [R2, np.sqrt(R2)], axis=0)


        #plt.plot(race_years, curr_state, 'o', label='Census Data')
        #plt.plot(pred_years, pred_hisp, '-', label='Regression')
        #plt.xlabel('Years')
        #plt.ylabel('Hispanic Population (in thousands)')
        #plt.title(f'Hispanic Population in State {state}')
        #plt.legend(loc='upper left')
        #plt.show()

    #white_res = pd.DataFrame(data=white_res)
    #whiteRs =  pd.DataFrame(data=whiteRs)
    #hisp_res = pd.DataFrame(data=hisp_res)
    #hispRs = pd.DataFrame(data=hispRs)

    #white_res.to_excel("WhiteRes.xlsx")
    #whiteRs.to_excel("WhiteR.xlsx")
    #hisp_res.to_excel("HispRes.xlsx")
    #hispRs.to_excel("HispR.xlsx")


    #df_marriage = pd.read_excel("MarriageAggregate.xlsx")
    #print(df_marriage)
    #marriage = df_marriage.to_numpy()
    #marriage_years = np.array(marriage[0, 12:0:-1], dtype=int)
    #marriage = np.array(marriage[1:53, 12:0:-1], dtype=int)
    ##print(marriage_years)
    ##print(marriage)
    #pred_years = np.arange(1988, 2024, 1,dtype=int)
    #mar_res = np.zeros((1, 36), dtype=float)
    #mar_r = np.zeros((1, 2), dtype=float)

    #for state in range(len(marriage)):

    #    curr_state = marriage[state]
    #    fit = np.polyfit(marriage_years, curr_state, 1)
    #    p = np.poly1d(fit)
    #    pred_mar = p(pred_years)
    #    R2 = r2_score(curr_state, p(marriage_years))
        
    #    mar_res = np.insert(mar_res, state, pred_mar, axis=0)
    #    mar_r = np.insert(mar_r, state, [R2, np.sqrt(R2)], axis=0)

    #    plt.plot(marriage_years, curr_state, 'o', label='Census Data')
    #    plt.plot(pred_years, pred_mar, '-', label='Regression')
    #    plt.xlabel('Years')
    #    plt.ylabel('Number of Married Families (in thousands)')
    #    plt.title(f'Number of Married Families in State {state}')
    #    plt.legend(loc='upper left')
    #    plt.show()

    #df_MAR = pd.DataFrame(data=mar_res)
    #df_MAR_R = pd.DataFrame(data=mar_r)
    #df_MAR.to_excel("marProj.xlsx")
    #df_MAR_R.to_excel("marR.xlsx")







    #vets = df_vet.to_numpy()
    #vets = vets[0:, 1:]
    #vet_years = np.array(vets[0, len(vets[0]):0:-1], dtype=int)
    #vets = np.array(vets[1:, len(vets[1]):0:-1], dtype=int)
    #pred_years = np.arange(1988, 2024, 1, dtype=int)
    #print(vet_years)
    #print(vets)

    #VetResults = np.zeros((1, 36), dtype=float)
    #VetRs = np.zeros((1, 2), dtype=float)

    #for state in range(len(vets)):
    #    curr_state = vets[state]
    #    fit = np.polyfit(vet_years, curr_state, 1)
    #    p = np.poly1d(fit)
    #    pred_vets = p(pred_years)
    #    R2 = r2_score(curr_state, p(vet_years))

    #    print(f'Vals:\n{pred_vets}\nR2/R:{R2} {np.sqrt(R2)}')

    #    VetResults = np.insert(VetResults, state, pred_vets, axis=0)
    #    VetRs = np.insert(VetRs, state, [R2, np.sqrt(R2)], axis=0)

    #    plt.plot(vet_years, curr_state, 'o', label='Census Data')
    #    plt.plot(pred_years, pred_vets, '-', label='Regression')
    #    plt.xlabel('Years')
    #    plt.ylabel('Number of Veterans (in thousands)')
    #    plt.title(f'Number of Veterans in State {state}')
    #    plt.legend(loc='upper left')
    #    plt.show()


    #df_VETS = pd.DataFrame(data = VetResults)
    #df_VETS.to_excel("VetProj.xlsx")
    #df_Rs = pd.DataFrame(data = VetRs)
    #df_Rs.to_excel("VetRs.xlsx")







    ##print(df_edu)
    #edu = df_edu.to_numpy()
    #edu_years = np.array(edu[0, len(edu[0]):0:-1], dtype=int)
    #edu = np.array(edu[1:, len(edu[1]):0:-1], dtype=int)
    #pred_years = np.arange(1988, 2024, 1, dtype=int)
    #results = np.zeros((1, 36), dtype=float)
    #r_vals = np.zeros((1, 2), dtype=float)
    ##print(edu_years)
    ##print(edu)
    
    #for state in range(len(edu)):
    #    curr_state = edu[state]
    #    twox_fit = np.polyfit(edu_years, curr_state, 2)
    #    p_two = np.poly1d(twox_fit)
    #    pred_vals = p_two(pred_years)
    #    R2 = r2_score(curr_state, p_two(edu_years))
        
    #    if (pred_vals[0] < 0 or pred_vals[0]*1.1 >= curr_state[0]):
    #        onex_fit = np.polyfit(edu_years, curr_state, 1)
    #        p_one = np.poly1d(onex_fit)
    #        pred_vals = p_one(pred_years)
    #        R2 = r2_score(curr_state, p_one(edu_years))
           

    #    print(f'Pred years:\n{pred_vals}')
    #    print(f'R2: {R2}\nR: {np.sqrt(R2)}')

    #    results = np.insert(results, state, pred_vals, axis=0)
    #    r_vals = np.insert(r_vals, state, [R2, np.sqrt(R2)], axis=0)

    #    plt.plot(pred_years, pred_vals, '-', label='Least Squares')
    #    plt.plot(edu_years, edu[state], 'o', label="Census Data")
    #    plt.xlabel('Years')
    #    plt.ylabel('Population in thousands')
    #    plt.title(f'Number of People in State {state} with at least a Bachelors Degree')
    #    plt.legend(loc='upper left')
    #    plt.show()


    #df_EDU = pd.DataFrame(data=results)
    #df_EDU.to_excel("EduProj.xlsx")
    #df_R = pd.DataFrame(data=r_vals)
    #df_R.to_excel("EduR.xlsx")









    ##act_years = np.arange(2000, 2022, 1, dtype=int)
    #income = df_income.to_numpy()
    #income_years = income[0, len(income[0]):0:-1]
    #income = income[1:, len(income[0]):0:-1]
    #state_len = len(income)
    #pred_years = np.arange(1988, 2024, 1, dtype=int)
    #print(income)
    #print(income_years)
    #income_data = np.zeros((1, 36), dtype=float)

    #for state in range(state_len):

    #    func = sp.interpolate.interp1d(income_years, income[state], kind='linear', fill_value='extrapolate')
    #    pred_vals = func(pred_years)
    #    income_data = np.insert(income_data, state, pred_vals, axis=0)
    #    #plt.plot(pred_years, pred_vals, '-', label='Interpolated')
    #    #plt.plot(income_years, income[state], 'o', label='SAIPE Med. Income Est.')
    #    #plt.title(f'Median Income, State {state}')
    #    #plt.xlabel('Year')
    #    #plt.ylabel('Income')
    #    #plt.legend(loc='upper left')
    #    #plt.show()


    #df_INCOME = pd.DataFrame(data=income_data)
    #df_INCOME.to_excel("IncomeProj.xlsx")







    #---------------------------------------THE GOAT-----------------------------------------------------------------
    #df_cohorts = pd.read_excel("CohortAgg.xlsx")
    #df_pop = pd.read_excel("EditedTotalProjExtrapolate.xlsx")

    #df_ages = df_cohorts.iloc[1:715, 1:].to_numpy(dtype=float)
    #print(df_ages)
    #df_statepop = df_cohorts.iloc[715:, 1:25].to_numpy(dtype=int)
    ##df_total = df_pop.iloc[0:, 2:].to_numpy(dtype=int)
    ##print(df_total)


    #PROJECTIONS_ALL = np.zeros((1, 34), dtype=float)
    #MAE = np.zeros((1, 1), dtype=float)
    #r_squares = []
    #pred_years = np.arange(1988, 2025, 1, dtype=int)
    ##pred_years = pred_years[0:, np.newaxis]
    #act_years = df_cohorts.iloc[0, 1:24].to_numpy()
    ##act_years = act_years[0:, np.newaxis]
    #print(pred_years)
    #print(act_years)


    
    #for index in range(len(df_ages)): 
        
    #    if (index == 0):
    #        state_index = 0
    #    else:
    #        state_index = floor(index/14)
        
    #    print(state_index)
    #    state_pop = df_statepop[state_index]
    #    state = df_ages[index, 0:30]
    #    years = act_years
        
    #    if (state[27] < 0):
    #        plot_years = np.arange(1750, 2022, 1, dtype=int)
    #        years = (plot_years - 1750) + 1
    #        #ages = state[0:23]
    #        #ages = np.insert(ages, 0, state[28])
    #        start_pop = state[28]
    #        total_time = 2022 - 1750
    #    else:
    #        plot_years = np.arange(state[27], 2022, 1, dtype=int)
    #        years = (plot_years - state[27]) + 1
    #        #ages = state[0:23]
    #        #ages = np.insert(ages, 0, state[28])
    #        start_pop = state[28]
    #        total_time = 2022 - state[27]

    #    #print(f"state\n{state}\nyears{years}\nages{ages}\n")
    #    results = []
    #    if (state[25] > 0):
    #        results = growth(years, total_time, start_pop)
    #    else:
    #        results = decay(years, total_time, start_pop)
    #    #print(results)
    #    mae = 0.0
    #    if(len(results) >= 34):  
    #        r = sp.stats.pearsonr(results[len(results)-23:], state[0:23])
    #        #print(results[len(results)-23:])
    #        #print(state[0:23])
    #        r = r.statistic**2
    #        r_squares.append(r)
    #        #print(f'r:{r}')

    #        for i in range(len(act_years)):
    #        #range of 22
    #            mae += abs(results[len(results)-(23-i)] - state[i])/23

    #        eval_citeria = mae/state[29]
    #        print(eval_citeria)
    #        MAE = np.insert(MAE, index, eval_citeria, axis=0)

    #    else:
    #            MAE = np.insert(MAE, index, -1, axis=0)
    #            r_squares.append(-1)


    #    if (len(results) > 34):
    #        results = results[len(results)-34:len(results)]
    #    elif (len(results) < 34):
    #        results = np.zeros((1,34), dtype=float)

    #    PROJECTIONS_ALL = np.insert(PROJECTIONS_ALL, index, results, axis=0)

    #    #if (len(plot_years)>150):
    #    #    results = results[len(results)-150:len(results)]
    #    #    projections.append(results)
    #    #    plot_years = plot_years[len(plot_years)-150:len(plot_years)]

    #    #plt.plot(act_years, state[0:23], 'o', label="Census Data")
    #    #plt.plot(plot_years, results, 'o', label= "Regression")
    #    #plt.title(f'Cohort {index}')
    #    #plt.xlabel('Year')
    #    #plt.ylabel('Population (in hundred-thousands)')
    #    #plt.legend(loc='upper left')
    #    #plt.show()

    
    ##print(f"projections\n{projections}")
    #df_PROJECTIONS = pd.DataFrame(data=PROJECTIONS_ALL)
    #df_MAE = pd.DataFrame(data=MAE)
    #df_r = pd.DataFrame(data=r_squares)
    #df_PROJECTIONS.to_excel("WriteCohortPopModel.xlsx")
    #df_MAE.to_excel("WriteCohortError.xlsx")
    #df_r.to_excel("CohortRVals.xlsx")
    #------------------------------------------------------------THE GOAT--------------------------------------------------------------

    #for state in range(len(df_ages)):
    #    X = act_years
    #    y = df_ages[state]
    #    print(y)
    #    #model = LinearRegression(fit_intercept=False)
    #    #model.fit(X, y)
    #    #pred_vals = model.predict(pred_years)
    #    #np.insert(projections, state, pred_vals)

    #    z = np.polyfit(X, y, 1)
    #    f= np.poly1d(z)
    #    y_new = f(pred_years)


    #    plt.plot(X,  y, 'o', label="Census Data")
    #    plt.plot(pred_years, y_new, 'o', label= "Regression")
    #    #plt.yticks(ticks=[300000, 350000, 400000, 450000], labels=[300, 350, 400, 450])
    #    #plt.xticks(ticks=[1980, 1990, 2000, 2010, 2020], labels=[1980, 1990, 2000, 2010, 2020])
    #    plt.xlabel('Year')
    #    plt.ylabel('Population (in thousands)')
    #    plt.title(f'State {state}')
    #    plt.legend(loc='upper left')
    #    plt.show()



    


    #national pop interpolation workss great, esp in scarce data from actual pop counts
    #years = df_ages[0, 0:]
    #cohort = df_ages[1, 0:]
    #func = sp.interpolate.interp1d(years, cohort, kind='linear', fill_value='extrapolate')
    #pred_vals = func(pred_years)
    #print(pred_vals)
    #plt.plot(years,  cohort, 'o', label="Census Data")
    #plt.plot(pred_years, pred_vals, '-', label="Interpolated Estimate")
    #plt.xlabel('Year')
    #plt.ylabel('Population (in thousands)')
    #plt.title('Alabama 18-24 Cohort Interpolation, by Year')
    #plt.legend(loc='upper left')
    #plt.show()


    
    

  



   
   


    #linear regression, good for nat, sucks bc local negative growth leads to horrible predictions for prior years
    #for cohort in range(1):
    #    X = df_ages[0, np.newaxis].T
    #    y = df_ages[1, np.newaxis].T
    #    print(f"X:\n{X}")
    #    print(f"y:\n{y}")
    #    model = LinearRegression().fit(X=X, y=y)
    #    predict = np.array(model.predict(pred_years)).T
    #    predict = np.squeeze(predict, axis=0)
    #    score = model.score(X=X, y=y)
    #    print(f"Model Score:\n{score}")
    #    print(f"Predictions:\n{predict}")
    #    plt.plot(pred_years, predict, '-', label="Regression Estimate")
    #    plt.plot(X, y, 'o', label='Existing Values')
    #    plt.xlabel('Year')
    #    plt.ylabel('Population')
    #    plt.title("Regression Estimate of Age Cohort 18-24 in Alabama")
    #    plt.show()



    #projections = np.squeeze(projections)
        
    #projections = np.zeros((714, 37), dtype=float)
    #projections = projections[:, np.newaxis]
    #pred_years = np.arange(1988, 2025, 1, dtype=int)

    #doesnt work for cohorts bc negative growth
    #das gupta cohort projection, doesnt work
    #for index in range(rows):
    #    t_total = abs((2022 - df_proj[index][2])) * 365
    #    print(t_total)
    #    #pop_est = ceil(df_proj[index][1])
    #    pop_act = df_proj[index][0]
    #    #pop_prev = df_proj[index][3]
    #    pop_start = df_proj[index][2]
    #    start_year = df_proj[index][1]
    #    year_range = int(abs(2022 - start_year))

    #    year_count = 0
    #    if (pop_start < pop_act):
    #        for year in range(year_range):
    #            t_curr = (year + 1) *365
    #            pop_next = (pop_act*(t_curr/t_total)) + (pop_start*((t_total-t_curr)/t_total))
    #            if (year_range - year <= 34):
    #                projections[index][year_count] = pop_next
    #                year_count += 1
 
           



    
    #sns.pairplot(df_pop_class, hue='Label', diag_kind='hist', vars=df_pop_class.columns[14:16], plot_kws=dict(alpha=0.5), diag_kws=dict(alpha=0.5))
    ##plt.title(label="18-44")
    #plt.show()



    #df_test = df_pop_class.iloc[0:51, 13:]
    #df_test_labels = df_test["Label"]
    #df_test.drop(columns="Label", inplace=True)
    
    #df_train = df_pop_class.iloc[51:len(df_pop_class.index), 13:]
    #df_train_labels = df_train["Label"]
    #df_train.drop(columns="Label", inplace=True)
    
    #print(df_test)
    #print(df_train)

    #tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    #tree.fit(X=df_train, y=df_train_labels)
    #print(f"18-24 Classifier: {tree.score(X=df_test, y=df_test_labels)}")



    #ticks=["Population_18_24", "Population_25_29", "Population_30_34	", "Population_35_39", "Population_45_49",	"Population_40_44",	"Population_50_54","Population_55_59","Population_60_64","Population_65_69","Population_70_74","Population_75_79","Population_80_84","Population_85_Over",	"Total_Population","Median_Income","Edu","Marriage",	"White","Minority",	"Hispanic","Veteran","Pop_Density",	"Label"],

     


    #sns.pairplot(df_dem_labels, hue='Label', diag_kind='hist', vars=df_dem_labels.columns[0:4], plot_kws=dict(alpha=0.5), diag_kws=dict(alpha=0.5))
    ##plt.title(label="18-44")
    #plt.show()

    #sns.pairplot(df_dem_labels, hue='Label', diag_kind='hist', vars=df_dem_labels.columns[4:9], plot_kws=dict(alpha=0.5), diag_kws=dict(alpha=0.5))
    ##plt.title(label="45-69")
    #plt.show()

    #sns.pairplot(df_dem_labels, hue='Label', diag_kind='hist', vars=df_dem_labels.columns[9:14], plot_kws=dict(alpha=0.5), diag_kws=dict(alpha=0.5))
    ##plt.title(label="70-85+, Total")
    #plt.show()
    
    
    


    #sns.displot(df_dem_labels, x="Population_18_24", bins=10)
    #plt.show()

    #sns.displot(df_dem_labels, x="Population_85_Over", bins=10)
    #plt.show()


 

if __name__ == "__main__":
    main()
