try:
    from Work import*
except:
    from Workspace import*
print('\n')


'''Also do derivative of confirmed to do a different way of finding the peak num of cases'''



#confirmed = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_19-covid-Confirmed.csv&filename=time_series_2019-ncov-Confirmed.csv'
#confirmed = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
confirmed = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
cases = pd.read_csv(confirmed)

#recovered = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_19-covid-Recovered.csv&filename=time_series_2019-ncov-Recovered.csv'
recovered = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
recov = pd.read_csv(recovered)

dead = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
deaths = pd.read_csv(dead)

dates = ['1/22/20','1/23/20','1/24/20','1/25/20','1/26/20','1/27/20','1/28/20','1/29/20','1/30/20','1/31/20',
         '2/1/20','2/2/20','2/3/20','2/4/20','2/5/20','2/6/20','2/7/20','2/8/20','2/9/20','2/10/20','2/11/20',
         '2/12/20','2/13/20','2/14/20','2/15/20','2/16/20','2/17/20','2/18/20','2/19/20','2/20/20','2/21/20',
         '2/22/20','2/23/20','2/24/20','2/25/20','2/26/20','2/27/20','2/28/20','2/29/20','3/1/20','3/2/20',
         '3/3/20','3/4/20','3/5/20','3/6/20','3/7/20','3/8/20','3/9/20','3/10/20','3/11/20','3/12/20','3/13/20',
         '3/14/20','3/15/20','3/16/20','3/17/20','3/18/20','3/19/20','3/20/20','3/21/20','3/22/20','3/23/20',
         '3/24/20','3/25/20','3/26/20','3/27/20','3/28/20','3/29/20','3/30/20','3/31/20','4/1/20','4/2/20',
         '4/3/20','4/4/20','4/5/20','4/6/20','4/7/20','4/8/20','4/9/20','4/10/20','4/11/20','4/12/20','4/13/20',
         '4/14/20','4/15/20','4/16/20','4/17/20','4/18/20','4/19/20','4/20/20','4/21/20','4/22/20','4/23/20',
         '4/24/20','4/25/20','4/26/20','4/27/20','4/28/20','4/29/20','4/30/20','5/1/20','5/2/20','5/3/20',
         '5/4/20','5/5/20','5/6/20','5/7/20','5/8/20','5/9/20','5/10/20','5/11/20','5/12/20','5/13/20','5/14/20',
         '5/15/20','5/16/20','5/17/20','5/18/20','5/19/20','5/20/20','5/21/20','5/22/20','5/23/20','5/24/20',
         '5/25/20','5/26/20','5/27/20','5/28/20','5/29/20','5/30/20','5/31/20','6/1/20','6/2/20','6/3/20',
         '6/4/20','6/5/20','6/6/20','6/7/20','6/8/20','6/9/20','6/10/20','6/11/20','6/12/20','6/13/20','6/14/20',
         '6/15/20','6/16/20','6/17/20','6/18/20','6/19/20','6/20/20','6/21/20','6/22/20','6/23/20','6/24/20',
         '6/25/20','6/26/20','6/27/20','6/28/20','6/29/20','6/30/20','7/1/20','7/2/20','7/3/20',
         '7/4/20','7/5/20','7/6/20','7/7/20','7/8/20','7/9/20','7/10/20','7/11/20','7/12/20','7/13/20','7/14/20',
         '7/15/20','7/16/20','7/17/20','7/18/20','7/19/20','7/20/20','7/21/20','7/22/20','7/23/20','7/24/20',
         '7/25/20','7/26/20','7/27/20','7/28/20','7/29/20','7/30/20','7/31/20','8/1/20','8/2/20','8/3/20',
         '8/4/20','8/5/20','8/6/20','8/7/20','8/8/20','8/9/20','8/10/20','8/11/20','8/12/20','8/13/20','8/14/20',
         '8/15/20','8/16/20','8/17/20','8/18/20','8/19/20','8/20/20','8/21/20','8/22/20','8/23/20','8/24/20',
         '8/25/20','8/26/20','8/27/20','8/28/20','8/29/20','8/30/20','8/31/20','9/1/20','9/2/20','9/3/20',
         '9/4/20','9/5/20','9/6/20','9/7/20','9/8/20','9/9/20','9/10/20','9/11/20','9/12/20','9/13/20','9/14/20',
         '9/15/20','9/16/20','9/17/20','9/18/20','9/19/20','9/20/20','9/21/20','9/22/20','9/23/20','9/24/20',
         '9/25/20','9/26/20','9/27/20','9/28/20','9/29/20','9/30/20']

date_list = cases.columns
date_list = date_list[4:]
dates_labels = dates

US_df = cases[cases['Country/Region']=='US']
Italy_df = cases[cases['Country/Region']=='Italy']
China_df = cases[cases['Country/Region']=='China']
Ex_China_df = cases[cases['Country/Region']!='China']
SK_df = cases[cases['Country/Region']=='Korea, South']
All_df = cases

US_cases = np.array(list(US_df.sum(axis=0))[4:])
Italy_cases = np.array(list(Italy_df.sum(axis=0))[4:])
China_cases = np.array(list(China_df.sum(axis=0))[4:])
Excluding_China_cases = np.array(list(Ex_China_df.sum(axis=0))[2:])
SK_cases = np.array(list(SK_df.sum(axis=0))[4:])
All_cases = np.array(list(All_df.sum(axis=0))[2:])


US_df = recov[recov['Country/Region']=='US']
Italy_df = recov[recov['Country/Region']=='Italy']
China_df = recov[recov['Country/Region']=='China']
Ex_China_df = recov[recov['Country/Region']!='China']
SK_df = recov[recov['Country/Region']=='Korea, South']
All_df = recov

US_recov = np.array(list(US_df.sum(axis=0))[4:])
Italy_recov = np.array(list(Italy_df.sum(axis=0))[4:])
China_recov = np.array(list(China_df.sum(axis=0))[4:])
Excluding_China_recov = np.array(list(Ex_China_df.sum(axis=0))[2:])
SK_recov = np.array(list(SK_df.sum(axis=0))[4:])
All_recov = np.array(list(All_df.sum(axis=0))[2:])

US_df = deaths[deaths['Country/Region']=='US']
Italy_df = deaths[deaths['Country/Region']=='Italy']
China_df = deaths[deaths['Country/Region']=='China']
Ex_China_df = deaths[deaths['Country/Region']!='China']
SK_df = deaths[deaths['Country/Region']=='Korea, South']
All_df = deaths

US_deaths = np.array(list(US_df.sum(axis=0))[4:])
Italy_deaths = np.array(list(Italy_df.sum(axis=0))[4:])
China_deaths = np.array(list(China_df.sum(axis=0))[4:])
Excluding_China_deaths = np.array(list(Ex_China_df.sum(axis=0))[2:])
SK_deaths = np.array(list(SK_df.sum(axis=0))[4:])
All_deaths = np.array(list(All_df.sum(axis=0))[2:])


len_dates = min(len(All_cases),len(All_recov))

US_cases = US_cases[:len_dates]
Italy_cases = Italy_cases[:len_dates]
China_cases = China_cases[:len_dates]
Excluding_China_cases = Excluding_China_cases[:len_dates]
SK_cases = SK_cases[:len_dates]
All_cases = All_cases[:len_dates]

US_recov = US_recov[:len_dates]
Italy_recov = Italy_recov[:len_dates]
China_recov = China_recov[:len_dates]
Excluding_China_recov = Excluding_China_recov[:len_dates]
SK_recov = SK_recov[:len_dates]
All_recov = All_recov[:len_dates]

US_deaths = US_deaths[:len_dates]
Italy_deaths = Italy_deaths[:len_dates]
China_deaths = China_deaths[:len_dates]
Excluding_China_deaths = Excluding_China_deaths[:len_dates]
SK_deaths = SK_deaths[:len_dates]
All_deaths = All_deaths[:len_dates]


print(date_list[-1])


dates_graph  = date_list

frmt = 's'

extn = 2

x_dates = np.arange(len(date_list))
x_ = np.linspace(0,len(date_list)*extn,int(1E4))


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])


plt.suptitle('Recovered')
plt.plot(dates_graph,US_recov,label='US')
plt.plot(dates_graph,Italy_recov,label='Italy')
plt.plot(dates_graph,China_recov,label='China')
plt.plot(dates_graph,Excluding_China_recov,label='All but China')
plt.legend()
plt.show()
plt.clf()


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

#For fit, use deriv_logist distribution formula
plt.suptitle('Confirmed - Removed')

diff_China = China_cases-China_recov-China_deaths
diff_Italy = Italy_cases-Italy_recov-Italy_deaths
diff_US = US_cases-US_recov-US_deaths
diff_Ex_China = Excluding_China_cases-Excluding_China_recov-Excluding_China_deaths

plt.plot(dates_graph,diff_US,label='US')
plt.plot(dates_graph,diff_Italy,label='Italy')
plt.plot(dates_graph,diff_China,label='China')
#plt.plot(dates_graph,diff_Ex_China,label='All but China')
plt.legend()
plt.show()
plt.clf()



#Data Fitting stuff

def logist(x,amp,rate,center,nu):
    return amp*(1+M.e**(-rate*(x-center)))**(-1/nu)

#logistic = lambdify([a,b,d,e,f],logist(a,b,d,e,f),'numpy')
logistic = logist


def deriv_log(x,amp,rate,center,nu):
    fn = amp*rate/nu*M.e**(-rate*(x-center))*(1+M.e**(-rate*(x-center)))**(-(nu+1)/nu)
    fn = np.nan_to_num(np.asarray(fn))
    return fn

# Take in the deriv_log function in sympy and array and spit out array
def fn_peak(fn,var,nums,guess):
    deriv_fn = diff(fn(var[0],*nums),var[0])
    deriv_2_fn = diff(deriv_fn,var[0])
    deriv_fn = lambdify(var[0],deriv_fn)
    deriv_2_fn = lambdify(var[0],deriv_2_fn)
    sol = fsolve(deriv_2_fn,guess)[0]
    return deriv_fn(sol)

deriv_logist = deriv_log
#deriv_logist = lambdify([a,b,d,e,f],deriv_log(a,b,d,e,f),'numpy')
#deriv_logist = numpyze_lambdify_fn(deriv_log)











All_curve = curve_fit(logistic,x_dates,All_recov,p0=[1.8E5,4E-2,4E0,1E-1],bounds=(0,[8E9,1E1,1E3,1E3]))[0]

res_All = All_recov-logistic(x_dates,*All_curve)
ss_res_All = np.sum(res_All**2)
ss_tot_All = np.sum((All_recov-np.mean(All_recov))**2)
r_2_All = 1 - ss_res_All/ss_tot_All

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,All_recov,frmt)
plt.plot(x_,logistic(x_,*All_curve))
plt.suptitle('All Recovered')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(All_curve[0],-2),All_curve[2],All_curve[1],All_curve[3]))
print('All R^2 = {:.4f}'.format(r_2_All))
plt.clf()


China_curve = curve_fit(logistic,x_dates,China_recov,p0=[8E4,1E-1,3E1,5E-1],bounds=(0,[1E6,1E1,1E3,1E3]),maxfev=1E3)[0]

res_China = China_recov-logistic(x_dates,*China_curve)
ss_res_China = np.sum(res_China**2)
ss_tot_China = np.sum((China_recov-np.mean(China_recov))**2)
r_2_China = 1 - ss_res_China/ss_tot_China

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,China_recov,frmt)
plt.plot(x_,logistic(x_,*China_curve))
plt.suptitle('China Recovered')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(China_curve[0],-2),China_curve[2],China_curve[1],China_curve[3]))
print('China R^2 = {:.4f}'.format(r_2_China))
plt.clf()


US_curve = curve_fit(logistic,x_dates,US_recov,p0=[7E5,1.8E0,7E1,6E0],bounds=(0,[4E8,1E2,1E3,1E3]),maxfev=2E3)[0]

res_US = US_recov-logistic(x_dates,*US_curve)
ss_res_US = np.sum(res_US**2)
ss_tot_US = np.sum((US_recov-np.mean(US_recov))**2)
r_2_US = 1 - ss_res_US/ss_tot_US

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,US_recov,frmt)
plt.plot(x_,logistic(x_,*US_curve))
plt.suptitle('US Recovered')
plt.yscale('log')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(US_curve[0],0),US_curve[2],US_curve[1],US_curve[3]))
print('US R^2 = {:.4f}'.format(r_2_US))
plt.clf()


Italy_curve = curve_fit(logistic,x_dates,Italy_recov,p0=[3E4,1E-1,6E1,5E-1],bounds=(0,[1E8,1E1,1E3,1E3]),maxfev=2E3)[0]

res_Italy = Italy_recov-logistic(x_dates,*Italy_curve)
ss_res_Italy = np.sum(res_Italy**2)
ss_tot_Italy = np.sum((Italy_recov-np.mean(Italy_recov))**2)
r_2_Italy = 1 - ss_res_Italy/ss_tot_Italy

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,Italy_recov,frmt)
plt.plot(x_,logistic(x_,*Italy_curve))
plt.suptitle('Italy Recovered')
plt.show()
print('Italy R^2 = {:.4f}'.format(r_2_Italy))
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(Italy_curve[0],-2),Italy_curve[2],Italy_curve[1],Italy_curve[3]))
plt.clf()


Excluding_China_curve = curve_fit(logistic,x_dates,Excluding_China_recov,p0=[3E6,1.2E0,7.5E1,7E0],bounds=(0,[1E9,1E1,2E3,1E3]),maxfev=1.5E3)[0]

res_Excluding_China = Excluding_China_recov-logistic(x_dates,*Excluding_China_curve)
ss_res_Excluding_China = np.sum(res_Excluding_China**2)
ss_tot_Excluding_China = np.sum((Excluding_China_recov-np.mean(Excluding_China_recov))**2)
r_2_Excluding_China = 1 - ss_res_Excluding_China/ss_tot_Excluding_China

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,Excluding_China_recov,frmt)
plt.plot(x_,logistic(x_,*Excluding_China_curve))
plt.suptitle('All Except China Recovered')
plt.yscale('log')
plt.show()
print('Excluding China R^2 = {:.4f}'.format(r_2_Excluding_China))
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(Excluding_China_curve[0],-2),Excluding_China_curve[2],Excluding_China_curve[1],Excluding_China_curve[3]))
plt.clf()



SK_curve = curve_fit(logistic,x_dates,SK_recov,p0=[2E4,9E-1,6E1,2E-1],bounds=(0,[2E8,1E1,1E3,1E3]),maxfev=2E3)[0]

res_SK = SK_recov-logistic(x_dates,*SK_curve)
ss_res_SK = np.sum(res_SK**2)
ss_tot_SK = np.sum((SK_recov-np.mean(SK_recov))**2)
r_2_SK = 1 - ss_res_SK/ss_tot_SK

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,SK_recov,frmt)
plt.plot(x_,logistic(x_,*SK_curve))
plt.suptitle('South Korea Recovered')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(SK_curve[0],-1),SK_curve[2],SK_curve[1],SK_curve[3]))
print('S. Korea R^2 = {:.4f}'.format(r_2_SK))
plt.clf()











#deriv_logist(x,amp,sd,center,nu)
#amp,rate,center,nu

China_norm = curve_fit(deriv_logist,x_dates,diff_China,p0=[1.5E6,1E-1,1E1,1E-1],bounds=(0,[1E9,1E3,1E3,1E3]),maxfev=1E5)[0]

res_China = (diff_China)-deriv_logist(x_dates,*China_norm)
ss_res_China = np.sum(res_China**2)
ss_tot_China = np.sum(((diff_China)-np.mean(diff_China))**2)
r_2_China = 1 - ss_res_China/ss_tot_China

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,diff_China,frmt)
plt.plot(x_,deriv_logist(x_,*China_norm))
plt.suptitle('China Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(China_norm[0],-2),China_norm[2],China_norm[1],China_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*China_norm],3E1)))
print('China R^2 = {:.4f}'.format(r_2_China))
plt.clf()

#for i in range(np.where(x_>=28)[0][0],np.where(x_<=29)[0][-1]):
#    print(x_[i],deriv_logist(x_,*China_norm)[i])

US_norm = curve_fit(deriv_logist,x_dates,diff_US,p0=[4E6,1.5E-1,7E1,4E-1],maxfev=int(2E3))[0]

res_US = (diff_US)-deriv_logist(x_dates,*US_norm)
ss_res_US = np.sum(res_US**2)
ss_tot_US = np.sum(((diff_US)-np.mean(diff_US))**2)
r_2_US = 1 - ss_res_US/ss_tot_US

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,diff_US,frmt)
plt.plot(x_,deriv_logist(x_,*US_norm))
plt.suptitle('US Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(US_norm[0],-2),US_norm[2],US_norm[1],US_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*US_norm],8E1)))
print('US R^2 = {:.4f}'.format(r_2_US))
plt.clf()


#print(logistic(x_dates,1E5,10,80,1))
#print(deriv_logist(x_dates,1E5,10,80,1))

Italy_norm = curve_fit(deriv_logist,x_dates,diff_Italy,p0=[2E6,1E-1,6E1,3E-1],bounds=(0,[1E9,1E3,1E3,1E3]),maxfev=1E5)[0]

res_Italy = (diff_Italy)-deriv_logist(x_dates,*Italy_norm)
ss_res_Italy = np.sum(res_Italy**2)
ss_tot_Italy = np.sum(((diff_Italy)-np.mean(diff_Italy))**2)
r_2_Italy = 1 - ss_res_Italy/ss_tot_Italy

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,diff_Italy,frmt)
plt.plot(x_,deriv_logist(x_,*Italy_norm))
plt.suptitle('Italy Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(Italy_norm[0],-1),Italy_norm[2],Italy_norm[1],Italy_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*Italy_norm],7E1)))
print('Italy R^2 = {:.4f}'.format(r_2_Italy))
plt.clf()


Excluding_China_norm = curve_fit(deriv_logist,x_dates,diff_Ex_China,p0=[1E7,2.5E-1,7.5E1,1.5],bounds=(0,[7E9,1E2,1E3,1E3]),maxfev=1E6)[0]

res_Excluding_China = (diff_Ex_China)-deriv_logist(x_dates,*Excluding_China_norm)
ss_res_Excluding_China = np.sum(res_Excluding_China**2)
ss_tot_Excluding_China = np.sum(((diff_Ex_China)-np.mean(diff_Ex_China))**2)
r_2_Excluding_China = 1 - ss_res_Excluding_China/ss_tot_Excluding_China

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,diff_Ex_China,frmt)
plt.plot(x_,deriv_logist(x_,*Excluding_China_norm))
plt.suptitle('Excluding_China Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(Excluding_China_norm[0],-2),Excluding_China_norm[2],Excluding_China_norm[1],Excluding_China_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*Excluding_China_norm],7E1)))
print('Excluding_China R^2 = {:.4f}'.format(r_2_Excluding_China))
plt.clf()



SK_norm = curve_fit(deriv_logist,x_dates,SK_cases-SK_recov,p0=[2.4E5,1E-1,1E-1,1E-2],bounds=(0,[1E9,1E3,1E3,1E3]),maxfev=1E3)[0]

res_SK = (SK_cases-SK_recov)-deriv_logist(x_dates,*SK_norm)
ss_res_SK = np.sum(res_SK**2)
ss_tot_SK = np.sum(((SK_cases-SK_recov)-np.mean(SK_cases-SK_recov))**2)
r_2_SK = 1 - ss_res_SK/ss_tot_SK

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates,SK_cases-SK_recov,frmt)
plt.plot(x_,deriv_logist(x_,*SK_norm))
plt.suptitle('SK Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(SK_norm[0],-1),SK_norm[2],SK_norm[1],SK_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*SK_norm],5E1)))
print('SK R^2 = {:.4f}'.format(r_2_SK))
plt.clf()

print('\n\n\nUS Recovered = {} on {}\n\n\n'.format(int(US_recov[-1]),date_list[-1]))







'''Derivatives'''








