#from __future__ import (print_function, division, absolute_import, unicode_literals)
#from builtins import *
import warnings
warnings.filterwarnings("ignore")

try:
    from Work import*
except:
    from Workspace import*
print('\n')

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
         '9/25/20','9/26/20','9/27/20','9/28/20','9/29/20','9/30/20','10/1/20','10/2/20','10/3/20',
         '10/4/20','10/5/20','10/6/20','10/7/20','10/8/20','10/9/20','10/10/20','10/11/20','10/12/20','10/13/20','10/14/20',
         '10/15/20','10/16/20','10/17/20','10/18/20','10/19/20','10/20/20','10/21/20','10/22/20','10/23/20','10/24/20',
         '10/25/20','10/26/20','10/27/20','10/28/20','10/29/20','10/30/20','10/31/20','11/1/20','11/2/20','11/3/20',
         '11/4/20','11/5/20','11/6/20','11/7/20','11/8/20','11/9/20','11/10/20','11/11/20','11/12/20','11/13/20','11/14/20',
         '11/15/20','11/16/20','11/17/20','11/18/20','11/19/20','11/20/20','11/21/20','11/22/20','11/23/20','11/24/20',
         '11/25/20','11/26/20','11/27/20','11/28/20','11/29/20','11/30/20','12/1/20','12/2/20','12/3/20',
         '12/4/20','12/5/20','12/6/20','12/7/20','12/8/20','12/9/20','12/10/20','12/11/20','12/12/20','12/13/20','12/14/20',
         '12/15/20','12/16/20','12/17/20','12/18/20','12/19/20','12/20/20','12/21/20','12/22/20','12/23/20','12/24/20',
         '12/25/20','12/26/20','12/27/20','12/28/20','12/29/20','12/30/20','12/31/20']

date_list = cases.columns
date_list = date_list[4:]
dates_labels = dates

#print(cases.sum(axis=0))

US_df = cases[cases['Country/Region']=='US']
Italy_df = cases[cases['Country/Region']=='Italy']
China_df = cases[cases['Country/Region']=='China']
Ex_China_df = cases[cases['Country/Region']!='China']
SK_df = cases[cases['Country/Region']=='Korea, South']
All_df = cases


US_cases = np.array(list(US_df.sum(axis=0))[4:])
Italy_cases = np.array(list(Italy_df.sum(axis=0))[4:])
China_cases = np.array(list(China_df.sum(axis=0))[4:])
Excluding_China_cases = np.array(list(Ex_China_df.sum(axis=0))[3:])
SK_cases = np.array(list(SK_df.sum(axis=0))[4:])
All_cases = np.array(list(All_df.sum(axis=0))[3:])


US_df = recov[recov['Country/Region']=='US']
Italy_df = recov[recov['Country/Region']=='Italy']
China_df = recov[recov['Country/Region']=='China']
Ex_China_df = recov[recov['Country/Region']!='China']
SK_df = recov[recov['Country/Region']=='Korea, South']
All_df = recov

US_recov = np.array(list(US_df.sum(axis=0))[4:])
Italy_recov = np.array(list(Italy_df.sum(axis=0))[4:])
China_recov = np.array(list(China_df.sum(axis=0))[4:])
Excluding_China_recov = np.array(list(Ex_China_df.sum(axis=0))[3:])
SK_recov = np.array(list(SK_df.sum(axis=0))[4:])
All_recov = np.array(list(All_df.sum(axis=0))[3:])

US_df = deaths[deaths['Country/Region']=='US']
Italy_df = deaths[deaths['Country/Region']=='Italy']
China_df = deaths[deaths['Country/Region']=='China']
Ex_China_df = deaths[deaths['Country/Region']!='China']
SK_df = deaths[deaths['Country/Region']=='Korea, South']
All_df = deaths

US_deaths = np.array(list(US_df.sum(axis=0))[4:])
Italy_deaths = np.array(list(Italy_df.sum(axis=0))[4:])
China_deaths = np.array(list(China_df.sum(axis=0))[4:])
Excluding_China_deaths = np.array(list(Ex_China_df.sum(axis=0))[3:])
SK_deaths = np.array(list(SK_df.sum(axis=0))[4:])
All_deaths = np.array(list(All_df.sum(axis=0))[3:])


diff_China = China_cases-China_recov-China_deaths
diff_Italy = Italy_cases-Italy_recov-Italy_deaths
diff_US = US_cases-US_recov-US_deaths


print(date_list[-1])
print('\n\n\n')

dates_graph = date_list

frmt = 's'

extn = 2

x_dates = np.arange(len(date_list))
x_ = np.linspace(0,len(date_list)*extn,int(1E4))


#Data Fitting stuff

def logistic(x,amp,rate,center,nu):
    return amp*(1+M.e**(-rate*(x-center)))**(-1/nu)

def base_rate(x,amp,base,center,nu):
    return amp*(1+base**(-(x-center)))**(-1/nu)

def just_r_sqr(xdata,ydata,fn_name):
    res = np.array(ydata)-fn_name(xdata)
    return (1 - np.sum(res**2)/np.sum((np.array(ydata)-np.mean(np.array(ydata)))**2))

def logist(x,amp,rate,center,nu):
    return amp*(1+M.e**(-rate*(x-center)))**(-1/nu)

#logistic = lambdify([a,b,d,e,f],logist(a,b,d,e,f),'numpy')
logistic = logist

def deriv_log(x,amp,rate,center,nu):
    fn=lambdify([X,a,b,d,f],diff(logist(X,a,b,d,f),X),'numpy')
    return fn(x,amp,rate,center,nu)

#def deriv_log(x,amp,rate,center,nu):
#    fn = amp*rate/nu*M.e**(-rate*(x-center))*(1+M.e**(-rate*(x-center)))**(-(nu+1)/nu)
#    fn = np.nan_to_num(np.asarray(fn))
#    return fn

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


US_base = curve_fit(base_rate,x_dates,US_cases,p0=[2E5,1.5,60,1],bounds=(0,[3.5E8,1E3,1000,1E3]),maxfev=1E4)[0]
China_base = curve_fit(base_rate,x_dates,China_cases,p0=[8E4,1.2,20,1],bounds=(0,[1E6,1E3,1000,1E3]),maxfev=1E4)[0]
Italy_base = curve_fit(base_rate,x_dates,Italy_cases,p0=[1E7,1.2,50,1],bounds=(0,[1E8,1E3,1000,1E2]),maxfev=1E4)[0]


fh = open('covid_lists.txt','r')
for i in fh.readlines():
  i = str(i).replace('-inf,','-np.inf,')
  exec(i)
fh.close()


ndays = 20


for i in range(ndays+len(US_list_changes),len(x_dates)+1):
    print(date_list[i-1])
    US_max_r = []
    US_max_peak = []
    
    China_max_r = []
    China_max_peak = []
    
    Italy_max_r = []
    Italy_max_peak = []
    
    for j in range(int(3E1),int(8E1)+1,1):
        try:
            US_max_r.append(r_sqr(x_dates[:i],US_cases[:i],logistic,[10**(j/1E1),0.2,60,1],(0,[3.5E8,1E3,1E3,1E3]),1E4))
        except:
            None
        try:
            China_max_r.append(r_sqr(x_dates[:i],China_cases[:i],logistic,[10**(j/1E1),0.4,20,1],(0,[2E9,1E3,1E3,1E3]),1E4))
        except:
            None
        try:
            Italy_max_r.append(r_sqr(x_dates[:i],Italy_cases[:i],logistic,[10**(j/1E1),0.25,40,1],(0,[1E8,1E1,1E3,1E3]),1E4))
        except:
            None
        try:
            US_max_peak.append(r_sqr(x_dates[:i],diff_US[:i],deriv_log,[10**(j/1E1),1E-1,6E1,2E-1],(0,np.inf),1E5))
        except:
            None
        try:
            China_max_peak.append(r_sqr(x_dates[:i],diff_China[:i],deriv_log,[10**(j/1E1),1E-1,2E1,2E-1],(0,np.inf),1E5))
        except:
            None
        try:
            Italy_max_peak.append(r_sqr(x_dates[:i],diff_Italy[:i],deriv_log,[10**(j/1E1),1E-1,6E1,2E-1],(0,np.inf),1E5))
        except:
            None
    
    
    US_list_changes.append(US_max_r[np.where(np.array(US_max_r)[:,0] == max(np.array(US_max_r)[:,0]))[0][0]])
    China_list_changes.append(China_max_r[np.where(np.array(China_max_r)[:,0] == max(np.array(China_max_r)[:,0]))[0][0]])
    Italy_list_changes.append(Italy_max_r[np.where(np.array(Italy_max_r)[:,0] == max(np.array(Italy_max_r)[:,0]))[0][0]])
    
    US_peak_chgs.append(US_max_peak[np.where(np.array(US_max_peak)[:,0] == max(np.array(US_max_peak)[:,0]))[0][0]])
    China_peak_chgs.append(China_max_peak[np.where(np.array(China_max_peak)[:,0] == max(np.array(China_max_peak)[:,0]))[0][0]])
    Italy_peak_chgs.append(Italy_max_peak[np.where(np.array(Italy_max_peak)[:,0] == max(np.array(Italy_max_peak)[:,0]))[0][0]])

fh = open('covid_lists.txt','w')
fh.writelines('US_list_changes = '+str(US_list_changes)+'\n')
fh.writelines('China_list_changes = '+str(China_list_changes)+'\n')
fh.writelines('Italy_list_changes = '+str(Italy_list_changes)+'\n')
fh.writelines('US_peak_chgs = '+str(US_peak_chgs)+'\n')
fh.writelines('China_peak_chgs = '+str(China_peak_chgs)+'\n')
fh.writelines('Italy_peak_chgs = '+str(Italy_peak_chgs)+'\n\n')
fh.close()

US_r_2_chgs = []
China_r_2_chgs = []
Italy_r_2_chgs = []

US_amp_chgs = []
China_amp_chgs = []
Italy_amp_chgs = []

US_rate_chgs = []
China_rate_chgs = []
Italy_rate_chgs = []

US_center_chgs = []
China_center_chgs = []
Italy_center_chgs = []

US_nu_chgs = []
China_nu_chgs = []
Italy_nu_chgs = []

US_peak = []
China_peak = []
Italy_peak = []


#fn_peak(logistic,[x,b,d,e,f],[*US_list_changes[j]],8E1)

for j in range(len(US_list_changes)):
    US_r_2_chgs.append(US_list_changes[j][0])
    US_amp_chgs.append(US_list_changes[j][1][0])
    US_rate_chgs.append(US_list_changes[j][1][1])
    US_center_chgs.append(US_list_changes[j][1][2])
    US_nu_chgs.append(US_list_changes[j][1][3])
    US_peak.append(1 if max(deriv_log(x_,*US_peak_chgs[j][1]))>=np.nan_to_num(np.inf)/10 else max(deriv_log(x_,*US_peak_chgs[j][1])))
    
#    1 if US_list_changes[j][0] >= np.nan_to_num(np.inf)/10 else: US_list_changes[j][0]
    
    China_r_2_chgs.append(China_list_changes[j][0])
    China_amp_chgs.append(China_list_changes[j][1][0])
    China_rate_chgs.append(China_list_changes[j][1][1])
    China_center_chgs.append(China_list_changes[j][1][2])
    China_nu_chgs.append(China_list_changes[j][1][3])
    China_peak.append(1 if max(deriv_log(x_,*China_peak_chgs[j][1]))>=np.nan_to_num(np.inf)/10 else max(deriv_log(x_,*China_peak_chgs[j][1])))
    
    Italy_r_2_chgs.append(Italy_list_changes[j][0])
    Italy_amp_chgs.append(Italy_list_changes[j][1][0])
    Italy_rate_chgs.append(Italy_list_changes[j][1][1])
    Italy_center_chgs.append(Italy_list_changes[j][1][2])
    Italy_nu_chgs.append(Italy_list_changes[j][1][3])
    Italy_peak.append(1 if max(deriv_log(x_,*Italy_peak_chgs[j][1]))>=np.nan_to_num(np.inf)/10 else max(deriv_log(x_,*Italy_peak_chgs[j][1])))






plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.suptitle('Confirmed Cases')
plt.plot(dates_graph,US_cases,label='US')
plt.plot(dates_graph,Italy_cases,label='Italy')
plt.plot(dates_graph,China_cases,label='China')
plt.plot(dates_graph,Excluding_China_cases,label='All but China')
plt.plot(dates_graph,SK_cases,label='South Korea')
plt.yscale('log')
plt.legend()
plt.show()
plt.clf()




len_dates = min(len(All_cases),len(All_recov))


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])


plt.suptitle('Recovered')
plt.plot(dates_graph,US_recov,label='US')
plt.plot(dates_graph,Italy_recov,label='Italy')
plt.plot(dates_graph,China_recov,label='China')
plt.plot(dates_graph,Excluding_China_recov,label='All but China')
plt.yscale('log')
plt.legend()
plt.show()
plt.clf()


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

#For fit, use deriv_logist distribution formula
plt.suptitle('Confirmed - Removed')

diff_China = China_cases-China_recov-China_deaths
diff_Italy = Italy_cases-Italy_recov-Italy_deaths
diff_US = US_cases-US_recov-US_deaths
diff_Ex_China = Excluding_China_cases-Excluding_China_recov-Excluding_China_deaths

plt.plot(dates_graph,diff_US,label='US')
plt.plot(dates_graph,diff_Italy,label='Italy')
plt.plot(dates_graph,diff_China,label='China')
plt.plot(dates_graph,diff_Ex_China,label='All but China')
plt.yscale('log')
plt.legend()
plt.show()
plt.clf()


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.suptitle('Confirmed deaths')
plt.plot(dates_graph,US_deaths,label='US')
plt.plot(dates_graph,Italy_deaths,label='Italy')
plt.plot(dates_graph,China_deaths,label='China')
plt.plot(dates_graph,Excluding_China_deaths,label='All but China')
plt.plot(dates_graph,SK_deaths,label='South Korea')
plt.yscale('log')
plt.legend()
plt.show()
plt.clf()




US_new = np.diff(US_cases)
Italy_new = np.diff(Italy_cases)
China_new = np.diff(China_cases)
Ex_China_new = np.diff(Excluding_China_cases)
SK_new = np.diff(SK_cases)
All_new = np.diff(All_cases)



plt.plot(US_cases[1:],US_new,'-',color = 'blue',label='US')
plt.plot(US_cases[-1],US_new[-1],'o',color = 'blue',markersize = '12')

plt.plot(China_cases[1:],China_new,'-',color = 'orange',label='China')
plt.plot(China_cases[-1],China_new[-1],'o',color = 'orange',markersize = '12')

plt.plot(Italy_cases[1:],Italy_new,'-',color = 'green',label='Italy')
plt.plot(Italy_cases[-1],Italy_new[-1],'o',color = 'green',markersize = '12')

plt.plot(Excluding_China_cases[1:],Ex_China_new,'-',color = 'red',label='Ex China')
plt.plot(Excluding_China_cases[-1],Ex_China_new[-1],'o',color = 'red',markersize = '12')

plt.plot(SK_cases[1:],SK_new,'-',color = 'purple',label='S. Korea')
plt.plot(SK_cases[-1],SK_new[-1],'o',color = 'purple',markersize = '12')

plt.plot(All_cases[1:],All_new,'-',color = 'brown',label='All')
plt.plot(All_cases[-1],All_new[-1],'o',color = 'brown',markersize = '12')

#plt.plot(np.arange(1E1,1E6),np.arange(1E1,1E6)/10,'-',color='yellow')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total Cases')
plt.ylabel('New Cases')
plt.show()
plt.clf()



US_new_deaths = np.diff(US_deaths)
Italy_new_deaths = np.diff(Italy_deaths)
China_new_deaths = np.diff(China_deaths)
Ex_China_new_deaths = np.diff(Excluding_China_deaths)
SK_new_deaths = np.diff(SK_deaths)
All_new_deaths = np.diff(All_deaths)



plt.plot(US_deaths[1:],US_new_deaths,'-',color = 'blue',label='US')
plt.plot(US_deaths[-1],US_new_deaths[-1],'o',color = 'blue',markersize = '12')

plt.plot(China_deaths[1:],China_new_deaths,'-',color = 'orange',label='China')
plt.plot(China_deaths[-1],China_new_deaths[-1],'o',color = 'orange',markersize = '12')

plt.plot(Italy_deaths[1:],Italy_new_deaths,'-',color = 'green',label='Italy')
plt.plot(Italy_deaths[-1],Italy_new_deaths[-1],'o',color = 'green',markersize = '12')

plt.plot(Excluding_China_deaths[1:],Ex_China_new_deaths,'-',color = 'red',label='Ex China')
plt.plot(Excluding_China_deaths[-1],Ex_China_new_deaths[-1],'o',color = 'red',markersize = '12')

plt.plot(SK_deaths[1:],SK_new_deaths,'-',color = 'purple',label='S. Korea')
plt.plot(SK_deaths[-1],SK_new_deaths[-1],'o',color = 'purple',markersize = '12')

plt.plot(All_deaths[1:],All_new_deaths,'-',color = 'brown',label='All')
plt.plot(All_deaths[-1],All_new_deaths[-1],'o',color = 'brown',markersize = '12')

#plt.plot(np.arange(1E1,1E6),np.arange(1E1,1E6)/10,'-',color='yellow')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total Deaths')
plt.ylabel('New Deaths')
plt.show()
plt.clf()



if np.any(US_cases<=0):
    US_cases[US_cases<=0] = 1E-9
    US_death_rate = US_deaths/US_cases*100
else:
    US_death_rate = US_deaths/US_cases*100

if np.any(Italy_cases<=0):
    Italy_cases[Italy_cases<=0] = 1E-9
    Italy_death_rate = Italy_deaths/Italy_cases*100
else:
    Italy_death_rate = Italy_deaths/Italy_cases*100

if np.any(China_cases<=0):
    China_cases[China_cases<=0] = 1E-9
    China_death_rate = China_deaths/China_cases*100
else:
    China_death_rate = China_deaths/China_cases*100

if np.any(Excluding_China_cases<=0):
    Excluding_China_cases[Excluding_China_cases<=0] = 1E-9
    Excluding_China_death_rate = Excluding_China_deaths/Excluding_China_cases*100
else:
    Ex_China_death_rate = Excluding_China_deaths/Excluding_China_cases*100

if np.any(SK_cases<=0):
    SK_cases[SK_cases<=0] = 1E-9
    SK_death_rate = SK_deaths/SK_cases*100
else:
    SK_death_rate = SK_deaths/SK_cases*100

if np.any(All_cases<=0):
    All_cases[All_cases<=0] = 1E-9
    All_death_rate = All_deaths/All_cases*100
else:
    All_death_rate = All_deaths/All_cases*100


plt.suptitle('Death Rates (%)')
plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])
plt.plot(date_list,US_death_rate,label='US')
plt.plot(date_list,Italy_death_rate,label='Italy')
plt.plot(date_list,China_death_rate,label='China')
plt.plot(date_list,Ex_China_death_rate,label='All but China')
plt.plot(date_list,SK_death_rate,label='S. Korea')
plt.plot(date_list,All_death_rate,label='All')
plt.legend()
plt.show()
plt.clf()


US_first_index = np.where(US_deaths != 0)[0][0]
Italy_first_index = np.where(Italy_deaths != 0)[0][0]
China_first_index = np.where(China_deaths != 0)[0][0]
Ex_China_first_index = np.where(Excluding_China_deaths != 0)[0][0]
SK_first_index = np.where(SK_deaths != 0)[0][0]
All_first_index = np.where(All_deaths != 0)[0][0]


print('\n\n')




print('\nCurrent Death Rate in US: {:.2f}%\n'.format(US_death_rate[-1]))
print('Current Death Rate in Italy: {:.2f}%\n'.format(Italy_death_rate[-1]))
print('Current Death Rate in China: {:.2f}%\n'.format(China_death_rate[-1]))
print('Current Death Rate Everywhere but China: {:.2f}%\n'.format(Ex_China_death_rate[-1]))
print('Current Death Rate in S. Korea: {:.2f}%\n'.format(SK_death_rate[-1]))
print('Current Death Rate in Total: {:.2f}%\n'.format(All_death_rate[-1]))

print('\n\n\n')
print('Deaths in Italy = {} on {}'.format(int(Italy_deaths[-1]),date_list[-1]))
print('Deaths in the US = {} on {}'.format(int(US_deaths[-1]),date_list[-1]))
print('\n\n')





All_base = curve_fit(base_rate,x_dates,All_cases,p0=[1E7,1.2,1E2,1],bounds=(0,[8E9,1E3,2E3,1E3]),maxfev=1E3)[0]
All_curve = r_sqr(x_dates,All_cases,logistic,[1E7,0.3,1E2,1],(0,[8E9,1E1,2E3,1E3]),maxfev=1E3)

deriv_1_All = lambdify(X,diff(logistic(X,*All_curve[1]),X,1))
deriv_2_All = lambdify(X,diff(logistic(X,*All_curve[1]),X,2))
deriv_rsqr_All = just_r_sqr(x_dates[1:],np.diff(All_cases),deriv_1_All)
peak_All_day = fsolve(deriv_2_All,All_curve[1][2]+7E1)[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

#plt.plot(date_list,All_cases,frmt)
plt.plot(x_dates,All_cases,frmt)
plt.plot(x_,logistic(x_,*All_curve[1]))
plt.suptitle('All')
#plt.yscale('log')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(All_curve[1][0],-2),All_curve[1][2],All_curve[1][1]))
print('All R^2 = {:.4f}'.format(All_curve[0]),)
plt.clf()



All_curve = curve_fit(logistic,x_dates,All_deaths,p0=[3E6,0.2,1E2,1],bounds=(0,[7.5E9,1E1,1000,1E3]),maxfev=1E3)[0]

res_All = All_deaths-logistic(x_dates,*All_curve)
ss_res_All = np.sum(res_All**2)
ss_tot_All = np.sum((All_deaths-np.mean(All_deaths))**2)
r_2_All = 1 - ss_res_All/ss_tot_All


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(date_list,All_deaths,frmt)
plt.plot(x_,logistic(x_,*All_curve))
plt.yscale('log')
plt.suptitle('All Deaths')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(All_curve[0],-1),All_curve[2],All_curve[1]))
print('All R^2 = {:.4f}'.format(r_2_All))
plt.clf()






US_max_predict = []
China_max_predict = []
Italy_max_predict = []
SK_max_predict = []





China_base = curve_fit(base_rate,x_dates,China_cases,p0=[80000,1.2,20,1],bounds=(0,[1E6,1E3,1000,1E3]))[0]
China_curve = r_sqr(x_dates,China_cases,logistic,[8E4,0.2,20,1],(0,[1E6,1E1,1E3,1E3]))

deriv_1_China = lambdify(X,diff(logistic(X,*China_curve[1]),X,1))
deriv_2_China = lambdify(X,diff(logistic(X,*China_curve[1]),X,2))
deriv_rsqr_China = just_r_sqr(x_dates[1:],np.diff(China_cases),deriv_1_China)
peak_China_day = fsolve(deriv_2_China,China_curve[1][2]+7E1)[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

#plt.plot(date_list,China_cases,frmt)
plt.plot(x_dates,China_cases,frmt)
plt.plot(x_,logistic(x_,*China_curve[1]))
plt.suptitle('China')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(China_curve[1][0],-2),China_curve[1][2],China_curve[1][1]))
print('China R^2 = {:.4f}'.format(China_curve[0]))
plt.clf()

China_max_predict.append(China_curve[1][0])


China_norm = curve_fit(deriv_logist,x_dates,diff_China,p0=[1.5E6,1E-1,1E1,1E-1],bounds=(0,[1E9,1E3,1E3,1E3]),maxfev=1E5)[0]

res_China = (diff_China)-deriv_logist(x_dates,*China_norm)
ss_res_China = np.sum(res_China**2)
ss_tot_China = np.sum(((diff_China)-np.mean(diff_China))**2)
r_2_China = 1 - ss_res_China/ss_tot_China

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(x_dates,diff_China,frmt)
plt.plot(x_,deriv_logist(x_,*China_norm))
plt.suptitle('China Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(China_norm[0],-2),China_norm[2],China_norm[1],China_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*China_norm],3E1)))
print('China R^2 = {:.4f}'.format(r_2_China))
plt.clf()

China_max_predict.append(China_norm[0])


China_curve = curve_fit(logistic,x_dates,China_deaths,p0=[1E4,0.2,20,1],bounds=(0,[1E6,1E1,1000,1E3]))[0]

res_China = China_deaths-logistic(x_dates,*China_curve)
ss_res_China = np.sum(res_China**2)
ss_tot_China = np.sum((China_deaths-np.mean(China_deaths))**2)
r_2_China = 1 - ss_res_China/ss_tot_China


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(date_list,China_deaths,frmt)
plt.plot(x_,logistic(x_,*China_curve))
plt.suptitle('China Deaths')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(China_curve[0],-1),China_curve[2],China_curve[1]))
print('China R^2 = {:.4f}'.format(r_2_China))
plt.clf()



US_base = curve_fit(base_rate,x_dates,US_cases,p0=[2E5,1.5,60,1],bounds=(0,[3.5E8,1E3,1000,1E3]),maxfev=9E2)[0]
US_curve = r_sqr(x_dates,US_cases,logistic,[2E5,0.4,60,1],(0,[3.5E8,1E1,1E3,1E3]))

deriv_1_US = lambdify(X,diff(logistic(X,*US_curve[1]),X,1))
deriv_2_US = lambdify(X,diff(logistic(X,*US_curve[1]),X,2))
deriv_rsqr_US = just_r_sqr(x_dates[1:],np.diff(US_cases),deriv_1_US)
peak_US_day = fsolve(deriv_2_US,US_curve[1][2]+7E1)[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(date_list,US_cases,frmt)
#plt.plot(x_dates,US_cases,frmt)
plt.plot(x_,logistic(x_,*US_curve[1]))
plt.suptitle('US')
#plt.yscale('log')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(US_curve[1][0],-2),US_curve[1][2],US_curve[1][1]))
print('US R^2 = {:.4f}'.format(US_curve[0]))
plt.clf()

US_max_predict.append(US_curve[1][0])


US_norm = curve_fit(deriv_logist,x_dates,diff_US,p0=[1E8,1.5E-1,1.9E2,4E-1],maxfev=int(1E4))[0]

res_US = (diff_US)-deriv_logist(x_dates,*US_norm)
ss_res_US = np.sum(res_US**2)
ss_tot_US = np.sum(((diff_US)-np.mean(diff_US))**2)
r_2_US = 1 - ss_res_US/ss_tot_US

US_peak_c_r = fn_peak(logist,[x,b,d,e,f],[*US_norm],1.5E2)

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(x_dates,diff_US,frmt)
plt.plot(x_,deriv_logist(x_,*US_norm))
plt.suptitle('US Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(US_norm[0],-2),US_norm[2],US_norm[1],US_norm[3]))
print('Peak # of Cases = {:.1f}'.format(US_peak_c_r))
print('US R^2 = {:.4f}'.format(r_2_US))
plt.clf()

US_max_predict.append(US_peak_c_r)

print('\n\n\nUS Recovered = {} on {}\n\n\n'.format(int(US_recov[-1]),date_list[-1]))



US_curve = curve_fit(logistic,x_dates,US_deaths,p0=[1E6,0.2,1E2,1],bounds=(0,[3.5E8,1E1,1000,1E3]),maxfev=2E3)[0]

res_US = US_deaths-logistic(x_dates,*US_curve)
ss_res_US = np.sum(res_US**2)
ss_tot_US = np.sum((US_deaths-np.mean(US_deaths))**2)
r_2_US = 1 - ss_res_US/ss_tot_US


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(date_list,US_deaths,frmt)
plt.plot(x_,logistic(x_,*US_curve))
plt.suptitle('US Deaths')
#plt.yscale('log')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(US_curve[0],-2),US_curve[2],US_curve[1]))
print('US R^2 = {:.4f}'.format(r_2_US))
plt.clf()



Italy_curve = r_sqr(x_dates,Italy_cases,logistic,[1E7,0.2,50,1],(0,[1E8,1E1,1000,1E3]),maxfev=1E3)
Italy_base = curve_fit(base_rate,x_dates,Italy_cases,p0=[1E7,1.2,50,1],bounds=(0,[1E8,1E3,1000,1E3]),maxfev=1E3)[0]

deriv_1_Italy = lambdify(X,diff(logistic(X,*Italy_curve[1]),X,1))
deriv_2_Italy = lambdify(X,diff(logistic(X,*Italy_curve[1]),X,2))
deriv_rsqr_Italy = just_r_sqr(x_dates[1:],np.diff(Italy_cases),deriv_1_Italy)
peak_Italy_day = fsolve(deriv_2_Italy,Italy_curve[1][2]+7E1)[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

#plt.plot(date_list,Italy_cases,frmt)
plt.plot(x_dates,Italy_cases,frmt)
plt.plot(x_,logistic(x_,*Italy_curve[1]))
plt.suptitle('Italy')
plt.show()
print('Italy R^2 = {:.4f}'.format(Italy_curve[0]))
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(Italy_curve[1][0],-2),Italy_curve[1][2],Italy_curve[1][1]))
plt.clf()

Italy_max_predict.append(Italy_curve[1][0])


Italy_norm = curve_fit(deriv_logist,x_dates,diff_Italy,p0=[2E6,1E-1,6E1,3E-1],bounds=(0,[1E9,1E3,1E3,1E3]),maxfev=1E5)[0]

res_Italy = (diff_Italy)-deriv_logist(x_dates,*Italy_norm)
ss_res_Italy = np.sum(res_Italy**2)
ss_tot_Italy = np.sum(((diff_Italy)-np.mean(diff_Italy))**2)
r_2_Italy = 1 - ss_res_Italy/ss_tot_Italy

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(x_dates,diff_Italy,frmt)
plt.plot(x_,deriv_logist(x_,*Italy_norm))
plt.suptitle('Italy Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(Italy_norm[0],-1),Italy_norm[2],Italy_norm[1],Italy_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*Italy_norm],8E1)))
print('Italy R^2 = {:.4f}'.format(r_2_Italy))
plt.clf()

Italy_max_predict.append(Italy_norm[0])


Italy_curve = curve_fit(logistic,x_dates,Italy_deaths,p0=[1E4,0.2,60,1],bounds=(0,[1E8,1E1,1000,1E3]))[0]

res_Italy = Italy_deaths-logistic(x_dates,*Italy_curve)
ss_res_Italy = np.sum(res_Italy**2)
ss_tot_Italy = np.sum((Italy_deaths-np.mean(Italy_deaths))**2)
r_2_Italy = 1 - ss_res_Italy/ss_tot_Italy


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(date_list,Italy_deaths,frmt)
plt.plot(x_,logistic(x_,*Italy_curve))
plt.suptitle('Italy Deaths')
plt.show()
print('Italy R^2 = {:.4f}'.format(r_2_Italy))
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(Italy_curve[0],-1),Italy_curve[2],Italy_curve[1]))
plt.clf()



Excluding_China_curve = r_sqr(x_dates,Excluding_China_cases,logistic,[1E8,0.2,20,1],(0,[1E9,1E1,2000,1E3]),maxfev=1.5E3)
Excluding_China_base = curve_fit(base_rate,x_dates,Excluding_China_cases,p0=[1E8,1.3,20,1],bounds=(0,[1E9,1E3,2000,1E3]),maxfev=1.5E3)[0]

deriv_1_Excluding_China = lambdify(X,diff(logistic(X,*Excluding_China_curve[1]),X,1))
deriv_2_Excluding_China = lambdify(X,diff(logistic(X,*Excluding_China_curve[1]),X,2))
deriv_rsqr_Excluding_China = just_r_sqr(x_dates[1:],np.diff(Excluding_China_cases),deriv_1_Excluding_China)
peak_Excluding_China_day = fsolve(deriv_2_Excluding_China,Excluding_China_curve[1][2]+7E1)[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

#plt.plot(date_list,Excluding_China_cases,frmt)
plt.plot(x_dates,Excluding_China_cases,frmt)
plt.plot(x_,logistic(x_,*Excluding_China_curve[1]))
plt.suptitle('All Except China')
plt.yscale('log')
plt.show()
print('Excluding China R^2 = {:.4f}'.format(Excluding_China_curve[0]))
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(Excluding_China_curve[1][0],-3),Excluding_China_curve[1][2],Excluding_China_curve[1][1]))
plt.clf()


Excluding_China_norm = curve_fit(deriv_logist,x_dates,diff_Ex_China,p0=[1E8,2.5E-1,1.3E2,1.5],bounds=(0,[7E9,1E2,1E3,1E3]),maxfev=1E6)[0]

res_Excluding_China = (diff_Ex_China)-deriv_logist(x_dates,*Excluding_China_norm)
ss_res_Excluding_China = np.sum(res_Excluding_China**2)
ss_tot_Excluding_China = np.sum(((diff_Ex_China)-np.mean(diff_Ex_China))**2)
r_2_Excluding_China = 1 - ss_res_Excluding_China/ss_tot_Excluding_China

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(x_dates,diff_Ex_China,frmt)
plt.plot(x_,deriv_logist(x_,*Excluding_China_norm))
plt.suptitle('Excluding China Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(Excluding_China_norm[0],-2),Excluding_China_norm[2],Excluding_China_norm[1],Excluding_China_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*Excluding_China_norm],1.2E2)))
print('Excluding_China R^2 = {:.4f}'.format(r_2_Excluding_China))
plt.clf()


Excluding_China_curve = curve_fit(logistic,x_dates,Excluding_China_deaths,p0=[1E5,0.2,60,1],bounds=(0,[7E9,1E1,2000,1E3]),maxfev=1.5E3)[0]

res_Excluding_China = Excluding_China_deaths-logistic(x_dates,*Excluding_China_curve)
ss_res_Excluding_China = np.sum(res_Excluding_China**2)
ss_tot_Excluding_China = np.sum((Excluding_China_deaths-np.mean(Excluding_China_deaths))**2)
r_2_Excluding_China = 1 - ss_res_Excluding_China/ss_tot_Excluding_China


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(date_list,Excluding_China_deaths,frmt)
plt.plot(x_,logistic(x_,*Excluding_China_curve))
plt.suptitle('All Except China Deaths')
plt.show()
print('Excluding China R^2 = {:.4f}'.format(r_2_Excluding_China))
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(Excluding_China_curve[0],-2),Excluding_China_curve[2],Excluding_China_curve[1]))
plt.clf()



SK_curve = r_sqr(x_dates,SK_cases,logistic,[2E5,0.4,60,1],(0,[2E8,1E1,1000,1E3]),maxfev=2E3)
SK_base = curve_fit(base_rate,x_dates,SK_cases,p0=[2E5,2,60,1],bounds=(0,[2E8,1E3,1000,1E3]),maxfev=2E3)[0]

deriv_1_SK = lambdify(X,diff(logistic(X,*SK_curve[1]),X,1))
deriv_2_SK = lambdify(X,diff(logistic(X,*SK_curve[1]),X,2))
deriv_rsqr_SK = just_r_sqr(x_dates[1:],np.diff(SK_cases),deriv_1_SK)
peak_SK_day = fsolve(deriv_2_SK,SK_curve[1][2]+7E1)[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

#plt.plot(date_list,SK_cases,frmt)
plt.plot(x_dates,SK_cases,frmt)
plt.plot(x_,logistic(x_,*SK_curve[1]))
plt.suptitle('South Korea')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(SK_curve[1][0],-1),SK_curve[1][2],SK_curve[1][1]))
print('S. Korea R^2 = {:.4f}'.format(SK_curve[0]))
plt.clf()

SK_max_predict.append(SK_curve[1][0])


SK_norm = curve_fit(deriv_logist,x_dates,SK_cases-SK_recov,p0=[2.4E5,1E-1,1E-1,1E-2],bounds=(0,[1E9,1E3,1E3,1E3]),maxfev=1E3)[0]

res_SK = (SK_cases-SK_recov)-deriv_logist(x_dates,*SK_norm)
ss_res_SK = np.sum(res_SK**2)
ss_tot_SK = np.sum(((SK_cases-SK_recov)-np.mean(SK_cases-SK_recov))**2)
r_2_SK = 1 - ss_res_SK/ss_tot_SK

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(x_dates,SK_cases-SK_recov,frmt)
plt.plot(x_,deriv_logist(x_,*SK_norm))
plt.suptitle('SK Cases - Removed')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(SK_norm[0],-1),SK_norm[2],SK_norm[1],SK_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*SK_norm],6E1)))
print('SK R^2 = {:.4f}'.format(r_2_SK))
plt.clf()

SK_max_predict.append(SK_norm[0])


SK_curve = curve_fit(logistic,x_dates,SK_deaths,p0=[1E2,0.2,50,1],bounds=(0,[2E8,1E1,1000,1E3]),maxfev=2E3)[0]

res_SK = SK_deaths-logistic(x_dates,*SK_curve)
ss_res_SK = np.sum(res_SK**2)
ss_tot_SK = np.sum((SK_deaths-np.mean(SK_deaths))**2)
r_2_SK = 1 - ss_res_SK/ss_tot_SK


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(date_list,SK_deaths,frmt)
plt.plot(x_,logistic(x_,*SK_curve))
plt.suptitle('South Korea Deaths')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(SK_curve[0],0),SK_curve[2],SK_curve[1]))
print('S. Korea R^2 = {:.4f}'.format(r_2_SK))
plt.clf()



print('\n\n')



print('1 infection leads to {:.3f} times more infections per week in the US'.format(US_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week in Italy'.format(Italy_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week in China'.format(China_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week everywhere but China'.format(Excluding_China_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week in S. Korea'.format(SK_base[1]**7))
print('1 infection leads to {:.3f} more infections per week'.format(All_base[1]**7))
print('\n\nUS confirmed cases = {} on {}\n\n'.format(int(US_cases[-1]),date_list[-1]))



ndays = 20


for i in range(ndays+len(US_death_changes),len(x_dates)+1):
    print(date_list[i-1])
    US_max_r = []
    
    All_max_r = []
    All_max_peak = []
    
    Italy_max_r = []
    Italy_max_peak = []
    
    for j in range(int(2E1),int(8E1)+1,1):
#        if j%1E1 ==0:
#            print(10**(j/1E1))
#        print(10**(j/1E1))
        try:
            US_max_r.append(r_sqr(x_dates[:i],US_deaths[:i],logistic,[10**(j/1E1),0.2,60,1],(0,[3.5E8,1E3,1E3,1E3]),1E4))
        except:
            None
        try:
            All_max_r.append(r_sqr(x_dates[:i],All_deaths[:i],logistic,[10**(j/1E1),0.4,20,1],(0,[2E9,1E3,1E3,1E3]),1E4))
        except:
            None
        try:
            Italy_max_r.append(r_sqr(x_dates[:i],Italy_deaths[:i],logistic,[10**(j/1E1),0.25,40,1],(0,[1E8,1E1,1E3,1E3]),1E4))
        except:
            None
    
    US_death_changes.append(US_max_r[np.where(np.array(US_max_r)[:,0] == max(np.array(US_max_r)[:,0]))[0][0]])
    All_death_changes.append(All_max_r[np.where(np.array(All_max_r)[:,0] == max(np.array(All_max_r)[:,0]))[0][0]])
    Italy_death_changes.append(Italy_max_r[np.where(np.array(Italy_max_r)[:,0] == max(np.array(Italy_max_r)[:,0]))[0][0]])
    
fh = open('covid_lists.txt','a')
fh.writelines('US_death_changes = '+str(US_death_changes)+'\n')
fh.writelines('Italy_death_changes = '+str(Italy_death_changes)+'\n')
fh.writelines('All_death_changes = '+str(All_death_changes)+'\n\n')
fh.close()

US_r_2_deaths_chgs = []
All_r_2_deaths_chgs = []
Italy_r_2_deaths_chgs = []

US_death_chgs = []
All_death_chgs = []
Italy_death_chgs = []



#fn_peak(logistic,[x,b,d,e,f],[*US_list_changes[j]],8E1)

for j in range(len(US_death_changes)):
    US_r_2_deaths_chgs.append(US_death_changes[j][0])
    US_death_chgs.append(US_death_changes[j][1][0])
    
    All_r_2_deaths_chgs.append(All_death_changes[j][0])
    All_death_chgs.append(All_death_changes[j][1][0])
    
    Italy_r_2_deaths_chgs.append(Italy_death_changes[j][0])
    Italy_death_chgs.append(Italy_death_changes[j][1][0])
    

frmt = '-'

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_r_2_chgs,frmt)
#plt.plot(x_dates[4:],US_r_2_deaths_chgs,frmt)
plt.suptitle('US $R^2$')
plt.yscale('log')
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_amp_chgs,frmt)
#plt.plot(x_dates[4:],US_death_chgs,frmt)
plt.suptitle('US Max Cases')
#plt.yscale('log')
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_amp_chgs,frmt)
#plt.plot(x_dates[4:],US_death_chgs,frmt)
plt.suptitle('US Max Cases')
plt.yscale('log')
plt.show()
plt.clf()

#plt.xticks(rotation=45)
#plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
#plt.plot(dates_graph[ndays-1:],US_rate_chgs,frmt)
##plt.plot(x_dates[4:],US_rate_chgs,frmt)
#plt.suptitle('US Growth Rate')
#plt.yscale('log')
#plt.show()
#plt.clf()
#
US_max_predict.append(US_amp_chgs[-1])
#
#plt.xticks(rotation=45)
#plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
#plt.plot(dates_graph[ndays-1:],US_center_chgs,frmt)
##plt.plot(x_dates[4:],US_center_chgs,frmt)
#plt.suptitle('US Turnover Dates')
#plt.show()
#plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_peak[0:],frmt)
#plt.plot(x_dates[4:],US_center_chgs,frmt)
plt.suptitle('US Peak # of Cases')
#plt.yscale('log')
plt.show()
plt.clf()

US_max_predict.append(US_peak[-1])

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_peak[0:],frmt)
#plt.plot(x_dates[4:],US_center_chgs,frmt)
plt.suptitle('US Peak # of Cases')
plt.yscale('log')
plt.show()
plt.clf()


print('\n\n')


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],China_r_2_chgs,frmt)
#plt.plot(x_dates[4:],China_r_2_chgs,frmt)
plt.suptitle('China $R^2$')
plt.yscale('log')
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],China_amp_chgs,frmt)
#plt.plot(x_dates[4:],China_amp_chgs,frmt)
plt.suptitle('China Max Cases')
#plt.yscale('log')
plt.show()
plt.clf()

China_max_predict.append(China_amp_chgs[-1])

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],China_amp_chgs,frmt)
#plt.plot(x_dates[4:],China_amp_chgs,frmt)
plt.suptitle('China Max Cases')
plt.yscale('log')
plt.show()
plt.clf()

#plt.xticks(rotation=45)
#plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
#plt.plot(dates_graph[ndays-1:],China_rate_chgs,frmt)
##plt.plot(x_dates[4:],China_rate_chgs,frmt)
#plt.suptitle('China Growth Rate')
#plt.show()
#plt.clf()
#
#plt.xticks(rotation=45)
#plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
#plt.plot(dates_graph[ndays-1:],China_center_chgs,frmt)
##plt.plot(x_dates[4:],China_center_chgs,frmt)
#plt.suptitle('China Turnover Dates')
#plt.show()
#plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays+30::14])
plt.plot(dates_graph[ndays-1+30:],China_peak[30:],frmt)
#plt.plot(x_dates[4:],US_center_chgs,frmt)
plt.suptitle('China Peak # of Cases')
#plt.yscale('log')
plt.show()
plt.clf()

China_max_predict.append(China_peak[-1])

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays+30::14])
plt.plot(dates_graph[ndays-1+30:],China_peak[30:],frmt)
#plt.plot(x_dates[4:],US_center_chgs,frmt)
plt.suptitle('China Peak # of Cases')
plt.yscale('log')
plt.show()
plt.clf()


print('\n\n')


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],Italy_r_2_chgs,frmt)
#plt.plot(x_dates[4:],Italy_r_2_deaths_chgs,frmt)
plt.suptitle('Italy $R^2$')
plt.yscale('log')
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],Italy_amp_chgs,frmt)
#plt.plot(x_dates[4:],Italy_death_chgs,frmt)
plt.suptitle('Italy Max Cases')
#plt.yscale('log')
plt.show()
plt.clf()

Italy_max_predict.append(Italy_amp_chgs[-1])

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],Italy_amp_chgs,frmt)
#plt.plot(x_dates[4:],Italy_amp_chgs,frmt)
plt.suptitle('Italy Max Cases')
plt.yscale('log')
plt.show()
plt.clf()

#plt.xticks(rotation=45)
#plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
#plt.plot(dates_graph[ndays-1:],Italy_rate_chgs,frmt)
##plt.plot(x_dates[4:],Italy_rate_chgs,frmt)
#plt.suptitle('Italy Growth Rate')
#plt.yscale('log')
#plt.show()
#plt.clf()
#
#plt.xticks(rotation=45)
#plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
#plt.plot(dates_graph[ndays-1:],Italy_center_chgs,frmt)
##plt.plot(x_dates[4:],Italy_center_chgs,frmt)
#plt.suptitle('Italy Turnover Dates')
#plt.show()
#plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],Italy_peak[0:],frmt)
#plt.plot(x_dates[4:],Italy_center_chgs,frmt)
plt.suptitle('Italy Peak # of Cases')
#plt.yscale('log')
plt.show()
plt.clf()

Italy_max_predict.append(Italy_peak[-1])

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],Italy_peak[0:],frmt)
#plt.plot(x_dates[4:],Italy_center_chgs,frmt)
plt.suptitle('Italy Peak # of Cases')
plt.yscale('log')
plt.show()
plt.clf()


print('\n\n')

print('1 infection leads to {:.3f} times more infections per week in the US'.format(US_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week in China'.format(China_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week in Italy'.format(Italy_base[1]**7))

print('\n\nUS confirmed cases = {} on {}'.format(int(US_cases[-1]),date_list[-1]))
print('\nChina confirmed cases = {} on {}'.format(int(China_cases[-1]),date_list[-1]))
print('\nItaly confirmed cases = {} on {}'.format(int(Italy_cases[-1]),date_list[-1]))


print('\n\n\n')



print('US Max Cases: {:.0f} (Confirmed)\t {:.0f} (Recovered)\t {:.0f} (Growth)'.format(US_max_predict[0],US_max_predict[1],US_max_predict[2]))
print('Peak US Cases (Current): {:.0f}'.format(US_max_predict[3]))
print('\n')

print('China Max Cases: {:.0f} (Confirmed)\t {:.0f} (Recovered)\t {:.0f} (Growth)'.format(China_max_predict[0],China_max_predict[1],China_max_predict[2]))
print('Peak China Cases (Current): {:.0f}'.format(China_max_predict[3]))
print('\n')

print('Italy Max Cases: {:.0f} (Confirmed)\t {:.0f} (Recovered)\t {:.0f} (Growth)'.format(Italy_max_predict[0],Italy_max_predict[1],Italy_max_predict[2]))
print('Peak Italy Cases (Current): {:.0f}'.format(Italy_max_predict[3]))
print('\n')

print('S. Korea Max Cases: {:.0f} (Confirmed)\t {:.0f} (Recovered)'.format(SK_max_predict[0],SK_max_predict[1]))
print('\n\n\n')



plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_r_2_deaths_chgs)
#plt.plot(x_dates[4:],US_r_2_chgs,frmt)
plt.suptitle('US Deaths $R^2$')
plt.yscale('log')
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_death_chgs)
#plt.plot(x_dates[4:],US_amp_chgs,frmt)
plt.suptitle('US Max Deaths')
plt.yscale('log')
plt.show()
plt.clf()


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],Italy_r_2_deaths_chgs)
#plt.plot(x_dates[4:],Italy_r_2_chgs,frmt)
plt.suptitle('Italy Deaths $R^2$')
plt.yscale('log')
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],Italy_death_chgs)
#plt.plot(x_dates[4:],Italy_amp_chgs,frmt)
plt.suptitle('Italy Max Deaths')
plt.yscale('log')
plt.show()
plt.clf()


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],All_r_2_deaths_chgs)
#plt.plot(x_dates[4:],All_r_2_deaths_chgs,frmt)
plt.suptitle('All Deaths $R^2$')
plt.yscale('log')
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],All_death_chgs)
#plt.plot(x_dates[4:],All_death_chgs,frmt)
plt.suptitle('All Max Deaths')
plt.yscale('log')
plt.show()
plt.clf()


print('\n\n\n')

plt.xticks(rotation=45)
start_=90
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[start_::7])

plt.plot(date_list[start_:],US_deaths[start_:],frmt,label='Reported Deaths')
plt.plot(dates_graph[start_:],US_death_chgs[start_-ndays+1:],label='Est. Max Deaths')
plt.suptitle('US Deaths')
plt.legend()
plt.show()
plt.clf()

print('\n')
print('Current Deaths in US: {}'.format(US_deaths[-1]))
print('\n\n\n')


hospitalizations_str='https://covidtracking.com/api/v1/us/daily.csv'
hospitalizations=pd.read_csv(hospitalizations_str)

hospitalizations=hospitalizations.sort_values(by='date').reset_index().drop(columns='index')
hospitalizations.date=pd.to_datetime((hospitalizations.date.astype('str')))
hospitalizations.hospitalizedCurrently=np.nan_to_num(hospitalizations.hospitalizedCurrently)

hosp_dates = hospitalizations.date
hosp_cases = hospitalizations.hospitalizedCurrently
plt.xticks(rotation=45)
start_= int(np.where(hosp_cases != 0.0)[0][0])
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[start_::7])

plt.plot(date_list[start_:],hosp_cases[start_:],frmt)
#plt.plot(dates_graph[start_:],US_death_chgs[start_-ndays+1:],label='Est. Max Deaths')
plt.suptitle('US Hospitalizations')
#plt.legend()
plt.show()
plt.clf()



print('\n\n\n')
