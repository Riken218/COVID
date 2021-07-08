try:
    from Work import*
except:
    from Workspace import*
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
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
         '9/25/20','9/26/20','9/27/20','9/28/20','9/29/20','9/30/20']

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


diff_China = China_cases-China_recov-China_deaths
diff_Italy = Italy_cases-Italy_recov-Italy_deaths
diff_US = US_cases-US_recov-US_deaths
diff_SK = SK_cases-SK_recov-SK_deaths
diff_Ex_China = Excluding_China_cases-Excluding_China_recov-Excluding_China_deaths
diff_All = All_cases-All_recov-All_deaths





print(date_list[-1])

dates_graph = date_list

frmt = 's'

extn = 2E0

x_dates = np.arange(len(date_list))
x_ = np.linspace(1,len(date_list)*extn,int(1E4))




trans = 8E-5
remov = 1E-2
amp = 1E6
def fitfn(xdata,amp,trans,remov):
    N = amp
    S = N-1
    I = 1
    R = 0
    def SIR(sir,x):
        S_,I_,R_ = sir
        return [-trans*S_*I_,trans*S_*I_-remov*I_,remov*I_]
    initSIR = [S,I,R]
    
    fnsol = odeint(SIR,initSIR,xdata)
    return fnsol[:,2]

def logistic(x,amp,rate,center,nu):
    return amp*(1+M.e**(-rate*(x-center)))**(-1/nu)

def deriv_log(x,amp,rate,center,nu):
    fn = amp*rate/nu*M.e**(-rate*(x-center))*(1+M.e**(-rate*(x-center)))**(-(nu+1)/nu)
    fn = np.nan_to_num(np.asarray(fn))
    return fn


ydata = US_cases


trans = 1E-10
remov = 1E-10
amp = 1E3

lg_fit_poss = []

#for i in range(int(7E0)):
#    print(i)
#    for j in range(int(1.3E1)):
#        print(j)
#        for k in range(int(1.3E1)):
#            lg_fit_poss.append(r_sqr(x_dates,ydata,fitfn,[amp*10**i,trans*10**j,remov*10**k],(0,[8E9,1E3,1E3]),maxfev=1E5))
#
#tr_fit = lg_fit_poss[np.where(np.array(lg_fit_poss)[:,0] == max(np.array(lg_fit_poss)[:,0]))[0][0]]
#tr_fit = r_sqr(x_dates,ydata,fitfn,[amp,trans,remov],(0,[8E9,1E2,1E2]),maxfev=1E4)
tr_fit = [0.9996771995257047, [149397428.6155831, 1.986412700724577e-09, 0.00020759010819830837]]
print(tr_fit)
lg_fit = r_sqr(x_dates,ydata,logistic,[1E6,0.4,70,1],(0,[8E9,1E3,1E3,1E3]),maxfev=1E4)
print(lg_fit)


#sol = odeint(SIR,y0,x_,args=(trans,remov))
#sol_array = odeint(SIR,[S,I,R],x_,args=(trans,remov))
#Sfn = fitfn(x_,trans,remov)[:,0]
#Ifn = fitfn(x_,trans,remov)[:,1]
#Rfn = fitfn(x_,trans,remov)[:,2]
plt.plot(x_,fitfn(x_,*tr_fit[1]),label='SIR Model')
plt.plot(x_,logistic(x_,*lg_fit[1]),label='Logistic Model')
plt.plot(x_dates,ydata,'s',label='Data')
#plt.yscale('log')
#plt.ylim(10**-3,10**6)
#plt.xscale('log')
plt.legend()
plt.show()


#quad(fitfn,0,10,args=(trans,remov))







