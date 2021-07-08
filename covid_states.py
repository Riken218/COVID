try:
    from Work import*
except:
    from Workspace import*
print('\n')


confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'             
cases = pd.read_csv(confirmed)

dead = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
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
date_list = date_list[11:]
dates_labels = dates

state = str(input('Name of State to Analyze: '))

US_df = cases[cases['Province_State']==state]

US_cases = np.array(list(US_df.sum(axis=0))[11:])

US_df = deaths[deaths['Province_State']==state]

US_deaths = np.array(list(US_df.sum(axis=0))[12:])

diff_US = US_cases-US_deaths


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


US_base = curve_fit(base_rate,x_dates,US_cases,p0=[5E3,1.5,60,1],bounds=(0,[3.5E8,1E3,1000,1E3]),maxfev=1E4)[0]

'''Updated as of 4/24/2020'''
US_list_changes = []

US_peak_chgs = []

ndays = np.where(US_cases>0)[0][0]
print('Start num days: ',ndays)

for i in range(ndays+len(US_list_changes),len(x_dates)): #len(x_dates)+1?
    #if i%10==0:
    #    print(date_list[i-1])
    print(date_list[i])
    US_max_r = []
    US_max_peak = []
    
    for j in range(int(2E1),int(5E1)+1,1):
#        print(j)
        try:
            US_max_r.append(r_sqr(x_dates[:i],US_cases[:i],logistic,[10**(j/1E1),0.2,1.3E2,1],(0,[3.5E6,1E3,1E3,1E3]),1E5))
        except:
            None
#            US_max_r.append([0,0])
#            print('Failed to Append R')
        try:
            US_max_peak.append(r_sqr(x_dates[:i],diff_US[:i],deriv_log,[10**(j/1E1),1E-1,1.3E2,1E0],(0,[3.5E6,1E3,1E3,1E3]),1E5))
        except:
            None
#            US_max_r.append([0,0])
#            print('Failed to Append Peak')

#    print(US_max_peak)
    US_list_changes.append(US_max_r[np.where(np.array(US_max_r)[:,0] == max(np.array(US_max_r)[:,0]))[0][0]])
    US_peak_chgs.append(US_max_peak[np.where(np.array(US_max_peak)[:,0] == max(np.array(US_max_peak)[:,0]))[0][0]])


US_r_2_chgs = []
US_amp_chgs = []
US_rate_chgs = []
US_center_chgs = []
US_nu_chgs = []
US_peak = []


for j in range(len(US_list_changes)):
    US_r_2_chgs.append(US_list_changes[j][0])
    US_amp_chgs.append(US_list_changes[j][1][0])
    US_rate_chgs.append(US_list_changes[j][1][1])
    US_center_chgs.append(US_list_changes[j][1][2])
    US_nu_chgs.append(US_list_changes[j][1][3])
    US_peak.append(1 if max(deriv_log(x_,*US_peak_chgs[j][1]))>=np.nan_to_num(np.inf)/10 else max(deriv_log(x_,*US_peak_chgs[j][1])))





plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.suptitle('Confirmed Cases')
plt.plot(dates_graph,US_cases,label=state)
plt.yscale('linear')
plt.legend()
plt.show()
plt.clf()





plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

#For fit, use deriv_logist distribution formula
plt.suptitle('Confirmed - Deaths')

plt.plot(dates_graph,diff_US,label=state)
plt.yscale('linear')
plt.legend()
plt.show()
plt.clf()


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.suptitle('Confirmed deaths')
plt.plot(dates_graph,US_deaths,label=state)
plt.yscale('linear')
plt.legend()
plt.show()
plt.clf()




US_new = np.diff(US_cases)


plt.plot(US_cases[1:],US_new,'-',color = 'blue',label=state)
plt.plot(US_cases[-1],US_new[-1],'o',color = 'blue',markersize = '12')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()
plt.clf()




if np.any(US_cases<=0):
    US_cases[US_cases<=0] = 1E-9
    US_death_rate = US_deaths/US_cases*100
else:
    US_death_rate = US_deaths/US_cases*100


plt.suptitle('Death Rates (%)')
plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])
plt.plot(date_list,US_death_rate,label=state)
plt.legend()
plt.show()
plt.clf()


US_first_index = np.where(US_deaths != 0)[0][0]

print('\n\n')


print('\nCurrent Death Rate in {}: {:.2f}%\n'.format(state,US_death_rate[-1]))
print('\n\n\n')
print('Deaths in {} = {} on {}'.format(state,int(US_deaths[-1]),date_list[-1]))
print('\n\n')



US_max_predict = []


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
plt.suptitle(state)
#plt.yscale('log')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(US_curve[1][0],-1),US_curve[1][2],US_curve[1][1]))
print('{} R^2 = {:.4f}'.format(state,US_curve[0]))
plt.clf()

US_max_predict.append(US_curve[1][0])


US_norm = curve_fit(deriv_logist,x_dates,diff_US,p0=[4E6,1.5E-1,9E1,7E-1],maxfev=int(1E4))[0]

res_US = (diff_US)-deriv_logist(x_dates,*US_norm)
ss_res_US = np.sum(res_US**2)
ss_tot_US = np.sum(((diff_US)-np.mean(diff_US))**2)
r_2_US = 1 - ss_res_US/ss_tot_US

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(x_dates,diff_US,frmt)
plt.plot(x_,deriv_logist(x_,*US_norm))
plt.suptitle('{} Cases - Deaths'.format(state))
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days, a growth rate of {:.4f}, and nu = {:.3f}'.format(round(US_norm[0],-2),US_norm[2],US_norm[1],US_norm[3]))
print('Peak # of Cases = {:.1f}'.format(fn_peak(logist,[x,b,d,e,f],[*US_norm],9E1))) #WI = 1.5E2, MN = 1.7E2, IA = 1.2E2, NY = 9E1
print('{} R^2 = {:.4f}'.format(state,r_2_US))
plt.clf()

US_max_predict.append(US_norm[0])



US_curve = curve_fit(logistic,x_dates,US_deaths,p0=[1E6,0.2,1E2,1],bounds=(0,[3.5E8,1E1,1000,1E3]),maxfev=2E3)[0]

res_US = US_deaths-logistic(x_dates,*US_curve)
ss_res_US = np.sum(res_US**2)
ss_tot_US = np.sum((US_deaths-np.mean(US_deaths))**2)
r_2_US = 1 - ss_res_US/ss_tot_US


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_labels[::14])

plt.plot(date_list,US_deaths,frmt)
plt.plot(x_,logistic(x_,*US_curve))
plt.suptitle('{} Deaths'.format(state))
#plt.yscale('log')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(US_curve[0],0),US_curve[2],US_curve[1]))
print('{} R^2 = {:.4f}'.format(state,r_2_US))
plt.clf()



print('\n\n')

print('1 infection leads to {:.3f} times more infections per week in {}'.format(US_base[1]**7,state))
print('\n\n{} confirmed cases = {} on {}\n\n'.format(state,int(US_cases[-1]),date_list[-1]))




'''Past Max Deaths'''

'''Updated as of 4/24/2020'''
US_death_changes = []


for i in range(ndays+len(US_death_changes),len(x_dates)+1):
    if i%10==0:
        print(date_list[i-1])
    US_max_r = []
    
    for j in range(int(2E1),int(6E1)+1,1):
        try:
            US_max_r.append(r_sqr(x_dates[:i],US_deaths[:i],logistic,[10**(j/1E1),0.2,1.4E2,1],(0,[3.5E8,1E3,1E3,1E3]),1E5))
        except:
            None
    US_death_changes.append(US_max_r[np.where(np.array(US_max_r)[:,0] == max(np.array(US_max_r)[:,0]))[0][0]])


US_r_2_deaths_chgs = []
US_death_chgs = []



for j in range(len(US_death_changes)):
    US_r_2_deaths_chgs.append(US_death_changes[j][0])
    US_death_chgs.append(US_death_changes[j][1][0])
    

frmt = '-'

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_r_2_chgs,frmt)
#plt.plot(x_dates[4:],US_r_2_deaths_chgs,frmt)
plt.suptitle('{} R^2'.format(state))
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_amp_chgs,frmt)
#plt.plot(x_dates[4:],US_death_chgs,frmt)
plt.suptitle('{} Max Cases'.format(state))
#plt.yscale('log')
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_amp_chgs,frmt)
#plt.plot(x_dates[4:],US_death_chgs,frmt)
plt.suptitle('{} Max Cases'.format(state))
plt.yscale('log')
plt.show()
plt.clf()


US_max_predict.append(US_amp_chgs[-1])

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_peak[0:],frmt)
#plt.plot(x_dates[4:],US_center_chgs,frmt)
plt.suptitle('{} Peak # of Cases'.format(state))
#plt.yscale('log')
plt.show()
plt.clf()

US_max_predict.append(US_peak[-1])

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_peak[0:],frmt)
#plt.plot(x_dates[4:],US_center_chgs,frmt)
plt.suptitle('{} Peak # of Cases'.format(state))
plt.yscale('log')
plt.show()
plt.clf()


print('\n\n')

print('1 infection leads to {:.3f} times more infections per week in the {}'.format(US_base[1]**7,state))

print('\n\n{} confirmed cases = {} on {}'.format(state,int(US_cases[-1]),date_list[-1]))


print('\n\n\n')



print('{} Max Cases: {:.0f} (Confirmed)\t {:.0f} (Growth)'.format(state,US_max_predict[0],US_max_predict[1]))
print('Peak {} Cases (Current): {:.0f}'.format(state,US_max_predict[2]))
print('\n')


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_r_2_deaths_chgs)
#plt.plot(x_dates[4:],US_r_2_chgs,frmt)
plt.suptitle('{} Deaths R^2'.format(state))
plt.show()
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,14),labels=dates_graph[ndays::14])
plt.plot(dates_graph[ndays-1:],US_death_chgs)
#plt.plot(x_dates[4:],US_amp_chgs,frmt)
plt.suptitle('{} Max Deaths'.format(state))
plt.yscale('log')
plt.show()
plt.clf()


print('\n\n\n')

axis = 'linear'
confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'             
cases = pd.read_csv(confirmed)

dead = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
deaths = pd.read_csv(dead)
print_counties=True
if state == 'Wisconsin':
  counties = ['Pierce','Eau Claire','Dunn','Pepin','Chippewa','St. Croix','Dane','Milwaukee']
  axis = 'log'

elif state == 'Iowa':
  counties = ['Johnson','Linn']
elif state == 'Minnesota':
  counties = ['Winona','Wabasha','Goodhue','Olmsted']
  axis='log'
else:
  print_counties=False

if print_counties:
  date_list = cases.columns
  date_list = date_list[11:]
  
  
  state_cases = cases[cases['Province_State']==state]
  state_deaths = deaths[deaths['Province_State']==state]
  
  state_cases = state_cases.drop(columns=['UID','iso2','iso3','code3','FIPS','Country_Region','Lat','Long_','Combined_Key','Province_State'])
  state_deaths = state_deaths.drop(columns=['UID','iso2','iso3','code3','FIPS','Country_Region','Lat','Long_','Combined_Key','Population','Province_State'])
  
  state_cases.set_index('Admin2').T.plot.line(y=counties,title=state+' Cases')
  plt.yscale(axis)
  plt.legend(loc='upper left')
  state_deaths.set_index('Admin2').T.plot.line(y=counties,title=state+' Deaths')
  plt.yscale(axis)
  plt.legend(loc='upper left')
else:
  print('\n\nDone\n\n')




