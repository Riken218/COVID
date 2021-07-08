try:
    from Work import*
except:
    from Workspace import*
print('\n')


#confirmed = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_19-covid-Confirmed.csv&filename=time_series_2019-ncov-Confirmed.csv'
#confirmed = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
confirmed = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

cases = pd.read_csv(confirmed)
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

print(date_list[-1])

dates_graph = date_list

frmt = 's'

extn = 1.1

x_dates = np.arange(len(date_list))
x_ = np.linspace(0,len(date_list)*extn,int(1E4))


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.suptitle('Confirmed Cases')
plt.plot(dates_graph,US_cases,label='US')
plt.plot(dates_graph,Italy_cases,label='Italy')
plt.plot(dates_graph,China_cases,label='China')
#plt.plot(dates_graph,Excluding_China_cases,label='All but China')
plt.plot(dates_graph,SK_cases,label='South Korea')
plt.legend()
plt.show()
plt.clf()

#Data Fitting stuff

def logistic(x,amp,rate,center,nu):
    return amp*(1+M.e**(-rate*(x-center)))**(-1/nu)

def base_rate(x,amp,base,center):
    return amp/(1+base**(-(x-center)))

def just_r_sqr(xdata,ydata,fn_name):
    res = np.array(ydata)-fn_name(xdata)
    return (1 - np.sum(res**2)/np.sum((np.array(ydata)-np.mean(np.array(ydata)))**2))

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(dates_graph,All_cases,frmt)
plt.suptitle('All')
plt.show()
plt.clf()



China_base = curve_fit(base_rate,x_dates,China_cases,p0=[80000,1.2,20],bounds=(0,[1E6,1E3,1000]))[0]
China_curve = r_sqr(x_dates,China_cases,logistic,[8E4,0.2,20,1],(0,[1E6,1E1,1E3,1E3]))

deriv_1_China = lambdify(X,diff(logistic(X,*China_curve[1]),X,1))
deriv_2_China = lambdify(X,diff(logistic(X,*China_curve[1]),X,2))
deriv_rsqr_China = just_r_sqr(x_dates[1:],np.diff(China_cases),deriv_1_China)
peak_China_day = fsolve(deriv_2_China,China_curve[1][2])[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

#plt.plot(date_list,China_cases,frmt)
plt.plot(x_dates,China_cases,frmt)
plt.plot(x_,logistic(x_,*China_curve[1]))
plt.suptitle('China')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(China_curve[1][0],-2),China_curve[1][2],China_curve[1][1]))
print('China R^2 = {:.4f}'.format(China_curve[0]))
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

#plt.plot(date_list,China_cases,frmt)
plt.plot(x_dates[1:],np.diff(China_cases),'s-')
plt.plot(x_,deriv_1_China(x_))
plt.suptitle('China New Cases')
plt.show()
print('Peak of {:.0f} at {:.1f} days'.format(deriv_1_China(peak_China_day),peak_China_day))
print('Peak China R^2 = {:.4f}'.format(deriv_rsqr_China))
plt.clf()



US_base = curve_fit(base_rate,x_dates,US_cases,p0=[2E5,1.5,60],bounds=(0,[3.5E8,1E3,1000]),maxfev=9E2)[0]
US_curve = r_sqr(x_dates,US_cases,logistic,[2E5,0.4,60,1],(0,[3.5E8,1E1,1E3,1E3]))

deriv_1_US = lambdify(X,diff(logistic(X,*US_curve[1]),X,1))
deriv_2_US = lambdify(X,diff(logistic(X,*US_curve[1]),X,2))
deriv_rsqr_US = just_r_sqr(x_dates[1:],np.diff(US_cases),deriv_1_US)
peak_US_day = fsolve(deriv_2_US,US_curve[1][2]+7E1)[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(date_list,US_cases,frmt)
#plt.plot(x_dates,US_cases,frmt)
plt.plot(x_,logistic(x_,*US_curve[1]))
plt.suptitle('US')
#plt.yscale('log')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(US_curve[1][0],-2),US_curve[1][2],US_curve[1][1]))
print('US R^2 = {:.4f}'.format(US_curve[0]))
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates[1:],np.diff(US_cases),'s-')
plt.plot(x_,deriv_1_US(x_))
plt.suptitle('US New Cases')
plt.show()
print('Peak of {:.0f} at {:.1f} days'.format(deriv_1_US(peak_US_day),peak_US_day))
print('Peak US R^2 = {:.4f}'.format(deriv_rsqr_US))
plt.clf()


Italy_curve = r_sqr(x_dates,Italy_cases,logistic,[1E7,0.2,50,1],(0,[1E8,1E1,1000,1E3]))
Italy_base = curve_fit(base_rate,x_dates,Italy_cases,p0=[1E7,1.2,50],bounds=(0,[1E8,1E3,1000]))[0]

res_Italy = Italy_cases-logistic(x_dates,*Italy_curve[1])
r_2_Italy = 1 - np.sum(res_Italy**2)/np.sum((Italy_cases-np.mean(Italy_cases))**2)

deriv_1_Italy = lambdify(X,diff(logistic(X,*Italy_curve[1]),X,1))
deriv_2_Italy = lambdify(X,diff(logistic(X,*Italy_curve[1]),X,2))
deriv_rsqr_Italy = just_r_sqr(x_dates[1:],np.diff(Italy_cases),deriv_1_Italy)
peak_Italy_day = fsolve(deriv_2_Italy,Italy_curve[1][2])[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

#plt.plot(date_list,Italy_cases,frmt)
plt.plot(x_dates,Italy_cases,frmt)
plt.plot(x_,logistic(x_,*Italy_curve[1]))
plt.suptitle('Italy')
plt.show()
print('Italy R^2 = {:.4f}'.format(r_2_Italy))
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(Italy_curve[1][0],-2),Italy_curve[1][2],Italy_curve[1][1]))
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates[1:],np.diff(Italy_cases),'s-')
plt.plot(x_,deriv_1_Italy(x_))
plt.suptitle('Italy New Cases')
plt.show()
print('Peak of {:.0f} at {:.1f} days'.format(deriv_1_Italy(peak_Italy_day),peak_Italy_day))
print('Peak Italy R^2 = {:.4f}'.format(deriv_rsqr_Italy))
plt.clf()


Excluding_China_curve = r_sqr(x_dates,Excluding_China_cases,logistic,[1E8,0.2,20,1],(0,[1E9,1E1,2E3,1E3]),maxfev=1.5E3)
Excluding_China_base = curve_fit(base_rate,x_dates,Excluding_China_cases,p0=[1E8,1.3,20],bounds=(0,[1E9,1E3,2000]),maxfev=1.5E3)[0]

res_Excluding_China = Excluding_China_cases-logistic(x_dates,*Excluding_China_curve[1])
r_2_Excluding_China = 1 - np.sum(res_Excluding_China**2)/np.sum((Excluding_China_cases-np.mean(Excluding_China_cases))**2)

deriv_1_Excluding_China = lambdify(X,diff(logistic(X,*Excluding_China_curve[1]),X,1))
deriv_2_Excluding_China = lambdify(X,diff(logistic(X,*Excluding_China_curve[1]),X,2))
deriv_rsqr_Excluding_China = just_r_sqr(x_dates[1:],np.diff(Excluding_China_cases),deriv_1_Excluding_China)
peak_Excluding_China_day = fsolve(deriv_2_Excluding_China,Excluding_China_curve[1][2])[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

#plt.plot(date_list,Excluding_China_cases,frmt)
plt.plot(x_dates,Excluding_China_cases,frmt)
plt.plot(x_,logistic(x_,*Excluding_China_curve[1]))
plt.suptitle('All Except China')
plt.yscale('log')
plt.show()
print('Excluding China R^2 = {:.4f}'.format(r_2_Excluding_China))
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(Excluding_China_curve[1][0],-3),Excluding_China_curve[1][2],Excluding_China_curve[1][1]))
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates[1:],np.diff(Excluding_China_cases),'s-')
plt.plot(x_,deriv_1_Excluding_China(x_))
plt.suptitle('All but China New Cases')
plt.show()
print('Peak of {:.0f} at {:.1f} days'.format(deriv_1_Excluding_China(peak_Excluding_China_day),peak_Excluding_China_day))
print('Peak All but China R^2 = {:.4f}'.format(deriv_rsqr_Excluding_China))
plt.clf()


SK_curve = r_sqr(x_dates,SK_cases,logistic,[2E4,0.4,6E1,1],(0,[2E8,1E1,1E3,1E3]),maxfev=2E3)
SK_base = curve_fit(base_rate,x_dates,SK_cases,p0=[2E5,2,60],bounds=(0,[2E8,1E3,1000]),maxfev=2E3)[0]

res_SK = SK_cases-logistic(x_dates,*SK_curve[1])
r_2_SK = 1 - np.sum(res_SK**2)/np.sum((SK_cases-np.mean(SK_cases))**2)

deriv_1_SK = lambdify(X,diff(logistic(X,*SK_curve[1]),X,1))
deriv_2_SK = lambdify(X,diff(logistic(X,*SK_curve[1]),X,2))
deriv_rsqr_SK = just_r_sqr(x_dates[1:],np.diff(SK_cases),deriv_1_SK)
peak_SK_day = fsolve(deriv_2_SK,SK_curve[1][2]+4E1)[0]


plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

#plt.plot(date_list,SK_cases,frmt)
plt.plot(x_dates,SK_cases,frmt)
plt.plot(x_,logistic(x_,*SK_curve[1]))
plt.suptitle('South Korea')
plt.show()
print('{:.0f} people with a turnaround at {:.1f} days and a growth rate of {:.4f}'.format(round(SK_curve[1][0],-1),SK_curve[1][2],SK_curve[1][1]))
print('S. Korea R^2 = {:.4f}'.format(r_2_SK))
plt.clf()

plt.xticks(rotation=45)
plt.xticks(ticks=np.arange(0,len(dates_graph)*extn+1,7),labels=dates_labels[::7])

plt.plot(x_dates[1:],np.diff(SK_cases),'s-')
plt.plot(x_,deriv_1_SK(x_))
plt.suptitle('S. Korea New Cases')
plt.show()
print('Peak of {:.0f} at {:.1f} days'.format(deriv_1_SK(peak_SK_day),peak_SK_day))
print('Peak S. Korea R^2 = {:.4f}'.format(deriv_rsqr_SK))
plt.clf()



print('\n\n')



print('1 infection leads to {:.3f} times more infections per week in the US'.format(US_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week in Italy'.format(Italy_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week in China'.format(China_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week everywhere but China'.format(Excluding_China_base[1]**7))
print('1 infection leads to {:.3f} times more infections per week in S. Korea'.format(SK_base[1]**7))
print('\n\nUS confirmed cases = {} on {}\n\n'.format(int(US_cases[-1]),date_list[-1]))
#print('1 infection leads to {:.2f} more infections per week'.format(All_base[1]**7))



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
plt.show()
plt.clf()


