# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:33:39 2020

@author: qx911590
"""
pl.close('all')

# six regions
gbi = [(-7.33,58.73),(-1.45,58.73),(2.08,52.83),(1.58,51.03),(-5.46,49.88),
       (-10.99,52.00)]
france = [(-5.02,48.63),(1.48,50.94),(2.45,51.13),(8.08,48.96),(7.55,47.60),
          (6.13,47.44),(7.04,45.93),(7.73,43.66),(3.23,42.49),(-1.87,43.42)]
italy = [(7.67,43.74),(6.86,45.87),(10.51,46.82),(13.66,46.63),(13.56,45.67),
         (12.32,45.33),(18.96,40.19),(16.00,37.71),(13.06,41.09),(9.25,43.90)]
balkans = [(13.66,46.59),(16.06,46.88),(18.65,45.81),(20.22,46.16),(22.65,44.13),
           (28.53,43.74),(27.98,41.01),(26.10,40.74),(22.99,40.24),(24.24,38.01),
            (20.56,38.24),(18.05,42.42),(13.48,45.39)]
sswenor = [(4.82,58.00),(4.38,62.04),(14.72,63.20),(17.70,61.82),(19.32,59.70),
           (14.62,55.43),(12.79,55.37),(10.67,58.82),(8.61,57.95)]
south_scand = [(4.82,58.00),(4.38,62.04),(14.72,63.20),(17.70,61.82),(19.32,59.70),
               (14.62,55.43),(12.79,55.37),(10.97,54.35),(8.53,54.35),(7.39,58.00)]
denngerhol = [(3.41,51.53),(4.77,53.52),(8.45,53.95),(7.75,57.56),(10.88,57.83),
              (10.80,54.54),(14.15,54.27),(14.62,50.94),(6.15,50.94)]
poland = [(14.23,53.98),(18.15,54.92),(22.77,54.41),(23.95,52.76),(22.46,49.04),
          (14.78,50.82)]
ngerpol = [(7.20,53.49),(20.26,54.43),(18.90,49.61),(14.88,51.08),(11.98,50.30),
           (6.15,50.30)]
iberia = [(-8.95,43.93),(-1.95,43.74),(4.28,42.04),(-1.2,35.99),(-5.55,35.23),
          (-10.09,36.93)]
westbalkans = [(13.70,46.55),(16.24,46.83),(18.81,45.79),(21.87,42.37),
               (21.94,39.01),(20.65,38.79),(19.22,40.35),(19.23,41.73),
                (13.71,44.81),(13.57,45.71)]
italy_aegean = [(7.67,43.74),(6.86,45.87),(10.51,46.82),(13.66,46.63),
                (17.22,46.03),(20.80,43.26),(21.87,42.37),(21.94,39.01),
                (20.65,38.79),(19.22,40.35),(16.00,37.71),(13.06,41.09),
                (9.25,43.90)]
romania = [(20.20,46.16),(21.17,46.41),(22.84,48.06),(24.95,47.79),(26.57,48.29),
           (28.23,46.62),(28.20,45.45),(29.67,45.21),(28.57,43.72),(22.59,44.21)]
northeast = [(14.28,53.97),(21.10,55.63),(22.05,59.00),(25.37,59.70),(28.21,59.62),
             (28.15,56.18),(28.15,48.91),(22.59,49.09),(14.83,50.92)]
baltics = [(21.10,55.63),(22.05,59.00),(25.37,59.70),(28.21,59.62),(28.15,56.18),
           (24.47,53.98),(22.98,54.40)]
east_europe = [(21.10,55.63),(22.05,59.00),(25.37,59.70),(28.21,59.62),(28.15,56.18),
               (30.15,55.89),(30.15,50.00),(34.02,46.35),(31.69,46.72),(26.68,48.28),
                (22.20,48.45),(23.60,51.60),(23.49,54.14)]
ukraine_belarus = [(23.52,53.95),(28.16,56.16),(30.85,55.64),(30.18,51.50),
                   (30.53,50.44),(34.02,46.82),(34.02,46.18),(31.71,46.63),
                    (30.09,46.47),(29.21,48.00),(27.70,48.53),(24.93,47.84),
                    (22.12,48.38),(24.00,50.89)]
hungary = [(16.18,46.91),(17.09,48.00),(18.73,47.87),(20.54,48.55),(22.14,48.44),
           (22.92,47.99),(22.03,47.61),(21.10,46.23),(19.60,46.17),(18.07,45.78)]

regions = [gbi,france,italy,ngerpol,hungary,balkans]
#edges = ['g','magenta','deeppink','gold','k','deepskyblue']
titles = ['b. Great Britain & Ireland','c. France','d. Italy',
          'e. Poland & North Germany','f. Hungary','g. Balkans']

V_an = pl.zeros([len(regions),len(years)])
S = pl.zeros([len(regions),len(teleindex),len(years)])
#ACT = pl.zeros([len(regions),2]) # actual trend
#TCT = pl.zeros([len(regions),len(teleindex),2]) # tc component trend
#RES = ACT.copy()
c = ['r','b','darkgoldenrod','hotpink']
x = pl.linspace(1,68,68)
cors_array = pl.zeros([4,len(regions),2])

for i in range(len(regions)):
    V = RegionCalc(regions[i],lon2,lat2,data)
    V_an[i] = V - pl.nanmean(V)
#    actual = stats.linregress(x,RegionCalc(regions[i],lon2,lat2,data)).slope
#    tc1_trnd = stats.linregress(x,RegionCalc(regions[i],lon2,lat2,tc_comp[0])).slope
#    tc2_trnd = stats.linregress(x,RegionCalc(regions[i],lon2,lat2,tc_comp[1])).slope
#    tc3_trnd = stats.linregress(x,RegionCalc(regions[i],lon2,lat2,tc_comp[2])).slope
#    tc4_trnd = stats.linregress(x,RegionCalc(regions[i],lon2,lat2,tc_comp[3])).slope
#    removed = data - (tc_comp[0]+tc_comp[1]+tc_comp[2]+tc_comp[3])
#    residual = stats.linregress(x,RegionCalc(regions[i],lon2,lat2,removed)).slope
#    
#    print round(actual,3), round(tc1_trnd,3), round(tc2_trnd,3), round(tc3_trnd,3), round(tc4_trnd,3), round(residual,3)
    # 

    #res = pl.zeros([2])
    for j in range(3):
        S[i,j] = RegionCalc(regions[i],lon2,lat2,tc_comp[j])
        cor = stats.pearsonr(V_an[i],S[i,j])
        #print round(cor[0],2), round(cor[1],3)
        cors_array[j,i,0], cors_array[j,i,1] = cor[0], cor[1]#3
#    
    #print round(pl.var(V_an[i]),2), round(pl.var(S[i,0]),2), round(pl.var(S[i,1]),2)
    #print round(pl.std(V_an[i]),2), round(pl.std(S[i,0]),2), round(pl.std(S[i,1]),2)
    
fig, ax = pl.subplots(2,4,figsize=(16,7))

proj = ccrs.PlateCarree()
ext = [-13,35,35,63]
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
ax1 = pl.subplot(241,projection=proj,extent=ext)
ax1.coastlines(linewidth=0.5,resolution='50m')
ax1.add_feature(borders_50m,linewidth=0.5,zorder=5)

#patches = []
for i in range(len(regions)):
    polygon = Polygon(regions[i],fc='none')#, ec='g')
    #patches.append(polygon)
    
    p = PatchCollection([polygon])
    p.set_edgecolor('k')
    p.set_facecolor('none')
    p.set_linewidth(2)
    ax1.add_collection(p)
GridLines(ax1,False,True,True,False)
pl.title('a. regions')

ax1.annotate('b',(-12.6,55.25),xycoords='data',color='k')
ax1.annotate('c',(1.89,45.25),xycoords='data',color='k')
ax1.annotate('d',(10.95,39.10),xycoords='data',color='k')
ax1.annotate('e',(10.57,51.52),xycoords='data',color='k')
ax1.annotate('f',(23.13,48.11),xycoords='data',color='k')
ax1.annotate('g',(24.60,44.71),xycoords='data',color='k')

for i in range(1,7):
    axx = pl.subplot(2,4,i+1)
    
    a = axx.plot(years,V_an[i-1],color='k',lw=1,label=name+' anomalies')
    b = axx.plot(years,S[i-1,0],color=c[0],lw=1.5,label='JFM '+caps[0]+' component')
    d = axx.plot(years,S[i-1,1],color=c[1],lw=2,label='JFM '+caps[1]+' component')
    e = axx.plot(years,S[i-1,2],color=c[2],lw=2,label='JFM '+caps[2][:3]+' component')
    #f = axx.plot(years,S[i-1,3],color=c[3],lw=2,label='GS '+caps[3]+' component')
    
    pl.grid(axis='y',ls='--',color='grey')
    pl.xlim(years[0].astype(int),years[-1].astype(int))
    pl.ylim(-60,40)
    pl.ylabel('days',fontsize=13,labelpad=-7)
    
    pl.title(titles[i-1])
    
#    if i == 1:
#        axx.legend(handles=[a[0],b[0]],
#                   labels=['gsr anomalies','GS NAO component',],
#                   loc=2)
#    if i == 2:
#        axx.legend(handles=[d[0],e[0]],
#                   labels=['GS EA component','GS SCA component'],
#                   loc=2)
    
    #if i < 6:
        #for k in range(2):
    if cors_array[0,i-1,1] <= 0.05:
        axx.annotate(str(round(cors_array[0,i-1,0],2)),(1951,-48),
                     color=c[0],size=12)
    if cors_array[1,i-1,1] <= 0.05:
        axx.annotate(str(round(cors_array[1,i-1,0],2)),(1951,-58),
                     color=c[1],size=12)
    if cors_array[2,i-1,1] <= 0.05:
        axx.annotate(str(round(cors_array[2,i-1,0],2)),(1961,-48),
                     color=c[2],size=12)
#    if cors_array[3,i-1,1] <= 0.05:
#        axx.annotate(str(round(cors_array[3,i-1,0],2)),(1998,-175),
#                     color=c[3],size=12)
    #if i == 6:
    #    axx.legend(loc=4)
#        #for k in range(2):
#        if cors_array[0,i-1,1] <= 0.05:
#            axx.annotate(str(round(cors_array[0,i-1,0],2)),(1951,-45),
#                         color=c[0],size=12)
#        if cors_array[1,i-1,1] <= 0.05:
#            axx.annotate(str(round(cors_array[1,i-1,0],2)),(1951,-50),
#                         color=c[1],size=12)
#        if cors_array[2,i-1,1] <= 0.05:
#            axx.annotate(str(round(cors_array[2,i-1,0],2)),(1951,-50),
#                         color=c[2],size=12)
    
    #if i == 4:
    #    m = [a[0],b[0],d[0]]
    #    labs = [name+' anomalies','GS '+caps[0]+' component','GS '+caps[1]+' component']
    #    axx.legend(m,labs,loc=2)
    #elif i == 5:
    #    m = [e[0],f[0]]
    #    labs = ['GS '+caps[2][:3]+' component','GS '+caps[3]+' component']
    #    axx.legend(m,labs,loc=2)
    
    if i == 6:
        axx.legend(loc=4)

ax8 = pl.subplot(248)

ax8.plot(years,tele_jfm[0],color=c[0],label='JFM NAO',lw=2)
ax8.plot(years,tele_jfm[1],color=c[1],label='JFM EA',lw=2)
ax8.plot(years,tele_jfm[2],color=c[2],label='JFM SCA',lw=2)
#ax8.plot(years,tele_ao[3],color=c[3],label='GS EAWR',lw=2)
ax8.grid(axis='y',ls='--',color='grey')
pl.xlim(years[0].astype(int),years[-1].astype(int))
pl.ylim(-2,2)
pl.ylabel('index',fontsize=13,labelpad=-5)
ax8.legend(loc=2,ncol=3,labelspacing=0.1,columnspacing=0.8)
pl.title('h. teleconnection indices')

pl.tight_layout()

pl.savefig(indecis+'figures/'+name+'_ts_panels_2x4_jfm_new.png',dpi=410)