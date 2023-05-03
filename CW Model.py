# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:11:53 2023

@author: Nisa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error 

#Function for creating simulation plots 
def plots(predict, exp_sheet):
    matplotlib.rcParams['font.family'] = 'serif'
    df= pd.read_excel (r'/Users/nisa/Library/Mobile Documents/com~apple~CloudDocs/Modeling Paper/Final Model/CW Model/Experimental Data.xlsx', sheet_name= exp_sheet)
    merged= pd.concat([predict, df], axis=1)
    simulated=[]
    experimental=[]
    influent=[]
    for index, row in merged.iterrows():
        if np.isfinite(row['Experimental']):
            simulated.append(row['Simulated_'])
            experimental.append(row['Experimental'])
            influent.append(row['Influent'])
    s= np.asarray(simulated)
    e= np.asarray(experimental)
    i= np.asarray(influent)
    re_eff_sim= np.mean((i-s)/i)
    re_eff_exp= np.mean((i-e)/i)
    pe= (re_eff_exp-re_eff_sim)/re_eff_exp
    
    rmse= np.around(mean_squared_error(e,s,squared=False),2)
    nrmse= np.around(rmse/(max(e)-min(e)),2)             
    
    x= np.array(np.arange(89,260, 1/24))
    y_exp= np.array(merged['Experimental'])
    y_sim= np.array(merged['Simulated_'])
    y_in=np.array(merged['Influent'])
    
    #df_control=pd.DataFrame ({'Day': x, 'Influent': y_in, 'Experimental': y_exp, 'Simulated': y_sim, 'NRMSE': nrmse})
    #df_control.to_excel('{}.xlsx'.format(os.path.basename(path).split('.')[0]))
    
    plt.ylim([0, max(i+100)])
    plt.xlim([80,265])
    plt.plot(x,y_sim, label='Simulated', markersize=3, color='darkslategrey')
    plt.plot(x,y_exp, '^', color='darkgreen', markersize=3, label='Experimental')
    plt.plot(x,y_in, 'x', color='maroon',markersize=3, label='Influent')
    plt.xlabel('Day')
    plt.ylabel('Conc(mg/L)')
    plt.text(150, max(i)+100, 'NRMSE={}'.format(nrmse), fontsize = 7)
    plt.text(150, max(i)+130, exp_sheet, fontsize = 10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          fancybox=False, shadow=False, ncol=3, fontsize=5, frameon=False)
    plt.box(on=False)
    plt.rcParams["figure.figsize"] = (3,2)
    plt.rcParams.update({'font.size': 7})
    #plt.savefig('{}'.format(os.path.basename(path).split('.')[0]), bbox_inches='tight', dpi=300)
    plt.show()

#Function for creating plots showing adsorbent efficiency
def adsorbent_effects (control, amended, a_b):
    x= np.array(np.arange(89,259, 1/24))
    y_control= np.array(control)
    y_amended= np.array(amended)
    
    #plt.ylim(ylim)
    plt.xlim([88,260])
    plt.plot(x,y_control, label='Unamended',markersize=4, color='darkslategrey')
    plt.plot(x,y_amended, label='Adsorbent-amended',markersize=4, color='lightsteelblue')
    
    plt.xlabel('Day')
    plt.ylabel('Conc(mg/L)')
    
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.17),
          fancybox=False, shadow=False, ncol=3, fontsize=8, frameon=False)
    plt.box(on=False)
    plt.rcParams["figure.figsize"] = (6,3)
    plt.rcParams.update({'font.size': 8})
    #plt.xticks(np.arange(min(x), max(x), 30))
    plt.set_cmap('hot')
    #plt.axvline(259, color='grey', linestyle='-')
    
    
    plt.title(a_b, fontsize='10', y=1.07)
    plt.show()

#The CW Model
def CW_Model ():
    #Reads data from the input data sheet (only excel formats)
    
    path= '/Users/nisa/Library/Mobile Documents/com~apple~CloudDocs/Modeling Paper/Final Model/CW Model/Hourly_Input.xlsx'
    # input_data=pd.read_excel(input('Insert Path of the Input Data Sheet (Format: .xls/.xlsx):' or path))
    input_data=pd.read_excel(path)
    
    #create arrays from input data
    day= np.asarray(list(input_data['Day']))
    time = np.asarray(list(input_data['Time']))
    P = np.asarray(list(input_data['Precip(m)']))
    temp = np.asarray(list(input_data['Temp']))
    COD_i = np.asarray(list(input_data['COD'])) 
    OrgN_i = np.asarray(list(input_data['Organic N']))       
    NH4_i = np.asarray(list(input_data['Ammonium']))  
    NO3_i = np.asarray(list(input_data['Nitrate']))   
    
    ########## Water Balance Functions ########
    #CW dimensions
    VF_area = input('VF-CW Area (m^2):')
    if VF_area == '':
        VF_area = 0.4195
    else:
        VF_area= float(VF_area)
    HF_area = input('HF-CW Area (m^2):' or '1.1')
    if HF_area == '':
        HF_area = 1.1
    else:
        HF_area= float(HF_area)
    

    #create inflow set the same size as the input data
    inflow = input('Inflow Rate (m^3/hr):' or '0.001')
    if inflow == '':
        inflow= 0.001
    else:
        inflow= float(inflow)

    #VF inflow equals inflow rate
    VF_Qi = inflow
    #VF outflow equals VF inflow 
    VF_Qo = VF_Qi
    #HF inflow equals VF outflow
    HF_Qi = VF_Qo
    #HF outflow equals HF inflow
    HF_Qo= HF_Qi
    
    #calculate evapotranspiration (m/hour)
    dl = 1    #hour
    #heat index
    I = 167.1  #C
    a = 6.75*(10**-7)*(I**3)-7.71*(10**-5)*(I**2)+1.792*(10**-2)*I+0.49239 
    #Thortwaite's equation
    ET = 16*dl/12*((10*temp/I)**a)/(30*100)/10
    
    ####Water Volume###
    VF_volume = input('VF-CW Volume (m^3):' or '0.015')
    if VF_volume == '':
        VF_volume = 0.015
    else:
        VF_volume= float(VF_volume)

    #water balance equation
    dVdt = VF_Qi-VF_Qo+(P*VF_area)-(ET*VF_area)
    #store result in array
    dVdt = np.asarray(dVdt)
    #volume equals intial plus change each day 
    VF_volume += dVdt  
    VF_volume= VF_volume*1000   
    
    HF_volume = input('HF-CW Volume (m^3):' or '0.14')
    if HF_volume == '':
        HF_volume = 0.14
    else:
        HF_volume= float(HF_volume)
    #water balance equation
    dVdt = HF_Qi-HF_Qo+(P*HF_area)-(ET*HF_area)
    #store result in array
    dVdt = np.asarray(dVdt)
    #volume equals intial plus change each day 
    HF_volume += dVdt  
    HF_volume= HF_volume*1000 

    #######Oxygen mass balance########
    DO_i = input ('Initial Dissolved Oxygen Concentration (mg/L):' or '1.5')
    if DO_i == '':
        DO_i = 1.5
    else:
        DO_i= float(DO_i)
    T = temp        #avg. temp of water (C)
    VF_kR = .5*1.08**(temp-20)/24  # (/hr) #(0-1)
    VF_kr=np.asarray(VF_kR)
    HF_kR = 0.001*1.08**(temp-20)/24  # (/hr) #(0-1)

    #calculate DO saturation
    DO_s = 14.652-0.41022*T+0.007991*T**2-0.00007777*T**3 #g/m3

    #calculate mass flux 
    VF_JO2 = VF_kR*(DO_s-DO_i)


    #VF Monod Parameters for heterotrophs
    VF_HT_Y = 1.23*1.08**(temp-20)/24
    VF_HT_DO_Ks = 1.3*1.08**(temp-20)/24
    VF_HT_TOC_Ks = 60*1.08**(temp-20)/24
    VF_HT_mu_max = 4*1.08**(temp-20)/24
    TOC = COD_i
    #VF Monod Parameters for autotrophs
    VF_NS_Y = 0.084 *1.08**(temp-20)/24
    VF_NS_DO_Ks = 0.7*1.08**(temp-20)/24
    VF_NS_NH4_Ks = 1.5*1.08**(temp-20)/24
    VF_NS_mu_max = .0005 *1.08**(temp-20)/24

    #VF Monod equations
    VF_HT_growth = VF_HT_mu_max*(TOC/(TOC+VF_HT_TOC_Ks))*(VF_HT_DO_Ks/(DO_i+VF_HT_DO_Ks))
    VF_HT_res = VF_HT_growth/VF_HT_Y
    VF_NS_growth = VF_NS_mu_max*(NH4_i/(NH4_i+VF_NS_NH4_Ks))*(DO_i/(DO_i+VF_NS_DO_Ks))
    VF_NS_res = VF_NS_growth/VF_NS_Y

    #HF Monod Parametersfor heterotrophs
    HF_HT_Y = 0.8*1.25**(temp-20)/24
    HF_HT_DO_Ks = 2*1.25**(temp-20)/24
    HF_HT_TOC_Ks = 40*1.08**(temp-20)/24
    HF_HT_mu_max = 12*1.08**(temp-20)/24
    TOC = COD_i
    #HF Monod Parameters for autotrophs
    HF_NS_Y = 0.085/24
    HF_NS_DO_Ks = 124/24
    HF_NS_NH4_Ks = 1/24
    HF_NS_mu_max = .01/24

    #HF Monod equations
    HF_HT_growth = HF_HT_mu_max*(TOC/(TOC+HF_HT_TOC_Ks))*(HF_HT_DO_Ks/(DO_i+HF_HT_DO_Ks))
    HF_HT_res = HF_HT_growth/HF_HT_Y
    HF_NS_growth = HF_NS_mu_max*(NH4_i/(NH4_i+HF_NS_NH4_Ks))*(DO_i/(DO_i+HF_NS_DO_Ks))
    HF_NS_res = HF_NS_growth/HF_NS_Y

    #DO mass balances
    VF_DO = DO_i + VF_JO2 - VF_HT_res - VF_NS_res
    HF_JO2 = HF_kR*(DO_s-VF_DO)
    HF_DO = VF_DO + HF_JO2 - HF_HT_res - HF_NS_res
    
    #Inflow and number of tanks
    Q = inflow*1000 #L
    n= input ("Value of 'n' (number of tanks in series)?" or '3')
    if n== '':
        n = 3
    else:
        n= float(n)
    
    #Decides to produce results for Unamended or Amended CWs
    ads= input('Do you want to run the model with adsorbent amendment(Y/N)?:')
    
    if ads == 'N':
        ####Pollutant Mass Balance####
        ##COD Balance##
        
        #Stover-Kincannon Parameters
        #VF
        VF_k=0.5
        VF_kd=VF_k*1.01**(temp-20)
        VF_mu_max= 990*1.01**(temp-20)/24*(VF_DO/(VF_kd+VF_DO)) #mg/L-hr
        VF_kB= 868   #mg/L-hr
        
        VF_COD = COD_i
        i=1
        while i<= n:
            
            VF_COD= VF_COD-VF_mu_max*VF_COD/(VF_kB+Q*VF_COD/VF_volume/n)
            i+=1
        df_VF_COD= pd.DataFrame(VF_COD, columns=['Simulated_'])
        plots(df_VF_COD, 'VF_COD')
        #HF
        HF_k=0.1
        HF_kd=HF_k*1.01**(temp-20)
        HF_mu_max= 1609*1.01**(temp-20)/24*(HF_DO/(HF_kd+HF_DO)) #mg/L-hr 
        HF_kB= 1177     #mg/L-hr 

        HF_COD = VF_COD
        i=1
        while i<= n: 
            HF_COD= HF_COD-HF_mu_max*HF_COD/(HF_kB+Q*HF_COD/HF_volume/n)
            i+=1
        df_HF_COD= pd.DataFrame(HF_COD, columns=['Simulated_'])
        plots(df_HF_COD, 'HF_COD')
        ##Nitrogen Balance##
        #Background concentrations (mg/L)
        OrgN_0 = 5
                  
        ######Rate constants (per hour)
        VF_k=0.1
        VF_kd=VF_k*1.01**(temp-20)
        HF_k=0.01
        HF_kd=HF_k*1.01**(temp-20)
        #plant decomposition
        VF_kpd = 0.005*1.04**(temp-20)   
        HF_kpd = 0.008*1.04**(temp-20)

        #mineralization
        km_a = 0.6*1.08**(temp-20) #aerobic   
        km_an = 0.12*1.08**(temp-20) #anaerobic


        #nitrification
        kn_a =  0.07*1.08**(temp-20)*(VF_DO/(VF_kd+VF_DO))  #anerobic 
        kn_an =  0.006*1.02**(temp-20)*(HF_DO/(HF_kd+HF_DO))  #anerobic
        

        #plant uptake of ammonia 
        VF_kpu_NH4 = 0.002*1.08**(temp-20)
        HF_kpu_NH4 = 0.003*1.08**(temp-20)


        #Organic N
        #VF
        VF_OrgN = OrgN_i
        i=1
        while i<= n: 
            VF_OrgN =(Q*VF_OrgN )/(Q-(VF_kpd*VF_volume/n)+(km_a*VF_volume/n))+OrgN_0
            i+=1
            
        #HF
        HF_OrgN = VF_OrgN
        i=1
        while i<= n: 
            HF_OrgN =(Q*HF_OrgN )/(Q-(HF_kpd*HF_volume/n)+(km_an*HF_volume/n))+OrgN_0
            i+=1

        ##Ammonia
        #VF
        VF_NH4=NH4_i
        i=1
        while i<=n:
            VF_NH4 =(Q*VF_NH4+(km_a*VF_OrgN*VF_volume/n))/(Q+kn_a*VF_volume/n+(VF_kpu_NH4*VF_volume/n))
            i+=1
        df_VF_NH4= pd.DataFrame(VF_NH4, columns=['Simulated_'])
        plots(df_VF_NH4, 'VF_Ammonia')
        #HF
        HF_NH4=VF_NH4
        i=1
        while i<=n:
            HF_NH4 =(Q*HF_NH4+(km_an*HF_OrgN*HF_volume/n))/(Q+kn_an*HF_volume/n+(HF_kpu_NH4*HF_volume/n))
            i+=1
        df_HF_NH4= pd.DataFrame(HF_NH4, columns=['Simulated_'])
        plots(df_HF_NH4, 'HF_Ammonia')
        
    else:
        ##Adsorbent-amended model##
        m_zeolite = input('Amount of Zeolite Added (g):' or '23000') #23000g
        if m_zeolite== '':
            m_zeolite = 23000
        else: 
            m_zeolite=float(m_zeolite)
    
        rho_zeolite = 877000 #g/m3
        r_zeolite = .00025 #m
        a_zeolite = (3/r_zeolite)*(m_zeolite/rho_zeolite/VF_volume) #m2/m3
        
        m_biochar = input('Amount of Biochar Added (g):' or '2600') #2600g
        if m_biochar== '':
            m_biochar = 2600
        else: 
            m_biochar=float(m_biochar)
        rho_biochar = 1340000 #g/m3
        r_biochar = .0015 #m
        a_biochar = (3/r_biochar)*(m_biochar/rho_biochar/HF_volume) #m2/m3
        
        
        
        ##COD Balance##
        
        #Stover-Kincannon Parameters
        #VF
        VF_k=0.1
        VF_kd=VF_k*1.01**(temp-20)
        A_VF_mu_max= 990*1.01**(temp-20)/24*(VF_DO/(VF_kd+VF_DO)) #mg/L-hr
        A_VF_kB= 868   #mg/L-hr
        VF_qcod= 0.29 #mg COD/g zeolite   (0.1-1)
        Ds_zeolite_COD = (4.77*10**-12)*864000/24 #m2/hr   
        VF_Jcod = -rho_zeolite*VF_qcod*Ds_zeolite_COD
        
        A_VF_COD = COD_i
        i=1
        while i<= n: 
            A_VF_COD= A_VF_COD-A_VF_mu_max*A_VF_COD/(A_VF_kB+Q*A_VF_COD/VF_volume/n)+(VF_Jcod*a_zeolite*VF_volume/n)/Q
            i+=1
        df_A_VF_COD= pd.DataFrame(A_VF_COD, columns=['Simulated_'])
        plots(df_A_VF_COD, 'A_VF_COD')
        #HF
        HF_k=0.01
        HF_kd=HF_k*1.01**(temp-20)
        A_HF_mu_max= 1609*1.01**(temp-20)/24*(HF_DO/(HF_kd+HF_DO)) #mg/L-hr 
        A_HF_kB= 1177     #mg/L-hr 
        HF_qcod= 12 #mg COD/g biochar  (12-35)
        Ds_biochar_COD = (3.37*10**-11)*864000/24 #m2/hr (4-8)
        HF_Jcod = rho_biochar*HF_qcod*Ds_biochar_COD
        
        A_HF_COD = A_VF_COD
        i=1
        while i<= n: 
            A_HF_COD= A_HF_COD-A_HF_mu_max*A_HF_COD/(A_HF_kB+Q*A_HF_COD/HF_volume/n)-(HF_Jcod*a_biochar*HF_volume/n)/Q
            i+=1
        df_A_HF_COD= pd.DataFrame(A_HF_COD, columns=['Simulated_'])
        plots(df_A_HF_COD, 'A_HF_COD')
        ##Nitrogen Balance##
        #Background concentrations (mg/L)
        OrgN_0 = 5
                  
        ######Rate constants (per hour)
        VF_k=0.1
        VF_kd=VF_k*1.01**(temp-20)
        HF_k=0.01
        HF_kd=HF_k*1.01**(temp-20)
        #plant decomposition
        VF_kpd = 0.005*1.04**(temp-20)   
        HF_kpd = 0.008*1.04**(temp-20)

        #mineralization
        km_a = 0.6*1.08**(temp-20) #aerobic   
        km_an = 0.12*1.08**(temp-20) #anaerobic   


        #nitrification
        kn_a =  0.09*1.08**(temp-20)*(VF_DO/(VF_kd+VF_DO)) #aerobic    
        kn_an =  0.009*1.08**(temp-20)*(HF_DO/(HF_kd+HF_DO)) #anerobic  
        

        #plant uptake of ammonia 
        VF_kpu_NH4 = 0.002*1.08**(temp-20)
        HF_kpu_NH4 = 0.025*1.08**(temp-20)

        VF_qNH4= 2 #mg/g 
        Ds_zeolite_ammonia = (2.99*10**-12)*864000/24 #m2/day 
        VF_JNH4 = -rho_zeolite*VF_qNH4*Ds_zeolite_ammonia
        
        HF_qNH4= 0.05 #mg/g  
        Ds_biochar_ammonia = (5.6*10**-11)*864000/24 #m2/hr   
        HF_JNH4 = -rho_biochar*HF_qNH4*Ds_biochar_ammonia
        #Organic N
        #VF
        VF_OrgN = OrgN_i
        i=1
        while i<= n: 
            VF_OrgN =(Q*VF_OrgN)/(Q-(VF_kpd*VF_volume/n)+(km_a*VF_volume/n))+OrgN_0
            i+=1
        #HF
        HF_OrgN = VF_OrgN
        i=1
        while i<= n: 
            HF_OrgN =(Q*HF_OrgN )/(Q-(HF_kpd*HF_volume/n)+(km_an*HF_volume/n))+OrgN_0
            i+=1
        ##Ammonia
        #VF
        A_VF_NH4=NH4_i
        i=1
        while i<=n:
            A_VF_NH4=(Q*A_VF_NH4+(km_a*VF_OrgN*VF_volume/n)+(VF_JNH4*a_zeolite*VF_volume/n))/(Q+(kn_a*VF_volume/n)+(VF_kpu_NH4*VF_volume/n))
            i+=1
        df_A_VF_NH4= pd.DataFrame(A_VF_NH4, columns=['Simulated_'])
        plots(df_A_VF_NH4, 'A_VF_Ammonia')
        #HF
        A_HF_NH4=A_VF_NH4
        i=1
        while i<=n:
            A_HF_NH4 =(Q*A_HF_NH4+(km_an*HF_OrgN*HF_volume/n)+(HF_JNH4*a_biochar*HF_volume/n))/(Q+(kn_an*HF_volume/n)+(HF_kpu_NH4*HF_volume/n))
            i+=1
        df_A_HF_NH4= pd.DataFrame(A_HF_NH4, columns=['Simulated_'])
        plots(df_A_HF_NH4, 'A_HF_Ammonia')
            
CW_Model()