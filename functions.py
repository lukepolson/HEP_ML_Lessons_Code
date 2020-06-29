import uproot # for reading .root files
import pandas as pd # to store data as dataframe
import time # to measure time to analyse
import math # for mathematical functions such as square root
import numpy as np # # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import AutoMinorLocator # for minor ticks

import infofile # local file containing info on cross-sections, sums of weights, dataset IDs


lumi = 10 # fb-1 # data_A,data_B,data_C,data_D
fraction = 1 # 0.9 # reduce this is you want the code to run quicker
tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/" # web address

samples = {
    'ZZ' : {
        'list' : ['llll'],
        'color' : "#ff0000" # red
    },

    r'$H \rightarrow ZZ \rightarrow \ell\ell\ell\ell$' : { # H -> ZZ -> llll
        'list' : ['ggH125_ZZ4lep','VBFH125_ZZ4lep','WH125_ZZ4lep','ZH125_ZZ4lep'],
        'color' : "#00cdff" # light blue
    }
}



def get_data_from_files():

    data = {}
    for s in samples:
        print('Processing '+s+' samples')
        frames = []
        for val in samples[s]['list']:
            prefix = "MC/mc_"
            if s == 'data':
                prefix = "Data/"
            else: prefix += str(infofile.infos[val]["DSID"])+"."
            fileString = tuple_path+prefix+val+".4lep.root"
            if fileString != "":
                temp = read_file(fileString,val)
                frames.append(temp)
            else:
                print("Error: "+val+" not found!")
        data[s] = pd.concat(frames)
    
    return data


def calc_weight(mcWeight,scaleFactor_PILEUP,scaleFactor_ELE,
                scaleFactor_MUON, scaleFactor_LepTRIGGER):
    return mcWeight*scaleFactor_PILEUP*scaleFactor_ELE*scaleFactor_MUON*scaleFactor_LepTRIGGER


def get_xsec_weight(totalWeight,sample):
    info = infofile.infos[sample]
    weight = (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"]) #*1000 to go from fb-1 to pb-1
    weight *= totalWeight
    return weight

def calc_mllll(lep_pt,lep_eta,lep_phi,lep_E):
    # first lepton is [0], 2nd lepton is [1] etc
    px_0 = lep_pt[0]*math.cos(lep_phi[0]) # x-component of lep[0] momentum
    py_0 = lep_pt[0]*math.sin(lep_phi[0]) # y-component of lep[0] momentum
    pz_0 = lep_pt[0]*math.sinh(lep_eta[0]) # z-component of lep[0] momentum
    px_1 = lep_pt[1]*math.cos(lep_phi[1]) # x-component of lep[1] momentum
    py_1 = lep_pt[1]*math.sin(lep_phi[1]) # y-component of lep[1] momentum
    pz_1 = lep_pt[1]*math.sinh(lep_eta[1]) # z-component of lep[1] momentum
    px_2 = lep_pt[2]*math.cos(lep_phi[2]) # x-component of lep[2] momentum
    py_2 = lep_pt[2]*math.sin(lep_phi[2]) # y-component of lep[2] momentum
    pz_2 = lep_pt[2]*math.sinh(lep_eta[2]) # z-component of lep[3] momentum
    px_3 = lep_pt[3]*math.cos(lep_phi[3]) # x-component of lep[3] momentum
    py_3 = lep_pt[3]*math.sin(lep_phi[3]) # y-component of lep[3] momentum
    pz_3 = lep_pt[3]*math.sinh(lep_eta[3]) # z-component of lep[3] momentum
    sumpx = px_0 + px_1 + px_2 + px_3 # x-component of 4-lepton momentum
    sumpy = py_0 + py_1 + py_2 + py_3 # y-component of 4-lepton momentum
    sumpz = pz_0 + pz_1 + pz_2 + pz_3 # z-component of 4-lepton momentum
    sumE = lep_E[0] + lep_E[1] + lep_E[2] + lep_E[3] # energy of 4-lepton system
    return math.sqrt(sumE**2 - sumpx**2 - sumpy**2 - sumpz**2)/1000 #/1000 to go from MeV to GeV


def lep_pt_0(lep_pt):
    return lep_pt[0]/1000 # /1000 to go from MeV to GeV

def lep_pt_1(lep_pt):
    return lep_pt[1]/1000 # /1000 to go from MeV to GeV

def lep_pt_2(lep_pt):
    return lep_pt[2]/1000 # /1000 to go from MeV to GeV

def lep_pt_3(lep_pt):
    return lep_pt[3]/1000 # /1000 to go from MeV to GeV


# cut on number of leptons
# paper: "selecting two pairs of isolated leptons"
def cut_lep_n(lep_n):
# exclamation mark (!) means "not"
# so != means "not equal to"
# throw away when number of leptons is not equal to 4 
    return lep_n != 4

# cut on lepton charge
# paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"
def cut_lep_charge(lep_charge):
# throw away when sum of lepton charges is not equal to 0
# first lepton is [0], 2nd lepton is [1] etc
    return lep_charge[0] + lep_charge[1] + lep_charge[2] + lep_charge[3] != 0

# cut on lepton type
# paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"
def cut_lep_type(lep_type):
# for an electron lep_type is 11
# for a muon lep_type is 13
# throw away when none of eeee, mumumumu, eemumu
    sum_lep_type = lep_type[0] + lep_type[1] + lep_type[2] + lep_type[3]
    return (sum_lep_type != 44) and (sum_lep_type != 48) and (sum_lep_type != 52)

#cut on transverse momentum of the leptons
# paper: " the second (third) lepton in pT order must satisfy pT > 15 GeV (pT > 10 GeV)"
def cut_lep_pt_012(lep_pt):
# throw away any events where lep_pt[1] < 15000 MeV
# throw away any events where lep_pt[2] < 10000 MeV
    return lep_pt[1]<15000 or lep_pt[2]<10000

def read_file(path,sample):
    start = time.time() # start the clock
    print("\tProcessing: "+sample) # print which sample is being processed
    data_all = pd.DataFrame() # define empty pandas DataFrame to hold all data for this sample
    tree = uproot.open(path)["mini"] # open the tree called mini
    numevents = uproot.numentries(path, "mini") # number of events
    for data in tree.iterate(["lep_n","lep_pt","lep_eta","lep_phi","lep_E","lep_charge","lep_type","lep_ptcone30",
                            "lep_etcone20", # add more variables here if you make cuts on them 
                            "mcWeight","scaleFactor_PILEUP","scaleFactor_ELE","scaleFactor_MUON",
                            "scaleFactor_LepTRIGGER"], # variables to calculate Monte Carlo weight
                           entrysteps=2500000, # number of events in a batch to process
                           outputtype=pd.DataFrame, # choose output type as pandas DataFrame
                           entrystop=numevents*fraction): # process up to numevents*fraction

        nIn = len(data.index) # number of events in this batch
        print('\t initial number of events:\t\t\t',nIn)

        if 'data' not in sample: # only do this for Monte Carlo simulation files
            # multiply all Monte Carlo weights and scale factors together to give total weight
            data['totalWeight'] = np.vectorize(calc_weight)(data.mcWeight,data.scaleFactor_PILEUP,
                                                            data.scaleFactor_ELE,data.scaleFactor_MUON,
                                                            data.scaleFactor_LepTRIGGER)
            # incorporate the cross-section weight into the total weight
            data['totalWeight'] = np.vectorize(get_xsec_weight)(data.totalWeight,sample)
            
        # drop the columns we don't need anymore from the dataframe
        data.drop(["mcWeight","scaleFactor_PILEUP","scaleFactor_ELE","scaleFactor_MUON","scaleFactor_LepTRIGGER"], 
                  axis=1, inplace=True)

        # cut on number of leptons using the function cut_lep_n defined above
        fail = data[ np.vectorize(cut_lep_n)(data.lep_n)].index
        data.drop(fail, inplace=True)
        print('\t after requiring 4 leptons:\t\t\t',len(data.index))

        # cut on lepton charge using the function cut_lep_charge defined above
        fail = data[ np.vectorize(cut_lep_charge)(data.lep_charge) ].index
        data.drop(fail, inplace=True)
        print('\t after requiring zero net charge:\t\t',len(data.index))

        # cut on lepton type using the function cut_lep_type defined above
        fail = data[ np.vectorize(cut_lep_type)(data.lep_type) ].index
        data.drop(fail, inplace=True)
        print('\t after requiring lepton pairs of same type:\t',len(data.index))

        #cut on the transverse momentum of the leptons using the function cut_lep_pt_012 defined above
        #fail =data[ np.vectorize(cut_lep_pt_012)(data.lep_pt)].index
        #data.drop(fail,inplace=True)
        #print('\t after requirements on lepton pt:\t\t',len(data.index))

        # calculation of 4-lepton invariant mass using the function calc_mllll defined above
        data['mllll'] = np.vectorize(calc_mllll)(data.lep_pt,data.lep_eta,data.lep_phi,data.lep_E)
        
        # return the individual lepton transverse momenta in GeV
        #data['lep_pt_0'] = np.vectorize(lep_pt_0)(data.lep_pt)
        data['lep_pt_1'] = np.vectorize(lep_pt_1)(data.lep_pt)
        data['lep_pt_2'] = np.vectorize(lep_pt_2)(data.lep_pt)
        #data['lep_pt_3'] = np.vectorize(lep_pt_3)(data.lep_pt)

        # dataframe contents can be printed at any stage like this
        #print(data)

        # dataframe column can be printed at any stage like this
        #print(data['lep_pt'])

        # multiple dataframe columns can be printed at any stage like this
        #print(data[['lep_pt','lep_eta']])

        nOut = len(data.index) # number of events passing cuts in this batch
        data_all = data_all.append(data) # append dataframe from this batch to the dataframe for the whole sample
        elapsed = time.time() - start # time taken to process
        print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in "+str(round(elapsed,1))+"s") # events before and after
    
    return data_all # return dataframe containing events passing all cuts