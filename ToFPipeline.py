import sys
import yaml
import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import find_peaks
from functools import partial

from pathlib import Path
import string, os, re

import time
from tqdm.notebook import tqdm

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


import contextlib
from IPython.display import clear_output
from timeit import default_timer as timer


class GlobalConfig:
    _config = {}

    @classmethod
    def load(cls, path):
        """Load a YAML config file into memory."""
        with open(path, "r") as f:
            cls._config = yaml.safe_load(f) or {}

    @classmethod
    def get_for_class(cls, cls_or_name):
        """Return the config dictionary for a given class."""
        name = cls_or_name if isinstance(cls_or_name, str) else cls_or_name.__name__
        return cls._config.get(name, {}).copy()

class Configurable:
    CONFIG_KEY = None
    def __init__(self, config=None):
        self.className = self.__class__.__name__
        class_key = self.CONFIG_KEY or self.__class__.__name__
        base = GlobalConfig.get_for_class(class_key)
        if config:
            base.update(config)
        self.config = base

    def getConfig(self, key, default=None):
        return self.config.get(key, default)

class Loader(Configurable):
    def load(self):
        raise NotImplemtedError
        
_RUN_CACHE = {}
class FLASHLoader(Loader):
    def __init__(self,proposal, runNo,config=None):
        super().__init__(config)
        if self.className == "FLASHLoader":
            #print("Using ",self.className)
            import dask.array as da
            from fab.magic import config, beamtime, ballchamber, opis, timing
            self.da = da
            self.ballchamber = ballchamber
            self.opis = opis
        self.proposal = proposal
        self.runNo = runNo
        self._run = None
        self.key = None
        self.data = None
        fullTheta = np.array(self.config.get("angles", np.linspace(0,2*np.pi,16,endpoint=False)))
        detectors = self.config.get("ToF",range(16))
        theta = fullTheta[detectors]
        self.angles = pd.DataFrame(detectors,columns=["detector"])
        self.angles["Angles"] = theta
        self.xmg = None
        self.photonEnergy = None
    """
    @property
    def run(self):
        global _RUN_CACHE
        cacheKey = (self.proposal, self.runNo)
        if self._run is not None:
            return self._run
        if cacheKey in _RUN_CACHE:
            self._run = _RUN_CACHE[cacheKey]
            return self._run

        print("opening run: ",self.runNo)
        self._run = ballchamber.load(daq_run=[self.runNo])
        _RUN_CACHE[cacheKey] = self._run
        return self._run
    """
    
    def load(self, key="all", trainStart=None, trainStop=None, trainStep = None, pulseStart=None, pulseStop=None, pulseStep=None,roi=[None,None]):
        self.key = key or self.config.get("key", "all")
        trainStart = trainStart or self.config.get("trainStart", None)
        trainStop = trainStop or self.config.get("trainStop", None)
        trainStep = trainStep or self.config.get("trainStep", None)
        pulseStart = pulseStart or self.config.get("pulseStart", None)
        pulseStop = pulseStop or self.config.get("pulseStop", None)
        pulseStep = pulseStep or self.config.get("pulseStep", None)
        
        traces = self.ballchamber.load(daq_run=self.runNo)
        
        if any(v is not None for v in [trainStart, trainStop, trainStep]):
            traces = traces.isel(train_id=slice(trainStart, trainStop, trainStep))
        if any(v is not None for v in [pulseStart, pulseStop, pulseStep]):
            traces = traces.isel(shot_id=slice(pulseStart, pulseStop, pulseStep))
        
        # ds is your original dataset
        
        # 1) Stack train_id + shot_id into a single pulse MultiIndex
        ds2 = traces.stack(pulse=("train_id", "shot_id"))
        
        # Optional: rename MultiIndex levels
        ds2 = ds2.rename({"train_id": "trainId", "shot_id": "pulseId"})
        ds2 = ds2.set_index(pulse=("trainId", "pulseId"))
        
        # 2) Rename tof_trace → sample
        ds2 = ds2.rename({"tof_trace": "sample"})
        
        # 3) Combine adcXX variables into a new detector dimension
        detector_names = [f"adc{str(i).zfill(2)}" for i in range(16)]
        
        ds_stacked = xr.concat([ds2[v] for v in detector_names], dim="detector")
        ds_stacked = ds_stacked.assign_coords(detector=np.arange(16))
        
        # 4) Final reorder
        self.run = ds_stacked.transpose("detector", "pulse", "sample")
        self.data = self.run.sel(sample=slice(roi[0],roi[1]))

        if self.key in ("all", "Photon Energy"):
            self.photonEnergy = self.opis.load(daq_run=self.runNo).to_dataframe().reset_index()
            self.photonEnergy.columns = ["trainId","daq_run","Photon Energy"]
            fullTrainIds = pd.RangeIndex(self.photonEnergy["trainId"].min(), self.photonEnergy["trainId"].max()+1)
            self.photonEnergy = self.photonEnergy.set_index("trainId").reindex(fullTrainIds)
            self.photonEnergy["Photon Energy"] = self.photonEnergy["Photon Energy"].interpolate()
            self.photonEnergy = (self.photonEnergy.reset_index().rename(columns={"index": "trainId"}))

        return self

    def defaultPreprocessing(self,ToF=None,baselineRegion=None,trainStart=None,trainStop=None):
        ToF = ToF or self.config.get("ToF", [0])
        self.data = self.data.sel(detector=ToF)
        baselineRegion = baselineRegion or self.config.get("baselineRegion",[-200,None])
        self.data = -(self.data - self.data.isel(sample=slice(baselineRegion[0],baselineRegion[1])).mean(dim="sample"))
        grouped = self.data.groupby("daq_run")
        sliced = xr.concat(
            [g.isel(pulse=slice(trainStart, trainStop)) for _, g in grouped],
            dim="pulse")
        self.data = sliced
        return self
        

class EuXFelLoader(Loader):
    def __init__(self,proposal, runNo,config=None):
        super().__init__(config)
        print("init...")
        if self.className == "EuXFelLoader":
            print("Using ",self.className)
            import extra_data as xd
            from euxfel_bunch_pattern import indices_at_sase
            from extra.components import AdqRawChannel
            self.xd = xd
            self.indices_at_sase = indices_at_sase
            self.AdqRawChannel = AdqRawChannel
        self.proposal = proposal
        self.runNo = runNo
        self._run = None
        self.key = None
        self.data = None
        self.xmg = None
        self.photonEnergy = None
        print("Done!")

    @property
    def run(self):
        global _RUN_CACHE
        cacheKey = (self.proposal, self.runNo)
        if self._run is not None:
            return self._run
        if cacheKey in _RUN_CACHE:
            self._run = _RUN_CACHE[cacheKey]
            return self._run

        print("opening run: ",self.runNo)
        self._run = self.xd.open_run(proposal=self.proposal, run=self.runNo)
        _RUN_CACHE[cacheKey] = self._run
        return self._run

    def filterByIntensity(self,intensityThreshold=None,):
        intensityThreshold = (
            intensityThreshold
            or self.config.get("intensityThreshold", None))
        if intensityThreshold is not None:
            mask = self.xgm > intensityThreshold
            self.data = self.data.where(mask, drop=True)                 
        return self

    
    def load(self, key="all", trainStart=None, trainStop=None, trainStep=None):
        self.key = key or self.config.get("key", "all")
        trainStart = trainStart or self.config.get("trainStart", 0)
        trainStop = trainStop or self.config.get("trainStop", -1)
        trainStep = trainStep or self.config.get("trainStep", 1)
        self.data = self.run[trainStart:trainStop:trainStep]
        if self.key in ("all", "XGM"):
            self.xgm = self.run[trainStart:trainStop:trainStep].select('SA3_XTD10_XGM/XGM/DOOCS:output')['SA3_XTD10_XGM/XGM/DOOCS:output', 'data.intensitySa3TD'].xarray()
        if self.key  in ("all", "Photon Energy"):
            self.photonEnergy = self.data.select("SA3_XTD10_UND/DOOCS/PHOTON_ENERGY_COLOR2","calibratedActualPosition").get_dataframe().reset_index()
            self.photonEnergy.columns = ["trainId","Photon Energy"]
        else:
            raise KeyError(f"{self.key} not found in loader")
        return self

    def detectors(self, proposal, run):
        """
        Provides detector information when given a run number.
        
        Parameters
        ----------
        run : unsigned int
            Run number within proposal
        
        Returns
        -------
        detinfo : structured ndarray
            Keys: name (detector name), 
                  digitizer,
                  channel,
                  angle (degrees, looking along the beam, 0 is right, increasing counter-clockwise),
                  sample_rate (GS/s)
        """
        confs = os.listdir(Path(__file__).parent / 'configurations' / str(proposal))
        groups = [re.search('(\d*)-(\d*).txt', f) for f in confs]
        for idx, gr in enumerate(groups):
            if gr is not None:
                a, b = gr.group(1, 2)
                if (run >= int(a)) & (run <= int(b)):
                    #print(f"Using configuration file: {Path(__file__).parent / 'configurations' / str(proposal) / confs[idx]}")
                    return  np.genfromtxt(Path(__file__).parent / 'configurations' / str(proposal) / confs[idx],
                                          names=True, dtype=('|U5', '|U4', '|U3', '<f8', '?', '<i4'))
        raise Exception(f'Did not find detector configuration file for run {run}')

    def offsets(self, proposal, run):
        from_, to, train_offset, pulse_length = np.genfromtxt(Path(__file__).parent / 'configurations' / str(proposal) / 'offsets.cfg', unpack=True)
        idx = np.argmax((run >= from_) & (run <= to))
        return train_offset[idx], int(pulse_length[idx])

    def defaultPreprocessing(self,ToF=None,
        pattern_noise_region=None,
        pattern_noise_sym=None):

        ToF = (
            ToF
            if ToF is not None
            else self.config.get("ToF", [0])
        )

        pattern_noise_region = (
            pattern_noise_region
            if pattern_noise_region is not None
            else self.config.get("pattern_noise_region", np.s_[:1000])
        )
        pattern_noise_sym = (
            pattern_noise_sym
            if pattern_noise_sym is not None
            else self.config.get("pattern_noise_sym", 8)
        )

        
        det = self.detectors(self.proposal, self.runNo)
        det = det[ToF]
        offs = self.offsets(self.proposal, self.runNo)

        self.data = xr.concat([
            self.AdqRawChannel(
                self.data,
                d['channel'],
                digitizer=f'SQS_DIGITIZER_{d["digitizer"]}',
                first_pulse_offset=d["first_pulse_offset"],
                single_pulse_length=offs[1],
                baseline=pattern_noise_region,
                cm_period=pattern_noise_sym
            ).pulse_data()[..., :offs[1]]
            for d in tqdm(det,position=1,leave=False,disable=True)
        ], dim=pd.Index(ToF,name="detector"))
        self.data = -self.data
        """
        if self.key == "all" or "XGM":
            lenTrain = 380
            pulseIds = self.data.pulseId.values.reshape(-1,lenTrain)[0]
            self.xgm = self.xgm.rename({"dim_0": "pulseId"}).isel(pulseId=slice(0,lenTrain))
            self.xgm = self.xgm.assign_coords(pulseId = pulseIds).stack(pulse=('trainId','pulseId'))
        """
        return self

class PeakFinder(Configurable):
    def __init__(self, data, config=None):
        super().__init__(config)
        self.data = data
        self.results = None

    def stack(self, stackTrains=None, trainStackSize=None, stackPulses=None, pulseStackStart = None, pulseStackStop=None,pulseStackSize=None):
        pulseIndex = self.data["pulse"].to_index()
        trainIds = pulseIndex.get_level_values("trainId").to_numpy()
        pulseIds = pulseIndex.get_level_values("pulseId").to_numpy()
        
        stackTrains = stackTrains if stackTrains is not None else self.config.get("stackTrains", True)
        trainStackSize = trainStackSize if trainStackSize is not None else self.config.get("trainStackSize", len(np.unique(trainIds)))

        stackPulses = stackPulses if stackPulses is not None else self.config.get("stackPulses", True)
        pulseStackStart = pulseStackStart if pulseStackStart is not None else self.config.get("pulseStackStart", 0)
        pulseStackStop = pulseStackStop if pulseStackStop is not None else self.config.get("pulseStackStop", len(np.unique(pulseIds)))
        pulseStackSize = pulseStackSize if pulseStackSize is not None else self.config.get("pulseStackSize", len(np.unique(pulseIds)))
        
        if trainStackSize is None:
            trainStackSize = len(np.unique(trainIds))
        if pulseStackStart is None:
            pulseStackStart = 0
        if pulseStackSize is None:
            pulseStackSize = len(np.unique(pulseIds))
        if pulseStackStop is None:
            pulseStackStop = len(np.unique(pulseIds))        
        
        if stackTrains == True:
            chunks = []
            uniqueTrainIds = np.unique(trainIds)
                
            nTrainStacks = len(uniqueTrainIds)//trainStackSize
            if nTrainStacks == 0:
                raise ValueError("pulseStackSize larger than available trains")
            
            for i in range(nTrainStacks):
                selTrainIds = uniqueTrainIds[i*trainStackSize:(i+1)*trainStackSize]
                trainMask = np.isin(self.data.trainId, selTrainIds)
                chunk = self.data.isel(pulse=trainMask).groupby("pulseId").mean()
                chunkTrainId = [int(selTrainIds[0])]
                chunk = chunk.expand_dims("trainId")
                chunk = chunk.assign_coords(trainId=("trainId",chunkTrainId))
                
                chunkTrainIds = selTrainIds[0]
                chunkPulseIds = chunk.pulseId.values
                chunkTrainIds = [int(chunkTrainIds)]*len(chunkPulseIds)
                chunk = chunk.rename({"trainId": "tid", "pulseId": "pulse"})
                multi_idx = pd.MultiIndex.from_arrays([chunkTrainIds, chunkPulseIds],names=["trainId", "pulseId"])
                chunk = chunk.assign_coords(pulse=multi_idx).drop_vars("tid").squeeze("tid") 
                chunks.append(chunk)
                
            stack = xr.concat(chunks,dim="pulse")
            self.data = stack
    
        if stackPulses:
            chunks = []
            uniquePulsIds = np.unique(pulseIds)
            nPulseStacks = len(uniquePulsIds)//pulseStackSize
            if nPulseStacks == 0:
                raise ValueError("pulseStackSize larger than available trains")
            
            for i in range(pulseStackStart,pulseStackStop,pulseStackSize):
                selPulseIds = uniquePulsIds[i:i+1]
                pulseMask = np.isin(self.data.pulseId, selPulseIds)
                chunk = self.data.isel(pulse=pulseMask).groupby("trainId").mean()
                chunkPulseId = [int(selPulseIds[0])]
                chunk = chunk.expand_dims("pulseId")
                chunk = chunk.assign_coords(pulseId=("pulseId",chunkPulseId))
                
                chunkPulseIds = selPulseIds[0]
                chunkTrainIds = chunk.trainId.values
                chunkPulseIds = [int(chunkPulseIds)]*len(chunkTrainIds)
                chunk = chunk.rename({"trainId": "pulse", "pulseId": "pid"})
                multi_idx = pd.MultiIndex.from_arrays([chunkTrainIds, chunkPulseIds],names=["trainId", "pulseId"])
                chunk = chunk.assign_coords(pulse=multi_idx).drop_vars("pid").squeeze("pid")  
                chunks.append(chunk)
    
            stack = xr.concat(chunks,dim="pulse")
            self.data = stack
        return self
        

    def normalize(self,ToF=None):
        if ToF != None:
            self.data = self.data / self.data.sel(detector=ToF).max().commpute()
        else:
            self.data = self.data/self.data.max().compute()
        self.data = self.data.persist()
        return self

    def process(self, threshold=None, peakNo=None,roi=None, distanceFactor=None, symmetric=None, minWidth=True):
        threshold = threshold if threshold is not None else self.config.get("threshold", 0)
        peakNo = peakNo if peakNo is not None else self.config.get("peakNo", 8)
        roi = roi if roi is not None else self.config.get("roi", [None,None])
        distanceFactor = distanceFactor if distanceFactor is not None else self.config.get("distanceFactor", 2)
        symmetric = symmetric if symmetric is not None else self.config.get("symmetric", True)
        
        peakFunc = partial(
            findPeaksInTrace_np,
            peakNo=peakNo,
            cutOff=threshold,
            widthFactor=distanceFactor,
            symmetric=symmetric,
            minWidth=minWidth
        )
    
        results_list = []
        for det in tqdm(range(self.data.sizes["detector"]), desc="Finding peaks in ToFs",position=2,leave=False,disable=True):
            sliceDet = self.data.isel(detector=det)
            sliceDet = sliceDet.sel(sample=slice(roi[0],roi[1]))
            
            sample_coords = sliceDet["sample"].values  # the actual sample indices
            def peakFuncWithCoords(trace):
                peaks = findPeaksInTrace_np(trace, peakNo=peakNo, cutOff=threshold,
                                            widthFactor=distanceFactor, symmetric=symmetric,
                                            minWidth=minWidth)
                # replace positions with real coordinates
                if peaks is not None and len(peaks) > 0:
                    peaks[:, 0] = sample_coords[peaks[:, 0].astype(int)]
                return peaks
        
            results_det = xr.apply_ufunc(
                peakFunc,
                sliceDet,
                input_core_dims=[["sample"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[object]
            )
            results_list.append(results_det)
    
        self.results = xr.concat(results_list, dim="detector")
        self.results = self.results.persist()
        return self


    def plot(self, trainIndex=None, pulseIndex=None, num=None, xmin=None, xmax=None, ymin=None, ymax=None, logScale=True):

        train_ids = self.data.indexes["pulse"].get_level_values("trainId")
        pulse_ids = self.data.indexes["pulse"].get_level_values("pulseId")
        

        randomTrainId = np.random.choice(train_ids)
        randomPulseId = np.random.choice(pulse_ids)
        
        
        trainId = trainIndex if trainIndex is not None else self.config.get("plotTrainIndex", randomTrainId)
        pulseId = pulseIndex if pulseIndex is not None else self.config.get("plotPulseIndex", randomPulseId)

        
        print(f"Random trainId: {trainId}, pulseId: {pulseId}")

        plotYNum = int(np.ceil(len(np.unique(self.data["detector"]))/4))
        fig, ax = plt.subplots(plotYNum,4,figsize=(12, 3*plotYNum),sharex='all', sharey='all')
        plt.ylabel ('Signal')
        plt.xlabel ('Sample')
        ax = ax.flatten()
        j=0
        if ymax is None:
            ymax = self.data.max() * 1.05
        
        for ToF in self.data["detector"].to_index():
            trace = self.data.sel(detector=ToF,pulse={"trainId": trainId, "pulseId": pulseId})
            ax[j].set_title(f"ToF: {ToF}")
            ax[j].grid(True)
            ax[j].plot(trace,marker='.', color = 'teal',  markersize=0 ,alpha=1,linewidth = 1)
            if logScale:
                #trace = np.clip(trace,1e-12,None)
                ax[j].set_yscale('symlog', linthresh=1e-2)
            ax[j].set_ylim([ymin, ymax])
            ax[j].set_xlim([xmin, xmax])
            try:
                for peakNo in self.results["peakNo"].unique():
                    peak = self.results[(self.results["detector"]==ToF)&(self.results["peakNo"]==peakNo)&(self.results["trainId"]==trainId)&(self.results["pulseId"]==pulseId)]
                    pos = peak["pos"].iloc[0]
                    height = peak["height"].iloc[0]
                    widthl = peak["width left"].iloc[0]
                    widthr = peak["width right"].iloc[0]
                    ax[j].hlines(y=height/2, xmin=pos+widthl, xmax=pos+widthr, colors="red")
                    ax[j].scatter(x=pos,y=height,color="red")
            except:
                print("Found no peaks in ToF ",ToF)
            j+=1
        plt.savefig("traces.png",dpi=600)
        return self


    def dataframe(self):
        """
        Convert the list-of-arrays results from peak finding into a pandas DataFrame
        with columns ["detector","trainId","pulseId","peakNo","pos","height","width left","width right","fwhm area"].
        """
        all_results = []
        all_metadata = []
    
        # iterate over detectors
        for det_idx, det in enumerate(self.results["detector"].values):
            # get pulse MultiIndex
            pulse_index = self.results["pulse"].to_index()
            data_det = self.results.isel(detector=det_idx).values
    
            # iterate over pulses (still required because peaks per pulse vary)
            for (trainId, pulseId), peaks in zip(pulse_index, data_det):
                if peaks is None or len(peaks) == 0:
                    continue
                peaks = np.array(peaks)  # shape: (num_peaks, 5)
                peakNos = np.arange(len(peaks)).reshape(-1,1)
                all_results.append(np.hstack([peakNos, peaks]))
                # replicate metadata for each peak
                all_metadata.append(np.tile([det, trainId, pulseId], (len(peaks), 1)))
    
        if len(all_results) == 0:
            self.results = pd.DataFrame(
                columns=["detector","trainId","pulseId","peakNo","pos","height","width left","width right","fwhm area"]
            )
            return self
    
        # stack results vertically
        all_results = np.vstack(all_results)
        all_metadata = np.vstack(all_metadata)
    
        # create DataFrame
        df = pd.DataFrame(
            np.hstack([all_metadata, all_results]),
            columns=["detector","trainId","pulseId","peakNo","pos","height","width left","width right","fwhm area"]
        )
    
        # convert appropriate columns to int
        df[["detector","trainId","pulseId","peakNo"]] = df[["detector","trainId","pulseId","peakNo"]].astype(int)
    
        self.results = df
        return self





class AuxFunc:
    def __init__(self, data):
        self.data = data

    def addData(self, moreData, key="Photon Energy", axis="trainId"):
        if axis not in self.data.columns:
            raise KeyError(f"{axis} not found in self.data")
        if key not in moreData.columns:
            raise KeyError(f"{key} not found in moreData")

        mapping = moreData.set_index(axis)[key]
        self.data[key] = self.data[axis].map(mapping)
        return self

def streamXarray(data):
    for trainId, group in data.groupby("trainId"):
        #time.sleep(0.01)
        yield group


class PhotonEnergyProcessor(Configurable):
    def __init__(self, proposal, runNo, loaderClass, config=None):
        super().__init__(config)
        self.loaderClass = loaderClass
        self.proposal = proposal
        self.runNo = runNo
        self.data = None
        self.photonEnergies = None
        self.firstTrainId = int
        self.results = []
        self.run = self.loaderClass(self.proposal, self.runNo)
        self.pf = PeakFinder(self.data,config=self.config)


    def getRunEnergies(self,singleRun=None, energyStart=None, energyStop=None, energyStep=None):
        singleRun = singleRun if singleRun is not None else self.config.get("singleRun", True)
        energyStart = energyStart if energyStart is not None else self.config.get("energyStart", 0)
        energyStop = energyStop if energyStop is not None else self.config.get("energyStop", None)
        energyStep = energyStep if energyStep is not None else self.config.get("energyStep", 1)
        if singleRun:
            self.run.load(key="Photon Energy",trainStart=None, trainStop=None, trainStep=None, pulseStart=None, pulseStop=None, pulseStep=None)
            self.firstTrainId = self.run.photonEnergy.trainId[0]
            self.photonEnergies = self.run.photonEnergy.groupby("Photon Energy",as_index=False).last()
        else:
            self.run = self.loaderClass(self.proposal, self.runNo,config=self.config)
            self.run.load(key="Photon Energy",trainStart=None, trainStop=None, trainStep=None, pulseStart=None, pulseStop=None, pulseStep=None)
            self.photonEnergies = self.run.photonEnergy#.groupby("Photon Energy",as_index=False))
            #self.photonEnergies["daq_run"] = self.photonEnergies["daq_run"].astype(int)
            #self.photonEnergies["trainId"] = self.photonEnergies["trainId"].astype(int)
        return self

        trainStart = trainStart or self.config.get("trainStart", None)
        trainStop = trainStop or self.config.get("trainStop", None)
        trainStep = trainStep or self.config.get("trainStep", None)
        pulseStart = pulseStart or self.config.get("pulseStart", None)
        pulseStop = pulseStop or self.config.get("pulseStop", None)
        pulseStop = pulseStep or self.config.get("pulseStep", None)
    
    def processEnergies(self, energyStart=None, energyStop=None, energyStep=None, trainSliceStop=None, singleRun=None, peakFinderConfig=None):
        energyStart = energyStart if energyStart is not None else self.config.get("energyStart", 0)
        energyStop = energyStop if energyStop is not None else self.config.get("energyStop", None)
        energyStep = energyStep if energyStep is not None else self.config.get("energyStep", 1)
        singleRun = singleRun if singleRun is not None else self.config.get("singleRun", True)

        #loaderConfig = loaderConfig or self.config.get("loaderConfig", {})
        #peakFinderConfig = peakFinderConfig or self.config.get("PeakFinder", {})
        #print("Start processing...")
        if energyStop == None:
            energyStop = len(self.photonEnergies)-1

        peakChunks = []
        for i in tqdm(np.arange(energyStart, energyStop, energyStep),desc="Processing trains",position=0):
            if singleRun:
                trainStart = self.photonEnergies.trainId[i] - self.firstTrainId
                trainSliceStop = self.run.config.get("trainStep",1)
                self.data = self.run.load(trainStart = trainStart, trainStop = int(trainStart+trainSliceStop)).defaultPreprocessing().data
            else:
                self.run = self.loaderClass(self.proposal, self.runNo[0], config=self.config)
                self.data = self.run.load().defaultPreprocessing().data
            peakChunk = PeakFinder(self.data,config=self.config).stack().normalize().process().dataframe().results
            AuxFunc(peakChunk).addData(self.run.photonEnergy)
            peakChunks.append(peakChunk)
        self.results = pd.concat(peakChunks)
        print("Done!")
        return self.results

class Calibrate(Configurable):
    def __init__(self, data, config=None):
        super().__init__(config)
        self.data = data
        self.results = []
        self.energyParam = []
        self.transmissionParam = []

    def madFilter(self, x, y, thresh=3):
        y_np = np.asarray(y)
        med = np.median(y_np)
        mad = np.median(np.abs(y_np - med))
    
        if mad == 0:
            return np.ones_like(y_np, dtype=bool)
    
        z = 0.6745 * (y_np - med) / mad
        return np.abs(z) <= thresh

        
    def energy(self,relPos=False,peakNo=None,guess=None):
        peakNo = (peakNo or self.config.get("peakNo", 1))
        guess = (guess or self.config.get("initial guess", [0,0.001,10000]))
        avgPos = self.data.groupby(["detector","peakNo","Photon Energy"])["pos"].mean().reset_index()
        energyParam = []
        transmissionParam = []
        for det in avgPos["detector"].unique():
            pos = pd.DataFrame(avgPos[(avgPos["detector"]==det)&(avgPos["peakNo"]==peakNo)]["pos"]).reset_index()["pos"]
            if relPos:
                pos0 = pd.DataFrame(avgPos[(avgPos["detector"]==det)&(avgPos["peakNo"]==0)]["pos"]).reset_index()
                pos = pos - pos0["pos"]
            energy = avgPos[(avgPos["detector"]==det)&(avgPos["peakNo"]==peakNo)]["Photon Energy"]
            if len(energy)<3:
                continue
            xdata = pos.values
            ydata = energy.values
            goodData = self.madFilter(xdata,ydata)
            guess = [min(ydata),0.001,max(ydata)]
            try:
                params, pcov = curve_fit(negExpFunc, xdata[goodData], ydata[goodData], p0=guess, maxfev=1000000)
                aFit, bFit, cFit = params
                perr = np.sqrt(np.diag(pcov))
                energyParam.append({"detector": det, "peakNo": peakNo, "a": aFit, "b": bFit, "c": cFit, "a error":perr[0], "b error":perr[1], "c error":perr[2]})
            except RuntimeError:
                # Fit failed
                energyParam.append({
                    'detector': det,
                    'peakNo': peakNo,
                    'a': np.nan,
                    'b': np.nan,
                    'c': np.nan,
                    "a error":np.nan,
                    "b error":np.nan,
                    "c error":np.nan,
            })
        self.energyParam = pd.DataFrame(energyParam)
        return self

    def transmission(self, peakNo=None, setBeta=None, setPhi=None, setPlin=None):
        transmissionParam = []
        """
        beta = beta or self.config.get("beta",0)
        peakNo = peakNo or self.config.get("Transmission PeakNo",0)
        """
        for energy in self.data["Photon Energy"].unique():
            for ToF in self.data["detector"].unique():
                selData = self.data[(self.data["peakNo"]==peakNo)&(self.data["Photon Energy"]==energy)&(self.data["detector"]==ToF)]
                trace = selData["fwhm area"].values
                theta = np.deg2rad(selData["Angles"].to_numpy()[0])
                
                g = polarization_model(theta, Plin=setPlin, phi=setPhi,beta2=setBeta)
                transPar = trace/g
                transmissionParam.append({"detector": ToF, "Photon Energy": energy, "Transmission coefficent": transPar[0]})
        self.transmissionParam = pd.DataFrame(transmissionParam)
        return self

    def plotTransmission(self,ymin=None,ymax=None):
        plotYNum = int(np.ceil(self.data["detector"].nunique()/4))
        fig, ax = plt.subplots(plotYNum,4,figsize=(12, 3*plotYNum),sharex='all', sharey='all')
        plt.ylabel ('Transmission coefficent')
        plt.xlabel ('Photon Energy')
        ax = ax.flatten()
        j=0
        for ToF in self.transmissionParam["detector"].unique():
            xdata = self.transmissionParam[(self.transmissionParam["detector"]==ToF)]["Photon Energy"]
            ydata = self.transmissionParam[(self.transmissionParam["detector"]==ToF)]["Transmission coefficent"]
            ax[j].set_title(f"ToF: {ToF}")
            ax[j].grid(True)
            ax[j].plot(xdata,ydata,marker='.', color = 'teal',  markersize=2 ,alpha=1,linewidth = 0)
            ax[j].set_ylim([ymin, ymax])
            j+=1
        plt.savefig("Transmission.png",dpi=600)
        return self
                

    def plotEnergy(self, peakNo = None, plotReg = True, relPos=False, ymin=None, ymax=None, xmin=None, xmax=None):
        peakNo = (peakNo or self.config.get("peakNo", 1))
        plotYNum = int(np.ceil(self.data["detector"].nunique()/4))
        fig, ax = plt.subplots(plotYNum,4,figsize=(12, 3*plotYNum),sharex='all', sharey='all')

        plt.ylabel ('Photon Energy')
        plt.xlabel ('Sample')
        ax = ax.flatten()
        j=0
            
        for i in self.data["detector"].unique():
            pos = self.data[(self.data["detector"]==i)&(self.data["peakNo"]==peakNo)]["pos"].reset_index()
            if relPos:
                pos0 = pd.DataFrame(self.data[(self.data["detector"]==i)&(self.data["peakNo"]==0)]["pos"]).reset_index()
                pos = pos - pos0
                        
            xdata = pos["pos"]
            ydata = self.data[(self.data["peakNo"]==peakNo)&(self.data["detector"]==i)]["Photon Energy"]
            
            if plotReg:
                goodData = self.madFilter(xdata,ydata)
                xFit = np.linspace(xdata[goodData].min(),xdata[goodData].max(),500)
                xFitExt = np.linspace(xdata[goodData].min(),xdata.max(),500)
                #print(xdata[goodData].min(),xdata[goodData].max())
                a, b, c = self.energyParam.loc[self.energyParam["detector"] == i, ["a", "b", "c"]].values[0]
                ax[j].plot(xFitExt, negExpFunc(xFitExt, a, b, c), color="salmon",linestyle="dashed", label="Fit", linewidth=0.9)
                ax[j].plot(xFit, negExpFunc(xFit, a, b, c), color="forestgreen", label="Fit")
                ax[j].set_xlim([xmin, xmax])
                ax[j].set_ylim([ymin, ymax])
            ax[j].set_title(f"ToF: {i}")
            ax[j].grid(True)
            ax[j].plot(xdata,ydata,marker='.', color = 'teal',  markersize=2 ,alpha=1,linewidth = 0)
            j+=1

        plt.savefig("Energy.png",dpi=600)
        plt.show()
        return self

class plotter(Configurable):
    def __init__(self, results, config=None):
        super().__init__(config)
        self.results = results

    def plotPol(self, transParam, peakNo=None,beta=0,intMethod="fwhm area"):
        peakNo = peakNo if peakNo is not None else self.config.get("peakNo", 0)
        
        fig, ax = plt.subplots(figsize=(6,4), subplot_kw={'projection': 'polar'})
        fullTheta = np.linspace(0,2*np.pi,16,endpoint=False)
        area = self.results[self.results["peakNo"]==peakNo][["fwhm area","detector","Angles"]]
        calib = transParam
        calibArea = pd.merge(area,calib,on="detector")
        calibArea["calibValue"] = calibArea[intMethod] * calibArea["Transmission coefficent"] / calibArea[intMethod].max()
            
        theta = calibArea["Angles"].values*np.pi/180
        trace = calibArea["calibValue"].values
        #ax.set_rlim(0,2)
        ax.plot(theta, trace, marker="o", linewidth=0, label='Data')
    
        def model(theta,Plin,phi,scale):
            return polarization_model(theta, Plin, phi, beta2=beta, scale=scale)
            
        initial_guess = [0.5, 0.0,1.0]  # [Plin, phi, scale]
        bounds = ([0, -np.pi,0], [2, np.pi,20])
        popt, pcov = curve_fit(model, theta, trace, p0=initial_guess, bounds=bounds)
        Plin_fit, phi_fit, scale_fit = popt
    
        theta_fit = np.linspace(0, 2*np.pi, 360)
        intensity_fit = model(theta_fit, Plin_fit, phi_fit, scale=scale_fit)
        
        #ax.set_rlim(0,1.2)
        if Plin_fit>0.015:
            ax.plot([phi_fit,phi_fit],[0,1],color="orange")
            ax.plot([phi_fit+np.pi,phi_fit+np.pi],[0,1],color="orange")
        ax.plot(theta_fit, intensity_fit, label=f"Fitted degree of linear polarization: {Plin_fit:.5f}",color="green")
            
        ax.set_yticks([])
        ax.set_theta_zero_location("N")  # 0° at top
        ax.set_theta_direction(-1)       # clockwise
        ax.legend(loc="lower right")
        plt.show()

class StreamTracePlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.ToF = None
        #self.ax.set_xlim(0,800)
        self.ax.set_ylim(1,0)

    def setup(self, ToF=0):
        """Optional: initialize the figure."""
        self.ToF = ToF
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Signal")
        self.ax.set_title("Live Stream")
        plt.show()

    def update(self, data, results=None, ToF=None, trainIndex=0, pulseIndex=0,xmin=0,xmax=800):
        """
        data: the current xarray chunk
        results: optional DataFrame with peak results
        """
        clear_output(wait=True)  # clears previous output so the plot refreshes

        index = data["pulse"].to_index()
        trainId = index.get_level_values("trainId")[0]
        pulseId = index.get_level_values("pulseId")[pulseIndex]

        ToF = ToF if ToF is not None else 0

        # select trace for given detector/train/pulse
        trace = data.sel(detector=ToF, pulse={"trainId": trainId, "pulseId": pulseId})

        # create figure fresh each time
        fig, ax = plt.subplots(figsize=(6,4))
        line, = ax.plot(trace, label=f"Train {trainId}, Pulse {pulseId}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Signal")
        ax.set_title("Streaming Trace")
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(-0.1,1)
        ax.legend(loc="lower right")
        
        
        # draw peaks if results exist
        if results is not None and not results.empty and "pos" in results.columns:
            peaks = results[
                  (results["detector"] == ToF)
                & (results["trainId"] == trainId)
                & (results["pulseId"] == pulseId)]
            
            for _, peak in peaks.iterrows():
                pos = peak["pos"]
                height = peak["height"]
                ax.scatter(pos, height, color="red")
                ax.hlines(y=height/2, xmin=pos+peak["width left"], xmax=pos+peak["width right"], colors="red")
        else:
            line.set_color("red")
        plt.show()


class StreamPolPlotter(Configurable):
    def __init__(self,calibration=None, config=None):
        super().__init__(config)
        self.calibData = None
        self.trace = None
        self.traceData = None
        self.fullTheta = np.linspace(0,2*np.pi,16,endpoint=False)
        self.calibration = calibration        
    

    def loadCalib(self):
        self.calibData = self.data.merge(calibParams,on["detector","Photon Energy"])
        self.calibData["calibrated area"] = self.calibData["fwhm area"]*self.calibData["Transmission coefficent"]
        return self

    def update(self, data, trainId=None):    
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6,4), subplot_kw={'projection': 'polar'})
    
        # select the data for this trainId and peakNo==0
        if data is not None and not data.empty and "detector" in data.columns:
            traceData = data[(data["peakNo"]==0)]
            if traceData is None or traceData.empty:
                print("false")
                
            dets = traceData["detector"].values
            theta = self.fullTheta[dets]
            if self.calibration is not None:
                #merge
                traceData_sorted = traceData.sort_values("Photon Energy")
                calib_sorted = self.calibration.sort_values("Photon Energy")
                traceData = pd.merge_asof(
                    traceData_sorted,
                    calib_sorted,
                    on="Photon Energy",
                    by="detector",
                    direction="nearest"
                ).dropna(subset=["Transmission coefficent", "fwhm area"])
                traceData["calibrated area"] = traceData["fwhm area"] * traceData["Transmission coefficent"]

                trace = traceData["calibrated area"].values
                theta = self.fullTheta[traceData["detector"].values]
                #Fit
                initial_guess = [0.5, 0.0, 1.0]  # [Plin, phi, scale]
                bounds = ([0, -np.pi, 0], [1, np.pi, np.inf])
                popt, pcov = curve_fit(polarization_model, theta, trace, p0=initial_guess, bounds=bounds)
                Plin_fit, phi_fit, scale_fit = popt

                theta_fit = np.linspace(0, 2*np.pi, 360)
                intensity_fit = polarization_model(theta_fit, Plin_fit, phi_fit, scale_fit)
                #ax.set_rlim(0,1.2)
                if Plin_fit>0.015:
                    ax.plot([phi_fit,phi_fit],[0,1],color="orange")
                    ax.plot([phi_fit+np.pi,phi_fit+np.pi],[0,1],color="orange")
                ax.plot(theta_fit, intensity_fit, label=f"Fitted degree of linear polarization: {Plin_fit:.3f}",color="green")

            else:
                trace = traceData["fwhm area"].values
                ax.set_rlim(0,2)
            ax.plot(theta, trace, marker="o", linewidth=0, label='Data')
        ax.set_yticks([])
        ax.set_theta_zero_location("N")  # 0° at top
        ax.set_theta_direction(-1)       # clockwise
        ax.legend(loc="lower right")
        plt.show()
        

def polarization_model(theta, Plin=1, phi=0, beta2=2, scale=1):
    return scale*(1 + (beta2 / 4) * (1 + 3 * Plin * np.cos(2 * (theta - phi))))

#Main Functions
#determines peakwidth of symetric peaks
def findSymmetricPeakWidth(trace,peak):
    '''
    Function to calculate the fwhm of a peak within a trace
    assuming the peak is symmetric.
    
    Parameters
    --------
    trace, array:
        trace with the peak
    peak, int:
        index of the peak which width shall be calculated.

    Returns
    --------
    peakWidth, int:
        half width at half maximum.
    '''
    peakWidth = 0
    maxWidth=20
    while peakWidth < maxWidth and (peak+peakWidth) < len(trace):
        if trace[peak]/2 <= trace[peak+peakWidth]:
            peakWidth +=1
        else:
            break
    
    return -peakWidth, peakWidth

#determines peakwidth of asymetric peaks
def findAsymmetricPeakWidth(trace,peak):
    '''
    Function to calculate the fwhm of a peak within a trace
    assuming the peak is asymmetric.
    
    Parameters
    --------
    trace, array:
        trace with the peak
    peak, int:
        index of the peak which width shall be calculated.

    Returns
    --------
    peakWidthL, int:
        width at half maximum left of the peak. (negative value)
    peakWidthR, int:
        width at half maximum right of the peak.
    '''
    peakWidthR = 0
    maxWidth = 20
    while peakWidthR < maxWidth and (peak+peakWidthR) < len(trace):
        if trace[peak]/2 <= trace[peak+peakWidthR]:
            peakWidthR +=1
        else:
            break
            
    peakWidthL = -peakWidthR
    if trace[peak]/2 <= trace[peak+peakWidthL]:
        for i in range(20):
            if trace[peak]/2 <= trace[peak+peakWidthL]:
                peakWidthL -=1
            else:
                break

    if trace[peak]/2 >= trace[peak+peakWidthL]:
        for i in range(20):
            if trace[peak]/2 >= trace[peak+peakWidthL]:
                peakWidthL +=1
            else:
                break          
    return peakWidthL, peakWidthR

def findPeak(trace, widthFactor=2, symmetric = True):
    '''
    Function to find the minimum in a trace.

    Parameters
    --------
    trace, 1D array
        Array with the trace

    Returns
    --------
    trace, 1D array
        Array of the trace without the peak
    peak, int
        Index of the peak
    '''
    trace = trace.copy()
    peak = trace.argmax()
    height = trace.max()
    if symmetric == True:
        peakWidthL, peakWidthR = findSymmetricPeakWidth(trace,peak)
    else:
        peakWidthL, peakWidthR = findAsymmetricPeakWidth(trace,peak)
        
    trace[peak+(peakWidthL*widthFactor):peak+(peakWidthR*widthFactor)] = 0
    return trace, peak, height, peakWidthL, peakWidthR

def findPeaksInTrace(trace, peakNo , cutOff = -100, widthFactor=2, symmetric = True):
    results = []
    traceCopy = trace.copy()
    if traceCopy.max() > cutOff:
        for i in range(peakNo+1):
            if traceCopy.max() > cutOff:
                traceCopy, pos, height, widthL, widthR = findPeak(traceCopy, widthFactor=widthFactor, symmetric = symmetric)
                a = trace[pos+widthL:pos+widthR].sum()
                results.append([pos,height,widthL,widthR,a])
    if len(results) == peakNo+1:
        results.sort()
        resultsDicts = [{"pos": p, "height": h, "width left": wl, "width right":wr, "fwhm area": a} for p, h, wl, wr, a in results]
    else:
        resultsDicts = None
    return resultsDicts

def findPeak_np(trace, widthFactor=2, symmetric=False, maxWidth=20, minWidth=False):
    """
    Find the largest peak in a trace and compute FWHM widths.
    Returns trace with peak zeroed, peak position, height, left width, right width.
    """
    trace = trace.copy()
    peak = trace.argmax()
    height = trace[peak]

    # Slice around peak
    left_slice = trace[max(0, peak-maxWidth):peak+1][::-1]  # reverse for left
    right_slice = trace[peak:peak+maxWidth+1]

    # Find first index below half max
    widthR = np.argmax(right_slice < height/2)
    if widthR == 0 and right_slice[0] >= height/2:
        widthR = min(maxWidth, len(right_slice)-1)

    widthL = -np.argmax(left_slice < height/2)
    if widthL == 0 and left_slice[0] >= height/2:
        widthL = -min(maxWidth, len(left_slice)-1)

    # For symmetric peaks, just use max of L/R
    if symmetric:
        w = max(abs(widthL), widthR)
        widthL, widthR = -w, w

    # Zero out peak region
    start_zero = max(0, peak + widthL*widthFactor)
    stop_zero = min(len(trace), peak + widthR*widthFactor)
    if minWidth:
        widthL = -min(abs(widthL),abs(widthR))
        widthR = min(abs(widthL),abs(widthR))
    # Area under peak
    start = max(0, peak + widthL)
    stop = min(len(trace), peak + widthR)
    area = trace[start:stop].sum()
    trace[start_zero:stop_zero] = 0

    return trace, peak, height, widthL, widthR, area

def findPeaksInTrace_np(trace, peakNo, cutOff=-100, widthFactor=2, symmetric=True, maxWidth=30, minWidth=False):
    results = []

    traceCopy = trace.copy()
    for _ in range(peakNo+1):
        if traceCopy.max() <= cutOff:
            break
        traceCopy, pos, height, widthL, widthR, area = findPeak_np(
            traceCopy, widthFactor=widthFactor, symmetric=symmetric, maxWidth=maxWidth,minWidth=minWidth)

        results.append([pos, height, widthL, widthR, area])

    if len(results) == peakNo + 1:
        results_arr = np.array(results, dtype=float)
        sorted_indices = np.argsort(results_arr[:, 0])  # sort by pos
        return results_arr[sorted_indices]


def findPeaksInTrace_sp(trace, peakNo, cutOff=0, widthFactor=2, symmetric=False, maxWidth=20):
    results = []

    traceCopy = trace.copy()
    pos, optRes = find_peaks(traceCopy, height=cutOff)

    i=0
    if len(pos)==peakNo+1:
        for peak in pos:
            
            height = optRes['peak_heights'][i]
            leftSlice = trace[max(0, peak-maxWidth):peak+1][::-1]  # reverse for left
            rightSlice = trace[peak:peak+maxWidth+1]
            
            widthR = np.argmax(rightSlice < height/2)
            if widthR == 0 and rightSlice[0] >= height/2:
                widthR = min(maxWidth, len(rightSlice)-1)
            
            widthL = -np.argmax(leftSlice < height/2)
            if widthL == 0 and leftSlice[0] >= height/2:
                widthL = -min(maxWidth, len(leftSlice)-1)
            widthL = -min(abs(widthL),abs(widthR))
            widthR = min(abs(widthL),abs(widthR))
            area = trace[peak+widthL:peak+widthR].sum()
            i+=1
        
            results.append([peak, height, widthL, widthR, area])

        results_arr = np.array(results, dtype=float)
        sorted_indices = np.argsort(results_arr[:, 0])  # sort by pos
        return results_arr[sorted_indices]



def negExpFunc(x, a, b, c):
    return a * np.exp(-b * x) + c