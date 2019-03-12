#!/usr/bin/env python
import numpy as np

from .refldata import ReflData
from . import nexusref
from .nexusref import data_as, fetch_str

def load_metadata(filename, file_obj=None):
    """
    Load the summary info for all entries in a NeXus file.
    """
    return load_entries(filename, file_obj=file_obj,
                        meta_only=True, entry_loader=Candor)

def load_entries(filename, file_obj=None, entries=None):
    return nexusref.load_nexus_entries(filename, file_obj=file_obj,
                                       meta_only=False, entry_loader=Candor)


class Candor(ReflData):
    """
    NeXus reflectometry entry.

    See :class:`refldata.ReflData` for details.
    """
    format = "NeXus"
    probe = "neutron"

    def __init__(self, entry, entryname, filename):
        super(Candor, self).__init__()
        nexusref.nexus_common(self, entry, entryname, filename)

    def load(self, entry):
        #print(entry['instrument'].values())
        das = entry['DAS_logs']
        n = self.points

        self.slit1.distance = data_as(entry, 'instrument/presample_slit1/distance', 'mm')
        self.slit2.distance = data_as(entry, 'instrument/presample_slit2/distance', 'mm')
        self.slit3.distance = data_as(entry, 'instrument/predetector_slit1/distance', 'mm')
        self.slit4.distance = data_as(entry, 'instrument/detector_mask/distance', 'mm')

        raw_intent = fetch_str(das, 'trajectoryData/_scanType')
        if raw_intent in nexusref.TRAJECTORY_INTENTS:
            self.intent = nexusref.TRAJECTORY_INTENTS[raw_intent]
        self.polarization = nexusref.get_pol(das, 'frontPolarization') \
                            + nexusref.get_pol(das, 'backPolarization')

        if np.isnan(self.slit1.distance):
            self.warn("Slit 1 distance is missing; using 4600 mm")
            self.slit1.distance = -4600
        if np.isnan(self.slit2.distance):
            self.warn("Slit 2 distance is missing; using 356 mm")
            self.slit2.distance = -356
        if np.isnan(self.slit3.distance):
            self.warn("Slit 3 distance is missing; using 356 mm")
            self.slit1.distance = 356
        if np.isnan(self.slit4.distance):
            self.warn("Slit 4 distance is missing; using 3496 mm")
            self.slit1.distance = 3496

        if np.isnan(self.detector.wavelength):
            self.warn("Wavelength is missing; using 4.75 A")
            self.detector.wavelength = 4.75
        if np.isnan(self.detector.wavelength_resolution):
            self.warn("Wavelength resolution is missing; using 1.5% dL/L FWHM")
            self.detector.wavelength_resolution = 0.015/2.35*self.detector.wavelength

        self.detector.counts = np.asarray(data_as(entry, 'instrument/detector/data', ''), 'd')
        self.detector.counts_variance = self.detector.counts.copy()
        self.detector.dims = self.detector.counts.shape[1:]
        self.detector.wavelength = data_as(entry, 'instrument/detector/wavelength','Ang', rep=n)
        self.detector.wavelength_resolution = data_as(entry, 'instrument/detector/wavelength_error','Ang', rep=n)
        for k, s in enumerate((self.slit1, self.slit2, self.slit3)):
            x = 'slitAperture%d/softPosition'%(k+1)
            x_target = 'slitAperture%d/desiredSoftPosition'%(k+1)
            s.x = data_as(das, x, 'mm', rep=n)
            s.x_target = data_as(das, x_target, 'mm', rep=n)
            #y = 'vertSlitAperture%d/softPosition'%(k+1)
            #y_target = 'vertSlitAperture%d/desiredSoftPosition'%(k+1)
            #s.y = data_as(das, y, 'mm', rep=n)
            #s.y_target = data_as(das, y_target, 'mm', rep=n)
        # Slit 4 on CANDOR is a fixed mask rather than continuous motors.
        self.slit4.x = data_as(entry, 'instrument/detector_mask/width', 'mm', rep=n)
        self.slit4.x_target = self.slit4.x
        #self.slit4.y = data_as(entry, 'instrument/detector_mask/height', 'mm', rep=n)
        #self.slit4.y_target = self.slit4.y
        self.sample.angle_x = data_as(das, 'sampleAngle/softPosition', 'degree', rep=n)
        self.sample.angle_x_target = data_as(das, 'sampleAngle/desiredSoftPosition', 'degree', rep=n)

        table_angle = data_as(das, 'detectorTableMotor/softPosition', 'degree', rep=n)
        table_angle_target = data_as(das, 'detectorTableMotor/desiredSoftPosition', 'degree', rep=n)
        #bank_angle = das['detectorTable/rowAngleOffset'].value
        bank_angle = np.arange(30)*0.1
        self.detector.angle_x = table_angle[:, None] + bank_angle[None, :]
        self.detector.angle_x = table_angle_target[:, None] + bank_angle[None, :]

from .nexusref import demo
if __name__ == "__main__":
    demo(loader=load_entries)
