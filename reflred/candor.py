#!/usr/bin/env python
import numpy as np

from .refldata import ReflData
from . import nexusref
from .nexusref import data_as, fetch_str
from .resolution import FWHM2sigma

def load_metadata(filename, file_obj=None):
    """
    Load the summary info for all entries in a NeXus file.
    """
    return load_entries(filename, file_obj=file_obj,
                        meta_only=True, entry_loader=Candor)

def load_entries(filename, file_obj=None, entries=None):
    return nexusref.load_nexus_entries(filename, file_obj=file_obj,
                                       meta_only=False, entry_loader=Candor)

#: Number of detector channels per detector tube on CANDOR
NUM_CHANNELS = 54

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
        self.monochromator.wavelength = data_as(entry, 'instrument/monochromator/wavelength', 'Ang', rep=n)
        self.monochromator.wavelength_resolution = data_as(entry, 'instrument/monochromator/wavelength_error','Ang', rep=n)
        divergence = data_as(das, 'detectorTable/angularSpreads', '')
        wavelength = data_as(das, 'detectorTable/wavelengths', '')
        wavelength_spread = data_as(das, 'detectorTable/wavelengthSpreads', '')
        efficiency = data_as(das, 'detectorTable/detectorEfficiencies', '')
        if np.isnan(efficiency):
            # Missing detector efficiency info
            efficiency = np.ones_like(divergence)
        divergence = divergence.reshape(NUM_CHANNELS, -1).T[None, :, :]
        wavelength = wavelength.reshape(NUM_CHANNELS, -1).T[None, :, :]
        wavelength_spread = wavelength_spread.reshape(NUM_CHANNELS, -1).T[None, :, :]
        efficiency = efficiency.reshape(NUM_CHANNELS, -1).T[None, :, :]
        self.detector.wavelength = wavelength
        self.detector.wavelength_resolution = FWHM2sigma(wavelength * wavelength_spread)
        self.detector.efficiency = efficiency
        # TODO: not using angular divergence?

        if np.isnan(self.monochromator.wavelength):
            # If not monochromatic beam then assume elastic scattering
            self.monochromator.wavelength = self.detector.wavelength
            self.monochromator.wavelength_resolution = self.detector.wavelength_resolution


        raw_intent = fetch_str(das, 'trajectoryData/_scanType')
        if raw_intent in nexusref.TRAJECTORY_INTENTS:
            self.intent = nexusref.TRAJECTORY_INTENTS[raw_intent]
        self.polarization = nexusref.get_pol(das, 'frontPolarization') \
                            + nexusref.get_pol(das, 'backPolarization')

        self.detector.counts = np.asarray(data_as(das, 'areaDetector/counts', ''), 'd')
        self.detector.counts_variance = self.detector.counts.copy()
        self.detector.dims = self.detector.counts.shape[1:]
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
        self.sample.angle_x = data_as(das, 'sampleAngleMotor/softPosition', 'degree', rep=n)
        self.sample.angle_x_target = data_as(das, 'sampleAngleMotor/desiredSoftPosition', 'degree', rep=n)
        # Add an extra dimension to sample angle
        self.sample.angle_x = self.sample.angle_x[:, None, None]
        self.sample.angle_x_target = self.sample.angle_x[:, None, None]

        table_angle = data_as(das, 'detectorTableMotor/softPosition', 'degree', rep=n)
        table_angle_target = data_as(das, 'detectorTableMotor/desiredSoftPosition', 'degree', rep=n)
        bank_angle = data_as(das, 'detectorTable/rowAngularOffsets', '')[0]
        #bank_angle = np.arange(30)*0.1
        self.detector.angle_x = table_angle[:, None, None] + bank_angle[None, :, None]
        self.detector.angle_x_target = table_angle_target[:, None, None] + bank_angle[None, :, None]

if __name__ == "__main__":
    from .nexusref import demo
    demo(loader=load_entries)
