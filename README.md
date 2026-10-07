this repository includes scripts for the visualisation and analysis of the MEG/EMG data from the paper Kornysheva et al. (2019) (https://pubmed.ncbi.nlm.nih.gov/30744987/)

made by Nicolo Marcello from UCL (nicolo.marcello.24@ucl.ac.uk).


## MEG/EMG DATA recordings:
MEG was recorded continuously at 1200 samples/second using a whole-head 275-channel axial gradiometer system (CTF Omega,
VSM MedTech) while participants sat upright in a magnetically shielded room. Head position coils were attached to nasion, left, and
right auricular sites to provide anatomical co-registration.

Participants were also fitted with four EMG electrodes to measure finger movement-related muscular activity. The electrodes were
placed above the flexor carpi radialis (FCR), abductor polices brevis (APB), abductor digiti minimi (ADM), first dorsal interossei (FDI).
FCR was recorded with a belly-belly montage, APB, ADM and FDI with a tendon-belly montage.

## Chanel naming convention
UPPT001 = parallel port trigger/stimulus channel
UPPT002 = show exactly when button press occured, different amplitudes mean different button was pressed

BG1-3 = background gradiometers
BP1-3 = background probes
G11-23 = (Reference Gradiometers): Channels measuring the spatial gradient of the background magnetic field
P11-23 = + (Reference Magnetometers / Probes): Channels measuring the raw magnetic field intensity along specific axes 
Q11-23 (Reference Magnetometers / Probes): Another set of background reference magnetometers orientation-mapped to isolate orthogonal noise components.
R11-23 (Reference Gradiometers): Additional high-order reference gradiometers used in the balancing array.

positions 31 to 304 represent all 275 MEG channels, starting with M

positional codes for MEG:
    first letter: M
    second letter (hemisphere): L (left), R (right), Z (midline)
    third letter (lobe region): F (frontal), C(central), P (parietal), O (occipital) or T (temporal)

EEG057-60 = EEG electrode channels

HADC001-3 = Head digital to analogue converter channels

HLC0011-0038 = Head localization coil

UADC = universal analogue-to0digital converter

## MEG and EMG preprocessing
MEG data analysis made use of SPM8 (Litvak et al., 2011, Wellcome Trust Centre for Neuroimaging, London, United Kingdom),
Fieldtrip (Oostenveld et al., 2011) Donders Institute for Brain Cognition and Behavior) and custom MATLAB code. MEG data
were downsampled to 1000 Hz, epoched for pre-processing into long trials spanning -2.8 to +12 s around the fractal cue to include
a baseline fixation, fractal cue, ‘Go’ cue, sequence production and feedback. A 48-52Hz stopband filter was then applied to remove
the 50 Hz power line noise within these long epochs. Channel artifacts were inspected in each participant, but no channels were
identified as corrupted in any of the datasets.

The EMG data were downsampled, epoched, and filtered in the same way as the
MEG data. No trials were removed from the dataset, so that the EMG data from the same trials as in the MEG dataset was submitted
to multivariate classification analysis.