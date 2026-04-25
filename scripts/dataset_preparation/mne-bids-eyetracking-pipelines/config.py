import mne

#0 = symbolsearch
#1 = freeView (all free view merged)
#2 = allTasks (all tasks merged)
EXPERIMENT = 1
USE_ICA = True

bids_root = "/path/to/mergedDataset"
deriv_root = bids_root + "/derivatives"
subjects_dir = None

subjects = "all"

ch_types = ["eeg"]
interactive = False
sessions = []
task = ["symbolsearch","freeView","allTasks"][EXPERIMENT]

runs = "all"
et_has_run = True
et_has_task = True

task_is_rest = True
epochs_tmin = 0 #forced for rest 
epochs_tmax = 1
rest_epochs_duration = 1 #forced for rest, make small so not too much gets rejected
rest_epochs_overlap = 0 #forced to be set
baseline = None # for later Unfold analysis

#raw_resample_sfreq: float | None = 250
# plenty of papers use these values for HBN EEG
# dimigen+ehinger paper uses 0.1 Hz for their FRPs
l_freq: float | None = 0.5
h_freq: float | None = 100

# data was recorded in the US
detect_freqs = [20, 60, 80, 100]

# determined by icalabel
ica_h_freq: float | None = 100
ica_l_freq = 1
eeg_reference = "average"


on_error = "continue"

######### Condition analysis, not suitable for unfold ############

# positive / negative feedback
#conditions = ["HAPPY", "SAD"]
#conditions = ["# Message: 12XX", "# Message: 13XX"]

#epochs_tmin: float = -0.5
#epochs_tmax: float = 2.6 # since feedback is so infrequent, long epochs are okay?

#baseline: tuple[float | None, float | None] | None = (-0.2, 0)
###############################################################


spatial_filter = None
ica_use_icalabel = False

if USE_ICA:
    spatial_filter = "ica"
    # ica_n_components = 96 ?
    # icalabel forces picard-extended_infomax or extended_infomax
    ica_algorithm = "picard-extended_infomax"
    ica_use_ecg_detection: bool = False
    ica_use_eog_detection: bool = False
    ica_use_icalabel = True
    


# other common values like 200 drop too many epochs for free viewing tasks
# this way we can retain epochs with blinks
ica_reject = dict(eeg=400e-6) #600 seems to be too high
#reject = "autoreject_local"

sync_eyelink = True
#sync_eventtype_regex = "\\d-trigger=10 Image moves"
#sync_eventtype_regex_et = "trigger=10 Image moves"

# unfortunately, syncing over full freeView is not possible
# as ET timestamps are absolute (start at 0) instead of being global from the full session

sync_eventtype_regex     = r"(?:instructed_toOpenEyes|instructed_toCloseEyes|fixpoint_ON|stim_ON|left_buttonPress|right_buttonPress|left_target|right_target|dot_no1_ON|dot_no2_ON|dot_no3_ON|dot_no4_ON|dot_no5_ON|dot_no6_ON|dot_no1_OFF|dot_no2_OFF|dot_no3_OFF|dot_no4_OFF|dot_no5_OFF|dot_no6_OFF|dot_no1_ON|dot_no2_ON|dot_no3_ON|dot_no4_ON|dot_no5_ON|dot_no6_ON|dot_no7_ON|dot_no8_ON|dot_no1_OFF|dot_no2_OFF|dot_no3_OFF|dot_no4_OFF|dot_no5_OFF|dot_no6_OFF|dot_no7_OFF|dot_no8_OFF|trialResponse|newPage|video_start|video_stop|video_start|video_stop|video_start|video_stop|video_start|video_stop)"
sync_eventtype_regex_et  = r"# Message: (?:4|8|9|11|12|13|14|15|16|17|18|20|21|22|23|24|25|26|27|28|30|81|82|83|84|101|102|103|104)"

eeg_bipolar_channels = {
    "HEOG": (["E8","E125","E109"], ["E25","E128","E40"]),   # left vs right outer canthus
    "VEOG": (["E21","E14","E22","E9"], ["E17","E126","E127"]),  # nasion vs left inner canthus
}
#eog channels: 8, 14, 17, 21, 25, 125, 126, 127, 128

eog_channels = ["HEOG", "VEOG"]
sync_heog_ch = "HEOG"
sync_et_ch = ("L POR X [px]", "R POR X [px]")

sync_plot_samps = 3000 # width of the xcorr plot

decode: bool = False
run_source_estimation = False

# montage is a GSN-HydroCel-128, but electrode positions are baked into the .set
# the baked-in positions are identical for all subjects (not fully tested)
#montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
#eeg_template_montage = montage

drop_channels = ["Cz"]

n_jobs = 1
