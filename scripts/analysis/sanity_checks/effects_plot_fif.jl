if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
	python_exe = Sys.which("python3")
	python_exe === nothing || (ENV["JULIA_PYTHONCALL_EXE"] = python_exe)
end

"""
This is a simple modification to Bene's script that is now able to read in the .fif and generate an identical effects plot
"""

using CSV
using DataFrames
using CairoMakie
using Unfold
using UnfoldMakie
using StatsBase
using BSplineKit
using PyMNE

const OUTPUT_PNG = joinpath(@__DIR__, "effects_plot_fif_bads_removed.png")
const SFREQ = 500
const FIR_BASIS_WINDOW = (-0.5,1)
const PLOT_CHANNEL = 76

subject = "NDARUF540ZJ1"

function gen_paths(_subject)
	paths = Dict()
	paths["dataset"] = joinpath(@__DIR__, "sample_subject", _subject, "processed")
	paths["fif"] = joinpath(paths["dataset"],"sub-$(_subject)_task-freeView_run-4_proc-eyelink_raw.fif")
	return paths
end

function annotations_to_dataframe(raw_mne)
	ann_pd = raw_mne.annotations.to_data_frame()
	ann_dict = ann_pd.to_dict("list")
	ann_keys = pyconvert(Vector{String}, pybuiltins.list(ann_dict.keys()))
	ann_pairs = map(key -> Symbol(key) => pyconvert(Vector, ann_dict[key]), ann_keys)
	ann_df = DataFrame(ann_pairs)
	rename!(ann_df, Symbol.(names(ann_df)))
	ann_df.onset = pyconvert(Vector{Float64}, raw_mne.annotations.onset)
	ann_df.duration = pyconvert(Vector{Float64}, raw_mne.annotations.duration)
	ann_df.description = pyconvert(Vector{String}, raw_mne.annotations.description)
	return ann_df
end

function convert_et(et,sync_et_zero)
	et_saccades = subset(et,:description => (x->occursin.("ET_Saccade",x)))
	et_saccades[!,"Event Type"] .= replace.(et_saccades.description,"ET_"=>"")
	raw_first_time = sync_et_zero
	et_saccades.latency = (et_saccades.onset .- raw_first_time) .* SFREQ
	#et_saccades.Amplitude = collect(winsor(et_saccades.Amplitude;prop=0.10))
	small_ix = et_saccades.Amplitude.<0.5
	@info "number of small saccades <0.5 deg: $(sum(small_ix))"
	et_saccades = et_saccades[.!small_ix,:]
	thr = quantile(et_saccades.Amplitude,0.98)
	large_ix = et_saccades.Amplitude.>thr
	@info "clamped 98% large saccades >$(thr) deg: $(sum(large_ix))"
	et_saccades.Amplitude[large_ix] .= thr
	return et_saccades
end

function read_fif(_paths)
	raw = PyMNE.io.read_raw_fif(_paths["fif"],preload=true,verbose="ERROR")
	raw.info["bads"] = pybuiltins.list()
	ann_df = annotations_to_dataframe(raw)
	raw_first_time = pyconvert(Float64,raw.first_time)
	data = pyconvert(Array,raw.copy().load_data().pick("eeg").set_eeg_reference().get_data(units="uV"))
	sync_eeg = DataFrame(:onset=>pyconvert(Array,raw.annotations.onset),
:description=>pyconvert(Array,raw.annotations.description))
	return data,sync_eeg,ann_df,raw_first_time
end

function calc_sync_et(_et)
	sync_et = subset(_et,:description => (x->occursin.("# Message:",x)))
	sync_et_zero = 0
	sync_et,sync_et_zero
	end

function fit_model(_data,_et_saccades;amplitude_effect_type="spline")
	_evts = deepcopy(_et_saccades)
	subset!(_evts,:latency => x->(x.>0))

	f = if amplitude_effect_type == "spline"
		@formula(0~1+spl(Amplitude,4))
	elseif amplitude_effect_type == "linear"
		@formula(0~1+Amplitude)
	elseif amplitude_effect_type == "quadratic"
		@formula(0~1+Amplitude + Amplitude^2)
	elseif amplitude_effect_type == "none"
		@formula(0~1)
	else
		error("notimplemented")
	end

	m = fit(UnfoldModel,["Saccade L"=>(f,firbasis((-0.5,1),SFREQ))], _evts,_data;eventcolumn = "Event Type")
	return m
end

function analysis_samples(_data,_sync_eeg;raw_first_time=0,sfreq=SFREQ)
	stop_ix = findall(_sync_eeg.description .== "video_stop")
	if isempty(stop_ix)
		return size(_data,2)
	end
	return min(size(_data,2),floor(Int,minimum(_sync_eeg.onset[stop_ix] .- raw_first_time) * sfreq))
end

function exclude_outside_firbasis(_et_saccades,_data;basis_window=FIR_BASIS_WINDOW,sfreq=SFREQ,n_samples=size(_data,2))
	min_latency = -basis_window[1] * sfreq
	max_latency = n_samples - basis_window[2] * sfreq
	n_before = nrow(_et_saccades)
	_evts = subset(_et_saccades,:latency => x->((x .>= min_latency) .& (x .<= max_latency)))
	@info "excluded saccades outside firbasis window: $(n_before - nrow(_evts))"
	return _evts
end

function main()
	paths = gen_paths(subject)

	data,sync_eeg,et,raw_first_time = read_fif(paths)
	sync_et,sync_et_zero = calc_sync_et(et)
	et_saccades = convert_et(et,raw_first_time)
	et_saccades = exclude_outside_firbasis(et_saccades,data;n_samples=analysis_samples(data,sync_eeg;raw_first_time=raw_first_time))

	m = fit_model(data,et_saccades,amplitude_effect_type="spline")

	effect_df = subset(dropmissing(effects(Dict(:Amplitude=>1:2:20),m)),:channel => ByRow(==(PLOT_CHANNEL)))
	fig = plot_erp(effect_df;mapping=(;color=:Amplitude,group=:Amplitude))
	save(OUTPUT_PNG, fig)
	@info "saved effects plot" OUTPUT_PNG

	return m
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
	main()
end
