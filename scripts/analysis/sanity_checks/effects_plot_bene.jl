if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
	python_exe = Sys.which("python3")
	python_exe === nothing || (ENV["JULIA_PYTHONCALL_EXE"] = python_exe)
end

"""
This script is mostly a copy of Benedikt Ehinger's prototype implementation, put into an executable script.
It uses the raw files and synchronizes them manually.
To make this exactly match with the effects plot created from the .fif, we add artificial shift and scale to the ET data stream, in order to match the latencies of the .fif exactly.
The sample recording used here already has very good synchronization.
"""

using CSV
using DataFrames
using CairoMakie
using Unfold
using UnfoldMakie
using StatsBase
using BSplineKit
using PyMNE

const OUTPUT_PNG = joinpath(@__DIR__, "effects_plot.png")
const SFREQ = 500
const FIR_BASIS_WINDOW = (-0.5,1)
const ET_ARTIFICIAL_SCALE = 1.000019217706295
const ET_ARTIFICIAL_SHIFT_SECONDS = -0.004179796721704972
const EEG_CROP_START_SAMPLES = 2
const PLOT_CHANNEL = 76

subject = "NDARUF540ZJ1"

function gen_paths(_subject)
	paths = Dict()
	paths["dataset"] = joinpath(@__DIR__, "sample_subject", _subject, "unprocessed")
	paths["eeg"] = joinpath(paths["dataset"],"sub-$(_subject)_task-DiaryOfAWimpyKid_eeg.set")
	paths["events"] = joinpath(paths["dataset"],"sub-$(_subject)_task-DiaryOfAWimpyKid_events.tsv")
	paths["et"] = joinpath(paths["dataset"],"$(_subject)_Video-WK_Events.txt")
	return paths
end

function convert_et(et,sync_et_zero)
	et_saccades = subset(et,:Column1 => (x->occursin.("Sacca",x)))[2:end,:]
	rename!(et_saccades,collect(et[4,:]))
	parselist = (:Trial=>Int,:Number=>Int,:Start=>Int,:End=>Int,:Duration=>Int,Symbol("Start Loc.X")=>Float64,Symbol("Start Loc.Y")=>Float64,Symbol("End Loc.X")=>Float64,Symbol("End Loc.Y")=>Float64,
	Symbol("Amplitude")=>Float64,Symbol("Peak Speed")=>Float64,
	Symbol("Peak Accel.")=>Float64,Symbol("Peak Speed At")=>Float64,
	Symbol("Peak Decel.")=>Float64,Symbol("Average Speed")=>Float64,
	Symbol("Average Accel.")=>Float64,)
	for (k,t) = parselist
		#@info k,t
		et_saccades[!,k] = parse.(t,et_saccades[:,k])
	end

	et_saccades.Start = (et_saccades.Start .- sync_et_zero) ./ 1000 ./ 1000
	et_saccades.End = (et_saccades.End .- sync_et_zero) ./ 1000 ./ 1000
	et_saccades.latency = et_saccades.Start .* SFREQ

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

function apply_et_timing!(_et_saccades;scale=ET_ARTIFICIAL_SCALE,shift_seconds=ET_ARTIFICIAL_SHIFT_SECONDS,sfreq=SFREQ)
	if scale != 1 || shift_seconds != 0
		_et_saccades.Start .= _et_saccades.Start .* scale .+ shift_seconds
		_et_saccades.End .= _et_saccades.End .* scale .+ shift_seconds
		_et_saccades.latency .= _et_saccades.Start .* sfreq
		@info "applied artificial ET timing" scale shift_seconds
	end
	return _et_saccades
end

function apply_eeg_crop(_data;start_samples=EEG_CROP_START_SAMPLES,sfreq=SFREQ)
	if start_samples == 0
		return _data,0.0
	end
	@info "cropped EEG start samples" start_samples
	return _data[:,(start_samples+1):end],start_samples / sfreq
end

function read_et(_paths)
	return CSV.read(_paths["et"],DataFrame,header=0,skipto=8,delim="\t",silencewarnings=true)
end

function read_eeg(_paths)
	eeglabdata = PyMNE.io.read_raw_eeglab(_paths["eeg"])
	eeg_only = eeglabdata.copy().load_data().drop_channels(["Cz"])
	data = pyconvert(Array,eeg_only.notch_filter([60,100]).filter(0.5,100).set_eeg_reference().get_data(units="uV"))
	sync_eeg = DataFrame(:onset=>pyconvert(Array,eeglabdata.annotations.onset), :description=>pyconvert(Array,eeglabdata.annotations.description))
	return data,sync_eeg
end

function calc_sync_et(_et)
	_ix = (_et.Column1 .== "UserEvent") .& occursin.("# Message:", coalesce.(_et.Column5,""))
	sync_et = DataFrame(:onset=>_et[_ix,:Column4]|>x->parse.(Int,x),:description=>replace.(_et[_ix,:Column5],"# Message: "=>"")|>x->parse.(Int,x))
	sync_et_zero = sync_et.onset[1]
	sync_et.onset = sync_et.onset .- sync_et_zero
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

	et = read_et(paths)
	sync_et,sync_et_zero = calc_sync_et(et)
	et_saccades = convert_et(et,sync_et_zero)
	apply_et_timing!(et_saccades)
	data,sync_eeg = read_eeg(paths)
	data,eeg_crop_seconds = apply_eeg_crop(data)
	et_saccades = exclude_outside_firbasis(et_saccades,data;n_samples=analysis_samples(data,sync_eeg;raw_first_time=eeg_crop_seconds))

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
