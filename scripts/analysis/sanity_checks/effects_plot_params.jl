using DataFrames
using CairoMakie
using Unfold
using UnfoldMakie
using StatsBase
using BSplineKit

if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
    python_exe = Sys.which("python3")
    python_exe === nothing || (ENV["JULIA_PYTHONCALL_EXE"] = python_exe)
end

"""
This is a more complex variant of effects_plot_fif that additionally allows for toggling between various parameters. In its default configuration, it will still generate the exact same model as Bene's script.  For the analyses done in the paper, not all default settings were used. The parameters that deviate are:

CLAMP_LIMIT_EYE = ["L"], because top saccades of the right eye should not impact which saccades of the left eye to consider

IGNORE_BAD_CHANNELS = false, bad channel markings should not be ignored (impacts rereferencing)

EXCLUDE_EDGE_SACCADES = false, when set to true, it only includes saccades that happened within the paradigm window (e.g. during the movie). In order to have more data for averaging, this was set to false

USE_GPU_SOLVER = true, only makes a negligible difference in result for massive speed gain

This also implements USE_FIXATION_ONSET, to allow comparison between the two onset alternatives
"""

using PythonCall
using CUDA
using CUDA.CUSPARSE

const EVENT_EYE = ["L"] # ["L"] or ["L", "R"]
const CLAMP_LIMIT_EYE = ["L", "R"] # ["L", "R"] = all saccades, ["L"] or ["R"] = one eye
const py_mne = pyimport("mne")
const OUTPUT_DIR = "."
const OUTPUT_PNG = joinpath(OUTPUT_DIR, "modified_effects_plot_NDARUF540ZJ1.png")
const DEFAULT_FIF_PATH = "sample_subject/NDARUF540ZJ1/processed/sub-NDARUF540ZJ1_task-freeView_run-4_proc-eyelink_raw.fif"
const IGNORE_BAD_CHANNELS = true
const EXCLUDE_EDGE_SACCADES = true
const USE_GPU_SOLVER = false
const USE_FIXATION_ONSET = false

"""
For Fixation Onset:
Search for Saccades and note the amplitude
Search forward for the next same-eye fixation.
Ignore opposite-eye annotations in between.
Drop the event if another same-eye saccade or blink happens before that fixation.
Use that fixation onset as the latency.
"""


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


# Workaround for Unfold GPU-QR path with current CUDA sparse matrices:
# Unfold.prepare_XTX currently tries CuArray{T,...}(CuSparseMatrixCSC), which can
# trigger scalar-indexing errors. Convert via CuArray(...) explicitly.
import Unfold: prepare_XTX
function Unfold.prepare_XTX(Ĥ, data::CuArray{T}, X::CuSparseMatrixCSC{T2}) where {T,T2}
    Xt = X'
    R_xx = CuArray(Xt * X)
    R_xy = similar(data, size(X, 2))
    return Ĥ, data, (Xt, R_xx, R_xy)
end

function read_fif(fif_path::AbstractString)
    raw_mne = py_mne.io.read_raw_fif(fif_path, preload = true, verbose = "ERROR")
    ann_df = annotations_to_dataframe(raw_mne)
    raw_first_time = pyconvert(Float64, raw_mne.first_time)
    eeg_only = raw_mne.copy().load_data().pick("eeg")
    if IGNORE_BAD_CHANNELS
        eeg_only.info["bads"] = pybuiltins.list()
    end
    eeg_only.set_eeg_reference()
    eeg_data = pyconvert(Array{Float64, 2}, eeg_only.get_data(units = "uV"))
    eeg_ch_names = pyconvert(Vector{String}, eeg_only.ch_names)
    sfreq = pyconvert(Float64, eeg_only.info["sfreq"])

    return eeg_data, eeg_ch_names, sfreq, ann_df, raw_first_time
end

function prepare_saccade_events(
    ann_df::DataFrame,
    sfreq::Real,
    raw_first_time::Real;
    eye = EVENT_EYE,
    clamp_eye = CLAMP_LIMIT_EYE,
)
    sorted_ann = sort(copy(ann_df), :onset)

    saccades = if USE_FIXATION_ONSET
        descriptions = string.(sorted_ann.description)
        saccade_idx = Int[]
        fixation_onsets = Float64[]

        for i in eachindex(descriptions)
            startswith(descriptions[i], "ET_Saccade ") || continue
            eye_name = replace(descriptions[i], "ET_Saccade " => "")
            fixation_description = "ET_Fixation $eye_name"
            next_saccade_description = "ET_Saccade $eye_name"
            blink_description = "ET_Blink $eye_name"

            for j in (i + 1):lastindex(descriptions)
                if descriptions[j] == fixation_description
                    push!(saccade_idx, i)
                    push!(fixation_onsets, sorted_ann.onset[j])
                    break
                elseif descriptions[j] == next_saccade_description || descriptions[j] == blink_description
                    break
                end
            end
        end

        isempty(saccade_idx) && error("USE_FIXATION_ONSET=true, but no saccade/fixation pairs were found.")
        paired_saccades = copy(sorted_ann[saccade_idx, :])
        paired_saccades.onset = fixation_onsets
        paired_saccades
    else
        subset(sorted_ann, :description => ByRow(x -> occursin("ET_Saccade", x)))
    end
    saccades = subset(saccades, :Amplitude => ByRow(x -> !ismissing(x) && isfinite(Float64(x))))

    saccade_events = DataFrame(
        latency = (saccades.onset .- raw_first_time) .* sfreq,
        Amplitude = Float64.(saccades.Amplitude),
    )
    saccade_events[!, "Event Type"] = replace.(saccades.description, "ET_" => "")
    saccade_events = subset(
        saccade_events,
        :latency => ByRow(>(0)),
        :Amplitude => ByRow(>=(0.5)),
    )
    clamp_event_types = Set("Saccade $eye_name" for eye_name in clamp_eye)
    event_types = Set("Saccade $eye_name" for eye_name in eye)

    clamp_source = subset(saccade_events, "Event Type" => ByRow(x -> x in clamp_event_types),)
    saccade_events = subset(saccade_events, "Event Type" => ByRow(x -> x in event_types),)

    clamp_threshold = quantile(clamp_source.Amplitude, 0.98)
    large_mask = saccade_events.Amplitude .> clamp_threshold
    saccade_events.Amplitude[large_mask] .= clamp_threshold

    return saccade_events
end

function analysis_samples(ann_df::DataFrame, n_samples::Integer, raw_first_time::Real, sfreq::Real)
    stop_ix = findall(ann_df.description .== "video_stop")
    isempty(stop_ix) && return n_samples
    return min(n_samples, floor(Int, minimum(ann_df.onset[stop_ix] .- raw_first_time) * sfreq))
end

function exclude_outside_firbasis(
    saccade_events::DataFrame,
    n_samples::Integer;
    basis_window = (-0.5, 1),
    sfreq::Real,
)
    min_latency = -basis_window[1] * sfreq
    max_latency = n_samples - basis_window[2] * sfreq
    return subset(saccade_events, :latency => ByRow(x -> min_latency <= x <= max_latency))
end

function fit_single_file_model(
    fif_path::AbstractString,
    eye = EVENT_EYE,
)
    eeg_data, _, sfreq, ann_df, raw_first_time = read_fif(fif_path)
    saccade_events = prepare_saccade_events(ann_df, sfreq, raw_first_time; eye = eye)
    if EXCLUDE_EDGE_SACCADES
        n_samples = analysis_samples(ann_df, size(eeg_data, 2), raw_first_time, sfreq)
        saccade_events = exclude_outside_firbasis(saccade_events, n_samples; sfreq = sfreq)
    end

    if USE_GPU_SOLVER
        gpu_solver = (x, y) -> Unfold.solver_predefined(x, y; solver = :qr)
        return fit(
            UnfoldModel,
            [
                "Saccade $eye_name" => (@formula(0 ~ 1 + spl(Amplitude, 4)), firbasis((-0.5, 1), sfreq)) for
                eye_name in eye
            ],
            saccade_events,
            CUDA.cu(eeg_data),
            solver = gpu_solver,
            eventcolumn = "Event Type",
        )
    end

    model = fit(
        UnfoldModel,
        [
            "Saccade $eye_name" => (@formula(0 ~ 1 + spl(Amplitude, 4)), firbasis((-0.5, 1), sfreq)) for
            eye_name in eye
        ],
        saccade_events,
        eeg_data,
        eventcolumn = "Event Type",
    )

    return model
end


function plot_model_effects(model)
    effect_df = dropmissing(effects(Dict(:Amplitude => 1:2:20), model))
    effect_df = subset(effect_df, :channel => ByRow(==(76)))
    return plot_erp(effect_df; mapping = (; color = :Amplitude, group = :Amplitude))
end


function main(fif_path::AbstractString = DEFAULT_FIF_PATH; output_png::AbstractString = OUTPUT_PNG)
    try
        model = fit_single_file_model(fif_path)
        fig = plot_model_effects(model)
        mkpath(OUTPUT_DIR)
        save(output_png, fig)
        display(fig)
        return model
    catch err
        showerror(stderr, err)
        println(stderr)
        return nothing
    end
end

function load_model(jld2_path::AbstractString)
    return Unfold.load(jld2_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
