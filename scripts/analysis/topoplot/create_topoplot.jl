using DataFrames
using Statistics
using CairoMakie
using Unfold
using UnfoldMakie

"""
By default, this creates the 3x5 overview of topoplots for the "ThePresent" recorings (freeView run 3)

Entirely LLM written
"""

const CONDAPKG_ENV_DIR = joinpath(@__DIR__, ".CondaPkg", "env")
const CONDAPKG_PYTHON_EXE = joinpath(CONDAPKG_ENV_DIR, "bin", "python")
const CONDAPKG_PYTHON_LIB = joinpath(CONDAPKG_ENV_DIR, "lib", "libpython3.so")

isfile(CONDAPKG_PYTHON_EXE) || error(
    "Expected CondaPkg Python executable at $(CONDAPKG_PYTHON_EXE). " *
    "Please instantiate the topoplots project so .CondaPkg is created.",
)
isfile(CONDAPKG_PYTHON_LIB) || error(
    "Expected CondaPkg Python library at $(CONDAPKG_PYTHON_LIB). " *
    "Please instantiate the topoplots project so .CondaPkg is created.",
)

ENV["JULIA_PYTHONCALL_EXE"] = CONDAPKG_PYTHON_EXE
ENV["JULIA_PYTHONCALL_LIB"] = CONDAPKG_PYTHON_LIB

using PyMNE
using PythonCall

include(joinpath(@__DIR__, "run3_paths.jl"))

function parse_run3_input_kind(value::AbstractString)
    kind = Symbol(lowercase(strip(value)))
    kind == :raw && return :eyelink
    kind in (:clean, :eyelink) ||
        error("Unsupported RUN3_INPUT_KIND: $(value). Expected clean, raw, eyelink, or both.")
    return kind
end

function parse_run3_input_kinds(value::AbstractString)
    stripped = lowercase(strip(value))
    if stripped in ("both", "all")
        return [:clean, :eyelink]
    end

    parsed_kinds = Symbol[]
    for part in split(value, ',')
        stripped_part = strip(part)
        isempty(stripped_part) && continue
        kind = parse_run3_input_kind(stripped_part)
        kind in parsed_kinds || push!(parsed_kinds, kind)
    end

    isempty(parsed_kinds) &&
        error("RUN3_INPUT_KIND did not contain any supported input kinds: $(value)")
    return parsed_kinds
end

function parse_analysis_mode(value::AbstractString)
    mode = Symbol(lowercase(strip(value)))
    if mode in (:allvariants, :all_variants)
        return :all_variants
    elseif mode == :exploratory
        return :exploratory
    end
    error(
        "Unsupported TOPOPLOT_ANALYSIS_MODE: $(value). Expected exploratory or all_variants.",
    )
end

function parse_lock_to(value::AbstractString)
    lock_to = Symbol(lowercase(strip(value)))
    lock_to in (:saccade, :fixation) ||
        error("Unsupported TOPOPLOT_LOCK_TO: $(value). Expected saccade or fixation.")
    return lock_to
end

const RUN3_INPUT_KINDS = parse_run3_input_kinds(get(ENV, "RUN3_INPUT_KIND", "both"))
const TOPOPLOT_ANALYSIS_MODE = parse_analysis_mode(
    get(ENV, "TOPOPLOT_ANALYSIS_MODE", "exploratory"),
)

const EVENT_EYE = ["L"]
const LOCK_OPTIONS = [:saccade, :fixation]
const DIRECTION_OPTIONS = [:all, :ltr]
const BASELINE_CORRECTION_OPTIONS = [false, true]
const DROP_SMALL_SACCADE_OPTIONS = [false, true]

const DEFAULT_LOCK_TO = parse_lock_to(get(ENV, "TOPOPLOT_LOCK_TO", "saccade"))
const DEFAULT_DIRECTION = :ltr
const DEFAULT_BASELINE_CORRECTION = false
const DEFAULT_DROP_SMALL_SACCADES = true

const EXPLORATORY_VARIANTS = [
    (
        DEFAULT_LOCK_TO,
        DEFAULT_BASELINE_CORRECTION,
        DEFAULT_DIRECTION,
        DEFAULT_DROP_SMALL_SACCADES,
    ),
]
const VARIANT_OPTIONS = if TOPOPLOT_ANALYSIS_MODE == :exploratory
    EXPLORATORY_VARIANTS
else
    collect(
        Iterators.product(
            LOCK_OPTIONS,
            BASELINE_CORRECTION_OPTIONS,
            DIRECTION_OPTIONS,
            DROP_SMALL_SACCADE_OPTIONS,
        ),
    )
end

const BIN_NUM = 15
const TOPOS_PER_ROW = 5
const NROWS = cld(BIN_NUM, TOPOS_PER_ROW)
const FIGURE_SIZE = (TOPOS_PER_ROW * 340, NROWS * 340 + 180)
const SINGLE_WINDOW_FIGURE_SIZE = (500, 420)
const COMPARISON_FIGURE_SIZE = (880, 420)
const EPOCH_ABS_THRESHOLD_UV = 100
const SACCADE_AMPLITUDE_THRESHOLD = 0.5
const SACCADE_START_X_COLUMN = Symbol("Start Loc.X")
const SACCADE_END_X_COLUMN = Symbol("End Loc.X")
const WINDOW_SPECS = [
    (name = "artifact", start_s = 0.0, stop_s = 1 / 15),
    (name = "lambda", start_s = 0.13, stop_s = 0.2),
]
const CHANNEL_RANK_WINDOW_NAME = "lambda"
const WARNED_ABOUT_MISSING_AMPLITUDE = Ref(false)
const WARNED_ABOUT_MISSING_DIRECTION_COLUMNS = Ref(false)

function fif_paths_for_input_kind(input_kind::Symbol)
    if input_kind == :clean
        return FREEVIEW_RUN3_CLEAN_FIF_PATHS
    elseif input_kind == :eyelink
        return FREEVIEW_RUN3_EYELINK_FIF_PATHS
    end
    error("Unsupported input kind: $(input_kind)")
end

output_input_kind_label(input_kind::Symbol) = input_kind == :eyelink ? "raw" : "clean"
title_input_kind_label(input_kind::Symbol) = input_kind == :eyelink ? "Raw" : "Clean"

function normalize_saccade_direction(direction::Symbol)
    direction in (:all, :ltr, :rtl) ||
        error(
            "Unsupported SACCADE_DIRECTION: $direction. Expected one of :all, :ltr, :rtl.",
        )
    return direction
end

function filename_number_component(value::Real)
    rounded_value = round(Float64(value); digits = 3)
    rounded_integer = round(Int, rounded_value)
    if isapprox(rounded_value, rounded_integer; atol = 1e-9)
        return string(rounded_integer)
    end
    return replace(string(rounded_value), "." => "p", "-" => "m")
end

filename_bool_component(value::Bool) = value ? "on" : "off"

function display_number_component(value::Real)
    rounded_value = round(Float64(value); digits = 3)
    rounded_integer = round(Int, rounded_value)
    if isapprox(rounded_value, rounded_integer; atol = 1e-9)
        return string(rounded_integer)
    end
    return string(rounded_value)
end

function exploratory_output_stem(
    input_kind::Symbol;
    lock_to::Symbol,
    direction::Symbol,
    baseline_correction::Bool,
    drop_small_saccades::Bool,
    epoch_abs_threshold_uv::Real = EPOCH_ABS_THRESHOLD_UV,
)
    threshold_label = filename_number_component(epoch_abs_threshold_uv)
    baseline_label = filename_bool_component(baseline_correction)
    amplitude_label = filename_bool_component(drop_small_saccades)
    return "exploratory_$(output_input_kind_label(input_kind))_lock-$(lock_to)_blcorr-$(baseline_label)_abs$(threshold_label)uV_dir-$(direction)_ampfilt-$(amplitude_label)"
end

function comparison_output_stem(
    input_kinds::AbstractVector{<:Symbol};
    lock_to::Symbol,
    direction::Symbol,
    baseline_correction::Bool,
    drop_small_saccades::Bool,
    epoch_abs_threshold_uv::Real = EPOCH_ABS_THRESHOLD_UV,
)
    threshold_label = filename_number_component(epoch_abs_threshold_uv)
    baseline_label = filename_bool_component(baseline_correction)
    amplitude_label = filename_bool_component(drop_small_saccades)
    input_label = join(output_input_kind_label.(input_kinds), "-vs-")
    return "exploratory_compare_$(input_label)_lock-$(lock_to)_blcorr-$(baseline_label)_abs$(threshold_label)uV_dir-$(direction)_ampfilt-$(amplitude_label)"
end

function window_filename_component(window_spec)
    start_label = filename_number_component(window_spec.start_s)
    stop_label = filename_number_component(window_spec.stop_s)
    return "$(window_spec.name)_$(start_label)to$(stop_label)s"
end

function window_display_label(window_spec)
    start_label = display_number_component(window_spec.start_s)
    stop_label = display_number_component(window_spec.stop_s)
    return "[$(start_label), $(stop_label)) s"
end

function normalize_lock_to(lock_to::Symbol)
    lock_to in LOCK_OPTIONS ||
        error("Unsupported lock option: $lock_to. Expected one of $(LOCK_OPTIONS).")
    return lock_to
end

function lock_label(lock_to::Symbol)
    lock_to = normalize_lock_to(lock_to)
    return lock_to == :saccade ? "saccade onset" : "fixation onset"
end

passes_saccade_amplitude_threshold(value) =
    !ismissing(value) && isfinite(Float64(value)) &&
    Float64(value) >= SACCADE_AMPLITUDE_THRESHOLD
valid_coordinate(value) = !ismissing(value) && isfinite(Float64(value))

function warn_missing_amplitude_once()
    if !WARNED_ABOUT_MISSING_AMPLITUDE[]
        @warn "Saccade annotations are missing :Amplitude; skipping amplitude-filtered variants for affected FIF(s)."
        WARNED_ABOUT_MISSING_AMPLITUDE[] = true
    end
end

function warn_missing_direction_columns_once()
    if !WARNED_ABOUT_MISSING_DIRECTION_COLUMNS[]
        @warn "Saccade annotations are missing direction columns; skipping left-to-right variants for affected FIF(s)."
        WARNED_ABOUT_MISSING_DIRECTION_COLUMNS[] = true
    end
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

function has_active_average_reference(raw_mne)
    projs = pyconvert(Vector{Py}, raw_mne.info["projs"])
    return any(projs) do proj
        pyconvert(Bool, proj["active"]) &&
            pyconvert(String, proj["desc"]) == "Average EEG reference"
    end
end

function has_custom_reference(raw_mne)
    custom_ref = raw_mne.info["custom_ref_applied"]

    try
        return pyconvert(Bool, custom_ref)
    catch
    end

    try
        return pyconvert(Int, custom_ref) != 0
    catch
    end

    return false
end

function ensure_average_reference!(raw_mne)
    if has_active_average_reference(raw_mne) || has_custom_reference(raw_mne)
        return raw_mne
    end

    raw_mne.set_eeg_reference(verbose = "ERROR")
    return raw_mne
end

function align_eeg_to_layout(eeg_only, eeg_data::AbstractMatrix, eeg_ch_names::Vector{String})
    layout_kwargs = (; exclude = pybuiltins.list(String[]))
    layout = PyMNE.channels.make_eeg_layout(eeg_only.info; layout_kwargs...)
    layout_names = pyconvert(Vector{String}, layout.names)
    positions = to_positions(eeg_only; layout_kwargs...)

    length(layout_names) == length(positions) ||
        error("Layout channel names and positions are inconsistent.")

    aligned_data = fill(NaN, length(layout_names), size(eeg_data, 2))
    eeg_index = Dict(name => idx for (idx, name) in pairs(eeg_ch_names))

    for (layout_idx, layout_name) in pairs(layout_names)
        data_idx = get(eeg_index, layout_name, nothing)
        isnothing(data_idx) && continue
        aligned_data[layout_idx, :] = eeg_data[data_idx, :]
    end

    bad_channels = Set(pyconvert(Vector{String}, eeg_only.info["bads"]))
    for (layout_idx, layout_name) in pairs(layout_names)
        layout_name in bad_channels || continue
        aligned_data[layout_idx, :] .= NaN
    end

    return aligned_data, layout_names, positions
end

function read_fif(fif_path::AbstractString)
    raw_mne = PyMNE.io.read_raw_fif(fif_path, preload = true, verbose = "ERROR")
    ann_df = annotations_to_dataframe(raw_mne)
    eeg_only = raw_mne.copy().pick("eeg")
    ensure_average_reference!(eeg_only)
    eeg_data = pyconvert(Array{Float64, 2}, eeg_only.get_data(units = "uV"))
    eeg_ch_names = pyconvert(Vector{String}, eeg_only.ch_names)
    sfreq = pyconvert(Float64, eeg_only.info["sfreq"])
    eeg_data, eeg_ch_names, positions = align_eeg_to_layout(eeg_only, eeg_data, eeg_ch_names)
    return eeg_data, eeg_ch_names, sfreq, ann_df, positions
end

function prepare_saccade_events(
    ann_df::DataFrame,
    sfreq::Real;
    eye = EVENT_EYE,
    direction::Symbol = :all,
    drop_small_saccades::Bool = false,
)
    direction = normalize_saccade_direction(direction)
    has_amplitude_column = :Amplitude in propertynames(ann_df)
    saccades = subset(
        sort(copy(ann_df), :onset),
        :description => ByRow(x -> occursin("ET_Saccade", x)),
    )

    if drop_small_saccades && has_amplitude_column
        saccades = subset(
            saccades,
            :Amplitude => ByRow(passes_saccade_amplitude_threshold),
        )
    elseif drop_small_saccades && !has_amplitude_column
        warn_missing_amplitude_once()
        return DataFrame(latency = Int[])
    end

    if direction != :all
        required_cols = Set([SACCADE_START_X_COLUMN, SACCADE_END_X_COLUMN])
        available_cols = Set(propertynames(ann_df))
        missing_cols = sort!(collect(setdiff(required_cols, available_cols)))
        if !isempty(missing_cols)
            warn_missing_direction_columns_once()
            return DataFrame(latency = Int[])
        end

        saccades = subset(
            saccades,
            SACCADE_START_X_COLUMN => ByRow(valid_coordinate),
            SACCADE_END_X_COLUMN => ByRow(valid_coordinate),
        )
    end

    saccade_events = DataFrame(
        latency = round.(Int, saccades.onset .* sfreq) .+ 1,
        Amplitude = has_amplitude_column ? Float64.(saccades[!, :Amplitude]) : fill(
            NaN,
            nrow(saccades),
        ),
    )
    saccade_events[!, "Event Type"] = replace.(saccades.description, "ET_" => "")
    if direction != :all
        saccade_events.start_loc_x = Float64.(saccades[!, SACCADE_START_X_COLUMN])
        saccade_events.end_loc_x = Float64.(saccades[!, SACCADE_END_X_COLUMN])
        saccade_events.horizontal_delta_x =
            saccade_events.end_loc_x .- saccade_events.start_loc_x
    end

    event_types = Set("Saccade $eye_name" for eye_name in eye)
    saccade_events = subset(
        saccade_events,
        :latency => ByRow(>(0)),
        "Event Type" => ByRow(x -> x in event_types),
    )

    if direction == :ltr
        saccade_events = subset(saccade_events, :horizontal_delta_x => ByRow(>(0)))
    elseif direction == :rtl
        saccade_events = subset(saccade_events, :horizontal_delta_x => ByRow(<(0)))
    end

    if direction != :all
        select!(saccade_events, Cols(:latency, :Amplitude, "Event Type"))
    end

    return saccade_events
end

function sorted_eye_annotations(ann_df::DataFrame, eye_name::AbstractString)
    eye_suffix = " $eye_name"
    return subset(
        sort(copy(ann_df), :onset),
        :description => ByRow(x -> !ismissing(x) && endswith(String(x), eye_suffix)),
    )
end

function prepare_fixation_events(
    ann_df::DataFrame,
    sfreq::Real;
    eye = EVENT_EYE,
    direction::Symbol = :all,
    drop_small_saccades::Bool = false,
)
    direction = normalize_saccade_direction(direction)
    has_amplitude_column = :Amplitude in propertynames(ann_df)

    if drop_small_saccades && !has_amplitude_column
        warn_missing_amplitude_once()
        return DataFrame(latency = Int[])
    end

    if direction != :all
        required_cols = Set([SACCADE_START_X_COLUMN, SACCADE_END_X_COLUMN])
        available_cols = Set(propertynames(ann_df))
        missing_cols = sort!(collect(setdiff(required_cols, available_cols)))
        if !isempty(missing_cols)
            warn_missing_direction_columns_once()
            return DataFrame(latency = Int[])
        end
    end

    fixation_latencies = Int[]

    for eye_name in eye
        eye_ann = sorted_eye_annotations(ann_df, eye_name)
        descriptions = String.(eye_ann.description)
        saccade_desc = "ET_Saccade $eye_name"
        fixation_desc = "ET_Fixation $eye_name"

        for idx in eachindex(descriptions)
            descriptions[idx] == fixation_desc || continue
            idx > firstindex(descriptions) || continue
            descriptions[idx - 1] == saccade_desc || continue

            if drop_small_saccades
                passes_saccade_amplitude_threshold(eye_ann[idx - 1, :Amplitude]) || continue
            end

            if direction != :all
                start_x = eye_ann[idx - 1, SACCADE_START_X_COLUMN]
                end_x = eye_ann[idx - 1, SACCADE_END_X_COLUMN]
                valid_coordinate(start_x) || continue
                valid_coordinate(end_x) || continue

                if direction == :ltr
                    Float64(end_x) > Float64(start_x) || continue
                elseif direction == :rtl
                    Float64(end_x) < Float64(start_x) || continue
                end
            end

            latency = round(Int, eye_ann.onset[idx] .* sfreq) + 1
            latency > 0 || continue
            push!(fixation_latencies, latency)
        end
    end

    return DataFrame(latency = fixation_latencies)
end

function collect_epochs(
    eeg_data::AbstractMatrix,
    latencies::AbstractVector{<:Integer},
    sfreq::Real;
    tmin = -0.2,
    tmax = 0.8,
    baseline_correction::Bool = false,
)
    isempty(latencies) &&
        return Array{Float64}(undef, size(eeg_data, 1), 0, 0), Float64[], 0

    events = DataFrame(latency = latencies)
    epochs, times = Unfold.epoch(data = eeg_data, tbl = events, τ = (tmin, tmax), sfreq = sfreq)
    time_ms = collect(times) .* 1000
    _, epochs = Unfold.drop_missing_epochs(events, epochs)
    size(epochs, 3) > 0 || return epochs, time_ms, 0

    if baseline_correction
        baseline_idx = findall(time_ms .<= 0)
        isempty(baseline_idx) &&
            error("No baseline samples are available in the requested epoch window.")
        for epoch_idx in axes(epochs, 3)
            epoch = @view epochs[:, :, epoch_idx]
            epoch .-= mean(epoch[:, baseline_idx], dims = 2)
        end
    end

    keep_mask = map(axes(epochs, 3)) do epoch_idx
        epoch = @view epochs[:, :, epoch_idx]
        all(value -> isnan(value) || abs(value) <= EPOCH_ABS_THRESHOLD_UV, epoch)
    end
    n_rejected = count(!, keep_mask)
    epochs = epochs[:, :, keep_mask]
    return epochs, time_ms, n_rejected
end

function align_epochs_to_channels(
    epochs::AbstractArray{<:Real, 3},
    eeg_ch_names::Vector{String},
    target_ch_names::Vector{String},
)
    aligned_epochs = fill(NaN, length(target_ch_names), size(epochs, 2), size(epochs, 3))
    eeg_index = Dict(name => idx for (idx, name) in pairs(eeg_ch_names))

    for (target_idx, target_name) in pairs(target_ch_names)
        source_idx = get(eeg_index, target_name, nothing)
        isnothing(source_idx) && continue
        aligned_epochs[target_idx, :, :] = epochs[source_idx, :, :]
    end

    return aligned_epochs
end

Base.@kwdef mutable struct VariantAccumulator
    sums::Union{Nothing, Matrix{Float64}} = nothing
    counts::Union{Nothing, Matrix{Int}} = nothing
    time_ms::Union{Nothing, Vector{Float64}} = nothing
    ch_names::Union{Nothing, Vector{String}} = nothing
    positions::Any = nothing
    sfreq::Union{Nothing, Float64} = nothing
    n_candidate_events::Int = 0
    n_epochs::Int = 0
    n_rejected_epochs::Int = 0
    n_skipped_files::Int = 0
    n_failed_files::Int = 0
end

function empty_variant_accumulators(variant_options)
    return Dict(combo => VariantAccumulator() for combo in variant_options)
end

function variant_label(
    lock_to::Symbol,
    baseline_correction::Bool,
    direction::Symbol,
    drop_small_saccades::Bool,
)
    return "lock=$(lock_to), blcorr=$(baseline_correction), dir=$(direction), ampfilt=$(drop_small_saccades)"
end

function ensure_accumulator_layout!(
    accumulator::VariantAccumulator,
    eeg_ch_names::Vector{String},
    positions,
    time_ms::Vector{Float64},
    sfreq::Real,
)
    if isnothing(accumulator.ch_names)
        accumulator.ch_names = copy(eeg_ch_names)
        accumulator.positions = collect(positions)
        accumulator.time_ms = copy(time_ms)
        accumulator.sfreq = Float64(sfreq)
        accumulator.sums = zeros(Float64, length(accumulator.ch_names), length(time_ms))
        accumulator.counts = zeros(Int, length(accumulator.ch_names), length(time_ms))
        return
    end

    isapprox(sfreq, accumulator.sfreq) || error("Sampling rates do not match across FIF files.")
    time_ms == accumulator.time_ms || error("Epoch time axes do not match across FIF files.")

    reference_channel_set = Set(accumulator.ch_names)
    extra_channels = [name for name in eeg_ch_names if name ∉ reference_channel_set]
    if !isempty(extra_channels)
        position_lookup = Dict(name => pos for (name, pos) in zip(eeg_ch_names, positions))
        append!(accumulator.ch_names, extra_channels)
        append!(accumulator.positions, [position_lookup[name] for name in extra_channels])
        accumulator.sums = vcat(
            accumulator.sums,
            zeros(Float64, length(extra_channels), size(accumulator.sums, 2)),
        )
        accumulator.counts = vcat(
            accumulator.counts,
            zeros(Int, length(extra_channels), size(accumulator.counts, 2)),
        )
    end
end

function accumulate_epochs!(
    accumulator::VariantAccumulator,
    epochs::AbstractArray{<:Real, 3},
    eeg_ch_names::Vector{String},
    positions,
    time_ms::Vector{Float64},
    sfreq::Real,
)
    size(epochs, 3) == 0 && return
    ensure_accumulator_layout!(accumulator, eeg_ch_names, positions, time_ms, sfreq)
    aligned_epochs = align_epochs_to_channels(epochs, eeg_ch_names, accumulator.ch_names)
    valid_mask = .!isnan.(aligned_epochs)
    epoch_sums = dropdims(sum(ifelse.(valid_mask, aligned_epochs, 0.0), dims = 3), dims = 3)
    epoch_counts = dropdims(sum(valid_mask, dims = 3), dims = 3)
    accumulator.sums .+= epoch_sums
    accumulator.counts .+= epoch_counts
    accumulator.n_epochs += size(aligned_epochs, 3)
end

function erp_from_accumulator(accumulator::VariantAccumulator)
    isnothing(accumulator.sums) && return nothing
    erp = fill(NaN, size(accumulator.sums))
    valid = accumulator.counts .> 0
    erp[valid] .= accumulator.sums[valid] ./ accumulator.counts[valid]
    channel_has_data = vec(any(accumulator.counts .> 0, dims = 2))
    any(channel_has_data) || return nothing
    return (
        erp[channel_has_data, :],
        accumulator.time_ms,
        accumulator.ch_names[channel_has_data],
        accumulator.positions[channel_has_data],
    )
end

function prepare_events_for_variant(
    ann_df::DataFrame,
    sfreq::Real;
    lock_to::Symbol,
    direction::Symbol,
    drop_small_saccades::Bool,
)
    lock_to = normalize_lock_to(lock_to)
    if lock_to == :saccade
        return prepare_saccade_events(
            ann_df,
            sfreq;
            direction = direction,
            drop_small_saccades = drop_small_saccades,
        )
    end

    return prepare_fixation_events(
        ann_df,
        sfreq;
        direction = direction,
        drop_small_saccades = drop_small_saccades,
    )
end

function average_fifs(
    fif_paths::AbstractVector{<:AbstractString};
    variant_options = VARIANT_OPTIONS,
)
    isempty(fif_paths) && error("FIF_PATHS must contain at least one file.")
    accumulators = empty_variant_accumulators(variant_options)

    for fif_path in fif_paths
        try
            eeg_data, eeg_ch_names, sfreq, ann_df, positions = read_fif(fif_path)
            for (lock_to, baseline_correction, direction, drop_small_saccades) in variant_options
                combo = (lock_to, baseline_correction, direction, drop_small_saccades)
                accumulator = accumulators[combo]
                try
                    events = prepare_events_for_variant(
                        ann_df,
                        sfreq;
                        lock_to = lock_to,
                        direction = direction,
                        drop_small_saccades = drop_small_saccades,
                    )
                    accumulator.n_candidate_events += nrow(events)
                    epochs, time_ms, n_rejected = collect_epochs(
                        eeg_data,
                        events.latency,
                        sfreq;
                        baseline_correction = baseline_correction,
                    )
                    accumulator.n_rejected_epochs += n_rejected

                    if size(epochs, 3) == 0
                        accumulator.n_skipped_files += 1
                        continue
                    end

                    accumulate_epochs!(accumulator, epochs, eeg_ch_names, positions, time_ms, sfreq)
                catch err
                    accumulator.n_failed_files += 1
                    @warn "Skipping FIF for one topoplot variant because it could not be processed." fif_path variant = variant_label(
                        lock_to,
                        baseline_correction,
                        direction,
                        drop_small_saccades,
                    ) error = sprint(showerror, err)
                end
            end
            println("Finished $(basename(fif_path))")
        catch err
            for accumulator in values(accumulators)
                accumulator.n_failed_files += 1
            end
            @warn "Skipping FIF because it could not be read." fif_path error = sprint(showerror, err)
            continue
        end
    end

    return accumulators
end

function erp_to_dataframe(erp::AbstractMatrix, eeg_ch_names::Vector{String}, time_ms::AbstractVector)
    df = eeg_array_to_dataframe(erp, eeg_ch_names)
    df.time = time_ms[df.time] ./ 1000
    return df
end

function position_components(position)
    coords = Tuple(position)
    return Float64(coords[1]), Float64(coords[2])
end

function window_mask(time_ms::AbstractVector, window_spec)
    start_ms = 1000 * Float64(window_spec.start_s)
    stop_ms = 1000 * Float64(window_spec.stop_s)
    mask = (time_ms .>= start_ms) .& (time_ms .< stop_ms)
    any(mask) || error("No samples fell into the requested time window $(window_display_label(window_spec)).")
    return mask
end

function symmetric_colorrange(values::AbstractVector{<:Real}; fallback = 1.0)
    valid_values = filter(value -> !isnan(value) && isfinite(value), collect(values))
    isempty(valid_values) && return (-fallback, fallback)
    max_abs = maximum(abs, valid_values)
    iszero(max_abs) && return (-fallback, fallback)
    return (-max_abs, max_abs)
end

function colorbar_ticks(colorrange)
    ticks = collect(LinRange(colorrange[1], colorrange[2], 5))
    rounded_ticks = round.(ticks, digits = 2)
    return (ticks, string.(rounded_ticks))
end

function no_data_figure(message::AbstractString; size = SINGLE_WINDOW_FIGURE_SIZE)
    fig = Figure(size = size)
    ax = Axis(fig[1, 1]; title = message)
    text!(
        ax,
        0.5,
        0.5;
        space = :relative,
        text = "No usable epochs",
        align = (:center, :center),
    )
    hidedecorations!(ax)
    hidespines!(ax)
    return fig
end

function window_average_data(accumulator::VariantAccumulator, window_spec)
    erp_data = erp_from_accumulator(accumulator)
    isnothing(erp_data) && return nothing

    erp, time_ms, eeg_ch_names, positions = erp_data
    mask = window_mask(time_ms, window_spec)
    window_values = vec(mean(erp[:, mask], dims = 2))
    valid_mask = map(value -> !isnan(value) && isfinite(value), window_values)
    any(valid_mask) || return nothing

    return (
        values = window_values[valid_mask],
        ch_names = eeg_ch_names[valid_mask],
        positions = positions[valid_mask],
    )
end

function window_dataframe(window_values::AbstractVector, eeg_ch_names::Vector{String}, window_spec)
    df = eeg_array_to_dataframe(reshape(window_values, :, 1), eeg_ch_names)
    df.window = fill(window_display_label(window_spec), nrow(df))
    return df
end

function plot_topoplot_variant(
    accumulator::VariantAccumulator;
    input_kind::Symbol,
    lock_to::Symbol,
    direction::Symbol,
    baseline_correction::Bool,
    drop_small_saccades::Bool,
)
    title =
        "$(title_input_kind_label(input_kind)) $(titlecase(lock_label(lock_to))) ERP (n = $(accumulator.n_epochs), dir = $(direction), blcorr = $(baseline_correction), ampfilt = $(drop_small_saccades))"
    fig = Figure(size = FIGURE_SIZE)

    erp_data = erp_from_accumulator(accumulator)
    if isnothing(erp_data)
        ax = Axis(fig[1, 1]; title = title)
        text!(
            ax,
            0.5,
            0.5;
            space = :relative,
            text = "No usable epochs for this combination",
            align = (:center, :center),
        )
        hidedecorations!(ax)
        hidespines!(ax)
        return fig
    end

    erp, time_ms, eeg_ch_names, positions = erp_data
    df = erp_to_dataframe(erp, eeg_ch_names, time_ms)
    plot_topoplotseries!(
        fig[1, 1],
        df;
        positions = positions,
        bin_num = BIN_NUM,
        nrows = NROWS,
        axis = (; xlabel = "Time relative to $(lock_label(lock_to)) [s]", title = title),
        visual = (; label_scatter = false, label_text = false),
        colorbar = (; label = "Voltage [µV]"),
        topolabels_rounding = (; digits = 2),
    )
    return fig
end

function plot_window_topoplot(
    accumulator::VariantAccumulator,
    window_spec;
    input_kind::Symbol,
    lock_to::Symbol,
    direction::Symbol,
    baseline_correction::Bool,
    drop_small_saccades::Bool,
    colorrange = nothing,
    show_colorbar::Bool = true,
    figure_size = SINGLE_WINDOW_FIGURE_SIZE,
)
    window_data = window_average_data(accumulator, window_spec)
    title =
        "$(title_input_kind_label(input_kind)) $(titlecase(lock_label(lock_to))) $(window_spec.name) topoplot $(window_display_label(window_spec))"

    if isnothing(window_data)
        return no_data_figure(title; size = figure_size)
    end

    figure = Figure(size = figure_size)
    window_range = isnothing(colorrange) ? symmetric_colorrange(window_data.values) : colorrange
    df = window_dataframe(window_data.values, window_data.ch_names, window_spec)
    plot_topoplotseries!(
        figure[1, 1],
        df;
        positions = window_data.positions,
        mapping = (; col = :window),
        axis = (; xlabel = "", title = title),
        visual = (;
            label_scatter = false,
            label_text = false,
            contours = (; linewidth = 1, color = :black),
            colorrange = window_range,
        ),
        colorbar = (;
            label = "Voltage [µV]",
            ticks = colorbar_ticks(window_range),
        ),
        layout = (; use_colorbar = show_colorbar),
        topolabels_rounding = (; digits = 3),
    )
    return figure
end

function preferred_input_kind_order(input_kinds::AbstractVector{<:Symbol})
    ordered = Symbol[]
    for preferred_kind in (:eyelink, :clean)
        preferred_kind in input_kinds && push!(ordered, preferred_kind)
    end
    for input_kind in input_kinds
        input_kind in ordered || push!(ordered, input_kind)
    end
    return ordered
end

function plot_window_comparison(
    accumulators_by_input_kind::Dict{Symbol, Dict};
    input_kinds::AbstractVector{<:Symbol},
    window_spec,
    lock_to::Symbol,
    direction::Symbol,
    baseline_correction::Bool,
    drop_small_saccades::Bool,
)
    ordered_input_kinds = preferred_input_kind_order(input_kinds)
    combo = (lock_to, baseline_correction, direction, drop_small_saccades)
    window_data_by_input = Dict{Symbol, Any}()
    combined_values = Float64[]

    for input_kind in ordered_input_kinds
        accumulator = accumulators_by_input_kind[input_kind][combo]
        window_data = window_average_data(accumulator, window_spec)
        if !isnothing(window_data)
            window_data_by_input[input_kind] = window_data
            append!(combined_values, window_data.values)
        end
    end

    comparison_range = symmetric_colorrange(combined_values)
    figure = Figure(size = COMPARISON_FIGURE_SIZE)

    for (column_idx, input_kind) in pairs(ordered_input_kinds)
        title =
            "$(title_input_kind_label(input_kind)) $(window_spec.name) $(window_display_label(window_spec))"
        if !haskey(window_data_by_input, input_kind)
            ax = Axis(figure[1, column_idx]; title = title)
            text!(
                ax,
                0.5,
                0.5;
                space = :relative,
                text = "No usable epochs",
                align = (:center, :center),
            )
            hidedecorations!(ax)
            hidespines!(ax)
            continue
        end

        window_data = window_data_by_input[input_kind]
        df = window_dataframe(window_data.values, window_data.ch_names, window_spec)
        accumulator = accumulators_by_input_kind[input_kind][combo]
        subplot_title = "$(title) (n = $(accumulator.n_epochs))"
        plot_topoplotseries!(
            figure[1, column_idx],
            df;
            positions = window_data.positions,
            mapping = (; col = :window),
            axis = (; xlabel = "", title = subplot_title),
            visual = (;
                label_scatter = false,
                label_text = false,
                contours = (; linewidth = 1, color = :black),
                colorrange = comparison_range,
            ),
            colorbar = (;
                label = "Voltage [µV]",
                ticks = colorbar_ticks(comparison_range),
            ),
            layout = (; use_colorbar = column_idx == length(ordered_input_kinds)),
            topolabels_rounding = (; digits = 3),
        )
    end

    return figure
end

function channel_window_dataframe(
    accumulator::VariantAccumulator,
    window_spec;
    input_kind::Symbol,
)
    window_data = window_average_data(accumulator, window_spec)
    if isnothing(window_data)
        return DataFrame(
            input_kind = String[],
            window = String[],
            window_start_s = Float64[],
            window_stop_s = Float64[],
            channel = String[],
            mean_voltage_uv = Float64[],
            abs_mean_voltage_uv = Float64[],
            x = Float64[],
            y = Float64[],
            positive_rank = Int[],
            negative_rank = Int[],
            absolute_rank = Int[],
        )
    end

    x_coords = Float64[]
    y_coords = Float64[]
    for position in window_data.positions
        x_coord, y_coord = position_components(position)
        push!(x_coords, x_coord)
        push!(y_coords, y_coord)
    end

    df = DataFrame(
        input_kind = fill(output_input_kind_label(input_kind), length(window_data.values)),
        window = fill(window_spec.name, length(window_data.values)),
        window_start_s = fill(window_spec.start_s, length(window_data.values)),
        window_stop_s = fill(window_spec.stop_s, length(window_data.values)),
        channel = window_data.ch_names,
        mean_voltage_uv = window_data.values,
        abs_mean_voltage_uv = abs.(window_data.values),
        x = x_coords,
        y = y_coords,
    )

    df.positive_rank = invperm(sortperm(df.mean_voltage_uv; rev = true))
    df.negative_rank = invperm(sortperm(df.mean_voltage_uv; rev = false))
    df.absolute_rank = invperm(sortperm(df.abs_mean_voltage_uv; rev = true))
    sort!(df, :absolute_rank)
    return df
end

function top_rank_dataframe(df::DataFrame, rank_column::Symbol, ranking_name::AbstractString; top_n = 10)
    if isempty(df)
        return DataFrame(
            ranking = String[],
            rank = Int[],
            input_kind = String[],
            window = String[],
            window_start_s = Float64[],
            window_stop_s = Float64[],
            channel = String[],
            mean_voltage_uv = Float64[],
            abs_mean_voltage_uv = Float64[],
            x = Float64[],
            y = Float64[],
            positive_rank = Int[],
            negative_rank = Int[],
            absolute_rank = Int[],
        )
    end
    ranked = subset(df, rank_column => ByRow(rank -> rank <= top_n))
    sort!(ranked, rank_column)
    ranked.ranking = fill(ranking_name, nrow(ranked))
    ranked.rank = ranked[!, rank_column]
    select!(
        ranked,
        :ranking,
        :rank,
        :input_kind,
        :window,
        :window_start_s,
        :window_stop_s,
        :channel,
        :mean_voltage_uv,
        :abs_mean_voltage_uv,
        :x,
        :y,
        :positive_rank,
        :negative_rank,
        :absolute_rank,
    )
    return ranked
end

function channel_region_summary_row(
    accumulator::VariantAccumulator,
    window_spec;
    input_kind::Symbol,
    lock_to::Symbol,
    direction::Symbol,
    baseline_correction::Bool,
    drop_small_saccades::Bool,
)
    window_data = window_average_data(accumulator, window_spec)
    isnothing(window_data) && return nothing

    y_coords = Float64[]
    for position in window_data.positions
        _, y_coord = position_components(position)
        push!(y_coords, y_coord)
    end

    anterior_cutoff = quantile(y_coords, 0.8)
    posterior_cutoff = quantile(y_coords, 0.2)
    anterior_mask = y_coords .>= anterior_cutoff
    posterior_mask = y_coords .<= posterior_cutoff
    any(anterior_mask) || return nothing
    any(posterior_mask) || return nothing

    anterior_values = window_data.values[anterior_mask]
    posterior_values = window_data.values[posterior_mask]
    anterior_mean_abs = mean(abs.(anterior_values))
    posterior_mean_abs = mean(abs.(posterior_values))

    return (
        input_kind = output_input_kind_label(input_kind),
        lock_to = String(lock_to),
        direction = String(direction),
        baseline_correction = baseline_correction,
        drop_small_saccades = drop_small_saccades,
        window = window_spec.name,
        window_start_s = window_spec.start_s,
        window_stop_s = window_spec.stop_s,
        anterior_n_channels = count(anterior_mask),
        posterior_n_channels = count(posterior_mask),
        anterior_mean_uv = mean(anterior_values),
        posterior_mean_uv = mean(posterior_values),
        anterior_mean_abs_uv = anterior_mean_abs,
        posterior_mean_abs_uv = posterior_mean_abs,
        anterior_max_abs_uv = maximum(abs.(anterior_values)),
        posterior_max_abs_uv = maximum(abs.(posterior_values)),
        anterior_to_posterior_mean_abs_ratio = iszero(posterior_mean_abs) ? missing :
                                               anterior_mean_abs / posterior_mean_abs,
    )
end

function variant_summary_row(
    accumulator::VariantAccumulator;
    input_kind::Symbol,
    n_input_files::Integer,
    lock_to::Symbol,
    direction::Symbol,
    baseline_correction::Bool,
    drop_small_saccades::Bool,
)
    return (
        input_kind = output_input_kind_label(input_kind),
        lock_to = String(lock_to),
        direction = String(direction),
        baseline_correction = baseline_correction,
        drop_small_saccades = drop_small_saccades,
        n_input_files = n_input_files,
        n_candidate_events = accumulator.n_candidate_events,
        n_used_epochs = accumulator.n_epochs,
        n_rejected_epochs = accumulator.n_rejected_epochs,
        n_skipped_files = accumulator.n_skipped_files,
        n_failed_files = accumulator.n_failed_files,
    )
end

function csv_escape(value)
    if ismissing(value)
        return ""
    end

    string_value = string(value)
    if occursin(',', string_value) || occursin('"', string_value) || occursin('\n', string_value)
        return "\"" * replace(string_value, "\"" => "\"\"") * "\""
    end
    return string_value
end

function write_dataframe_csv(path::AbstractString, df::DataFrame)
    open(path, "w") do io
        println(io, join(string.(names(df)), ","))
        for row in eachrow(df)
            row_values = [csv_escape(row[column_name]) for column_name in names(df)]
            println(io, join(row_values, ","))
        end
    end
end

function export_exploratory_outputs!(
    accumulators_by_input_kind::Dict{Symbol, Dict};
    input_kinds::AbstractVector{<:Symbol},
    lock_to::Symbol,
    direction::Symbol,
    baseline_correction::Bool,
    drop_small_saccades::Bool,
)
    combo = (lock_to, baseline_correction, direction, drop_small_saccades)
    variant_summary_df = DataFrame(
        input_kind = String[],
        lock_to = String[],
        direction = String[],
        baseline_correction = Bool[],
        drop_small_saccades = Bool[],
        n_input_files = Int[],
        n_candidate_events = Int[],
        n_used_epochs = Int[],
        n_rejected_epochs = Int[],
        n_skipped_files = Int[],
        n_failed_files = Int[],
    )
    region_summary_df = DataFrame(
        input_kind = String[],
        lock_to = String[],
        direction = String[],
        baseline_correction = Bool[],
        drop_small_saccades = Bool[],
        window = String[],
        window_start_s = Float64[],
        window_stop_s = Float64[],
        anterior_n_channels = Int[],
        posterior_n_channels = Int[],
        anterior_mean_uv = Float64[],
        posterior_mean_uv = Float64[],
        anterior_mean_abs_uv = Float64[],
        posterior_mean_abs_uv = Float64[],
        anterior_max_abs_uv = Float64[],
        posterior_max_abs_uv = Float64[],
        anterior_to_posterior_mean_abs_ratio = Union{Missing, Float64}[],
    )

    for input_kind in preferred_input_kind_order(input_kinds)
        accumulator = accumulators_by_input_kind[input_kind][combo]
        output_stem = exploratory_output_stem(
            input_kind;
            lock_to = lock_to,
            direction = direction,
            baseline_correction = baseline_correction,
            drop_small_saccades = drop_small_saccades,
        )

        series_path = "$(output_stem)_topoplotseries.png"
        series_figure = plot_topoplot_variant(
            accumulator;
            input_kind = input_kind,
            lock_to = lock_to,
            direction = direction,
            baseline_correction = baseline_correction,
            drop_small_saccades = drop_small_saccades,
        )
        save(series_path, series_figure)
        println("Saved $(abspath(series_path))")

        for window_spec in WINDOW_SPECS
            window_path = "$(output_stem)_window-$(window_filename_component(window_spec)).png"
            window_figure = plot_window_topoplot(
                accumulator,
                window_spec;
                input_kind = input_kind,
                lock_to = lock_to,
                direction = direction,
                baseline_correction = baseline_correction,
                drop_small_saccades = drop_small_saccades,
            )
            save(window_path, window_figure)
            println("Saved $(abspath(window_path))")

            region_summary_row_value = channel_region_summary_row(
                accumulator,
                window_spec;
                input_kind = input_kind,
                lock_to = lock_to,
                direction = direction,
                baseline_correction = baseline_correction,
                drop_small_saccades = drop_small_saccades,
            )
            isnothing(region_summary_row_value) || push!(region_summary_df, region_summary_row_value)
        end

        ranking_window_spec = only(
            filter(window_spec -> window_spec.name == CHANNEL_RANK_WINDOW_NAME, WINDOW_SPECS),
        )
        ranking_df = channel_window_dataframe(
            accumulator,
            ranking_window_spec;
            input_kind = input_kind,
        )
        ranking_path =
            "$(output_stem)_window-$(window_filename_component(ranking_window_spec))_channel_ranking.csv"
        write_dataframe_csv(ranking_path, ranking_df)
        println("Saved $(abspath(ranking_path))")

        top10_absolute_path =
            "$(output_stem)_window-$(window_filename_component(ranking_window_spec))_top10_absolute.csv"
        write_dataframe_csv(
            top10_absolute_path,
            top_rank_dataframe(ranking_df, :absolute_rank, "absolute"),
        )
        println("Saved $(abspath(top10_absolute_path))")

        top10_positive_path =
            "$(output_stem)_window-$(window_filename_component(ranking_window_spec))_top10_positive.csv"
        write_dataframe_csv(
            top10_positive_path,
            top_rank_dataframe(ranking_df, :positive_rank, "positive"),
        )
        println("Saved $(abspath(top10_positive_path))")

        top10_negative_path =
            "$(output_stem)_window-$(window_filename_component(ranking_window_spec))_top10_negative.csv"
        write_dataframe_csv(
            top10_negative_path,
            top_rank_dataframe(ranking_df, :negative_rank, "negative"),
        )
        println("Saved $(abspath(top10_negative_path))")

        push!(
            variant_summary_df,
            variant_summary_row(
                accumulator;
                input_kind = input_kind,
                n_input_files = length(fif_paths_for_input_kind(input_kind)),
                lock_to = lock_to,
                direction = direction,
                baseline_correction = baseline_correction,
                drop_small_saccades = drop_small_saccades,
            ),
        )
    end

    for window_spec in WINDOW_SPECS
        comparison_path =
            "$(comparison_output_stem(
                preferred_input_kind_order(input_kinds);
                lock_to = lock_to,
                direction = direction,
                baseline_correction = baseline_correction,
                drop_small_saccades = drop_small_saccades,
            ))_window-$(window_filename_component(window_spec)).png"
        comparison_figure = plot_window_comparison(
            accumulators_by_input_kind;
            input_kinds = input_kinds,
            window_spec = window_spec,
            lock_to = lock_to,
            direction = direction,
            baseline_correction = baseline_correction,
            drop_small_saccades = drop_small_saccades,
        )
        save(comparison_path, comparison_figure)
        println("Saved $(abspath(comparison_path))")
    end

    summary_stem = comparison_output_stem(
        preferred_input_kind_order(input_kinds);
        lock_to = lock_to,
        direction = direction,
        baseline_correction = baseline_correction,
        drop_small_saccades = drop_small_saccades,
    )
    variant_summary_path = "$(summary_stem)_variant_summary.csv"
    region_summary_path = "$(summary_stem)_region_summary.csv"
    write_dataframe_csv(variant_summary_path, variant_summary_df)
    println("Saved $(abspath(variant_summary_path))")
    write_dataframe_csv(region_summary_path, region_summary_df)
    println("Saved $(abspath(region_summary_path))")
end

function main()
    println("RUN3_INPUT_KIND: $(join(output_input_kind_label.(RUN3_INPUT_KINDS), ", "))")
    println("TOPOPLOT_ANALYSIS_MODE: $(TOPOPLOT_ANALYSIS_MODE)")
    println("TOPOPLOT_LOCK_TO: $(DEFAULT_LOCK_TO)")
    println("Variants to process: $(length(VARIANT_OPTIONS))")

    accumulators_by_input_kind = Dict{Symbol, Dict}()

    for input_kind in preferred_input_kind_order(RUN3_INPUT_KINDS)
        println("Processing $(title_input_kind_label(input_kind)) recordings...")
        fif_paths = fif_paths_for_input_kind(input_kind)
        accumulators = average_fifs(fif_paths; variant_options = VARIANT_OPTIONS)
        accumulators_by_input_kind[input_kind] = accumulators

        for (lock_to, baseline_correction, direction, drop_small_saccades) in VARIANT_OPTIONS
            accumulator = accumulators[(lock_to, baseline_correction, direction, drop_small_saccades)]
            println(
                "Summary $(output_input_kind_label(input_kind)) / $(variant_label(lock_to, baseline_correction, direction, drop_small_saccades)): epochs=$(accumulator.n_epochs), candidate_events=$(accumulator.n_candidate_events), rejected=$(accumulator.n_rejected_epochs), skipped=$(accumulator.n_skipped_files), failed=$(accumulator.n_failed_files)",
            )
        end
    end

    export_exploratory_outputs!(
        accumulators_by_input_kind;
        input_kinds = RUN3_INPUT_KINDS,
        lock_to = DEFAULT_LOCK_TO,
        direction = DEFAULT_DIRECTION,
        baseline_correction = DEFAULT_BASELINE_CORRECTION,
        drop_small_saccades = DEFAULT_DROP_SMALL_SACCADES,
    )
end

main()
