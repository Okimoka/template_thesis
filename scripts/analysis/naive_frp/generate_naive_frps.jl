using DataFrames
using CairoMakie
using Unfold

"""
Fully LLM generated
"""

# Let PythonCall use the project-managed CondaPkg environment by default.
# If you need a specific external Python, set JULIA_PYTHONCALL_EXE yourself
# before launching Julia.
using PythonCall

const RUN3_PATHS_FILE = get(ENV, "RUN3_PATHS_FILE", "./run3_paths.jl")
include(joinpath(@__DIR__, "..", RUN3_PATHS_FILE))

const py_mne = pyimport("mne")

const EVENT_EYE = "L"
const TARGET_CHANNELS = ["E82", "E8"]
const LOCK_OPTIONS = [:saccade, :fixation]
const DIRECTION_OPTIONS = [:all, :ltr]
const TMIN = -0.2
const TMAX = 0.8
const MIN_SACCADE_AMPLITUDE = 0.5
const FIGURE_SIZE = (1100, 700)
const OUTPUT_DIR = @__DIR__
const DATASET_LABELS = ["clean", "eyelink"]
const DATASET_PATHS = Dict(
    "clean" => FREEVIEW_RUN3_CLEAN_FIF_PATHS,
    "eyelink" => FREEVIEW_RUN3_EYELINK_FIF_PATHS,
)
const DATASET_COLORS = Dict(
    "clean" => :steelblue,
    "eyelink" => :firebrick,
)
const MAX_FILES_PER_DATASET = let raw_value = get(ENV, "NAIVE_FRP_MAX_FILES", "")
    isempty(raw_value) ? nothing : parse(Int, raw_value)
end

Base.@kwdef mutable struct RunningAverage
    sum::Union{Nothing, Matrix{Float64}} = nothing
    count::Union{Nothing, Matrix{Int}} = nothing
    time_ms::Union{Nothing, Vector{Float64}} = nothing
    n_candidate_events::Int = 0
    n_used_epochs::Int = 0
    n_dropped_epochs::Int = 0
    n_failed_files::Int = 0
    n_files_with_epochs::Int = 0
    n_existing_files::Int = 0
    n_missing_files::Int = 0
end

event_description(kind::Symbol, eye_name::AbstractString = EVENT_EYE) =
    kind == :saccade ? "ET_Saccade $eye_name" :
    kind == :fixation ? "ET_Fixation $eye_name" :
    error("Unsupported event kind: $kind")

function normalize_lock_to(lock_to::Symbol)
    lock_to in LOCK_OPTIONS || error("Unsupported lock option: $lock_to")
    return lock_to
end

function normalize_direction(direction::Symbol)
    direction in DIRECTION_OPTIONS || error("Unsupported direction option: $direction")
    return direction
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

function read_fif(fif_path::AbstractString)
    raw_mne = py_mne.io.read_raw_fif(fif_path, preload = true, verbose = "ERROR")
    ann_df = annotations_to_dataframe(raw_mne)
    eeg_only = raw_mne.copy().set_eeg_reference().pick("eeg")
    eeg_data = pyconvert(Array{Float64, 2}, eeg_only.get_data(units = "uV"))
    eeg_ch_names = pyconvert(Vector{String}, eeg_only.ch_names)
    sfreq = pyconvert(Float64, eeg_only.info["sfreq"])
    bad_channels = Set(pyconvert(Vector{String}, eeg_only.info["bads"]))

    for bad_channel in bad_channels
        idx = findfirst(==(bad_channel), eeg_ch_names)
        idx === nothing && continue
        eeg_data[idx, :] .= NaN
    end

    return eeg_data, eeg_ch_names, sfreq, ann_df
end

function extract_target_channel_data(eeg_data::AbstractMatrix, eeg_ch_names::Vector{String})
    target_data = fill(NaN, length(TARGET_CHANNELS), size(eeg_data, 2))
    eeg_index = Dict(name => idx for (idx, name) in pairs(eeg_ch_names))

    for (target_idx, channel_name) in pairs(TARGET_CHANNELS)
        source_idx = get(eeg_index, channel_name, nothing)
        isnothing(source_idx) && continue
        target_data[target_idx, :] = eeg_data[source_idx, :]
    end

    return target_data
end

valid_amplitude(value) = !ismissing(value) && isfinite(Float64(value)) && Float64(value) >= MIN_SACCADE_AMPLITUDE
valid_coordinate(value) = !ismissing(value) && isfinite(Float64(value))

function require_direction_columns(ann_df::DataFrame)
    required_cols = [Symbol("Start Loc.X"), Symbol("End Loc.X")]
    missing_cols = [col for col in required_cols if col ∉ propertynames(ann_df)]
    isempty(missing_cols) || error(
        "Missing saccade direction columns: $(join(string.(missing_cols), ", ")).",
    )
end

function sorted_eye_annotations(ann_df::DataFrame, eye_name::AbstractString = EVENT_EYE)
    eye_suffix = " $eye_name"
    return subset(
        sort(copy(ann_df), :onset),
        :description => ByRow(x -> !ismissing(x) && endswith(String(x), eye_suffix)),
    )
end

function prepare_saccade_events(
    ann_df::DataFrame,
    sfreq::Real;
    direction::Symbol = :all,
    eye_name::AbstractString = EVENT_EYE,
)
    direction = normalize_direction(direction)
    saccade_desc = event_description(:saccade, eye_name)
    saccades = subset(
        sort(copy(ann_df), :onset),
        :description => ByRow(==(saccade_desc)),
        :Amplitude => ByRow(valid_amplitude),
    )

    if direction == :ltr
        require_direction_columns(ann_df)
        saccades = subset(
            saccades,
            Symbol("Start Loc.X") => ByRow(valid_coordinate),
            Symbol("End Loc.X") => ByRow(valid_coordinate),
            [Symbol("Start Loc.X"), Symbol("End Loc.X")] => ByRow((start_x, end_x) -> Float64(end_x) > Float64(start_x)),
        )
    end

    events = DataFrame(latency = round.(Int, saccades.onset .* sfreq) .+ 1)
    return subset(events, :latency => ByRow(>(0)))
end

function prepare_fixation_events(
    ann_df::DataFrame,
    sfreq::Real;
    direction::Symbol = :all,
    eye_name::AbstractString = EVENT_EYE,
)
    direction = normalize_direction(direction)
    direction == :ltr && require_direction_columns(ann_df)

    eye_ann = sorted_eye_annotations(ann_df, eye_name)
    descriptions = String.(eye_ann.description)
    saccade_desc = event_description(:saccade, eye_name)
    fixation_desc = event_description(:fixation, eye_name)
    fixation_latencies = Int[]

    for idx in eachindex(descriptions)
        descriptions[idx] == fixation_desc || continue
        idx > firstindex(descriptions) || continue
        descriptions[idx - 1] == saccade_desc || continue
        valid_amplitude(eye_ann.Amplitude[idx - 1]) || continue

        if direction == :ltr
            start_x = eye_ann[idx - 1, Symbol("Start Loc.X")]
            end_x = eye_ann[idx - 1, Symbol("End Loc.X")]
            valid_coordinate(start_x) || continue
            valid_coordinate(end_x) || continue
            Float64(end_x) > Float64(start_x) || continue
        end

        latency = round(Int, eye_ann.onset[idx] * sfreq) + 1
        latency > 0 || continue
        push!(fixation_latencies, latency)
    end

    return DataFrame(latency = fixation_latencies)
end

function collect_epochs(
    eeg_data::AbstractMatrix,
    latencies::AbstractVector{<:Integer},
    sfreq::Real;
    tmin::Real = TMIN,
    tmax::Real = TMAX,
)
    isempty(latencies) && return Array{Float64}(undef, size(eeg_data, 1), 0, 0), Float64[], 0

    events = DataFrame(latency = latencies)
    epochs, times = Unfold.epoch(data = eeg_data, tbl = events, τ = (tmin, tmax), sfreq = sfreq)
    time_ms = collect(times) .* 1000
    kept_events, epochs = Unfold.drop_missing_epochs(events, epochs)
    n_dropped = nrow(events) - nrow(kept_events)

    return epochs, time_ms, n_dropped
end

function accumulate_epochs!(summary::RunningAverage, epochs::AbstractArray{<:Real, 3}, time_ms::Vector{Float64})
    size(epochs, 3) == 0 && return

    if isnothing(summary.sum)
        summary.sum = zeros(Float64, size(epochs, 1), size(epochs, 2))
        summary.count = zeros(Int, size(epochs, 1), size(epochs, 2))
        summary.time_ms = copy(time_ms)
    else
        length(time_ms) == length(summary.time_ms) || error("Epoch time axes do not match.")
        all(isapprox.(time_ms, summary.time_ms; atol = 1e-9)) || error("Epoch time axes do not match.")
    end

    valid_mask = .!isnan.(epochs)
    epoch_sums = dropdims(sum(ifelse.(valid_mask, epochs, 0.0), dims = 3), dims = 3)
    epoch_counts = dropdims(sum(valid_mask, dims = 3), dims = 3)
    summary.sum .+= epoch_sums
    summary.count .+= epoch_counts
    summary.n_used_epochs += size(epochs, 3)
    summary.n_files_with_epochs += 1
end

function average_waveform(summary::RunningAverage)
    isnothing(summary.sum) && return fill(NaN, length(TARGET_CHANNELS), 0)

    mean_waveform = fill(NaN, size(summary.sum))
    valid = summary.count .> 0
    mean_waveform[valid] .= summary.sum[valid] ./ summary.count[valid]
    return mean_waveform
end

function resolve_fif_paths(fif_paths::AbstractVector{<:AbstractString})
    existing_paths = filter(isfile, fif_paths)
    missing_count = length(fif_paths) - length(existing_paths)

    if !isnothing(MAX_FILES_PER_DATASET)
        existing_paths = first(existing_paths, min(MAX_FILES_PER_DATASET, length(existing_paths)))
    end

    return existing_paths, missing_count
end

function empty_combo_summaries()
    return Dict((lock_to, direction) => RunningAverage() for lock_to in LOCK_OPTIONS for direction in DIRECTION_OPTIONS)
end

function combo_event_count_summary(event_tables::Dict{Tuple{Symbol, Symbol}, DataFrame})
    parts = String[]
    for lock_to in LOCK_OPTIONS, direction in DIRECTION_OPTIONS
        count = nrow(event_tables[(lock_to, direction)])
        push!(parts, "$(lock_to)/$(direction)=$(count)")
    end
    return join(parts, ", ")
end

function process_dataset(dataset_label::AbstractString, fif_paths::AbstractVector{<:AbstractString})
    summaries = empty_combo_summaries()
    existing_paths, missing_count = resolve_fif_paths(fif_paths)

    for summary in values(summaries)
        summary.n_existing_files = length(existing_paths)
        summary.n_missing_files = missing_count
    end

    println("Processing $(dataset_label): $(length(existing_paths)) file(s) ($(missing_count) missing)")

    for (file_idx, fif_path) in pairs(existing_paths)
        if file_idx == 1 || file_idx % 25 == 0 || file_idx == length(existing_paths)
            println("  $(dataset_label): file $(file_idx) / $(length(existing_paths))")
        end

        started_at = time()
        try
            eeg_data, eeg_ch_names, sfreq, ann_df = read_fif(fif_path)
            target_data = extract_target_channel_data(eeg_data, eeg_ch_names)
            event_tables = Dict(
                (:saccade, :all) => prepare_saccade_events(ann_df, sfreq; direction = :all),
                (:saccade, :ltr) => prepare_saccade_events(ann_df, sfreq; direction = :ltr),
                (:fixation, :all) => prepare_fixation_events(ann_df, sfreq; direction = :all),
                (:fixation, :ltr) => prepare_fixation_events(ann_df, sfreq; direction = :ltr),
            )

            for combo in keys(summaries)
                summary = summaries[combo]
                event_df = event_tables[combo]
                summary.n_candidate_events += nrow(event_df)
                epochs, time_ms, n_dropped = collect_epochs(target_data, event_df.latency, sfreq)
                summary.n_dropped_epochs += n_dropped
                accumulate_epochs!(summary, epochs, time_ms)
            end

            elapsed_seconds = round(time() - started_at; digits = 1)
            println(
                "  finished $(dataset_label): $(file_idx) / $(length(existing_paths)) | $(basename(fif_path)) | $(combo_event_count_summary(event_tables)) | elapsed=$(elapsed_seconds)s",
            )
        catch err
            message = sprint(showerror, err)
            elapsed_seconds = round(time() - started_at; digits = 1)
            println(
                "  failed $(dataset_label): $(file_idx) / $(length(existing_paths)) | $(basename(fif_path)) | elapsed=$(elapsed_seconds)s | $(message)",
            )
            for summary in values(summaries)
                summary.n_failed_files += 1
            end
        end

        file_idx % 25 == 0 && GC.gc()
    end

    return summaries
end

function lock_label(lock_to::Symbol)
    lock_to = normalize_lock_to(lock_to)
    return lock_to == :saccade ? "saccade onset" : "fixation onset"
end

function direction_label(direction::Symbol)
    direction = normalize_direction(direction)
    return direction == :all ? "all saccades" : "left-to-right saccades"
end

function output_png_path(channel_name::AbstractString, lock_to::Symbol, direction::Symbol)
    lock_to = normalize_lock_to(lock_to)
    direction = normalize_direction(direction)
    limit_suffix = isnothing(MAX_FILES_PER_DATASET) ? "" : "_maxfiles-$(MAX_FILES_PER_DATASET)"
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_channel-$(channel_name)_lock-$(lock_to)_direction-$(direction)$(limit_suffix).png",
    )
end

function plot_naive_frp(
    summaries_by_dataset::Dict{String, Dict{Tuple{Symbol, Symbol}, RunningAverage}},
    channel_name::AbstractString,
    lock_to::Symbol,
    direction::Symbol,
)
    lock_to = normalize_lock_to(lock_to)
    direction = normalize_direction(direction)
    channel_idx = findfirst(==(channel_name), TARGET_CHANNELS)
    channel_idx === nothing && error("Unsupported channel: $channel_name")

    fig = Figure(size = FIGURE_SIZE)
    ax = Axis(
        fig[1, 1];
        xlabel = "Time relative to $(lock_label(lock_to)) [ms]",
        ylabel = "Voltage [uV]",
        title = "Naive FRP | $channel_name | $(lock_label(lock_to)) | $(direction_label(direction))",
    )
    hlines!(ax, [0.0], color = (:black, 0.3), linestyle = :dash)
    vlines!(ax, [0.0], color = (:black, 0.3), linestyle = :dash)

    plotted_anything = false

    for dataset_label in DATASET_LABELS
        summary = summaries_by_dataset[dataset_label][(lock_to, direction)]
        mean_waveform = average_waveform(summary)
        isnothing(summary.time_ms) && continue
        waveform = vec(mean_waveform[channel_idx, :])
        all(isnan, waveform) && continue

        label = "$(dataset_label) | epochs=$(summary.n_used_epochs) | events=$(summary.n_candidate_events) | failed=$(summary.n_failed_files)"
        lines!(
            ax,
            summary.time_ms,
            waveform;
            color = DATASET_COLORS[dataset_label],
            linewidth = 2.5,
            label = label,
        )
        plotted_anything = true
    end

    if plotted_anything
        axislegend(ax; position = :rb)
    else
        text!(
            ax,
            0.5,
            0.5;
            space = :relative,
            text = "No usable epochs for this combination",
            align = (:center, :center),
        )
    end

    return fig
end

function print_dataset_summary(dataset_label::AbstractString, summaries::Dict{Tuple{Symbol, Symbol}, RunningAverage})
    println("Summary for $(dataset_label):")
    for lock_to in LOCK_OPTIONS, direction in DIRECTION_OPTIONS
        summary = summaries[(lock_to, direction)]
        println(
            "  lock=$(lock_to), direction=$(direction), events=$(summary.n_candidate_events), epochs=$(summary.n_used_epochs), dropped=$(summary.n_dropped_epochs), files_with_epochs=$(summary.n_files_with_epochs), failed=$(summary.n_failed_files)",
        )
    end
end

function main()
    mkpath(OUTPUT_DIR)
    py_mne.set_log_level("ERROR")

    summaries_by_dataset = Dict(
        dataset_label => process_dataset(dataset_label, DATASET_PATHS[dataset_label]) for
        dataset_label in DATASET_LABELS
    )

    for dataset_label in DATASET_LABELS
        print_dataset_summary(dataset_label, summaries_by_dataset[dataset_label])
    end

    for channel_name in TARGET_CHANNELS, lock_to in LOCK_OPTIONS, direction in DIRECTION_OPTIONS
        fig = plot_naive_frp(summaries_by_dataset, channel_name, lock_to, direction)
        output_path = output_png_path(channel_name, lock_to, direction)
        save(output_path, fig)
        println("Saved $(output_path)")
    end
end

main()
