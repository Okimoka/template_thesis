using DataFrames
using CairoMakie
using Dates
using Unfold

"""
Entirely LLM written
"""

const CONDAPKG_ENV_DIR = joinpath(@__DIR__, ".CondaPkg", "env")
const CONDAPKG_PYTHON_EXE = joinpath(CONDAPKG_ENV_DIR, "bin", "python")
const CONDAPKG_PYTHON_LIB = joinpath(CONDAPKG_ENV_DIR, "lib", "libpython3.so")

isfile(CONDAPKG_PYTHON_EXE) || error(
    "Expected CondaPkg Python executable at $(CONDAPKG_PYTHON_EXE). " *
    "Please instantiate the project so .CondaPkg is created.",
)
isfile(CONDAPKG_PYTHON_LIB) || error(
    "Expected CondaPkg Python library at $(CONDAPKG_PYTHON_LIB). " *
    "Please instantiate the project so .CondaPkg is created.",
)

ENV["JULIA_PYTHONCALL_EXE"] = CONDAPKG_PYTHON_EXE
ENV["JULIA_PYTHONCALL_LIB"] = CONDAPKG_PYTHON_LIB

using PythonCall

const py_mne = pyimport("mne")
const pybuiltins = pyimport("builtins")

const EVENT_EYE = "L"
const TARGET_CHANNEL = "E82"
const LOCK_OPTIONS = [:saccade, :fixation]
const TMIN = -0.5
const TMAX = 1.0
const SACCADE_AMPLITUDE_THRESHOLD = 0.5
const FIXED_PEAK_THRESHOLD_UV = let raw_value = get(ENV, "NAIVE_FRP_FIXED_THRESHOLD_UV", "")
    isempty(raw_value) ? 100.0 : parse(Float64, raw_value)
end
const OUTPUT_DIR = joinpath(@__DIR__, "naive_frp_age_quartiles")
const GROUPS_FILE = get(
    ENV,
    "NAIVE_FRP_GROUPS_FILE",
    joinpath(@__DIR__, "more", "generated_recording_groups2.jl"),
)
const MAX_SUBJECTS_PER_DATASET = let raw_value = get(ENV, "NAIVE_FRP_MAX_SUBJECTS", "")
    isempty(raw_value) ? nothing : parse(Int, raw_value)
end
const MATCH_DATASET_SUBJECTS = lowercase(get(ENV, "NAIVE_FRP_MATCH_DATASET_SUBJECTS", "1")) in
                               ("1", "true", "yes")
const FIGURE_SIZE = (1700, 900)
const PLOT_LINEWIDTH = 3.0
const PATH_ALIASES = [
    (
        "/data/work/st156392/mergedDataset/derivatives",
        "/home/oki/ehlers-work2/mergedDataset/derivatives",
    ),
    (
        "/home/oki/ehlers-work2/mergedDataset/derivatives",
        "/data/work/st156392/mergedDataset/derivatives",
    ),
]
const PARTICIPANTS_TSV_CANDIDATES = filter(
    !isempty,
    [
        get(ENV, "NAIVE_FRP_PARTICIPANTS_TSV", ""),
        "participants.tsv",
    ],
)
const DATASET_GROUP_NAMES = Dict("clean" => "CLEAN_GROUPS", "eyelink" => "RAW_GROUPS")
const DATASET_TITLES = Dict("clean" => "Clean", "eyelink" => "Eyelink")
const DATASET_COLORS = Dict("clean" => :steelblue, "eyelink" => :firebrick)

CairoMakie.activate!()

Base.@kwdef mutable struct SubjectWaveformSummary
    sum::Union{Nothing, Vector{Float64}} = nothing
    count::Union{Nothing, Vector{Int}} = nothing
    time_ms::Union{Nothing, Vector{Float64}} = nothing
    n_candidate_events::Int = 0
    n_used_epochs::Int = 0
    n_dropped_epochs::Int = 0
    n_threshold_filtered_epochs::Int = 0
    n_failed_files::Int = 0
    n_missing_files::Int = 0
end

Base.@kwdef mutable struct QuartileWaveformSummary
    sum::Union{Nothing, Vector{Float64}} = nothing
    count::Union{Nothing, Vector{Int}} = nothing
    time_ms::Union{Nothing, Vector{Float64}} = nothing
    n_subjects_total::Int = 0
    n_subjects_used::Int = 0
    n_candidate_events::Int = 0
    n_used_epochs::Int = 0
    n_dropped_epochs::Int = 0
    n_threshold_filtered_epochs::Int = 0
    n_failed_files::Int = 0
    n_missing_files::Int = 0
end

function resolve_participants_tsv()
    for candidate in PARTICIPANTS_TSV_CANDIDATES
        isfile(candidate) && return candidate
    end

    joined = join(PARTICIPANTS_TSV_CANDIDATES, ", ")
    error("Could not find participants.tsv. Tried: $(joined)")
end

function resolve_input_path(path::AbstractString)
    isfile(path) && return String(path)

    for (from_root, to_root) in PATH_ALIASES
        startswith(path, from_root) || continue
        suffix = path[(lastindex(from_root) + 1):end]
        candidate = to_root * suffix
        isfile(candidate) && return candidate
    end

    return String(path)
end

function subject_from_path(path::AbstractString)
    match_obj = match(r"(sub-[A-Za-z0-9]+)", path)
    match_obj === nothing && error("Could not determine subject id from path: $(path)")
    return match_obj.match
end

function normalize_group_collection(groups_any)
    groups = Vector{NamedTuple{(:subject_id, :paths), Tuple{String, Vector{String}}}}()

    for raw_group in groups_any
        paths = [String(path) for path in raw_group]
        isempty(paths) && continue
        subject_id = subject_from_path(first(paths))

        for path in paths
            subject_from_path(path) == subject_id ||
                error("Mixed subject ids detected in one group: $(join(paths, ", "))")
        end

        push!(groups, (subject_id = subject_id, paths = paths))
    end

    return groups
end

function extract_group_block(file_text::AbstractString, variable_name::AbstractString)
    collected_lines = String[]
    collecting = false

    for line in eachline(IOBuffer(file_text))
        stripped = strip(line)

        if !collecting
            if startswith(stripped, variable_name * " = [")
                collecting = true
            end
            continue
        end

        stripped == "]" && break
        push!(collected_lines, line)
    end

    collecting || error("Expected $(variable_name) in generated group file.")
    return join(collected_lines, '\n')
end

function parse_groups_from_block(block_text::AbstractString)
    raw_groups = Vector{Tuple{Vararg{String}}}()

    for line in eachline(IOBuffer(block_text))
        stripped = strip(line)
        startswith(stripped, "(") || continue
        paths = [match_obj.captures[1] for match_obj in eachmatch(r"\"([^\"]+)\"", stripped)]
        isempty(paths) && continue
        push!(raw_groups, Tuple(paths))
    end

    return raw_groups
end

function load_group_definitions(groups_file::AbstractString)
    isfile(groups_file) || error("Recording-group file does not exist: $(groups_file)")
    file_text = read(groups_file, String)

    groups_by_dataset = Dict(
        "clean" => normalize_group_collection(
            parse_groups_from_block(extract_group_block(file_text, "CLEAN_GROUPS")),
        ),
        "eyelink" => normalize_group_collection(
            parse_groups_from_block(extract_group_block(file_text, "RAW_GROUPS")),
        ),
    )

    if MATCH_DATASET_SUBJECTS
        clean_subject_ids = Set(entry.subject_id for entry in groups_by_dataset["clean"])
        eyelink_subject_ids = Set(entry.subject_id for entry in groups_by_dataset["eyelink"])
        matched_subject_ids = intersect(clean_subject_ids, eyelink_subject_ids)

        for dataset_label in keys(groups_by_dataset)
            groups_by_dataset[dataset_label] = filter(
                entry -> entry.subject_id in matched_subject_ids,
                groups_by_dataset[dataset_label],
            )
        end
    end

    if !isnothing(MAX_SUBJECTS_PER_DATASET)
        for dataset_label in keys(groups_by_dataset)
            groups_by_dataset[dataset_label] = first(
                groups_by_dataset[dataset_label],
                min(MAX_SUBJECTS_PER_DATASET, length(groups_by_dataset[dataset_label])),
            )
        end
    end

    return groups_by_dataset
end

function load_ages(participants_tsv::AbstractString)
    lines = readlines(participants_tsv)
    isempty(lines) && error("participants.tsv is empty: $(participants_tsv)")

    header = split(chomp(first(lines)), '\t'; keepempty = true)
    participant_idx = findfirst(==("participant_id"), header)
    age_idx = findfirst(==("age"), header)
    participant_idx === nothing && error("Missing participant_id column in $(participants_tsv)")
    age_idx === nothing && error("Missing age column in $(participants_tsv)")

    ages = Dict{String, Float64}()

    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = split(chomp(line), '\t'; keepempty = true)
        max(length(fields), max(participant_idx, age_idx)) == length(fields) || continue

        participant_id = strip(fields[participant_idx])
        age_text = strip(fields[age_idx])
        isempty(participant_id) && continue
        isempty(age_text) && continue

        parsed_age = tryparse(Float64, age_text)
        parsed_age === nothing && continue
        ages[participant_id] = parsed_age
    end

    return ages
end

function split_into_quartiles(subject_groups, ages::Dict{String, Float64})
    eligible = NamedTuple{(:subject_id, :age, :paths), Tuple{String, Float64, Vector{String}}}[]
    missing_age = String[]

    for entry in subject_groups
        age = get(ages, entry.subject_id, nothing)
        if isnothing(age)
            push!(missing_age, entry.subject_id)
            continue
        end

        push!(eligible, (subject_id = entry.subject_id, age = age, paths = entry.paths))
    end

    isempty(eligible) && error("No groups with valid ages were found.")
    sort!(eligible, by = item -> (item.age, item.subject_id))

    n_quartiles = min(4, length(eligible))
    base_size, remainder = divrem(length(eligible), n_quartiles)
    quartiles = NamedTuple[
        (
            group_index = index,
            age_min = entries[1].age,
            age_max = entries[end].age,
            entries = entries,
        ) for (index, entries) in enumerate(begin
            chunks = Vector{Vector{typeof(first(eligible))}}()
            start_idx = 1
            for chunk_index in 1:n_quartiles
                chunk_size = base_size + (chunk_index <= remainder ? 1 : 0)
                stop_idx = start_idx + chunk_size - 1
                push!(chunks, eligible[start_idx:stop_idx])
                start_idx = stop_idx + 1
            end
            chunks
        end)
    ]

    return quartiles, missing_age
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

function read_target_channel_fif(fif_path::AbstractString)
    raw_mne = py_mne.io.read_raw_fif(fif_path, preload = true, verbose = "ERROR")
    ann_df = annotations_to_dataframe(raw_mne)
    eeg_only = raw_mne.copy().set_eeg_reference().pick("eeg")
    eeg_data = pyconvert(Array{Float64, 2}, eeg_only.get_data(units = "uV"))
    eeg_ch_names = pyconvert(Vector{String}, eeg_only.ch_names)
    sfreq = pyconvert(Float64, eeg_only.info["sfreq"])
    bad_channels = Set(pyconvert(Vector{String}, eeg_only.info["bads"]))

    channel_idx = findfirst(==(TARGET_CHANNEL), eeg_ch_names)
    channel_data = if isnothing(channel_idx)
        fill(NaN, size(eeg_data, 2))
    else
        collect(eeg_data[channel_idx, :])
    end

    if TARGET_CHANNEL in bad_channels
        channel_data .= NaN
    end

    return channel_data, sfreq, ann_df
end

has_numeric_amplitude(value) = !ismissing(value) && isfinite(Float64(value))
passes_saccade_amplitude_threshold(value) =
    has_numeric_amplitude(value) && Float64(value) >= SACCADE_AMPLITUDE_THRESHOLD

event_description(kind::Symbol, eye_name::AbstractString = EVENT_EYE) =
    kind == :saccade ? "ET_Saccade $eye_name" :
    kind == :fixation ? "ET_Fixation $eye_name" :
    error("Unsupported event kind: $kind")

function sorted_eye_annotations(ann_df::DataFrame, eye_name::AbstractString = EVENT_EYE)
    eye_suffix = " $eye_name"
    return subset(
        sort(copy(ann_df), :onset),
        :description => ByRow(x -> !ismissing(x) && endswith(String(x), eye_suffix)),
    )
end

function prepare_saccade_events(ann_df::DataFrame, sfreq::Real; eye_name::AbstractString = EVENT_EYE)
    saccade_desc = event_description(:saccade, eye_name)
    saccades = subset(
        sort(copy(ann_df), :onset),
        :description => ByRow(==(saccade_desc)),
        :Amplitude => ByRow(passes_saccade_amplitude_threshold),
    )

    latencies = round.(Int, saccades.onset .* sfreq) .+ 1
    return latencies[latencies .> 0]
end

function prepare_fixation_events(ann_df::DataFrame, sfreq::Real; eye_name::AbstractString = EVENT_EYE)
    eye_ann = sorted_eye_annotations(ann_df, eye_name)
    descriptions = String.(eye_ann.description)
    saccade_desc = event_description(:saccade, eye_name)
    fixation_desc = event_description(:fixation, eye_name)
    fixation_latencies = Int[]

    for idx in eachindex(descriptions)
        descriptions[idx] == fixation_desc || continue
        idx > firstindex(descriptions) || continue
        descriptions[idx - 1] == saccade_desc || continue
        passes_saccade_amplitude_threshold(eye_ann.Amplitude[idx - 1]) || continue

        latency = round(Int, eye_ann.onset[idx] * sfreq) + 1
        latency > 0 || continue
        push!(fixation_latencies, latency)
    end

    return fixation_latencies
end

function collect_epochs(
    channel_data::AbstractVector{<:Real},
    latencies::AbstractVector{<:Integer},
    sfreq::Real;
    tmin::Real = TMIN,
    tmax::Real = TMAX,
)
    isempty(latencies) && return Array{Float64}(undef, 1, 0, 0), Float64[], 0

    events = DataFrame(latency = latencies)
    data_matrix = reshape(Float64.(channel_data), 1, :)
    epochs, times = Unfold.epoch(data = data_matrix, tbl = events, τ = (tmin, tmax), sfreq = sfreq)
    time_ms = collect(times) .* 1000
    kept_events, epochs = Unfold.drop_missing_epochs(events, epochs)
    n_dropped = nrow(events) - nrow(kept_events)

    return epochs, time_ms, n_dropped
end

function compute_epoch_peak_abs(epochs::AbstractArray{<:Real, 3})
    peaks = fill(NaN, size(epochs, 3))

    for epoch_idx in axes(epochs, 3)
        epoch_peak = NaN

        for value in @view epochs[1, :, epoch_idx]
            isnan(value) && continue
            abs_value = abs(Float64(value))
            epoch_peak = isnan(epoch_peak) ? abs_value : max(epoch_peak, abs_value)
        end

        peaks[epoch_idx] = epoch_peak
    end

    return peaks
end

function accumulate_epochs!(summary::SubjectWaveformSummary, epochs::AbstractArray{<:Real, 3}, time_ms)
    size(epochs, 3) == 0 && return

    if isnothing(summary.sum)
        summary.sum = zeros(Float64, size(epochs, 2))
        summary.count = zeros(Int, size(epochs, 2))
        summary.time_ms = collect(time_ms)
    else
        length(time_ms) == length(summary.time_ms) || error("Epoch time axes do not match.")
        all(isapprox.(time_ms, summary.time_ms; atol = 1e-9)) ||
            error("Epoch time axes do not match.")
    end

    channel_epochs = @view epochs[1, :, :]
    valid_mask = .!isnan.(channel_epochs)
    epoch_sums = vec(sum(ifelse.(valid_mask, channel_epochs, 0.0), dims = 2))
    epoch_counts = vec(sum(valid_mask, dims = 2))
    summary.sum .+= epoch_sums
    summary.count .+= epoch_counts
    summary.n_used_epochs += size(epochs, 3)
end

function average_waveform(sum_values, count_values)
    isnothing(sum_values) && return Float64[]

    waveform = fill(NaN, length(sum_values))
    valid = count_values .> 0
    waveform[valid] .= sum_values[valid] ./ count_values[valid]
    return waveform
end

function finalize_subject_waveform(summary::SubjectWaveformSummary)
    isnothing(summary.sum) && return nothing
    return (time_ms = summary.time_ms, waveform = average_waveform(summary.sum, summary.count))
end

function accumulate_subject_waveform!(
    summary::QuartileWaveformSummary,
    time_ms::AbstractVector{<:Real},
    waveform::AbstractVector{<:Real},
)
    if isnothing(summary.sum)
        summary.sum = zeros(Float64, length(waveform))
        summary.count = zeros(Int, length(waveform))
        summary.time_ms = collect(Float64.(time_ms))
    else
        length(waveform) == length(summary.sum) || error("Subject waveform lengths do not match.")
        all(isapprox.(time_ms, summary.time_ms; atol = 1e-9)) ||
            error("Subject waveform time axes do not match.")
    end

    valid = .!isnan.(waveform)
    summary.sum[valid] .+= Float64.(waveform[valid])
    summary.count[valid] .+= 1
    summary.n_subjects_used += 1
end

function empty_subject_summaries()
    return Dict(lock_to => SubjectWaveformSummary() for lock_to in LOCK_OPTIONS)
end

function empty_quartile_summaries(quartiles)
    return Dict(lock_to => [QuartileWaveformSummary() for _ in quartiles] for lock_to in LOCK_OPTIONS)
end

function empty_dataset_summaries()
    return Dict(lock_to => QuartileWaveformSummary() for lock_to in LOCK_OPTIONS)
end

function process_subject_group(subject_id::AbstractString, paths::Vector{String})
    summaries = empty_subject_summaries()

    for path in paths
        resolved_path = resolve_input_path(path)

        if !isfile(resolved_path)
            for summary in values(summaries)
                summary.n_missing_files += 1
            end
            println("  missing $(subject_id): $(path)")
            continue
        end

        started_at = time()

        try
            channel_data, sfreq, ann_df = read_target_channel_fif(resolved_path)

            for lock_to in LOCK_OPTIONS
                latencies = lock_to == :saccade ?
                    prepare_saccade_events(ann_df, sfreq) :
                    prepare_fixation_events(ann_df, sfreq)

                summary = summaries[lock_to]
                summary.n_candidate_events += length(latencies)
                epochs, time_ms, n_dropped = collect_epochs(channel_data, latencies, sfreq)
                summary.n_dropped_epochs += n_dropped

                size(epochs, 3) == 0 && continue

                peaks = compute_epoch_peak_abs(epochs)
                finite_peak_mask = isfinite.(peaks)
                keep_mask = finite_peak_mask .& (peaks .<= FIXED_PEAK_THRESHOLD_UV)
                summary.n_threshold_filtered_epochs += count(finite_peak_mask .& .!keep_mask)
                accumulate_epochs!(summary, epochs[:, :, keep_mask], time_ms)
            end

            elapsed_seconds = round(time() - started_at; digits = 1)
            println(
                "  finished $(subject_id) | $(basename(resolved_path)) | elapsed=$(elapsed_seconds)s",
            )
        catch err
            message = sprint(showerror, err)
            elapsed_seconds = round(time() - started_at; digits = 1)
            println(
                "  failed $(subject_id) | $(basename(resolved_path)) | elapsed=$(elapsed_seconds)s | $(message)",
            )
            for summary in values(summaries)
                summary.n_failed_files += 1
            end
        end
    end

    return summaries
end

function quartile_label(quartile_summary, quartile)
    return "Q$(quartile.group_index) ($(round(quartile.age_min; digits = 2))-" *
           "$(round(quartile.age_max; digits = 2))y, n=$(quartile_summary.n_subjects_total))"
end

quartile_palette(n_groups::Integer) = collect(Makie.wong_colors()[1:n_groups])

function lock_label(lock_to::Symbol)
    return lock_to == :saccade ? "Saccade onset" : "Fixation onset"
end

function plot_output_path(dataset_label::AbstractString)
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_age_quartiles_$(dataset_label)_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV.png",
    )
end

function summary_output_path(dataset_label::AbstractString)
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_age_quartiles_$(dataset_label)_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV.tsv",
    )
end

function comparison_output_path()
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_all_subjects_dataset_comparison_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV.png",
    )
end

function comparison_summary_output_path()
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_all_subjects_dataset_comparison_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV.tsv",
    )
end

function quartile_series_output_path(dataset_label::AbstractString)
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_age_quartiles_$(dataset_label)_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV_series.tsv",
    )
end

function comparison_series_output_path()
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_all_subjects_dataset_comparison_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV_series.tsv",
    )
end

function subject_assignments_output_path(dataset_label::AbstractString)
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_age_quartiles_$(dataset_label)_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV_subject_assignments.tsv",
    )
end

function subject_summaries_output_path(dataset_label::AbstractString)
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_age_quartiles_$(dataset_label)_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV_subject_summaries.tsv",
    )
end

function subject_waveforms_output_path(dataset_label::AbstractString)
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_age_quartiles_$(dataset_label)_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV_subject_waveforms.tsv",
    )
end

function metadata_output_path()
    threshold_label = replace(string(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)), "." => "p")
    return joinpath(
        OUTPUT_DIR,
        "naive_frp_run_metadata_channel-$(TARGET_CHANNEL)_amp-gt-$(SACCADE_AMPLITUDE_THRESHOLD)_peak-le-$(threshold_label)uV.tsv",
    )
end

format_real(value::Real; digits::Integer = 6) =
    isfinite(Float64(value)) ? string(round(Float64(value); digits = digits)) : string(Float64(value))

format_real_vector(values::AbstractVector{<:Real}; digits::Integer = 6) =
    join((format_real(value; digits = digits) for value in values), ',')

sanitize_tsv_text(value::AbstractString) = replace(replace(String(value), '\t' => ' '), '\n' => ' ')

function initialize_subject_assignments_io(dataset_label::AbstractString)
    output_path = subject_assignments_output_path(dataset_label)
    mkpath(dirname(output_path))
    io = open(output_path, "w")
    println(
        io,
        join(
            [
                "dataset",
                "subject_id",
                "age",
                "group_index",
                "age_min",
                "age_max",
                "n_input_paths",
                "input_paths",
            ],
            '\t',
        ),
    )
    return io, output_path
end

function initialize_subject_summaries_io(dataset_label::AbstractString)
    output_path = subject_summaries_output_path(dataset_label)
    mkpath(dirname(output_path))
    io = open(output_path, "w")
    println(
        io,
        join(
            [
                "dataset",
                "subject_id",
                "age",
                "group_index",
                "age_min",
                "age_max",
                "lock_to",
                "n_input_paths",
                "n_candidate_events",
                "n_used_epochs",
                "n_dropped_epochs",
                "n_threshold_filtered_epochs",
                "n_failed_files",
                "n_missing_files",
            ],
            '\t',
        ),
    )
    return io, output_path
end

function initialize_subject_waveforms_io(dataset_label::AbstractString)
    output_path = subject_waveforms_output_path(dataset_label)
    mkpath(dirname(output_path))
    io = open(output_path, "w")
    println(
        io,
        join(
            [
                "dataset",
                "subject_id",
                "age",
                "group_index",
                "age_min",
                "age_max",
                "lock_to",
                "n_timepoints",
                "time_ms_csv",
                "waveform_uv_csv",
            ],
            '\t',
        ),
    )
    return io, output_path
end

function write_subject_assignment_row(io::IO, dataset_label::AbstractString, quartile, entry)
    println(
        io,
        join(
            [
                dataset_label,
                entry.subject_id,
                format_real(entry.age; digits = 4),
                string(quartile.group_index),
                format_real(quartile.age_min; digits = 4),
                format_real(quartile.age_max; digits = 4),
                string(length(entry.paths)),
                join((sanitize_tsv_text(path) for path in entry.paths), '|'),
            ],
            '\t',
        ),
    )
end

function write_subject_summary_rows(
    io::IO,
    dataset_label::AbstractString,
    quartile,
    entry,
    subject_summaries,
)
    for lock_to in LOCK_OPTIONS
        summary = subject_summaries[lock_to]
        println(
            io,
            join(
                [
                    dataset_label,
                    entry.subject_id,
                    format_real(entry.age; digits = 4),
                    string(quartile.group_index),
                    format_real(quartile.age_min; digits = 4),
                    format_real(quartile.age_max; digits = 4),
                    String(lock_to),
                    string(length(entry.paths)),
                    string(summary.n_candidate_events),
                    string(summary.n_used_epochs),
                    string(summary.n_dropped_epochs),
                    string(summary.n_threshold_filtered_epochs),
                    string(summary.n_failed_files),
                    string(summary.n_missing_files),
                ],
                '\t',
            ),
        )
    end
end

function write_subject_waveform_rows(
    io::IO,
    dataset_label::AbstractString,
    quartile,
    entry,
    finalized_by_lock,
)
    for lock_to in LOCK_OPTIONS
        finalized = get(finalized_by_lock, lock_to, nothing)
        isnothing(finalized) && continue

        println(
            io,
            join(
                [
                    dataset_label,
                    entry.subject_id,
                    format_real(entry.age; digits = 4),
                    string(quartile.group_index),
                    format_real(quartile.age_min; digits = 4),
                    format_real(quartile.age_max; digits = 4),
                    String(lock_to),
                    string(length(finalized.time_ms)),
                    format_real_vector(finalized.time_ms; digits = 4),
                    format_real_vector(finalized.waveform; digits = 6),
                ],
                '\t',
            ),
        )
    end
end

function write_metadata_tsv(groups_by_dataset, participants_tsv::AbstractString)
    output_path = metadata_output_path()
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(io, "key\tvalue")

        metadata_rows = [
            ("generated_at", Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")),
            ("target_channel", TARGET_CHANNEL),
            ("lock_options", join(String.(LOCK_OPTIONS), ",")),
            ("tmin_seconds", format_real(TMIN; digits = 4)),
            ("tmax_seconds", format_real(TMAX; digits = 4)),
            (
                "saccade_amplitude_threshold_degrees",
                format_real(SACCADE_AMPLITUDE_THRESHOLD; digits = 4),
            ),
            ("fixed_peak_threshold_uv", format_real(FIXED_PEAK_THRESHOLD_UV; digits = 4)),
            ("groups_file", GROUPS_FILE),
            ("participants_tsv", participants_tsv),
            ("match_dataset_subjects", string(MATCH_DATASET_SUBJECTS)),
            (
                "max_subjects_per_dataset",
                isnothing(MAX_SUBJECTS_PER_DATASET) ? "<all>" : string(MAX_SUBJECTS_PER_DATASET),
            ),
            ("julia_version", string(VERSION)),
            ("clean_subject_groups", string(length(groups_by_dataset["clean"]))),
            ("eyelink_subject_groups", string(length(groups_by_dataset["eyelink"]))),
        ]

        for (key, value) in metadata_rows
            println(io, "$(sanitize_tsv_text(key))\t$(sanitize_tsv_text(value))")
        end
    end

    return output_path
end

function write_summary_tsv(dataset_label::AbstractString, quartiles, summaries_by_lock)
    output_path = summary_output_path(dataset_label)
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(
            io,
            join(
                [
                    "dataset",
                    "lock_to",
                    "group_index",
                    "age_min",
                    "age_max",
                    "n_subjects_total",
                    "n_subjects_used",
                    "n_candidate_events",
                    "n_used_epochs",
                    "n_dropped_epochs",
                    "n_threshold_filtered_epochs",
                    "n_failed_files",
                    "n_missing_files",
                ],
                '\t',
            ),
        )

        for lock_to in LOCK_OPTIONS
            for (quartile_idx, quartile) in pairs(quartiles)
                summary = summaries_by_lock[lock_to][quartile_idx]
                println(
                    io,
                    join(
                        [
                            dataset_label,
                            String(lock_to),
                            string(quartile.group_index),
                            string(round(quartile.age_min; digits = 4)),
                            string(round(quartile.age_max; digits = 4)),
                            string(summary.n_subjects_total),
                            string(summary.n_subjects_used),
                            string(summary.n_candidate_events),
                            string(summary.n_used_epochs),
                            string(summary.n_dropped_epochs),
                            string(summary.n_threshold_filtered_epochs),
                            string(summary.n_failed_files),
                            string(summary.n_missing_files),
                        ],
                        '\t',
                    ),
                )
            end
        end
    end

    return output_path
end

function write_quartile_series_tsv(dataset_label::AbstractString, quartiles, summaries_by_lock)
    output_path = quartile_series_output_path(dataset_label)
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(
            io,
            join(
                [
                    "dataset",
                    "lock_to",
                    "group_index",
                    "group_label",
                    "age_min",
                    "age_max",
                    "n_subjects_total",
                    "n_subjects_used",
                    "time_ms",
                    "voltage_uv",
                ],
                '\t',
            ),
        )

        for lock_to in LOCK_OPTIONS
            for (quartile_idx, quartile) in pairs(quartiles)
                summary = summaries_by_lock[lock_to][quartile_idx]
                waveform = average_waveform(summary.sum, summary.count)
                isempty(waveform) && continue
                all(isnan, waveform) && continue

                for (time_ms, voltage_uv) in zip(summary.time_ms, waveform)
                    println(
                        io,
                        join(
                            [
                                dataset_label,
                                String(lock_to),
                                string(quartile.group_index),
                                quartile_label(summary, quartile),
                                format_real(quartile.age_min; digits = 4),
                                format_real(quartile.age_max; digits = 4),
                                string(summary.n_subjects_total),
                                string(summary.n_subjects_used),
                                format_real(time_ms; digits = 4),
                                format_real(voltage_uv; digits = 6),
                            ],
                            '\t',
                        ),
                    )
                end
            end
        end
    end

    return output_path
end

function plot_dataset(dataset_label::AbstractString, quartiles, summaries_by_lock)
    fig = Figure(size = FIGURE_SIZE)
    axes = Dict{Symbol, Axis}()
    palette = quartile_palette(length(quartiles))

    for (column_idx, lock_to) in enumerate(LOCK_OPTIONS)
        ax = Axis(
            fig[1, column_idx];
            title = lock_label(lock_to),
            xlabel = "Time [ms]",
            ylabel = column_idx == 1 ? "Voltage [uV]" : "",
        )
        hlines!(ax, [0.0], color = (:black, 0.3), linestyle = :dash)
        vlines!(ax, [0.0], color = (:black, 0.3), linestyle = :dash)
        xlims!(ax, TMIN * 1000, TMAX * 1000)
        axes[lock_to] = ax

        for (quartile_idx, quartile) in pairs(quartiles)
            summary = summaries_by_lock[lock_to][quartile_idx]
            waveform = average_waveform(summary.sum, summary.count)
            isempty(waveform) && continue
            all(isnan, waveform) && continue

            lines!(
                ax,
                summary.time_ms,
                waveform;
                color = palette[quartile_idx],
                linewidth = PLOT_LINEWIDTH,
                label = quartile_label(summary, quartile),
            )
        end
    end

    linkyaxes!(axes[:saccade], axes[:fixation])
    axislegend(axes[:fixation]; position = :rb)

    Label(
        fig[0, :],
        "Naive FRP | $(DATASET_TITLES[dataset_label]) age quartiles | $(TARGET_CHANNEL) | " *
        "saccades >= $(SACCADE_AMPLITUDE_THRESHOLD) deg | epoch peak <= $(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)) uV",
        fontsize = 22,
    )

    return fig
end

function plot_dataset_comparison(dataset_summaries_by_lock)
    fig = Figure(size = FIGURE_SIZE)
    axes = Dict{Symbol, Axis}()

    for (column_idx, lock_to) in enumerate(LOCK_OPTIONS)
        ax = Axis(
            fig[1, column_idx];
            title = lock_label(lock_to),
            xlabel = "Time [ms]",
            ylabel = column_idx == 1 ? "Voltage [uV]" : "",
        )
        hlines!(ax, [0.0], color = (:black, 0.3), linestyle = :dash)
        vlines!(ax, [0.0], color = (:black, 0.3), linestyle = :dash)
        xlims!(ax, TMIN * 1000, TMAX * 1000)
        axes[lock_to] = ax

        for dataset_label in ["clean", "eyelink"]
            summary = dataset_summaries_by_lock[dataset_label][lock_to]
            waveform = average_waveform(summary.sum, summary.count)
            isempty(waveform) && continue
            all(isnan, waveform) && continue

            label = "$(DATASET_TITLES[dataset_label]) (n=$(summary.n_subjects_total))"
            lines!(
                ax,
                summary.time_ms,
                waveform;
                color = DATASET_COLORS[dataset_label],
                linewidth = PLOT_LINEWIDTH,
                label = label,
            )
        end
    end

    linkyaxes!(axes[:saccade], axes[:fixation])
    axislegend(axes[:fixation]; position = :rb)

    subject_scope = MATCH_DATASET_SUBJECTS ? "matched subjects" : "all available subjects"
    Label(
        fig[0, :],
        "Naive FRP | dataset comparison | $(subject_scope) | $(TARGET_CHANNEL) | " *
        "saccades >= $(SACCADE_AMPLITUDE_THRESHOLD) deg | epoch peak <= $(round(FIXED_PEAK_THRESHOLD_UV; digits = 1)) uV",
        fontsize = 22,
    )

    return fig
end

function write_comparison_summary_tsv(dataset_summaries_by_lock)
    output_path = comparison_summary_output_path()
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(
            io,
            join(
                [
                    "dataset",
                    "lock_to",
                    "n_subjects_total",
                    "n_subjects_used",
                    "n_candidate_events",
                    "n_used_epochs",
                    "n_dropped_epochs",
                    "n_threshold_filtered_epochs",
                    "n_failed_files",
                    "n_missing_files",
                ],
                '\t',
            ),
        )

        for dataset_label in ["clean", "eyelink"], lock_to in LOCK_OPTIONS
            summary = dataset_summaries_by_lock[dataset_label][lock_to]
            println(
                io,
                join(
                    [
                        dataset_label,
                        String(lock_to),
                        string(summary.n_subjects_total),
                        string(summary.n_subjects_used),
                        string(summary.n_candidate_events),
                        string(summary.n_used_epochs),
                        string(summary.n_dropped_epochs),
                        string(summary.n_threshold_filtered_epochs),
                        string(summary.n_failed_files),
                        string(summary.n_missing_files),
                    ],
                    '\t',
                ),
            )
        end
    end

    return output_path
end

function write_comparison_series_tsv(dataset_summaries_by_lock)
    output_path = comparison_series_output_path()
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(
            io,
            join(
                [
                    "dataset",
                    "lock_to",
                    "label",
                    "n_subjects_total",
                    "n_subjects_used",
                    "time_ms",
                    "voltage_uv",
                ],
                '\t',
            ),
        )

        for dataset_label in ["clean", "eyelink"], lock_to in LOCK_OPTIONS
            summary = dataset_summaries_by_lock[dataset_label][lock_to]
            waveform = average_waveform(summary.sum, summary.count)
            isempty(waveform) && continue
            all(isnan, waveform) && continue
            label = "$(DATASET_TITLES[dataset_label]) (n=$(summary.n_subjects_total))"

            for (time_ms, voltage_uv) in zip(summary.time_ms, waveform)
                println(
                    io,
                    join(
                        [
                            dataset_label,
                            String(lock_to),
                            label,
                            string(summary.n_subjects_total),
                            string(summary.n_subjects_used),
                            format_real(time_ms; digits = 4),
                            format_real(voltage_uv; digits = 6),
                        ],
                        '\t',
                    ),
                )
            end
        end
    end

    return output_path
end

function print_dataset_summary(dataset_label::AbstractString, quartiles, summaries_by_lock)
    println("Summary for $(dataset_label):")
    for lock_to in LOCK_OPTIONS
        println("  lock=$(lock_to)")
        for (quartile_idx, quartile) in pairs(quartiles)
            summary = summaries_by_lock[lock_to][quartile_idx]
            println(
                "    Q$(quartile.group_index) ages=$(round(quartile.age_min; digits = 2))-$(round(quartile.age_max; digits = 2)) | subjects=$(summary.n_subjects_total) used_subjects=$(summary.n_subjects_used) events=$(summary.n_candidate_events) epochs=$(summary.n_used_epochs) dropped=$(summary.n_dropped_epochs) threshold_rejected=$(summary.n_threshold_filtered_epochs) missing_files=$(summary.n_missing_files) failed=$(summary.n_failed_files)",
            )
        end
    end
end

function process_dataset(dataset_label::AbstractString, subject_groups, ages)
    quartiles, missing_age = split_into_quartiles(subject_groups, ages)
    summaries_by_lock = empty_quartile_summaries(quartiles)
    dataset_summaries_by_lock = empty_dataset_summaries()
    assignments_io, assignments_path = initialize_subject_assignments_io(dataset_label)
    subject_summaries_io, subject_summaries_path = initialize_subject_summaries_io(dataset_label)
    subject_waveforms_io, subject_waveforms_path = initialize_subject_waveforms_io(dataset_label)

    println(
        "Processing $(dataset_label): $(length(subject_groups)) subject groups, $(length(missing_age)) missing ages",
    )

    total_subjects = sum(length(quartile.entries) for quartile in quartiles)

    for (quartile_idx, quartile) in pairs(quartiles)
        for lock_to in LOCK_OPTIONS
            summaries_by_lock[lock_to][quartile_idx].n_subjects_total = length(quartile.entries)
            dataset_summaries_by_lock[lock_to].n_subjects_total = total_subjects
        end
    end

    subject_counter = 0

    try
        for (quartile_idx, quartile) in pairs(quartiles)
            for entry in quartile.entries
                subject_counter += 1
                println(
                    "$(dataset_label): subject $(subject_counter) / $(total_subjects) | $(entry.subject_id) | age=$(round(entry.age; digits = 2)) | quartile=Q$(quartile.group_index)",
                )

                write_subject_assignment_row(assignments_io, dataset_label, quartile, entry)
                subject_summaries = process_subject_group(entry.subject_id, entry.paths)
                write_subject_summary_rows(
                    subject_summaries_io,
                    dataset_label,
                    quartile,
                    entry,
                    subject_summaries,
                )

                finalized_by_lock = Dict{Symbol, Any}()

                for lock_to in LOCK_OPTIONS
                    subject_summary = subject_summaries[lock_to]
                    quartile_summary = summaries_by_lock[lock_to][quartile_idx]
                    dataset_summary = dataset_summaries_by_lock[lock_to]
                    quartile_summary.n_candidate_events += subject_summary.n_candidate_events
                    quartile_summary.n_used_epochs += subject_summary.n_used_epochs
                    quartile_summary.n_dropped_epochs += subject_summary.n_dropped_epochs
                    quartile_summary.n_threshold_filtered_epochs +=
                        subject_summary.n_threshold_filtered_epochs
                    quartile_summary.n_failed_files += subject_summary.n_failed_files
                    quartile_summary.n_missing_files += subject_summary.n_missing_files
                    dataset_summary.n_candidate_events += subject_summary.n_candidate_events
                    dataset_summary.n_used_epochs += subject_summary.n_used_epochs
                    dataset_summary.n_dropped_epochs += subject_summary.n_dropped_epochs
                    dataset_summary.n_threshold_filtered_epochs +=
                        subject_summary.n_threshold_filtered_epochs
                    dataset_summary.n_failed_files += subject_summary.n_failed_files
                    dataset_summary.n_missing_files += subject_summary.n_missing_files

                    finalized = finalize_subject_waveform(subject_summary)
                    finalized_by_lock[lock_to] = finalized
                    isnothing(finalized) && continue
                    accumulate_subject_waveform!(
                        quartile_summary,
                        finalized.time_ms,
                        finalized.waveform,
                    )
                    accumulate_subject_waveform!(
                        dataset_summary,
                        finalized.time_ms,
                        finalized.waveform,
                    )
                end

                write_subject_waveform_rows(
                    subject_waveforms_io,
                    dataset_label,
                    quartile,
                    entry,
                    finalized_by_lock,
                )

                subject_counter % 25 == 0 && GC.gc()
            end
        end
    finally
        close(assignments_io)
        close(subject_summaries_io)
        close(subject_waveforms_io)
    end

    return (
        quartiles = quartiles,
        quartile_summaries = summaries_by_lock,
        dataset_summaries = dataset_summaries_by_lock,
        assignments_path = assignments_path,
        subject_summaries_path = subject_summaries_path,
        subject_waveforms_path = subject_waveforms_path,
    )
end

function main()
    mkpath(OUTPUT_DIR)
    py_mne.set_log_level("ERROR")

    groups_by_dataset = load_group_definitions(GROUPS_FILE)
    participants_tsv = resolve_participants_tsv()
    ages = load_ages(participants_tsv)
    dataset_results = Dict{String, NamedTuple}()
    metadata_path = write_metadata_tsv(groups_by_dataset, participants_tsv)
    println("Saved $(metadata_path)")

    for dataset_label in ["clean", "eyelink"]
        dataset_result = process_dataset(
            dataset_label,
            groups_by_dataset[dataset_label],
            ages,
        )

        quartiles = dataset_result.quartiles
        summaries_by_lock = dataset_result.quartile_summaries
        dataset_summaries_by_lock = dataset_result.dataset_summaries

        print_dataset_summary(dataset_label, quartiles, summaries_by_lock)

        fig = plot_dataset(dataset_label, quartiles, summaries_by_lock)
        output_path = plot_output_path(dataset_label)
        save(output_path, fig)
        println("Saved $(output_path)")

        summary_path = write_summary_tsv(dataset_label, quartiles, summaries_by_lock)
        println("Saved $(summary_path)")

        quartile_series_path = write_quartile_series_tsv(dataset_label, quartiles, summaries_by_lock)
        println("Saved $(quartile_series_path)")
        println("Saved $(dataset_result.assignments_path)")
        println("Saved $(dataset_result.subject_summaries_path)")
        println("Saved $(dataset_result.subject_waveforms_path)")

        dataset_results[dataset_label] = dataset_result
    end

    comparison_summaries = Dict(
        dataset_label => dataset_results[dataset_label].dataset_summaries for dataset_label in ["clean", "eyelink"]
    )
    comparison_fig = plot_dataset_comparison(comparison_summaries)
    comparison_output = comparison_output_path()
    save(comparison_output, comparison_fig)
    println("Saved $(comparison_output)")

    comparison_summary = write_comparison_summary_tsv(comparison_summaries)
    println("Saved $(comparison_summary)")

    comparison_series = write_comparison_series_tsv(comparison_summaries)
    println("Saved $(comparison_series)")
end

if get(ENV, "NAIVE_FRP_SKIP_MAIN", "0") != "1"
    main()
end
