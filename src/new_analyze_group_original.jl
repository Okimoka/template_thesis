using CSV
using CairoMakie
using DataFrames
using Statistics
using BSplineKit
using Unfold
using UnfoldMakie

"""
Given an input folder (DEFAULT_MODEL_DIR), this script 
computes the average effects plot of all models (.jld2) contained in that folder.
Various parameters can be specified in how the effects plot should be computed

Mostly written by LLM
"""

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_MODEL_DIR = joinpath(PROJECT_ROOT, "final_fitted_models_baseline_clean_synced")
const DEFAULT_OUTPUT_BASE_DIR = joinpath(@__DIR__, "visual_outputs")
const DEFAULT_SUFFIX = [".jld2"]
const DEFAULT_AMPLITUDES = 2:2:10
const PLOT_CHANNEL = 82
const PLOT_ELECTRODE = "E82"
const EXTREME_THRESHOLD = 50.0
const GROUP_WINSOR_PROPORTION = 0.10
const USE_FAST_CHANNEL_EXTRACTION = true

const ROI_CHANNELS = nothing #[81, 88, 73, 82, 74, 94, 89, 83, 75, 70, 69, 68]

CairoMakie.activate!()

normalize_suffix(suffix::AbstractString) = endswith(suffix, ".jld2") ? suffix : suffix * ".jld2"

normalize_suffixes(suffix::AbstractString) = [normalize_suffix(suffix)]

function normalize_suffixes(suffixes)
    normalized_suffixes = [normalize_suffix(suffix) for suffix in suffixes]
    isempty(normalized_suffixes) && error("At least one filename suffix must be provided.")
    return normalized_suffixes
end

function print_usage()
    println("Usage: julia --project=. new_analyze_group.jl [model_dir] [filename_suffix] [reference_model_dir]")
    println("Defaults:")
    println("  model_dir = $(DEFAULT_MODEL_DIR)")
    println("  filename_suffix = $(DEFAULT_SUFFIX)")
    println("  reference_model_dir = <none>")
end

sanitize_path_component(path::AbstractString) = replace(
    basename(normpath(path)),
    r"[^A-Za-z0-9._-]+" => "_",
)

default_output_dir(model_dir::AbstractString) = joinpath(
    DEFAULT_OUTPUT_BASE_DIR,
    sanitize_path_component(model_dir),
)

comparison_output_dir(output_dir::AbstractString, reference_model_dir::AbstractString) = joinpath(
    output_dir,
    "compare_vs_" * sanitize_path_component(reference_model_dir),
)

function find_model_files(model_dir::AbstractString, suffix::AbstractString)
    isdir(model_dir) || error("Model directory does not exist: $model_dir")

    normalized_suffix = normalize_suffix(suffix)
    model_files = filter(file -> endswith(file, normalized_suffix), readdir(model_dir; join = true))
    sort!(model_files)

    isempty(model_files) && error("No model files in $model_dir matched suffix $normalized_suffix")
    return model_files, normalized_suffix
end

function model_file_index(model_files)
    index = Dict{String, String}()
    for model_file in model_files
        index[basename(model_file)] = model_file
    end
    return index
end

function match_model_files(primary_files, reference_files)
    primary_index = model_file_index(primary_files)
    reference_index = model_file_index(reference_files)

    matched_names = sort!(collect(intersect(keys(primary_index), keys(reference_index))))
    primary_only = sort!(collect(setdiff(keys(primary_index), keys(reference_index))))
    reference_only = sort!(collect(setdiff(keys(reference_index), keys(primary_index))))

    return (
        matched_primary = [primary_index[name] for name in matched_names],
        matched_reference = [reference_index[name] for name in matched_names],
        matched_names = matched_names,
        primary_only = primary_only,
        reference_only = reference_only,
    )
end

function subject_from_model_file(model_file::AbstractString)
    stem = first(splitext(basename(model_file)))
    parts = split(stem, '|')
    return isempty(parts) ? stem : parts[1]
end

function add_subject!(df::DataFrame, subject::AbstractString)
    df[!, :subject] = fill(String(subject), nrow(df))
    return df
end

function add_series!(df::DataFrame)
    df[!, :series] = string.(df.eventname, " / ", df.Amplitude)
    return df
end

valid_effect_yhat(y) = !ismissing(y) && isfinite(y)

function filter_valid_effect_rows(
    effect_df::DataFrame;
    channels = nothing,
    context::AbstractString = "effect extraction",
)
    selected_df = isnothing(channels) ? effect_df : effect_df[in.(effect_df.channel, Ref(channels)), :]
    valid_mask = valid_effect_yhat.(selected_df.yhat)
    dropped_count = sum(.!valid_mask)

    if dropped_count > 0
        dropped_amplitudes = if :Amplitude in names(selected_df)
            sort(unique(selected_df[.!valid_mask, :Amplitude]))
        else
            Any[]
        end

        @warn "Dropped non-finite effect rows" context dropped_count dropped_amplitudes
    end

    return selected_df[valid_mask, :]
end

function selected_channels(; channel = PLOT_CHANNEL, roi_channels = ROI_CHANNELS)
    isnothing(roi_channels) && return [channel]

    channels = collect(roi_channels)
    isempty(channels) && error("ROI_CHANNELS must be `nothing` or contain at least one channel.")
    return channels
end

function channel_selection_text(; channel = PLOT_CHANNEL, electrode = PLOT_ELECTRODE, roi_channels = ROI_CHANNELS)
    isnothing(roi_channels) && return "channel $(channel) ($(electrode))"
    return "ROI average across channels $(collect(roi_channels))"
end

function average_roi_effects(effect_df::DataFrame; channel = PLOT_CHANNEL)
    roi_effect_df = combine(
        groupby(effect_df, [:Amplitude, :time, :eventname]),
        :yhat => mean => :yhat,
    )
    roi_effect_df[!, :channel] = fill(channel, nrow(roi_effect_df))
    select!(roi_effect_df, [:channel, :Amplitude, :time, :eventname, :yhat])
    return roi_effect_df
end

function channel_effects_public(
    model;
    amplitudes = DEFAULT_AMPLITUDES,
    channels = [PLOT_CHANNEL],
)
    effect_df = effects(Dict(:Amplitude => amplitudes), model)
    return filter_valid_effect_rows(
        effect_df;
        channels = channels,
        context = "public effects",
    )
end

function channel_effects_fast(
    model;
    amplitudes = DEFAULT_AMPLITUDES,
    channels = [PLOT_CHANNEL],
)
    reference_grid = Unfold.expand_grid(Dict(:Amplitude => amplitudes))
    forms = Unfold.formulas(model)
    model_matrices = Unfold.modelmatrix(model, false)
    forms_typical = Unfold._typify(typeof(model), reference_grid, forms, model_matrices, mean)
    reference_grids = fill(reference_grid, length(forms_typical))
    channel_coefs = @view coef(model)[channels, :]
    predicted = Unfold.predict_no_overlap(model, channel_coefs, forms_typical, reference_grids)
    effect_df = Unfold.result_to_table(
        predicted,
        reference_grids,
        Unfold.times(model),
        Unfold.eventnames(model),
    )
    effect_df[!, :channel] = getindex.(Ref(channels), Int.(effect_df.channel))
    return filter_valid_effect_rows(
        effect_df;
        channels = channels,
        context = "fast effects",
    )
end

function extract_model_effects(model_file::AbstractString; amplitudes = DEFAULT_AMPLITUDES, channel = PLOT_CHANNEL, roi_channels = ROI_CHANNELS)
    channels = selected_channels(; channel = channel, roi_channels = roi_channels)
    model = Unfold.load(model_file, UnfoldModel; generate_Xs = false)

    if any(x -> !isfinite(x), coef(model))
        @warn "Model coefficients contain non-finite values; invalid predicted rows will be dropped" model_file = basename(model_file)
    end

    effect_df = if USE_FAST_CHANNEL_EXTRACTION
        try
            channel_effects_fast(model; amplitudes = amplitudes, channels = channels)
        catch err
            println(stderr, "Fast selected-channel extraction failed for $(basename(model_file)); falling back.")
            Base.display_error(stderr, err, catch_backtrace())
            println(stderr)
            channel_effects_public(model; amplitudes = amplitudes, channels = channels)
        end
    else
        channel_effects_public(model; amplitudes = amplitudes, channels = channels)
    end

    nrow(effect_df) > 0 || error("No effect rows found for $(channel_selection_text(; channel = channel, roi_channels = roi_channels)) in $model_file")

    if !isnothing(roi_channels)
        effect_df = average_roi_effects(effect_df; channel = channel)
    end

    add_subject!(effect_df, subject_from_model_file(model_file))
    add_series!(effect_df)
    effect_df[!, :model_file] = fill(basename(model_file), nrow(effect_df))
    return effect_df
end

function collect_all_effects(model_files; amplitudes = DEFAULT_AMPLITUDES, channel = PLOT_CHANNEL, roi_channels = ROI_CHANNELS)
    effect_frames = Vector{Union{Nothing,DataFrame}}(nothing, length(model_files))
    loaded_subjects = Vector{Union{Nothing,String}}(nothing, length(model_files))
    error_messages = Vector{Union{Nothing,String}}(nothing, length(model_files))
    total_files = length(model_files)

    if Threads.nthreads() == 1
        for (i, model_file) in enumerate(model_files)
            println("[$i/$total_files] Loading $(basename(model_file))")
            flush(stdout)

            try
                effect_df = extract_model_effects(
                    model_file;
                    amplitudes = amplitudes,
                    channel = channel,
                    roi_channels = roi_channels,
                )

                if nrow(effect_df) == 0
                    continue
                end

                subject = subject_from_model_file(model_file)
                effect_frames[i] = effect_df
                loaded_subjects[i] = subject
                println("    extracted $(nrow(effect_df)) rows for $subject")
            catch err
                Base.display_error(stderr, err, catch_backtrace())
                println(stderr)
                println(stderr, "Skipping failed model and continuing: ", model_file)
            end
        end
    else
        println("Loading $total_files models using $(Threads.nthreads()) Julia threads")
        flush(stdout)

        Threads.@threads for i in eachindex(model_files)
            model_file = model_files[i]
            try
                effect_df = extract_model_effects(
                    model_file;
                    amplitudes = amplitudes,
                    channel = channel,
                    roi_channels = roi_channels,
                )

                if nrow(effect_df) == 0
                    continue
                end

                effect_frames[i] = effect_df
                loaded_subjects[i] = subject_from_model_file(model_file)
            catch err
                error_messages[i] = sprint() do io
                    Base.display_error(io, err, catch_backtrace())
                end
            end
        end

        failed_files = findall(x -> !isnothing(x), error_messages)
        for i in failed_files
            print(stderr, error_messages[i])
            println(stderr)
            println(stderr, "Skipping failed model and continuing: ", model_files[i])
        end
    end

    valid_frames = [df for df in effect_frames if !isnothing(df)]
    valid_subjects = [subject for subject in loaded_subjects if !isnothing(subject)]

    isempty(valid_frames) && error("No effect data could be extracted from the matching model files.")
    return reduce(vcat, valid_frames), collect(valid_subjects)
end

function warn_duplicate_subjects(subjects::Vector{String})
    counts = Dict{String, Int}()
    for subject in subjects
        counts[subject] = get(counts, subject, 0) + 1
    end

    duplicates = sort(filter(pair -> last(pair) > 1, collect(counts)); by = first)
    isempty(duplicates) && return nothing

    duplicate_text = join(["$(subject) ($(count)x)" for (subject, count) in duplicates], ", ")
    @warn "Some subjects have multiple matching model files and will contribute multiple models" duplicates = duplicate_text
    return nothing
end

function detect_bad_subjects(all_effects::DataFrame; threshold = EXTREME_THRESHOLD)
    bad_rows = subset(
        all_effects,
        :yhat => ByRow(x -> valid_effect_yhat(x) && abs(x) > threshold),
    )
    return sort(unique(String.(bad_rows.subject)))
end

function resolve_group_statistic(; statistic::Union{Nothing,Symbol} = nothing, use_median::Bool = false)
    resolved_statistic = isnothing(statistic) ? (use_median ? :median : :mean) : statistic
    resolved_statistic in (:mean, :median, :wins) || error(
        "Unsupported group statistic: $resolved_statistic. Expected one of [:mean, :median, :wins].",
    )
    return resolved_statistic
end

function group_statistic_label(statistic::Symbol)
    statistic == :mean && return "mean"
    statistic == :median && return "median"
    statistic == :wins && return "wins"
    error("Unsupported group statistic: $statistic")
end

function group_statistic_display(statistic::Symbol)
    statistic == :wins && return "winsorized mean"
    return group_statistic_label(statistic)
end

function winsorized_mean(values; proportion = GROUP_WINSOR_PROPORTION)
    0.0 <= proportion < 0.5 || error("GROUP_WINSOR_PROPORTION must be in [0, 0.5).")

    numeric_values = sort(Float64.(collect(values)))
    n_values = length(numeric_values)
    n_values > 0 || error("Cannot compute a winsorized mean from an empty collection.")

    n_winsor = floor(Int, proportion * n_values)
    if n_winsor == 0 || n_values == 1
        return mean(numeric_values)
    end

    lower = numeric_values[n_winsor + 1]
    upper = numeric_values[n_values - n_winsor]
    return mean(clamp.(numeric_values, lower, upper))
end

function group_statistic_reducer(statistic::Symbol)
    statistic == :mean && return mean
    statistic == :median && return median
    statistic == :wins && return winsorized_mean
    error("Unsupported group statistic: $statistic")
end

function aggregate_group_effects(
    all_effects::DataFrame;
    use_median::Bool = false,
    statistic::Union{Nothing,Symbol} = nothing,
    bad_subjects = String[],
)
    clean_effects = subset(
        all_effects,
        :subject => ByRow(x -> x ∉ bad_subjects),
        :yhat => ByRow(valid_effect_yhat),
    )
    nrow(clean_effects) > 0 || error("No effect rows remained after subject filtering.")

    resolved_statistic = resolve_group_statistic(; statistic = statistic, use_median = use_median)
    reducer = group_statistic_reducer(resolved_statistic)
    group_effects = combine(
        groupby(clean_effects, [:channel, :Amplitude, :time, :eventname, :series]),
        :yhat => reducer => :yhat,
    )

    return group_effects, sort(unique(String.(clean_effects.subject)))
end

function difference_summary(values)
    isempty(values) && return (
        n = 0,
        max_abs = missing,
        mean_abs = missing,
        rmse = missing,
        mean_signed = missing,
    )

    numeric_values = Float64.(collect(values))
    abs_values = abs.(numeric_values)
    return (
        n = length(numeric_values),
        max_abs = maximum(abs_values),
        mean_abs = mean(abs_values),
        rmse = sqrt(mean(numeric_values .^ 2)),
        mean_signed = mean(numeric_values),
    )
end

function build_effect_join(primary_effects::DataFrame, reference_effects::DataFrame)
    primary_df = copy(primary_effects)
    reference_df = copy(reference_effects)
    rename!(primary_df, :yhat => :yhat_primary)
    rename!(reference_df, :yhat => :yhat_reference)

    joined = innerjoin(
        primary_df,
        reference_df,
        on = [:subject, :model_file, :channel, :Amplitude, :time, :eventname, :series],
    )
    joined.diff = joined.yhat_primary .- joined.yhat_reference
    joined.abs_diff = abs.(joined.diff)
    return joined
end

function summarize_model_differences(joined_effects::DataFrame)
    return combine(
        groupby(joined_effects, [:subject, :model_file]),
        :diff => length => :n_rows,
        :diff => (x -> maximum(abs.(x))) => :max_abs_diff,
        :diff => (x -> mean(abs.(x))) => :mean_abs_diff,
        :diff => (x -> sqrt(mean(x .^ 2))) => :rmse_diff,
        :diff => mean => :mean_signed_diff,
    )
end

function summarize_subject_differences(model_differences::DataFrame)
    return combine(
        groupby(model_differences, :subject),
        :model_file => length => :n_models,
        :max_abs_diff => maximum => :max_abs_diff,
        :mean_abs_diff => mean => :mean_abs_diff,
        :rmse_diff => mean => :mean_rmse_diff,
        :mean_signed_diff => mean => :mean_signed_diff,
    )
end

function summarize_group_differences(joined_effects::DataFrame)
    group_diff = combine(
        groupby(joined_effects, [:channel, :Amplitude, :time, :eventname, :series]),
        :yhat_primary => mean => :yhat_primary,
        :yhat_reference => mean => :yhat_reference,
        :diff => mean => :yhat_difference,
        :abs_diff => mean => :yhat_abs_difference,
        :subject => (x -> length(unique(x))) => :n_subjects,
        :model_file => (x -> length(unique(x))) => :n_models,
    )

    signed_plot_df = select(
        copy(group_diff),
        [:channel, :Amplitude, :time, :eventname, :series, :n_subjects, :n_models],
    )
    signed_plot_df.yhat = group_diff.yhat_difference

    abs_plot_df = select(
        copy(group_diff),
        [:channel, :Amplitude, :time, :eventname, :series, :n_subjects, :n_models],
    )
    abs_plot_df.yhat = group_diff.yhat_abs_difference

    return group_diff, signed_plot_df, abs_plot_df
end

function write_table(output_file::AbstractString, table::DataFrame)
    mkpath(dirname(output_file))
    CSV.write(output_file, table)
    return output_file
end

function comparison_filename(suffix::AbstractString, stem::AbstractString)
    suffix_stem = first(splitext(basename(suffix)))
    return "$(stem)_$(suffix_stem).csv"
end

function comparison_plot_filename(suffix::AbstractString, stem::AbstractString)
    suffix_stem = first(splitext(basename(suffix)))
    return "$(stem)_$(suffix_stem).png"
end

function compare_model_directories(
    primary_effects::DataFrame,
    reference_effects::DataFrame,
    normalized_suffix::AbstractString;
    output_dir::AbstractString,
)
    joined_effects = build_effect_join(primary_effects, reference_effects)
    nrow(joined_effects) > 0 || error("No matched effect rows were available for comparison.")

    model_differences = sort(summarize_model_differences(joined_effects), :rmse_diff, rev = true)
    subject_differences = sort(
        summarize_subject_differences(model_differences),
        :mean_rmse_diff,
        rev = true,
    )
    group_difference_table, signed_plot_df, abs_plot_df = summarize_group_differences(joined_effects)
    group_diff_summary = difference_summary(joined_effects.diff)

    write_table(
        joinpath(output_dir, comparison_filename(normalized_suffix, "model_differences")),
        model_differences,
    )
    write_table(
        joinpath(output_dir, comparison_filename(normalized_suffix, "subject_differences")),
        subject_differences,
    )
    write_table(
        joinpath(output_dir, comparison_filename(normalized_suffix, "group_differences")),
        group_difference_table,
    )

    save_group_effects(
        signed_plot_df,
        joinpath(
            output_dir,
            comparison_plot_filename(normalized_suffix, "group_effects_mean_difference"),
        ),
    )
    save_group_effects(
        abs_plot_df,
        joinpath(
            output_dir,
            comparison_plot_filename(normalized_suffix, "group_effects_mean_abs_difference"),
        ),
    )

    return model_differences, subject_differences, group_diff_summary
end

function output_filename(
    suffix::AbstractString;
    use_median::Bool = false,
    statistic::Union{Nothing,Symbol} = nothing,
    filter_extremes::Bool,
)
    suffix_stem = first(splitext(basename(suffix)))
    resolved_statistic = resolve_group_statistic(; statistic = statistic, use_median = use_median)
    stat_label = group_statistic_label(resolved_statistic)
    filter_label = filter_extremes ? "filtered" : "unfiltered"
    return "group_effects_$(suffix_stem)_$(stat_label)_$(filter_label).png"
end

function save_group_effects(group_effects::DataFrame, output_file::AbstractString)
    fig = plot_erp(
        group_effects;
        mapping = (; y = :yhat, color = :Amplitude, group = :series),
    )
    save(output_file, fig)
    return output_file
end

function process_suffix(
    model_dir::AbstractString,
    normalized_suffix::AbstractString;
    output_dir::AbstractString,
    reference_model_dir::Union{Nothing,AbstractString} = nothing,
)
    println()
    println("Using filename suffix $(normalized_suffix)")
    model_files, normalized_suffix = find_model_files(model_dir, normalized_suffix)
    println("Found $(length(model_files)) matching model files")

    all_effects, loaded_subjects = collect_all_effects(model_files)
    warn_duplicate_subjects(loaded_subjects)

    unique_subjects = sort(unique(loaded_subjects))
    all_bad_subjects = detect_bad_subjects(all_effects; threshold = EXTREME_THRESHOLD)

    println("Loaded $(length(unique_subjects)) unique subjects")
    println("Subjects beyond +/-$(Int(EXTREME_THRESHOLD)) uV: $(length(all_bad_subjects))")

    combinations = [
        (statistic = :mean, filter_extremes = false),
        (statistic = :mean, filter_extremes = true),
        (statistic = :median, filter_extremes = false),
        (statistic = :median, filter_extremes = true),
        (statistic = :wins, filter_extremes = false),
        (statistic = :wins, filter_extremes = true),
    ]

    for (i, combo) in enumerate(combinations)
        bad_subjects = combo.filter_extremes ? all_bad_subjects : String[]
        stat_label = group_statistic_display(combo.statistic)
        output_file = joinpath(
            output_dir,
            output_filename(
                normalized_suffix;
                statistic = combo.statistic,
                filter_extremes = combo.filter_extremes,
            ),
        )

        aggregate_result = try
            aggregate_group_effects(
                all_effects;
                statistic = combo.statistic,
                bad_subjects = bad_subjects,
            )
        catch err
            if err isa ErrorException && sprint(showerror, err) == "No effect rows remained after subject filtering."
                println(
                    "[$i/$(length(combinations))] Skipping $(basename(output_file)) " *
                    "(stat = $stat_label, filter_extremes = $(combo.filter_extremes), " *
                    "kept_subjects = 0, removed_subjects = $(length(bad_subjects)); no rows remained)",
                )
                flush(stdout)
                continue
            end
            rethrow()
        end

        group_effects, kept_subjects = aggregate_result
        println(
            "[$i/$(length(combinations))] Saving $(basename(output_file)) " *
            "(stat = $stat_label, filter_extremes = $(combo.filter_extremes), " *
            "kept_subjects = $(length(kept_subjects)), removed_subjects = $(length(bad_subjects)))",
        )
        flush(stdout)

        save_group_effects(group_effects, output_file)
    end

    if !isnothing(reference_model_dir)
        println()
        println("Comparing against $(abspath(reference_model_dir))")
        reference_files, _ = find_model_files(reference_model_dir, normalized_suffix)
        matched = match_model_files(model_files, reference_files)
        println(
            "Matched $(length(matched.matched_names)) model files " *
            "(primary_only = $(length(matched.primary_only)), " *
            "reference_only = $(length(matched.reference_only)))",
        )

        comparison_dir = comparison_output_dir(output_dir, reference_model_dir)
        mkpath(comparison_dir)

        isempty(matched.matched_primary) && error("No matched model files were found between the two directories.")

        matched_primary_names = Set(matched.matched_names)
        matched_primary_effects = subset(
            all_effects,
            :model_file => ByRow(x -> x in matched_primary_names),
        )
        reference_effects, loaded_reference_subjects = collect_all_effects(matched.matched_reference)
        warn_duplicate_subjects(loaded_reference_subjects)

        model_differences, subject_differences, group_diff_summary = compare_model_directories(
            matched_primary_effects,
            reference_effects,
            normalized_suffix;
            output_dir = comparison_dir,
        )

        coverage_report = DataFrame([
            (
                suffix = normalized_suffix,
                primary_model_dir = abspath(model_dir),
                reference_model_dir = abspath(reference_model_dir),
                n_primary_files = length(model_files),
                n_reference_files = length(reference_files),
                n_matched_files = length(matched.matched_names),
                n_primary_only_files = length(matched.primary_only),
                n_reference_only_files = length(matched.reference_only),
                n_loaded_primary_models = length(unique(String.(matched_primary_effects.model_file))),
                n_loaded_reference_models = length(unique(String.(reference_effects.model_file))),
                n_compared_models = nrow(model_differences),
                primary_only_files = join(matched.primary_only, ";"),
                reference_only_files = join(matched.reference_only, ";"),
            ),
        ])
        write_table(
            joinpath(comparison_dir, comparison_filename(normalized_suffix, "coverage")),
            coverage_report,
        )

        println(
            "Comparison summary: max_abs_diff = $(group_diff_summary.max_abs), " *
            "mean_abs_diff = $(group_diff_summary.mean_abs), " *
            "rmse_diff = $(group_diff_summary.rmse)",
        )

        if nrow(model_differences) > 0
            top_models = first(model_differences, min(5, nrow(model_differences)))
            println("Top model mismatches:")
            show(stdout, "text/plain", top_models)
            println()
        end

        if nrow(subject_differences) > 0
            top_subjects = first(subject_differences, min(5, nrow(subject_differences)))
            println("Top subject mismatches:")
            show(stdout, "text/plain", top_subjects)
            println()
        end
    end

    return nothing
end

function main(
    ;
    model_dir::AbstractString = DEFAULT_MODEL_DIR,
    suffix = DEFAULT_SUFFIX,
    output_dir::Union{Nothing,AbstractString} = nothing,
    reference_model_dir::Union{Nothing,AbstractString} = nothing,
)
    normalized_suffixes = normalize_suffixes(suffix)
    resolved_output_dir = isnothing(output_dir) ? default_output_dir(model_dir) : output_dir

    println("Looking for models in $(abspath(model_dir))")
    println("Using filename suffixes $(normalized_suffixes)")
    println("Using amplitudes $(collect(DEFAULT_AMPLITUDES)) and $(channel_selection_text())")
    if !isnothing(reference_model_dir)
        println("Reference models: $(abspath(reference_model_dir))")
    end

    mkpath(resolved_output_dir)

    for normalized_suffix in normalized_suffixes
        process_suffix(
            model_dir,
            normalized_suffix;
            output_dir = resolved_output_dir,
            reference_model_dir = reference_model_dir,
        )
    end

    println("Finished. Outputs written to $(abspath(resolved_output_dir))")
    return nothing
end

if !(isdefined(Main, :CODEX_NO_AUTORUN) && getfield(Main, :CODEX_NO_AUTORUN) === true) && abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 3
        print_usage()
        error("Too many command line arguments.")
    end

    model_dir = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_MODEL_DIR
    suffix = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_SUFFIX
    reference_model_dir = length(ARGS) >= 3 ? ARGS[3] : nothing

    main(; model_dir = model_dir, suffix = suffix, reference_model_dir = reference_model_dir)
end
