using CSV
using CairoMakie
using DataFrames

"""
Helper script for run_full_group_analyses.py

LLM Written
"""

env_or_default(varname::AbstractString, default_value::AbstractString) = get(ENV, varname, default_value)

const DEFAULT_GROUP_OUTPUTS_ROOT = env_or_default(
    "UNFOLD_GROUP_OUTPUTS_ROOT",
    joinpath(@__DIR__, "visual_outputs"),
)
const DEFAULT_SUFFIX_STEM = "proc-clean_raw"
const DEFAULT_STATISTIC = "wins"
const DEFAULT_FILTER_LABEL = "filtered"
const DEFAULT_AMPLITUDES = 2:2:10
const PLOT_LINEWIDTH = 3

CairoMakie.activate!()

function summary_stem(summary_file::AbstractString)
    stem = first(splitext(basename(summary_file)))
    return replace(stem, r"_groups$" => "")
end

function default_output_dir(summary_file::AbstractString)
    return joinpath(
        DEFAULT_GROUP_OUTPUTS_ROOT,
        summary_stem(summary_file) * "_amplitude_comparisons",
    )
end

function default_output_prefix(summary_file::AbstractString)
    return summary_stem(summary_file) * "_comparison"
end

function group_effects_csv_path(
    group_name::AbstractString;
    suffix_stem = DEFAULT_SUFFIX_STEM,
    statistic = DEFAULT_STATISTIC,
    filter_label = DEFAULT_FILTER_LABEL,
)
    return joinpath(
        DEFAULT_GROUP_OUTPUTS_ROOT,
        group_name,
        "group_effects_$(suffix_stem)_$(statistic)_$(filter_label).csv",
    )
end

function load_group_summary(summary_file::AbstractString)
    isfile(summary_file) || error("Group summary file does not exist: $summary_file")
    summary_df = CSV.read(summary_file, DataFrame; delim = '\t')
    required_columns = ["group_index", "group_name"]
    missing_columns = setdiff(required_columns, names(summary_df))
    isempty(missing_columns) || error("Missing required column(s) in $(summary_file): $(join(missing_columns, ", "))")

    if :group_label ∉ names(summary_df)
        if (:age_min ∈ names(summary_df)) && (:age_max ∈ names(summary_df))
            summary_df[!, :group_label] = [
                "Q$(row.group_index) ($(round(row.age_min; digits = 2))-$(round(row.age_max; digits = 2))y)"
                for row in eachrow(summary_df)
            ]
        else
            summary_df[!, :group_label] = String.(summary_df.group_name)
        end
    end

    sort!(summary_df, :group_index)
    return summary_df
end

function load_group_effects(
    summary_df::DataFrame;
    suffix_stem = DEFAULT_SUFFIX_STEM,
    statistic = DEFAULT_STATISTIC,
    filter_label = DEFAULT_FILTER_LABEL,
)
    effect_frames = DataFrame[]

    for row in eachrow(summary_df)
        input_file = group_effects_csv_path(
            row.group_name;
            suffix_stem = suffix_stem,
            statistic = statistic,
            filter_label = filter_label,
        )
        isfile(input_file) || error("Expected group-effects table is missing: $input_file")

        effect_df = CSV.read(input_file, DataFrame)
        effect_df[!, :group_index] = fill(Int(row.group_index), nrow(effect_df))
        effect_df[!, :group_name] = fill(String(row.group_name), nrow(effect_df))
        effect_df[!, :group_label] = fill(String(row.group_label), nrow(effect_df))
        push!(effect_frames, effect_df)
    end

    isempty(effect_frames) && error("No group-effects tables were loaded.")
    return reduce(vcat, effect_frames)
end

function ensure_single_event(all_effects::DataFrame)
    if :eventname ∉ names(all_effects)
        return nothing
    end

    event_names = sort(unique(String.(all_effects.eventname)))
    length(event_names) <= 1 && return nothing
    error(
        "Expected a single eventname for the overlay plots, but found: " *
        join(event_names, ", "),
    )
end

group_palette(n_groups::Integer) = collect(Makie.wong_colors()[1:n_groups])

function save_with_extensions(fig::Figure, output_base::AbstractString)
    mkpath(dirname(output_base))
    output_files = String[]
    for ext in ("png", "svg")
        output_file = output_base * "." * ext
        save(output_file, fig)
        push!(output_files, output_file)
    end
    return output_files
end

function amplitude_output_base(
    amplitude::Integer;
    output_dir::AbstractString,
    output_prefix::AbstractString,
    suffix_stem = DEFAULT_SUFFIX_STEM,
    statistic = DEFAULT_STATISTIC,
    filter_label = DEFAULT_FILTER_LABEL,
)
    output_name = "$(output_prefix)_amp_$(lpad(string(amplitude), 2, '0'))_$(suffix_stem)_$(statistic)_$(filter_label)"
    return joinpath(output_dir, output_name)
end

function legend_output_base(
    ;
    output_dir::AbstractString,
    output_prefix::AbstractString,
    suffix_stem = DEFAULT_SUFFIX_STEM,
    statistic = DEFAULT_STATISTIC,
    filter_label = DEFAULT_FILTER_LABEL,
)
    output_name = "$(output_prefix)_legend_$(suffix_stem)_$(statistic)_$(filter_label)"
    return joinpath(output_dir, output_name)
end

function plot_amplitude_overlay(
    all_effects::DataFrame,
    summary_df::DataFrame,
    amplitude::Integer;
    output_dir::AbstractString,
    output_prefix::AbstractString,
    suffix_stem = DEFAULT_SUFFIX_STEM,
    statistic = DEFAULT_STATISTIC,
    filter_label = DEFAULT_FILTER_LABEL,
)
    amplitude_df = subset(all_effects, :Amplitude => ByRow(==(amplitude)))
    nrow(amplitude_df) > 0 || error("No rows found for amplitude $amplitude")

    fig = Figure(size = (1200, 800))
    ax = Axis(
        fig[1, 1];
        title = "",
        xlabel = "",
        ylabel = "",
        titlegap = 0,
        xlabelpadding = 0,
        ylabelpadding = 0,
    )

    palette = group_palette(nrow(summary_df))

    for (color_index, row) in enumerate(eachrow(summary_df))
        group_df = subset(amplitude_df, :group_index => ByRow(==(Int(row.group_index))))
        nrow(group_df) > 0 || error("No rows found for group $(row.group_name) at amplitude $amplitude")
        sorted_group = sort(group_df, :time)
        lines!(
            ax,
            sorted_group.time,
            sorted_group.yhat;
            color = palette[color_index],
            linewidth = PLOT_LINEWIDTH,
        )
    end

    output_base = amplitude_output_base(
        amplitude;
        output_dir = output_dir,
        output_prefix = output_prefix,
        suffix_stem = suffix_stem,
        statistic = statistic,
        filter_label = filter_label,
    )
    return save_with_extensions(fig, output_base)
end

function save_standalone_legend(
    summary_df::DataFrame;
    output_dir::AbstractString,
    output_prefix::AbstractString,
    suffix_stem = DEFAULT_SUFFIX_STEM,
    statistic = DEFAULT_STATISTIC,
    filter_label = DEFAULT_FILTER_LABEL,
)
    labels = String.(summary_df.group_label)
    palette = group_palette(length(labels))
    elements = [LineElement(color = palette[i], linewidth = PLOT_LINEWIDTH) for i in eachindex(labels)]

    fig = Figure(size = (560, max(220, 80 * length(labels))))
    Legend(
        fig[1, 1],
        elements,
        labels;
        orientation = :vertical,
        framevisible = false,
        tellheight = true,
        tellwidth = true,
    )

    output_base = legend_output_base(
        output_dir = output_dir,
        output_prefix = output_prefix,
        suffix_stem = suffix_stem,
        statistic = statistic,
        filter_label = filter_label,
    )
    return save_with_extensions(fig, output_base)
end

function main(
    ;
    group_summary_file::AbstractString,
    suffix_stem::AbstractString = DEFAULT_SUFFIX_STEM,
    statistic::AbstractString = DEFAULT_STATISTIC,
    filter_label::AbstractString = DEFAULT_FILTER_LABEL,
    output_dir::AbstractString = default_output_dir(group_summary_file),
    output_prefix::AbstractString = default_output_prefix(group_summary_file),
    amplitudes = DEFAULT_AMPLITUDES,
)
    summary_df = load_group_summary(group_summary_file)
    all_effects = load_group_effects(
        summary_df;
        suffix_stem = suffix_stem,
        statistic = statistic,
        filter_label = filter_label,
    )
    ensure_single_event(all_effects)

    all_output_files = String[]
    for amplitude in amplitudes
        output_files = plot_amplitude_overlay(
            all_effects,
            summary_df,
            amplitude;
            output_dir = output_dir,
            output_prefix = output_prefix,
            suffix_stem = suffix_stem,
            statistic = statistic,
            filter_label = filter_label,
        )
        append!(all_output_files, output_files)
        for output_file in output_files
            println("Saved $(output_file)")
        end
    end

    legend_files = save_standalone_legend(
        summary_df;
        output_dir = output_dir,
        output_prefix = output_prefix,
        suffix_stem = suffix_stem,
        statistic = statistic,
        filter_label = filter_label,
    )
    for output_file in legend_files
        println("Saved $(output_file)")
    end

    println("Finished. Wrote $(length(all_output_files)) overlay plot file(s) and $(length(legend_files)) legend file(s) to $(output_dir)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !(1 <= length(ARGS) <= 5)
        error(
            "Usage: julia --project=. sync_quality/plot_group_amplitude_comparisons.jl " *
            "<group_summary_file> [suffix_stem] [statistic_filter] [output_dir] [output_prefix]"
        )
    end

    group_summary_file = ARGS[1]
    suffix_stem = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_SUFFIX_STEM
    statistic_filter = length(ARGS) >= 3 ? ARGS[3] : "$(DEFAULT_STATISTIC)_$(DEFAULT_FILTER_LABEL)"
    output_dir = length(ARGS) >= 4 ? ARGS[4] : default_output_dir(group_summary_file)
    output_prefix = length(ARGS) >= 5 ? ARGS[5] : default_output_prefix(group_summary_file)

    parts = split(statistic_filter, "_"; limit = 2)
    length(parts) == 2 || error("Expected statistic_filter in the form <statistic>_<filter_label>, got: $statistic_filter")
    statistic, filter_label = parts

    main(
        ;
        group_summary_file = group_summary_file,
        suffix_stem = suffix_stem,
        statistic = statistic,
        filter_label = filter_label,
        output_dir = output_dir,
        output_prefix = output_prefix,
    )
end
