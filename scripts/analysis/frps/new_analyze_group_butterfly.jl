const CODEX_NO_AUTORUN = true
include(joinpath(@__DIR__, "plot_creation", "new_analyze_group_original.jl"))

using CSV
using DataFrames

"""
Fully written by LLM
"""

const BUTTERFLY_DEFAULT_MODEL_DIR = joinpath(@__DIR__, "TRUE_CLEAN_GROUPS")
const BUTTERFLY_DEFAULT_OUTPUT_BASE_DIR = joinpath(@__DIR__, "visual_outputs")
const BUTTERFLY_DEFAULT_AMPLITUDE = 4
const BUTTERFLY_DEFAULT_STATISTIC = :wins
const BUTTERFLY_DEFAULT_FILTER_EXTREMES = true
const BUTTERFLY_Y_LIMITS = (-30.0, 30.0)
const BUTTERFLY_Y_TICK_STEP = 5.0
const BUTTERFLY_X_TICK_STEP = 0.1
const BUTTERFLY_FIGURE_WIDTH = 980
const BUTTERFLY_SUPPORTED_SUFFIXES = [
    "proc-clean_raw.jld2",
    "proc-eyelink_raw.jld2",
]
const BUTTERFLY_AVERAGE_LINE_COLOR = RGBAf(0.866, 0.20, 0.28, 1.0)
const BUTTERFLY_INDIVIDUAL_LINE_ALPHA = 0.24
const BUTTERFLY_INDIVIDUAL_LINE_WIDTH = 0.7
const BUTTERFLY_AVERAGE_LINE_WIDTH = 1.5
const BUTTERFLY_INDIVIDUAL_GRADIENT_START = RGBAf(0.78, 0.96, 0.68, BUTTERFLY_INDIVIDUAL_LINE_ALPHA)
const BUTTERFLY_INDIVIDUAL_GRADIENT_END = RGBAf(0.02, 0.28, 0.08, BUTTERFLY_INDIVIDUAL_LINE_ALPHA)

function print_butterfly_usage()
    default_output_file = joinpath(
        butterfly_default_output_dir(BUTTERFLY_DEFAULT_MODEL_DIR),
        butterfly_output_filename("proc-clean_raw.jld2"),
    )
    println("Usage: julia --project=. new_analyze_group_butterfly.jl [model_dir] [filename_suffix|auto] [output_file]")
    println("Defaults:")
    println("  model_dir = $(BUTTERFLY_DEFAULT_MODEL_DIR)")
    println("  filename_suffix = auto-detect")
    println("  output_file = $(default_output_file)")
end

resolve_local_path(path::AbstractString) = isabspath(path) ? normpath(path) : normpath(joinpath(@__DIR__, path))

function detect_model_suffix(model_dir::AbstractString)
    isdir(model_dir) || error("Model directory does not exist: $model_dir")

    matching_suffixes = String[]
    for suffix in BUTTERFLY_SUPPORTED_SUFFIXES
        normalized_suffix = normalize_suffix(suffix)
        matches = filter(file -> endswith(file, normalized_suffix), readdir(model_dir; join = true))
        isempty(matches) || push!(matching_suffixes, normalized_suffix)
    end

    supported_suffix_text = join(BUTTERFLY_SUPPORTED_SUFFIXES, ", ")
    matching_suffix_text = join(matching_suffixes, ", ")
    isempty(matching_suffixes) && error(
        "Could not detect a supported suffix in $model_dir. Supported suffixes: $supported_suffix_text.",
    )
    length(matching_suffixes) == 1 || error(
        "Found multiple supported suffixes in $model_dir: $matching_suffix_text. Please pass the suffix explicitly.",
    )

    return only(matching_suffixes)
end

butterfly_default_output_dir(model_dir::AbstractString) = joinpath(
    BUTTERFLY_DEFAULT_OUTPUT_BASE_DIR,
    sanitize_path_component(model_dir),
)

function butterfly_output_filename(
    suffix::AbstractString;
    amplitude::Integer = BUTTERFLY_DEFAULT_AMPLITUDE,
    statistic::Symbol = BUTTERFLY_DEFAULT_STATISTIC,
    filter_extremes::Bool = BUTTERFLY_DEFAULT_FILTER_EXTREMES,
)
    suffix_stem = first(splitext(basename(suffix)))
    stat_label = group_statistic_label(statistic)
    filter_label = filter_extremes ? "filtered" : "unfiltered"
    return "group_effects_butterfly_amp$(amplitude)_$(suffix_stem)_$(stat_label)_$(filter_label).png"
end

output_stem(path::AbstractString) = joinpath(dirname(path), first(splitext(basename(path))))

mix_component(a::Real, b::Real, t::Real) = (1 - t) * Float32(a) + t * Float32(b)

function mix_color(c1::RGBAf, c2::RGBAf, t::Real; alpha = mix_component(c1.alpha, c2.alpha, t))
    return RGBAf(
        mix_component(c1.r, c2.r, t),
        mix_component(c1.g, c2.g, t),
        mix_component(c1.b, c2.b, t),
        Float32(alpha),
    )
end

function butterfly_individual_palette(n::Integer)
    n > 0 || return RGBAf[]

    if n == 1
        return [BUTTERFLY_INDIVIDUAL_GRADIENT_END]
    end

    palette = RGBAf[]
    for t in range(0.0, 1.0; length = n)
        # Bias the interpolation slightly toward the endpoints so nearby lines separate more clearly.
        stretched_t = 0.5 * (1 - cos(pi * t))
        color = mix_color(
            BUTTERFLY_INDIVIDUAL_GRADIENT_START,
            BUTTERFLY_INDIVIDUAL_GRADIENT_END,
            stretched_t;
            alpha = BUTTERFLY_INDIVIDUAL_LINE_ALPHA,
        )
        push!(palette, color)
    end
    return palette
end

function butterfly_individual_color_lookup(model_files::Vector{String})
    palette = butterfly_individual_palette(length(model_files))
    return Dict(model_files .=> palette)
end

function butterfly_x_limits(average_effects::DataFrame, individual_effects::DataFrame)
    all_x = vcat(Float64.(average_effects.time), Float64.(individual_effects.time))
    isempty(all_x) && error("No time values were available for plotting.")

    x_min, x_max = extrema(all_x)
    if x_min == x_max
        delta = BUTTERFLY_X_TICK_STEP / 2
        return (x_min - delta, x_max + delta)
    end

    return (x_min, x_max)
end

function butterfly_tick_values(min_value::Real, max_value::Real, step::Real; digits::Integer)
    start_tick = ceil(min_value / step) * step
    end_tick = floor(max_value / step) * step
    if start_tick > end_tick
        midpoint = round((min_value + max_value) / 2; digits = digits)
        return [midpoint]
    end

    n_ticks = floor(Int, ((end_tick - start_tick) / step) + 1 + 1e-9)
    return [round(start_tick + i * step; digits = digits) for i in 0:(n_ticks - 1)]
end

function filter_butterfly_effects(
    all_effects::DataFrame,
    amplitude::Integer;
    bad_subjects = String[],
)
    clean_effects = subset(
        all_effects,
        :subject => ByRow(x -> x ∉ bad_subjects),
        :yhat => ByRow(valid_effect_yhat),
        :Amplitude => ByRow(==(amplitude)),
    )
    nrow(clean_effects) > 0 || error("No amplitude $(amplitude) rows remained after filtering.")
    sort!(clean_effects, [:eventname, :model_file, :time])
    return clean_effects
end

function annotate_average_effects(
    average_effects::DataFrame;
    model_dir::AbstractString,
    suffix::AbstractString,
    amplitude::Integer,
    statistic::Symbol,
    filter_extremes::Bool,
    n_model_files::Integer,
    n_loaded_models::Integer,
    n_subjects_loaded::Integer,
    n_subjects_kept::Integer,
    n_subjects_removed_extreme::Integer,
)
    annotated = copy(average_effects)
    annotated[!, :variant] = fill(basename(normpath(model_dir)), nrow(annotated))
    annotated[!, :model_dir] = fill(abspath(model_dir), nrow(annotated))
    annotated[!, :suffix] = fill(String(suffix), nrow(annotated))
    annotated[!, :target_amplitude] = fill(amplitude, nrow(annotated))
    annotated[!, :statistic] = fill(String(group_statistic_label(statistic)), nrow(annotated))
    annotated[!, :filter_extremes] = fill(filter_extremes, nrow(annotated))
    annotated[!, :n_model_files] = fill(n_model_files, nrow(annotated))
    annotated[!, :n_loaded_models] = fill(n_loaded_models, nrow(annotated))
    annotated[!, :n_subjects_loaded] = fill(n_subjects_loaded, nrow(annotated))
    annotated[!, :n_subjects_kept] = fill(n_subjects_kept, nrow(annotated))
    annotated[!, :n_subjects_removed_extreme] = fill(n_subjects_removed_extreme, nrow(annotated))
    sort!(annotated, [:eventname, :time])
    return annotated
end

function annotate_individual_effects(
    individual_effects::DataFrame;
    model_dir::AbstractString,
    suffix::AbstractString,
    amplitude::Integer,
    statistic::Symbol,
    filter_extremes::Bool,
    n_model_files::Integer,
    n_loaded_models::Integer,
    n_subjects_loaded::Integer,
    n_subjects_kept::Integer,
    n_subjects_removed_extreme::Integer,
)
    annotated = copy(individual_effects)
    annotated[!, :variant] = fill(basename(normpath(model_dir)), nrow(annotated))
    annotated[!, :model_dir] = fill(abspath(model_dir), nrow(annotated))
    annotated[!, :suffix] = fill(String(suffix), nrow(annotated))
    annotated[!, :target_amplitude] = fill(amplitude, nrow(annotated))
    annotated[!, :statistic] = fill(String(group_statistic_label(statistic)), nrow(annotated))
    annotated[!, :filter_extremes] = fill(filter_extremes, nrow(annotated))
    annotated[!, :n_model_files] = fill(n_model_files, nrow(annotated))
    annotated[!, :n_loaded_models] = fill(n_loaded_models, nrow(annotated))
    annotated[!, :n_subjects_loaded] = fill(n_subjects_loaded, nrow(annotated))
    annotated[!, :n_subjects_kept] = fill(n_subjects_kept, nrow(annotated))
    annotated[!, :n_subjects_removed_extreme] = fill(n_subjects_removed_extreme, nrow(annotated))
    sort!(annotated, [:eventname, :model_file, :time])
    return annotated
end

function prepare_butterfly_plot_data(
    model_dir::AbstractString;
    suffix::Union{Nothing,AbstractString} = nothing,
    amplitude::Integer = BUTTERFLY_DEFAULT_AMPLITUDE,
    statistic::Symbol = BUTTERFLY_DEFAULT_STATISTIC,
    filter_extremes::Bool = BUTTERFLY_DEFAULT_FILTER_EXTREMES,
)
    resolved_model_dir = resolve_local_path(model_dir)
    resolved_suffix = isnothing(suffix) ? detect_model_suffix(resolved_model_dir) : normalize_suffix(suffix)

    model_files, _ = find_model_files(resolved_model_dir, resolved_suffix)
    all_effects, loaded_subjects = collect_all_effects(model_files)
    warn_duplicate_subjects(loaded_subjects)

    bad_subjects = filter_extremes ? detect_bad_subjects(all_effects; threshold = EXTREME_THRESHOLD) : String[]
    average_effects, kept_subjects = aggregate_group_effects(
        all_effects;
        statistic = statistic,
        bad_subjects = bad_subjects,
    )
    average_effects = subset(average_effects, :Amplitude => ByRow(==(amplitude)))
    nrow(average_effects) > 0 || error("No aggregated amplitude $(amplitude) rows were available.")
    sort!(average_effects, [:eventname, :time])

    individual_effects = filter_butterfly_effects(
        all_effects,
        amplitude;
        bad_subjects = bad_subjects,
    )

    unique_loaded_subjects = sort(unique(String.(loaded_subjects)))
    plotted_subjects = sort(unique(String.(individual_effects.subject)))
    plotted_models = sort(unique(String.(individual_effects.model_file)))

    annotated_average = annotate_average_effects(
        average_effects;
        model_dir = resolved_model_dir,
        suffix = resolved_suffix,
        amplitude = amplitude,
        statistic = statistic,
        filter_extremes = filter_extremes,
        n_model_files = length(model_files),
        n_loaded_models = length(unique(String.(all_effects.model_file))),
        n_subjects_loaded = length(unique_loaded_subjects),
        n_subjects_kept = length(kept_subjects),
        n_subjects_removed_extreme = length(bad_subjects),
    )
    annotated_individual = annotate_individual_effects(
        individual_effects;
        model_dir = resolved_model_dir,
        suffix = resolved_suffix,
        amplitude = amplitude,
        statistic = statistic,
        filter_extremes = filter_extremes,
        n_model_files = length(model_files),
        n_loaded_models = length(unique(String.(all_effects.model_file))),
        n_subjects_loaded = length(unique_loaded_subjects),
        n_subjects_kept = length(kept_subjects),
        n_subjects_removed_extreme = length(bad_subjects),
    )

    return (
        model_dir = resolved_model_dir,
        suffix = resolved_suffix,
        amplitude = amplitude,
        statistic = statistic,
        filter_extremes = filter_extremes,
        all_effects = all_effects,
        average_effects = average_effects,
        individual_effects = individual_effects,
        annotated_average = annotated_average,
        annotated_individual = annotated_individual,
        loaded_subjects = unique_loaded_subjects,
        kept_subjects = kept_subjects,
        plotted_subjects = plotted_subjects,
        plotted_models = plotted_models,
        bad_subjects = bad_subjects,
        n_model_files = length(model_files),
        n_loaded_models = length(unique(String.(all_effects.model_file))),
    )
end

function build_butterfly_figure(plot_data)
    average_effects = plot_data.average_effects
    individual_effects = plot_data.individual_effects
    eventnames = sort(unique(String.(average_effects.eventname)))
    isempty(eventnames) && error("No event names were available for plotting.")

    x_limits = butterfly_x_limits(average_effects, individual_effects)
    x_ticks = butterfly_tick_values(x_limits[1], x_limits[2], BUTTERFLY_X_TICK_STEP; digits = 1)
    y_ticks = butterfly_tick_values(BUTTERFLY_Y_LIMITS[1], BUTTERFLY_Y_LIMITS[2], BUTTERFLY_Y_TICK_STEP; digits = 0)
    figure_height = max(420, 280 * length(eventnames) + 20)
    fig = Figure(size = (BUTTERFLY_FIGURE_WIDTH, figure_height))
    model_colors = butterfly_individual_color_lookup(plot_data.plotted_models)

    first_axis = nothing
    for (row_index, eventname) in enumerate(eventnames)
        ax = Axis(
            fig[row_index, 1];
            xticks = x_ticks,
            yticks = y_ticks,
            xtickformat = values -> string.(round.(Float64.(values); digits = 1)),
            ytickformat = values -> string.(round.(Int, Float64.(values))),
            xautolimitmargin = (0.0, 0.0),
            yautolimitmargin = (0.0, 0.0),
        )

        if !isnothing(first_axis)
            linkxaxes!(first_axis, ax)
            linkyaxes!(first_axis, ax)
        else
            first_axis = ax
        end

        event_individuals = subset(individual_effects, :eventname => ByRow(==(eventname)))
        for line_df in groupby(event_individuals, :model_file)
            model_file = String(first(line_df.model_file))
            lines!(
                ax,
                Float64.(line_df.time),
                Float64.(line_df.yhat);
                color = model_colors[model_file],
                linewidth = BUTTERFLY_INDIVIDUAL_LINE_WIDTH,
            )
        end

        event_average = subset(average_effects, :eventname => ByRow(==(eventname)))
        lines!(
            ax,
            Float64.(event_average.time),
            Float64.(event_average.yhat);
            color = BUTTERFLY_AVERAGE_LINE_COLOR,
            linewidth = BUTTERFLY_AVERAGE_LINE_WIDTH,
        )

        vlines!(ax, [0.0]; color = :gray70, linestyle = :dash, linewidth = 1.0)
        hlines!(ax, [0.0]; color = :gray85, linewidth = 1.0)
        xlims!(ax, x_limits...)
        ylims!(ax, BUTTERFLY_Y_LIMITS...)
    end

    resize_to_layout!(fig)
    return fig
end

function butterfly_metadata(
    plot_data,
    output_png::AbstractString,
    output_svg::AbstractString,
    average_csv::AbstractString,
    individual_csv::AbstractString,
)
    return DataFrame([
        (
            variant = basename(normpath(plot_data.model_dir)),
            model_dir = abspath(plot_data.model_dir),
            suffix = plot_data.suffix,
            amplitude = plot_data.amplitude,
            statistic = String(group_statistic_label(plot_data.statistic)),
            filter_extremes = plot_data.filter_extremes,
            n_model_files = plot_data.n_model_files,
            n_loaded_models = plot_data.n_loaded_models,
            n_subjects_loaded = length(plot_data.loaded_subjects),
            n_subjects_kept = length(plot_data.kept_subjects),
            n_subjects_removed_extreme = length(plot_data.bad_subjects),
            n_plotted_models = length(plot_data.plotted_models),
            n_plotted_subjects = length(plot_data.plotted_subjects),
            plot_png = abspath(output_png),
            plot_svg = abspath(output_svg),
            average_csv = abspath(average_csv),
            individual_csv = abspath(individual_csv),
        ),
    ])
end

function butterfly_main(
    ;
    model_dir::AbstractString = BUTTERFLY_DEFAULT_MODEL_DIR,
    suffix::Union{Nothing,AbstractString} = nothing,
    output_file::Union{Nothing,AbstractString} = nothing,
    amplitude::Integer = BUTTERFLY_DEFAULT_AMPLITUDE,
    statistic::Symbol = BUTTERFLY_DEFAULT_STATISTIC,
    filter_extremes::Bool = BUTTERFLY_DEFAULT_FILTER_EXTREMES,
)
    resolved_model_dir = resolve_local_path(model_dir)
    resolved_suffix = isnothing(suffix) ? detect_model_suffix(resolved_model_dir) : normalize_suffix(suffix)
    resolved_output_stem = isnothing(output_file) ? joinpath(
        butterfly_default_output_dir(resolved_model_dir),
        butterfly_output_filename(
            resolved_suffix;
            amplitude = amplitude,
            statistic = statistic,
            filter_extremes = filter_extremes,
        ),
    ) : resolve_local_path(output_file)
    resolved_output_stem = output_stem(resolved_output_stem)
    resolved_output_png = resolved_output_stem * ".png"
    resolved_output_svg = resolved_output_stem * ".svg"

    println("Looking for models in $(abspath(resolved_model_dir))")
    println("Using filename suffix $(resolved_suffix)")
    println("Computing amplitude $(amplitude) butterfly plot with $(group_statistic_display(statistic)) and filter_extremes = $(filter_extremes)")

    plot_data = prepare_butterfly_plot_data(
        resolved_model_dir;
        suffix = resolved_suffix,
        amplitude = amplitude,
        statistic = statistic,
        filter_extremes = filter_extremes,
    )

    mkpath(dirname(resolved_output_png))
    figure = build_butterfly_figure(plot_data)
    save(resolved_output_png, figure)
    save(resolved_output_svg, figure)

    average_csv = resolved_output_stem * "_average.csv"
    individual_csv = resolved_output_stem * "_individual.csv"
    metadata_csv = resolved_output_stem * "_metadata.csv"

    write_table(average_csv, plot_data.annotated_average)
    write_table(individual_csv, plot_data.annotated_individual)
    write_table(
        metadata_csv,
        butterfly_metadata(plot_data, resolved_output_png, resolved_output_svg, average_csv, individual_csv),
    )

    println("Saved PNG:       $(abspath(resolved_output_png))")
    println("Saved SVG:       $(abspath(resolved_output_svg))")
    println("Saved average:   $(abspath(average_csv))")
    println("Saved lines:     $(abspath(individual_csv))")
    println("Saved metadata:  $(abspath(metadata_csv))")

    return (
        output_png = resolved_output_png,
        output_svg = resolved_output_svg,
        average_csv = average_csv,
        individual_csv = individual_csv,
        metadata_csv = metadata_csv,
        plot_data = plot_data,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 3
        print_butterfly_usage()
        error("Too many command line arguments.")
    end

    model_dir = length(ARGS) >= 1 ? ARGS[1] : BUTTERFLY_DEFAULT_MODEL_DIR
    suffix = if length(ARGS) >= 2
        ARGS[2] == "auto" ? nothing : ARGS[2]
    else
        nothing
    end
    output_file = length(ARGS) >= 3 ? ARGS[3] : nothing

    butterfly_main(; model_dir = model_dir, suffix = suffix, output_file = output_file)
end
