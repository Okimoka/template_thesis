using CairoMakie
using Unfold
using UnfoldMakie
using BSplineKit
using DataFrames
using Statistics
using StatsBase

"""
This was used for the images illustrating an introduction to Unfold.jl and effects plot from a single fitted model

After the `.jld2` model is loaded, the script writes
- a TSV summary with model metadata, amplitude distribution statistics, and spline breakpoints
- a coefficient plot for one EEG channel `plot_erp(coeftable(model)[channel, ...])`
- a spline-basis plot for the fitted model `plot_splines(model)`
- an effects plot for user-selected stimulus amplitudes at the same channel `plot_erp(effects(...)[channel, ...])`

From the repo root:
julia --project=render_subject_trio.jl

The script writes:
- `<SUBJECT>_channel-<N>_coefficients.svg`
- `<SUBJECT>_channel-<N>_splines.svg`
- `<SUBJECT>_channel-<N>_effects.svg`
- `<SUBJECT>_channel-<N>_summary.tsv`

LLM Code
"""

const DEFAULT_MODEL_DIR = "/home/oki/Desktop/EHLERS/unfold_bene/TRUE_CLEAN_GROUPS"

#joinpath(
#    dirname(@__DIR__),
#    "NOT_FIXED",
#    "final_fitted_models_baseline_clean_synced"#,
#)
const DEFAULT_OUTPUT_DIR = @__DIR__
const DEFAULT_SUBJECT = "NDARDZ147ETZ"
const DEFAULT_CHANNEL = 82
const DEFAULT_AMPLITUDES = Float64[2, 4, 6, 8, 10]

const USAGE = """
Usage:
  julia --project=/home/oki/Desktop/EHLERS/unfold_bene splines_plots/render_subject_trio.jl [SUBJECT_ID] [options]

Options:
  --subject SUBJECT_ID   Subject ID with or without the `sub-` prefix.
  --channel CHANNEL      EEG channel index to plot. Default: 82
  --model-dir PATH       Directory containing saved single-subject Unfold models.
  --output-dir PATH      Directory where the SVG outputs should be written.
  --amplitudes A,B,C     Comma-separated amplitudes for the effects plot. Default: 2,4,6,8,10
  --list                 Print available subject IDs in the model directory and exit.
  --help, -h             Show this message.

Examples:
  julia --project=/home/oki/Desktop/EHLERS/unfold_bene splines_plots/render_subject_trio.jl
  julia --project=/home/oki/Desktop/EHLERS/unfold_bene splines_plots/render_subject_trio.jl NDARYY694NE7
  julia --project=/home/oki/Desktop/EHLERS/unfold_bene splines_plots/render_subject_trio.jl --subject NDARBU183TDJ --channel 76
"""

CairoMakie.activate!()

normalize_subject_id(subject::AbstractString) = replace(subject, r"^sub-" => "")

function parse_amplitudes(text::AbstractString)
    stripped = strip(text)
    isempty(stripped) && error("Amplitude list cannot be empty.")
    return parse.(Float64, split(stripped, ","))
end

function parse_args(args)
    subject = DEFAULT_SUBJECT
    channel = DEFAULT_CHANNEL
    model_dir = DEFAULT_MODEL_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    amplitudes = copy(DEFAULT_AMPLITUDES)
    list_only = false
    positional_subject_seen = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            println(USAGE)
            exit(0)
        elseif arg == "--list"
            list_only = true
        elseif arg == "--subject"
            i += 1
            i <= length(args) || error("Missing value after --subject")
            subject = args[i]
        elseif arg == "--channel"
            i += 1
            i <= length(args) || error("Missing value after --channel")
            channel = parse(Int, args[i])
        elseif arg == "--model-dir"
            i += 1
            i <= length(args) || error("Missing value after --model-dir")
            model_dir = args[i]
        elseif arg == "--output-dir"
            i += 1
            i <= length(args) || error("Missing value after --output-dir")
            output_dir = args[i]
        elseif arg == "--amplitudes"
            i += 1
            i <= length(args) || error("Missing value after --amplitudes")
            amplitudes = parse_amplitudes(args[i])
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            positional_subject_seen && error("Only one positional SUBJECT_ID is allowed.")
            subject = arg
            positional_subject_seen = true
        end
        i += 1
    end

    return (
        subject = normalize_subject_id(subject),
        channel = channel,
        model_dir = abspath(model_dir),
        output_dir = abspath(output_dir),
        amplitudes = amplitudes,
        list_only = list_only,
    )
end

function available_model_files(model_dir::AbstractString)
    isdir(model_dir) || error("Model directory does not exist: $model_dir")
    model_files = filter(
        file -> endswith(file, ".jld2"),
        readdir(model_dir; join = true),
    )
    sort!(model_files)
    isempty(model_files) && error("No model files found in $model_dir")
    return model_files
end

subject_from_model_file(model_file::AbstractString) = begin
    subject_match = Base.match(r"^sub-([^_]+)_", basename(model_file))
    subject_match === nothing && error("Could not extract subject ID from $(basename(model_file))")
    return subject_match.captures[1]
end

function list_available_subjects(model_dir::AbstractString)
    model_files = available_model_files(model_dir)
    println("Available subjects in $model_dir")
    for model_file in model_files
        println(subject_from_model_file(model_file), '\t', basename(model_file))
    end
end

function find_subject_model_file(model_dir::AbstractString, subject::AbstractString)
    subject_id = normalize_subject_id(subject)
    matches = filter(
        model_file -> startswith(basename(model_file), "sub-$(subject_id)_"),
        available_model_files(model_dir),
    )
    isempty(matches) && error("No saved model found for subject $subject_id in $model_dir")
    length(matches) == 1 || error(
        "Found $(length(matches)) saved models for $subject_id. Please resolve manually:\n" *
        join(basename.(matches), "\n"),
    )
    return only(matches)
end

function extract_spline_term(model)
    terms = reduce(vcat, Unfold.terms.(getfield.(Unfold.formulas(model), :rhs)))
    spline_terms = filter(term -> term isa Unfold.AbstractSplineTerm, terms)
    length(spline_terms) == 1 || error(
        "Expected exactly one spline term, found $(length(spline_terms)).",
    )
    return only(spline_terms)
end

function event_table(model)
    return Unfold.events(designmatrix(model))[1]
end

function subject_summary_lines(model, model_file::AbstractString; channel::Integer, amplitudes)
    ev = event_table(model)
    amp = collect(skipmissing(ev.Amplitude))
    isempty(amp) && error("No amplitude values found in event table for $(basename(model_file))")

    spline_term = extract_spline_term(model)
    run_count = length(collect(eachmatch(r"__concat__", splitext(basename(model_file))[1]))) + 1

    return [
        "subject_id\t$(subject_from_model_file(model_file))",
        "model_file\t$(basename(model_file))",
        "model_dir\t$(dirname(model_file))",
        "channel\t$channel",
        "amplitudes\t$(join(amplitudes, ","))",
        "run_count\t$run_count",
        "n_events\t$(nrow(ev))",
        "amp_min\t$(round(minimum(amp); digits = 3))",
        "amp_p10\t$(round(quantile(amp, 0.10); digits = 3))",
        "amp_median\t$(round(median(amp); digits = 3))",
        "amp_mean\t$(round(mean(amp); digits = 3))",
        "amp_p90\t$(round(quantile(amp, 0.90); digits = 3))",
        "amp_max\t$(round(maximum(amp); digits = 3))",
        "amp_unique_count\t$(length(unique(amp)))",
        "amp_at_or_below_0_6\t$(count(x -> x <= 0.6, amp))",
        "spline_breakpoints\t$(join(round.(Float64.(spline_term.breakpoints); digits = 3), ","))",
    ]
end

function save_summary_file(model, model_file::AbstractString, output_dir::AbstractString; channel::Integer, amplitudes)
    summary_path = joinpath(
        output_dir,
        "$(subject_from_model_file(model_file))_channel-$(channel)_summary.tsv",
    )
    open(summary_path, "w") do io
        println(io, join(subject_summary_lines(model, model_file; channel = channel, amplitudes = amplitudes), "\n"))
    end
    return summary_path
end

function coefficient_plot(model; channel::Integer)
    coef_df = subset(coeftable(model), :channel => ByRow(==(channel)))
    nrow(coef_df) > 0 || error("No coefficient rows found for channel $channel")
    fig = plot_erp(coef_df; mapping = (; color = :coefname, group = :coefname))
    return fig
end

function spline_plot(model)
    return plot_splines(model)
end

function effects_plot(model; channel::Integer, amplitudes)
    effect_df = subset(
        effects(Dict(:Amplitude => amplitudes), model),
        :channel => ByRow(==(channel)),
        :yhat => ByRow(x -> !ismissing(x)),
    )
    nrow(effect_df) > 0 || error(
        "No effects rows found for channel $channel and amplitudes $(join(amplitudes, ", "))",
    )
    fig = plot_erp(effect_df; mapping = (; color = :Amplitude, group = :Amplitude))
    return fig
end

function svg_output_path(output_dir::AbstractString, model_file::AbstractString, channel::Integer, label::AbstractString)
    subject_id = subject_from_model_file(model_file)
    return joinpath(output_dir, "$(subject_id)_channel-$(channel)_$(label).svg")
end

function save_plot(fig, path::AbstractString)
    mkpath(dirname(path))
    save(path, fig)
    println("Wrote $(abspath(path))")
    return path
end

function main(args)
    config = parse_args(args)
    if config.list_only
        list_available_subjects(config.model_dir)
        return
    end

    model_file = find_subject_model_file(config.model_dir, config.subject)
    model = Unfold.load(model_file, UnfoldModel)

    println("Using subject $(subject_from_model_file(model_file))")
    println("Model file: $(model_file)")
    println("Channel: $(config.channel)")
    println("Amplitudes: $(join(config.amplitudes, ", "))")

    summary_path = save_summary_file(
        model,
        model_file,
        config.output_dir;
        channel = config.channel,
        amplitudes = config.amplitudes,
    )
    coeff_path = save_plot(
        coefficient_plot(model; channel = config.channel),
        svg_output_path(config.output_dir, model_file, config.channel, "coefficients"),
    )
    spline_path = save_plot(
        spline_plot(model),
        svg_output_path(config.output_dir, model_file, config.channel, "splines"),
    )
    effects_path = save_plot(
        effects_plot(model; channel = config.channel, amplitudes = config.amplitudes),
        svg_output_path(config.output_dir, model_file, config.channel, "effects"),
    )

    println()
    println("Summary: $(summary_path)")
    println("Coefficient plot: $(coeff_path)")
    println("Spline plot: $(spline_path)")
    println("Effects plot: $(effects_path)")
end

main(ARGS)
