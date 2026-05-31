using CairoMakie
using DataFrames
using Unfold
using UnfoldMakie
using BSplineKit

const MODEL_OUTPUT_DIR = "."
const DEFAULT_FIF_PATH = "sample_subject/NDARUF540ZJ1/processed/sub-NDARUF540ZJ1_task-freeView_run-4_proc-eyelink_raw.fif"
const DEFAULT_MODEL_FILE = joinpath(MODEL_OUTPUT_DIR, splitext(basename(DEFAULT_FIF_PATH))[1] * ".jld2")
const OUTPUT_PNG = joinpath(".", "modified_effects_plot_NDARUF540ZJ1.png")
const PLOT_CHANNEL = 76
const PLOT_AMPLITUDES = 1:2:20

function plot_model_effects(model)
    effect_df = dropmissing(effects(Dict(:Amplitude => PLOT_AMPLITUDES), model))
    effect_df = subset(effect_df, :channel => ByRow(==(PLOT_CHANNEL)))
    return plot_erp(effect_df; mapping = (; color = :Amplitude, group = :Amplitude))
end

function main(model_file::AbstractString = DEFAULT_MODEL_FILE; output_png::AbstractString = OUTPUT_PNG)
    model = Unfold.load(model_file, UnfoldModel)
    fig = plot_model_effects(model)
    mkpath(dirname(output_png))
    save(output_png, fig)
    display(fig)
    @info "saved effects plot" output_png
    return fig
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    model_file = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_MODEL_FILE
    output_png = length(ARGS) >= 2 ? ARGS[2] : OUTPUT_PNG
    main(model_file; output_png = output_png)
end
