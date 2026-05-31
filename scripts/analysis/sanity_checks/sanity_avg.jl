using CairoMakie
using DataFrames
using Statistics
using Unfold
using UnfoldMakie
using BSplineKit

const MODEL_DIR = "unsynced"
const OUTPUT_DIR = "visual_outputs"
const PLOT_CHANNEL = 82
const AMPLITUDES = 2:2:10
const EXTREME_THRESHOLD = 100.0
const WINSOR_PROPORTION = 0.10
const USE_FIXED_AXIS_LIMITS = true
const FIXED_X_LIMITS = (-0.5, 1.0)
const FIXED_Y_LIMITS = (-3.0, 4.0)

CairoMakie.activate!()

@eval Unfold begin
    """
    modified effects function (from effects.jl) that allows to predict only for a specific channel or set of channels, instead of all channels.
    this allows for a huge speed up, while giving the same plot as the original effects for a given channel
    """
    function effects(design::AbstractDict, model::T; typical = mean, channel = $PLOT_CHANNEL) where {T<:UnfoldModel}
        if isempty(design)
            return effects(Dict(:dummy => [:dummy]), model; typical, channel)
        end
        reference_grid = expand_grid(design)
        form = Unfold.formulas(model) # get formula

        # replace non-specified fields with "constants"
        m = Unfold.modelmatrix(model, false) # get the modelmatrix without timeexpansion
        #@debug "type form[1]", typeof(form[1])

        form_typical = _typify(T, reference_grid, form, m, typical)
        @debug typeof(form_typical) typeof(form_typical[1])

        #form_typical = vec(form_typical)
        reference_grids = repeat([reference_grid], length(form_typical))

        eff = if isnothing(channel)
            predict(model, form_typical, reference_grids; overlap = false)
        else
            channels = channel isa Integer ? [channel] : collect(channel)
            channel_coefs = @view coef(model)[channels, :]
            predict_no_overlap(model, channel_coefs, form_typical, reference_grids)
        end
        if :latency ∈ unique(vcat(names.(reference_grids)...))
            reference_grids = select.(reference_grids, Ref(DataFrames.Not(:latency)))
        end
        @debug "effects" size(eff[1]) reference_grid size(times(model)[1]) eventnames(model)
        effect_df = result_to_table(eff, reference_grids, times(model), eventnames(model))

        if !isnothing(channel)
            channels = channel isa Integer ? [channel] : collect(channel)
            effect_df[!, :channel] = getindex.(Ref(channels), Int.(effect_df.channel))
        end

        return effect_df
    end
end

function model_files()
    files = filter(file -> endswith(file, ".jld2"), readdir(MODEL_DIR; join = true))
    sort!(files)
    isempty(files) && error("No .jld2 files found in $MODEL_DIR")
    return files
end

subject_from_model_file(model_file::AbstractString) = first(splitext(basename(model_file)))

function add_subject!(df::DataFrame, subject::AbstractString)
    df[!, :subject] = fill(subject, nrow(df))
    return df
end

# behavior of original
#function model_effects(model_file::AbstractString)
#    model = Unfold.load(model_file, UnfoldModel; generate_Xs = false)
#    effect_df = dropmissing(effects(Dict(:Amplitude => AMPLITUDES), model; channel = PLOT_CHANNEL))
#    return add_subject!(effect_df, subject_from_model_file(model_file))
#end

# drops ismissing and non-finite
function model_effects(model_file::AbstractString)
    model = Unfold.load(model_file, UnfoldModel; generate_Xs = false)
    effect_df = effects(Dict(:Amplitude => AMPLITUDES), model; channel = PLOT_CHANNEL)
    effect_df = subset(effect_df, :yhat => ByRow(y -> !ismissing(y) && isfinite(y)))
    return add_subject!(effect_df, subject_from_model_file(model_file))
end


function winsorized_mean(values)
    x = sort(Float64.(collect(skipmissing(values))))
    isempty(x) && return missing

    n_winsor = floor(Int, WINSOR_PROPORTION * length(x))
    n_winsor == 0 && return mean(x)

    return mean(clamp.(x, x[n_winsor + 1], x[end - n_winsor]))
end

function save_effects_plot(effect_df::DataFrame, filename::AbstractString)
    axis_settings = USE_FIXED_AXIS_LIMITS ? (; limits = (FIXED_X_LIMITS, FIXED_Y_LIMITS)) : (;)
    fig = plot_erp(
        effect_df;
        mapping = (; color = :Amplitude, group = :Amplitude),
        axis = axis_settings,
    )
    mkpath(OUTPUT_DIR)
    output_file = joinpath(OUTPUT_DIR, filename)
    save(output_file, fig)
    @info "saved effects plot" output_file
    return output_file
end

function main()
    allEffects = map(model_effects, model_files()) |> effects -> reduce(vcat, effects)
    bad_subjects = subset(
        allEffects,
        :yhat => ByRow(y -> abs(y) > EXTREME_THRESHOLD),
    ).subject |> unique

    cleanEffects = subset(
        allEffects,
        :subject => ByRow(subject -> subject ∉ bad_subjects),
    )

    group_cols = [:channel, :Amplitude, :time, :eventname]

    effects_mean = combine(
        groupby(cleanEffects, group_cols),
        :yhat => (x -> mean(skipmissing(x))) => :yhat,
    )
    effects_median = combine(
        groupby(cleanEffects, group_cols),
        :yhat => (x -> median(skipmissing(x))) => :yhat,
    )
    effects_winsor = combine(
        groupby(cleanEffects, group_cols),
        :yhat => winsorized_mean => :yhat,
    )

    save_effects_plot(effects_mean, "effects_mean.png")
    save_effects_plot(effects_median, "effects_median.png")
    save_effects_plot(effects_winsor, "effects_winsorized_mean.png")

    @info "averaged models" total = length(unique(allEffects.subject)) kept = length(unique(cleanEffects.subject)) removed = length(bad_subjects)
    return effects_mean, effects_median, effects_winsor
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
