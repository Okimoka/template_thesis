const CODEX_NO_AUTORUN = true
include(joinpath(@__DIR__, "new_analyze_group_original.jl"))

"""
Fully written by LLM
"""

using Random

const OUTPUT_DIR = joinpath(@__DIR__, "baseline_clean_synced_convergence")
const NORMALIZED_SUFFIX = only(normalize_suffixes(DEFAULT_SUFFIX))
const CONVERGENCE_STATISTIC = :wins
const FILTER_EXTREMES = true
const TARGET_SAMPLE_SIZE = 300
const N_RANDOM_SUBSETS = 40
const RNG_SEED = 20260407
const FULL_JOIN_KEYS = [:channel, :Amplitude, :time, :eventname, :series]
const SQUARE_FIGURE_SIZE = (820, 820)
const AMPLITUDE_COLORS = Dict(
    2 => "#440154",
    4 => "#3b528b",
    6 => "#21918c",
    8 => "#5ec962",
    10 => "#fde725",
)

function planned_sample_sizes(n_subjects::Integer)
    raw_steps = [
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        75,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        275,
        300,
        350,
        400,
        500,
        600,
        700,
        800,
        n_subjects,
    ]
    return unique(sort(filter(x -> 1 <= x <= n_subjects, raw_steps)))
end

function select_subject_rows(effect_df::DataFrame, subjects::AbstractVector{<:AbstractString})
    subject_set = Set(String.(subjects))
    return effect_df[in.(effect_df.subject, Ref(subject_set)), :]
end

function aggregate_subject_subset(
    clean_effects::DataFrame,
    subjects::AbstractVector{<:AbstractString};
    statistic::Symbol = CONVERGENCE_STATISTIC,
)
    subset_effects = select_subject_rows(clean_effects, subjects)
    aggregated, kept_subjects = aggregate_group_effects(
        subset_effects;
        statistic = statistic,
        bad_subjects = String[],
    )
    return aggregated, kept_subjects
end

function align_group_effects(primary::DataFrame, reference::DataFrame)
    primary_df = copy(primary)
    reference_df = copy(reference)
    rename!(primary_df, :yhat => :yhat_primary)
    rename!(reference_df, :yhat => :yhat_reference)
    return innerjoin(primary_df, reference_df, on = FULL_JOIN_KEYS)
end

function summarize_group_effect_difference(primary::DataFrame, reference::DataFrame)
    joined = align_group_effects(primary, reference)
    joined.diff = joined.yhat_primary .- joined.yhat_reference
    diff_summary = difference_summary(joined.diff)
    coverage = nrow(reference) == 0 ? missing : nrow(joined) / nrow(reference)
    return joined, (; diff_summary..., coverage = coverage)
end

function summarize_metric(values::AbstractVector{<:Real})
    sorted_values = sort(Float64.(collect(values)))
    n_values = length(sorted_values)
    n_values > 0 || error("Cannot summarize an empty metric vector.")

    quantile_index(p) = clamp(ceil(Int, p * n_values), 1, n_values)
    return (
        min = first(sorted_values),
        p10 = sorted_values[quantile_index(0.10)],
        p50 = sorted_values[quantile_index(0.50)],
        p90 = sorted_values[quantile_index(0.90)],
        max = last(sorted_values),
        mean = mean(sorted_values),
    )
end

function summarize_draws(draw_metrics::DataFrame)
    grouped = groupby(draw_metrics, :n_subjects)

    summary_rows = [
        begin
        rmse_stats = summarize_metric(subset_df.rmse)
        max_abs_stats = summarize_metric(subset_df.max_abs)
        coverage_stats = summarize_metric(subset_df.coverage)

        (
            n_subjects = first(subset_df.n_subjects),
            n_draws = nrow(subset_df),
            rmse_min = rmse_stats.min,
            rmse_p10 = rmse_stats.p10,
            rmse_p50 = rmse_stats.p50,
            rmse_p90 = rmse_stats.p90,
            rmse_max = rmse_stats.max,
            rmse_mean = rmse_stats.mean,
            max_abs_min = max_abs_stats.min,
            max_abs_p10 = max_abs_stats.p10,
            max_abs_p50 = max_abs_stats.p50,
            max_abs_p90 = max_abs_stats.p90,
            max_abs_max = max_abs_stats.max,
            max_abs_mean = max_abs_stats.mean,
            coverage_p50 = coverage_stats.p50,
            coverage_p90 = coverage_stats.p90,
        )
        end
        for subset_df in grouped
    ]

    summary_df = sort(DataFrame(summary_rows), :n_subjects)
    first_rmse = first(summary_df.rmse_p90)
    first_max_abs = first(summary_df.max_abs_p90)
    summary_df.rmse_p90_relative = first_rmse == 0 ? ones(nrow(summary_df)) : summary_df.rmse_p90 ./ first_rmse
    summary_df.max_abs_p90_relative = first_max_abs == 0 ? ones(nrow(summary_df)) : summary_df.max_abs_p90 ./ first_max_abs
    return summary_df
end

function rounded_axis_limit(value::Real, step::Real)
    return ceil(value / step) * step
end

function make_convergence_plot(summary_df::DataFrame, output_file::AbstractString; target_n::Integer)
    upper_y = rounded_axis_limit(maximum(summary_df.max_abs_p90), 0.5)
    ytick_values = collect(0.0:0.5:upper_y)
    fig = Figure(size = SQUARE_FIGURE_SIZE)

    ax_raw = Axis(
        fig[1, 1];
        xlabel = "Included subjects",
        ylabel = "Difference to full reference (uV)",
        yticks = ytick_values,
    )

    rmse_color = Makie.wong_colors()[1]
    max_abs_color = Makie.wong_colors()[2]

    lines!(ax_raw, summary_df.n_subjects, summary_df.rmse_p90; color = rmse_color, linewidth = 3)
    scatter!(ax_raw, summary_df.n_subjects, summary_df.rmse_p90; color = rmse_color, markersize = 12)
    lines!(ax_raw, summary_df.n_subjects, summary_df.max_abs_p90; color = max_abs_color, linewidth = 3)
    scatter!(ax_raw, summary_df.n_subjects, summary_df.max_abs_p90; color = max_abs_color, markersize = 12)
    vlines!(ax_raw, [target_n]; color = :gray40, linestyle = :dash, linewidth = 2)
    axislegend(
        ax_raw,
        [
            LineElement(color = rmse_color, linewidth = 3),
            LineElement(color = max_abs_color, linewidth = 3),
        ],
        ["RMSE (p90)", "max |Δ| (p90)"];
        position = :rt,
    )
    ylims!(ax_raw, 0, upper_y)
    save(output_file, fig)
    return output_file
end

function amplitude_legend_title(diff_df::DataFrame)
    amplitudes = sort(unique(diff_df.Amplitude))
    return "Amplitude (" * join(string.(amplitudes), ", ") * ")"
end

function make_difference_plot(
    diff_df::DataFrame,
    output_file::AbstractString;
    target_n::Integer,
    full_n::Integer,
)
    eventnames = sort(unique(String.(diff_df.eventname)))
    n_events = length(eventnames)
    fig = Figure(size = n_events == 1 ? SQUARE_FIGURE_SIZE : (820, 820 * n_events))

    amplitude_values = sort(unique(diff_df.Amplitude))
    amplitude_colors = Dict(
        amplitude => get(AMPLITUDE_COLORS, Int(amplitude), "#000000")
        for amplitude in amplitude_values
    )

    lower_y = min(-0.4, floor(minimum(diff_df.yhat) / 0.2) * 0.2)
    upper_y = max(1.0, rounded_axis_limit(maximum(diff_df.yhat), 0.2))
    ytick_values = collect(lower_y:0.2:upper_y)

    for (row_index, eventname) in enumerate(eventnames)
        ax = Axis(
            fig[row_index, 1];
            xlabel = "Time",
            ylabel = "Δyhat (uV)",
            yticks = ytick_values,
        )
        hlines!(ax, [0.0]; color = :gray40, linestyle = :dash, linewidth = 2)

        event_df = diff_df[diff_df.eventname .== eventname, :]
        for amplitude in amplitude_values
            amplitude_df = sort(
                event_df[event_df.Amplitude .== amplitude, [:time, :yhat]],
                :time,
            )
            isempty(amplitude_df) && continue
            lines!(
                ax,
                amplitude_df.time,
                amplitude_df.yhat;
                color = amplitude_colors[amplitude],
                linewidth = 1.5,
                label = string(amplitude),
            )
        end

        ylims!(ax, lower_y, upper_y)
        row_index == 1 && axislegend(ax; title = amplitude_legend_title(diff_df), position = :rt)
    end

    save(output_file, fig)
    return output_file
end

function representative_draw(draw_metrics::DataFrame, target_n::Integer)
    subset_df = draw_metrics[draw_metrics.n_subjects .== target_n, :]
    nrow(subset_df) > 0 || error("No draw metrics were found for target_n = $target_n.")
    target_rmse = summarize_metric(subset_df.rmse).p50
    subset_df.distance_to_target = abs.(subset_df.rmse .- target_rmse)
    sort!(subset_df, [:distance_to_target, :draw_id])
    return first(subset_df, 1)
end

function main()
    mkpath(OUTPUT_DIR)
    draw_metrics_path = joinpath(OUTPUT_DIR, "draw_metrics.csv")
    draw_subjects_path = joinpath(OUTPUT_DIR, "draw_subjects.csv")

    println("Looking for models in $(abspath(DEFAULT_MODEL_DIR))")
    model_files, normalized_suffix = find_model_files(DEFAULT_MODEL_DIR, NORMALIZED_SUFFIX)
    println("Using suffix $normalized_suffix")
    println("Found $(length(model_files)) matching model files")

    all_effects, loaded_subjects = collect_all_effects(model_files)
    warn_duplicate_subjects(loaded_subjects)

    all_bad_subjects = FILTER_EXTREMES ? detect_bad_subjects(all_effects; threshold = EXTREME_THRESHOLD) : String[]
    clean_effects = subset(
        all_effects,
        :subject => ByRow(x -> x ∉ all_bad_subjects),
        :yhat => ByRow(valid_effect_yhat),
    )
    nrow(clean_effects) > 0 || error("No effect rows remained after filtering.")

    full_reference, kept_subjects = aggregate_group_effects(
        clean_effects;
        statistic = CONVERGENCE_STATISTIC,
        bad_subjects = String[],
    )

    full_n = length(kept_subjects)
    target_n = min(TARGET_SAMPLE_SIZE, full_n)
    sample_sizes = planned_sample_sizes(full_n)

    println("Loaded $(length(unique(loaded_subjects))) unique model subjects")
    println("Filtered out $(length(all_bad_subjects)) subjects with extreme values")
    println("Full reference uses $full_n clean subjects")
    println("Sampling sizes: $(sample_sizes)")
    println("Random subsets per size: $N_RANDOM_SUBSETS")
    println("Target comparison size: $target_n")

    if isfile(draw_metrics_path) && isfile(draw_subjects_path)
        println("Reusing cached draw metrics from $(abspath(draw_metrics_path))")
        draw_metrics = sort(CSV.read(draw_metrics_path, DataFrame), [:n_subjects, :draw_id])
        draw_subjects = sort(CSV.read(draw_subjects_path, DataFrame), [:n_subjects, :draw_id])
    else
        rng = MersenneTwister(RNG_SEED)
        draw_rows = NamedTuple[]
        draw_subject_rows = NamedTuple[]

        for n_subjects in sample_sizes
            println()
            println("Evaluating $n_subjects / $full_n subjects")
            for draw_id in 1:N_RANDOM_SUBSETS
                sampled_subjects = sort(shuffle(rng, kept_subjects)[1:n_subjects])
                subset_effects, _ = aggregate_subject_subset(clean_effects, sampled_subjects)
                _, diff_metrics = summarize_group_effect_difference(subset_effects, full_reference)

                push!(
                    draw_rows,
                    (
                        n_subjects = n_subjects,
                        draw_id = draw_id,
                        rmse = diff_metrics.rmse,
                        max_abs = diff_metrics.max_abs,
                        mean_abs = diff_metrics.mean_abs,
                        mean_signed = diff_metrics.mean_signed,
                        coverage = diff_metrics.coverage,
                    ),
                )

                push!(
                    draw_subject_rows,
                    (
                        n_subjects = n_subjects,
                        draw_id = draw_id,
                        subjects = join(sampled_subjects, ";"),
                    ),
                )
            end
        end

        draw_metrics = sort(DataFrame(draw_rows), [:n_subjects, :draw_id])
        draw_subjects = sort(DataFrame(draw_subject_rows), [:n_subjects, :draw_id])
        write_table(draw_metrics_path, draw_metrics)
        write_table(draw_subjects_path, draw_subjects)
    end

    summary_df = summarize_draws(draw_metrics)

    representative_row = representative_draw(draw_metrics, target_n)
    representative_subject_string = only(
        draw_subjects[
            (draw_subjects.n_subjects .== representative_row.n_subjects[1]) .&
            (draw_subjects.draw_id .== representative_row.draw_id[1]),
            :subjects,
        ],
    )
    representative_subjects = String.(split(representative_subject_string, ';'))

    representative_effects, _ = aggregate_subject_subset(clean_effects, representative_subjects)
    representative_join, representative_diff_metrics = summarize_group_effect_difference(
        representative_effects,
        full_reference,
    )

    representative_diff_df = select(
        representative_join,
        FULL_JOIN_KEYS,
    )
    representative_diff_df.yhat = representative_join.diff
    sort!(representative_diff_df, FULL_JOIN_KEYS)

    write_table(joinpath(OUTPUT_DIR, "summary_metrics.csv"), summary_df)
    write_table(joinpath(OUTPUT_DIR, "difference_300_vs_full.csv"), representative_diff_df)
    write_table(
        joinpath(OUTPUT_DIR, "representative_300_subjects.csv"),
        DataFrame(subject = representative_subjects),
    )
    write_table(
        joinpath(OUTPUT_DIR, "representative_300_metrics.csv"),
        DataFrame([
            (
                target_n = target_n,
                full_n = full_n,
                draw_id = representative_row.draw_id[1],
                rmse = representative_diff_metrics.rmse,
                max_abs = representative_diff_metrics.max_abs,
                mean_abs = representative_diff_metrics.mean_abs,
                coverage = representative_diff_metrics.coverage,
            ),
        ]),
    )

    make_convergence_plot(
        summary_df,
        joinpath(OUTPUT_DIR, "baseline_clean_synced_convergence_rmse_maxabs.svg");
        target_n = target_n,
    )
    make_difference_plot(
        representative_diff_df,
        joinpath(OUTPUT_DIR, "baseline_clean_synced_300_minus_full.svg");
        target_n = target_n,
        full_n = full_n,
    )

    println()
    println("Finished. Outputs written to $(abspath(OUTPUT_DIR))")
    return nothing
end

function render_cached_figures(; output_dir::AbstractString = OUTPUT_DIR)
    summary_df = CSV.read(joinpath(output_dir, "summary_metrics.csv"), DataFrame)
    representative_diff_df = CSV.read(joinpath(output_dir, "difference_300_vs_full.csv"), DataFrame)
    representative_metrics = CSV.read(joinpath(output_dir, "representative_300_metrics.csv"), DataFrame)

    target_n = Int(representative_metrics.target_n[1])
    full_n = Int(representative_metrics.full_n[1])

    make_convergence_plot(
        summary_df,
        joinpath(output_dir, "baseline_clean_synced_convergence_rmse_maxabs.svg");
        target_n = target_n,
    )
    make_difference_plot(
        representative_diff_df,
        joinpath(output_dir, "baseline_clean_synced_300_minus_full.svg");
        target_n = target_n,
        full_n = full_n,
    )

    println("Rendered cached SVG figures in $(abspath(output_dir))")
    return nothing
end

if !(isdefined(Main, :CODEX_NO_AUTORUN) && getfield(Main, :CODEX_NO_AUTORUN) === true) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end
