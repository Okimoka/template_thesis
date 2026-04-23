using DataFrames
using Unfold
using StatsBase
using StatsModels: ContinuousTerm
using BSplineKit
using SHA

if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
    python_exe = Sys.which("python3")
    python_exe === nothing || (ENV["JULIA_PYTHONCALL_EXE"] = python_exe)
end

using PythonCall
using CUDA
using CUDA.CUSPARSE

const EVENT_EYE = ["L"] # ["L"] or ["L", "R"]
const CLAMP_LIMIT_EYE = ["L"] # ["L", "R"] = all saccades, ["L"] or ["R"] = one eye
const FIXATION_POSITION_X_COLUMN = Symbol("Location X")
const FIXATION_POSITION_Y_COLUMN = Symbol("Location Y")
const DEFAULT_SOLVER_BACKEND = :gpu
const DEFAULT_FORMULA_MODE = :regular
const DEFAULT_EVENT_ONSET = :saccade
const VALID_EVENT_EYES = ("L", "R")
const PY_MNE_REF = Ref{Union{Nothing,Py}}(nothing)
const MODEL_OUTPUT_DIR = joinpath(@__DIR__, "final_fitted_models_baseline_interp")
const FIR_TIME_WINDOW = (-0.5, 1.0)
const MODEL_NAME_HASH_HEX_LENGTH = 16
const MODEL_NAME_LABEL_MAX_LENGTH = 96

eye_event_description(kind::AbstractString, eye_name::AbstractString) = "ET_$(kind) $(eye_name)"
eye_event_name(kind::AbstractString, eye_name::AbstractString) = "$(kind) $(eye_name)"

function normalize_event_eye(eye)
    raw_parts = if eye isa AbstractString
        split(replace(uppercase(strip(eye)), " " => ""), ",")
    elseif eye isa AbstractVector
        [uppercase(strip(string(part))) for part in eye]
    else
        error("Unsupported event-eye selection $(repr(eye)). Use \"L\", \"R\", \"L,R\", or a vector of those eye labels.")
    end

    provided_parts = Set(filter(!isempty, raw_parts))
    isempty(provided_parts) && error("At least one event eye must be selected.")

    unknown_parts = setdiff(provided_parts, Set(VALID_EVENT_EYES))
    isempty(unknown_parts) || error(
        "Unsupported event eye selection $(join(sort!(collect(unknown_parts)), ", ")). Use only L and/or R.",
    )

    normalized_parts = [eye_name for eye_name in VALID_EVENT_EYES if eye_name in provided_parts]
    isempty(normalized_parts) && error("At least one event eye must be selected.")
    return normalized_parts
end

function normalize_solver_backend(solver_backend::Symbol)
    solver_backend in (:cpu, :gpu) || error("Unsupported solver backend $(solver_backend). Use :cpu or :gpu.")
    return solver_backend
end

normalize_solver_backend(solver_backend::AbstractString) = normalize_solver_backend(Symbol(lowercase(strip(solver_backend))))
solver_uses_gpu(solver_backend) = normalize_solver_backend(solver_backend) == :gpu

function normalize_formula_mode(formula_mode::Symbol)
    formula_mode in (:regular, :fixpos) || error("Unsupported formula mode $(formula_mode). Use :regular or :fixpos.")
    return formula_mode
end

normalize_formula_mode(formula_mode::AbstractString) = normalize_formula_mode(Symbol(lowercase(strip(formula_mode))))
formula_uses_fixation_position(formula_mode) = normalize_formula_mode(formula_mode) == :fixpos

function normalize_event_onset(event_onset::Symbol)
    event_onset in (:saccade, :fixation) || error(
        "Unsupported event onset $(event_onset). Use :saccade or :fixation.",
    )
    return event_onset
end

normalize_event_onset(event_onset::AbstractString) = normalize_event_onset(Symbol(lowercase(strip(event_onset))))
event_uses_fixation_onset(event_onset) = normalize_event_onset(event_onset) == :fixation

function event_onset_config_suffix(event_onset = DEFAULT_EVENT_ONSET)
    resolved_event_onset = normalize_event_onset(event_onset)
    resolved_event_onset == DEFAULT_EVENT_ONSET && return ""
    return "__event-$(string(resolved_event_onset))-onset"
end

function config_tag(;
    eye = EVENT_EYE,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_onset = DEFAULT_EVENT_ONSET,
)
    eye_label = join(normalize_event_eye(eye), "_")
    solver_label = string(normalize_solver_backend(solver_backend))
    formula_label = string(normalize_formula_mode(formula_mode))
    onset_suffix = event_onset_config_suffix(event_onset)
    return "eye-$(eye_label)__solver-$(solver_label)__formula-$(formula_label)$(onset_suffix)"
end

function configured_output_dir(
    output_dir::AbstractString;
    eye = EVENT_EYE,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_onset = DEFAULT_EVENT_ONSET,
)
    return joinpath(
        output_dir,
        config_tag(;
            eye = eye,
            solver_backend = solver_backend,
            formula_mode = formula_mode,
            event_onset = event_onset,
        ),
    )
end

function build_model_formula(formula_mode = DEFAULT_FORMULA_MODE)
    resolved_formula_mode = normalize_formula_mode(formula_mode)

    if formula_uses_fixation_position(resolved_formula_mode)
        return @formula(
            0 ~ 1 +
                spl(Amplitude, 4) +
                spl(fixation_position_x, 4) +
                spl(fixation_position_y, 4)
        )
    end

    return @formula(0 ~ 1 + spl(Amplitude, 4))
end

function continuous_hints(formula_mode = DEFAULT_FORMULA_MODE)
    hints = Dict{Symbol,Any}(
        :Amplitude => ContinuousTerm,
    )

    if formula_uses_fixation_position(formula_mode)
        hints[:fixation_position_x] = ContinuousTerm
        hints[:fixation_position_y] = ContinuousTerm
    end

    return hints
end

function required_annotation_columns(formula_mode = DEFAULT_FORMULA_MODE)
    columns = Symbol[]

    if formula_uses_fixation_position(formula_mode)
        append!(columns, [FIXATION_POSITION_X_COLUMN, FIXATION_POSITION_Y_COLUMN])
    end

    return columns
end

function validate_required_columns(ann_df::DataFrame, formula_mode = DEFAULT_FORMULA_MODE)
    required_columns = required_annotation_columns(formula_mode)
    isempty(required_columns) && return nothing

    available_columns = Set(propertynames(ann_df))
    missing_columns = sort!(collect(setdiff(Set(required_columns), available_columns)))
    isempty(missing_columns) || error(
        "Annotations are missing required columns for formula mode $(normalize_formula_mode(formula_mode)): " *
        "$(join(string.(missing_columns), ", ")).",
    )

    return nothing
end

function build_model_design(
    sfreq::Real;
    eye = EVENT_EYE,
    formula_mode = DEFAULT_FORMULA_MODE,
)
    model_formula = build_model_formula(formula_mode)
    return [
        eye_event_name("Saccade", eye_name) => (model_formula, firbasis(FIR_TIME_WINDOW, sfreq)) for
        eye_name in normalize_event_eye(eye)
    ]
end

function get_py_mne()
    if isnothing(PY_MNE_REF[])
        PY_MNE_REF[] = pyimport("mne")
    end

    return PY_MNE_REF[]::Py
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

# Workaround for Unfold GPU-QR path with current CUDA sparse matrices:
# Unfold.prepare_XTX currently tries CuArray{T,...}(CuSparseMatrixCSC), which can
# trigger scalar-indexing errors. Convert via CuArray(...) explicitly.
import Unfold: prepare_XTX
function Unfold.prepare_XTX(Ĥ, data::CuArray{T}, X::CuSparseMatrixCSC{T2}) where {T,T2}
    Xt = X'
    R_xx = CuArray(Xt * X)
    R_xy = similar(data, size(X, 2))
    return Ĥ, data, (Xt, R_xx, R_xy)
end

function has_active_average_reference(raw_mne)
    projs = pyconvert(Vector{Py}, raw_mne.info["projs"])
    return any(projs) do proj
        pyconvert(Bool, proj["active"]) &&
            pyconvert(String, proj["desc"]) == "Average EEG reference"
    end
end

function interpolate_bad_eeg_channels!(raw_mne)
    bad_channels = pyconvert(Vector{String}, raw_mne.info["bads"])
    isempty(bad_channels) && return raw_mne
    raw_mne.interpolate_bads(reset_bads = false, verbose = "ERROR")
    return raw_mne
end

function read_fif(fif_path::AbstractString)
    raw_mne = get_py_mne().io.read_raw_fif(fif_path, preload = true, verbose = "ERROR")
    ann_df = annotations_to_dataframe(raw_mne)
    eeg_only = raw_mne.copy().pick("eeg")
    # Interpolate marked EEG bads before rereferencing so they do not bias the average.
    interpolate_bad_eeg_channels!(eeg_only)
    has_active_average_reference(raw_mne) || eeg_only.set_eeg_reference(verbose = "ERROR")
    eeg_data = pyconvert(Array{Float64, 2}, eeg_only.get_data(units = "uV"))
    eeg_ch_names = pyconvert(Vector{String}, eeg_only.ch_names)
    sfreq = pyconvert(Float64, eeg_only.info["sfreq"])

    return eeg_data, eeg_ch_names, sfreq, ann_df
end

function read_fif_group(fif_paths::Tuple)
    eeg_data_parts = Matrix{Float64}[]
    ann_parts = DataFrame[]
    boundary_samples = Int[]
    sample_offset = 0
    sfreq = nothing

    for (i, fif_path) in pairs(fif_paths)
        eeg_data, _, sfreq, ann_df = read_fif(fif_path)
        ann_shifted = copy(ann_df)
        ann_shifted.onset .+= sample_offset / sfreq

        push!(eeg_data_parts, eeg_data)
        push!(ann_parts, ann_shifted)
        sample_offset += size(eeg_data, 2)
        i < length(fif_paths) && push!(boundary_samples, sample_offset)
    end

    eeg_data = hcat(eeg_data_parts...)
    ann_df = vcat(ann_parts..., cols = :union)

    return eeg_data, sfreq, ann_df, boundary_samples
end

function latency_crosses_boundary(
    latency::Real,
    boundary_samples::AbstractVector{<:Integer},
    lower_sample::Int,
    upper_sample::Int,
)
    for boundary_sample in boundary_samples
        if latency + lower_sample < boundary_sample <= latency + upper_sample
            return true
        end
    end

    return false
end

function filter_saccade_latency_and_boundaries(
    saccade_events::DataFrame,
    sfreq::Real;
    boundary_samples = Int[],
)
    saccade_events = subset(
        saccade_events,
        :latency => ByRow(>(0)),
        :Amplitude => ByRow(>=(0.5)),
    )

    if !isempty(boundary_samples)
        lower_sample = round(Int, FIR_TIME_WINDOW[1] * sfreq)
        upper_sample = round(Int, FIR_TIME_WINDOW[2] * sfreq)
        valid_mask = [
            !latency_crosses_boundary(latency, boundary_samples, lower_sample, upper_sample) for
            latency in saccade_events.latency
        ]
        saccade_events = saccade_events[valid_mask, :]
    end

    return saccade_events
end

function clamp_source_events(
    ann_df::DataFrame,
    sfreq::Real;
    clamp_eye = CLAMP_LIMIT_EYE,
    boundary_samples = Int[],
)
    clamp_event_types = Set(eye_event_name("Saccade", eye_name) for eye_name in normalize_event_eye(clamp_eye))
    clamp_source = subset(
        sort(copy(ann_df), :onset),
        :description => ByRow(x -> replace(string(x), "ET_" => "") in clamp_event_types),
        :Amplitude => ByRow(x -> !ismissing(x) && isfinite(Float64(x))),
    )

    clamp_events = DataFrame(
        latency = clamp_source.onset .* sfreq,
        Amplitude = Float64.(clamp_source.Amplitude),
    )
    clamp_events = filter_saccade_latency_and_boundaries(
        clamp_events,
        sfreq;
        boundary_samples = boundary_samples,
    )
    nrow(clamp_events) > 0 || error("No saccade events remain after applying the clamp-eye selection.")

    return clamp_events
end

function prepare_saccade_events(
    ann_df::DataFrame,
    sfreq::Real;
    eye = EVENT_EYE,
    clamp_eye = CLAMP_LIMIT_EYE,
    boundary_samples = Int[],
    formula_mode = DEFAULT_FORMULA_MODE,
    event_onset = DEFAULT_EVENT_ONSET,
)
    resolved_eye = normalize_event_eye(eye)
    resolved_formula_mode = normalize_formula_mode(formula_mode)
    resolved_event_onset = normalize_event_onset(event_onset)
    validate_required_columns(ann_df, resolved_formula_mode)

    clamp_events = clamp_source_events(
        ann_df,
        sfreq;
        clamp_eye = clamp_eye,
        boundary_samples = boundary_samples,
    )
    clamp_threshold = quantile(clamp_events.Amplitude, 0.98)

    sorted_ann = sort(copy(ann_df), :onset)
    descriptions = sorted_ann.description
    event_frames = DataFrame[]

    for eye_name in resolved_eye
        saccade_description = eye_event_description("Saccade", eye_name)
        saccade_event_name = eye_event_name("Saccade", eye_name)
        needs_fixation_pair = formula_uses_fixation_position(resolved_formula_mode) ||
            event_uses_fixation_onset(resolved_event_onset)

        if needs_fixation_pair
            fixation_description = eye_event_description("Fixation", eye_name)
            fixation_idx = findall(i ->
                descriptions[i] == fixation_description &&
                i > firstindex(descriptions) &&
                descriptions[i - 1] == saccade_description,
                eachindex(descriptions)
            )
            saccade_idx = fixation_idx .- 1

            skipped_saccades = count(==(saccade_description), descriptions) - length(saccade_idx)
            if skipped_saccades > 0
                @warn "Dropping saccades without direct following fixation" eye = eye_name skipped_saccades
            end

            isempty(saccade_idx) && continue

            saccade_rows = sorted_ann[saccade_idx, :]
            fixation_rows = sorted_ann[fixation_idx, :]

            valid_mask = map(eachindex(saccade_idx)) do i
                amplitude = saccade_rows.Amplitude[i]
                amplitude_is_valid = !ismissing(amplitude) && isfinite(Float64(amplitude)) && Float64(amplitude) >= 0.5
                amplitude_is_valid || return false

                if formula_uses_fixation_position(resolved_formula_mode)
                    fixation_x = fixation_rows[i, FIXATION_POSITION_X_COLUMN]
                    fixation_y = fixation_rows[i, FIXATION_POSITION_Y_COLUMN]
                    return !ismissing(fixation_x) && isfinite(Float64(fixation_x)) &&
                        !ismissing(fixation_y) && isfinite(Float64(fixation_y))
                end

                return true
            end

            invalid_pairs = count(!, valid_mask)
            if invalid_pairs > 0
                @warn "Dropping invalid saccade/fixation pairs" eye = eye_name invalid_pairs
            end

            saccade_rows = saccade_rows[valid_mask, :]
            fixation_rows = fixation_rows[valid_mask, :]
            nrow(saccade_rows) == 0 && continue

            modeled_latency = if event_uses_fixation_onset(resolved_event_onset)
                fixation_rows.onset .* sfreq
            else
                saccade_rows.onset .* sfreq
            end

            saccade_events = DataFrame(
                latency = modeled_latency,
                Amplitude = Float64.(saccade_rows.Amplitude),
            )

            if formula_uses_fixation_position(resolved_formula_mode)
                saccade_events[!, :fixation_position_x] = Float64.(fixation_rows[!, FIXATION_POSITION_X_COLUMN])
                saccade_events[!, :fixation_position_y] = Float64.(fixation_rows[!, FIXATION_POSITION_Y_COLUMN])
            end
        else
            saccade_rows = subset(
                sorted_ann,
                :description => ByRow(==(saccade_description)),
                :Amplitude => ByRow(x -> !ismissing(x) && isfinite(Float64(x)) && Float64(x) >= 0.5),
            )
            nrow(saccade_rows) == 0 && continue

            saccade_events = DataFrame(
                latency = saccade_rows.onset .* sfreq,
                Amplitude = Float64.(saccade_rows.Amplitude),
            )
        end

        saccade_events[!, "Event Type"] = fill(saccade_event_name, nrow(saccade_events))
        saccade_events = filter_saccade_latency_and_boundaries(
            saccade_events,
            sfreq;
            boundary_samples = boundary_samples,
        )
        nrow(saccade_events) == 0 && continue

        large_mask = saccade_events.Amplitude .> clamp_threshold
        saccade_events.Amplitude[large_mask] .= clamp_threshold
        push!(event_frames, saccade_events)
    end

    isempty(event_frames) || return sort!(reduce((a, b) -> vcat(a, b; cols = :union), event_frames), :latency)
    error(
        "No modeled saccade events were created for formula mode $(resolved_formula_mode) with event onset $(resolved_event_onset).",
    )
end

function fit_model_from_data(
    eeg_data,
    sfreq::Real,
    saccade_events::DataFrame;
    eye = EVENT_EYE,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_onset = DEFAULT_EVENT_ONSET,
)
    resolved_solver_backend = normalize_solver_backend(solver_backend)
    design = build_model_design(sfreq; eye = eye, formula_mode = formula_mode)
    hints = continuous_hints(formula_mode)

    if solver_uses_gpu(resolved_solver_backend)
        gpu_solver = (x, y) -> Unfold.solver_predefined(x, y; solver = :qr)
        return fit(
            UnfoldModel,
            design,
            saccade_events,
            CUDA.cu(eeg_data),
            solver = gpu_solver,
            eventcolumn = "Event Type",
            contrasts = hints,
        )
    end

    return fit(
        UnfoldModel,
        design,
        saccade_events,
        eeg_data,
        eventcolumn = "Event Type",
        contrasts = hints,
    )
end

function fit_grouped_file_model(
    fif_paths::Tuple;
    eye = EVENT_EYE,
    clamp_eye = CLAMP_LIMIT_EYE,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_onset = DEFAULT_EVENT_ONSET,
)
    eeg_data, sfreq, ann_df, boundary_samples = read_fif_group(fif_paths)
    saccade_events = prepare_saccade_events(
        ann_df,
        sfreq;
        eye = eye,
        clamp_eye = clamp_eye,
        boundary_samples = boundary_samples,
        formula_mode = formula_mode,
        event_onset = event_onset,
    )

    return fit_model_from_data(
        eeg_data,
        sfreq,
        saccade_events;
        eye = eye,
        solver_backend = solver_backend,
        formula_mode = formula_mode,
        event_onset = event_onset,
    )
end

function strip_solver_history!(model::UnfoldModel)
    fit_info = model.modelfit.info

    if fit_info isa AbstractVector && length(fit_info) >= 3
        # GPU QR stores iteration history as views into CuArrays here, which makes
        # the saved JLD2 file depend on CUDA types when loading later.
        lean_info = Any[fit_info[1], fit_info[2], nothing]
        model.modelfit = typeof(model.modelfit)(
            model.modelfit.estimate,
            lean_info,
            model.modelfit.standarderror,
        )
    end

    return model
end

function ensure_finite_coefficients!(model::UnfoldModel)
    if any(x -> !isfinite(x), coef(model))
        error("Model coefficients contain non-finite values.")
    end

    return model
end

function release_fit_resources!(solver_backend = DEFAULT_SOLVER_BACKEND)
    GC.gc(true)
    solver_uses_gpu(solver_backend) && CUDA.reclaim()
    return nothing
end

function fif_group_label(fif_paths::Tuple)
    return join(basename.(collect(fif_paths)), " + ")
end

function model_stems(fif_paths::Tuple)
    return map(path -> splitext(basename(path))[1], fif_paths)
end

function legacy_model_output_path(fif_paths::Tuple; output_dir::AbstractString = MODEL_OUTPUT_DIR)
    model_name = join(model_stems(fif_paths), "__concat__") * ".jld2"
    return joinpath(output_dir, model_name)
end

function model_output_hash(fif_paths::Tuple)
    fingerprint_source = join(String.(collect(fif_paths)), "\n")
    return bytes2hex(sha1(fingerprint_source))[1:MODEL_NAME_HASH_HEX_LENGTH]
end

function shortened_model_label(fif_paths::Tuple)
    stems = model_stems(fif_paths)
    group_suffix = "__n$(length(stems))"
    stem_budget = max(16, MODEL_NAME_LABEL_MAX_LENGTH - length(group_suffix))
    first_stem = first(stems)
    shortened_first_stem = length(first_stem) <= stem_budget ? first_stem : first(first_stem, stem_budget)
    return shortened_first_stem * group_suffix
end

function model_output_path(fif_paths::Tuple; output_dir::AbstractString = MODEL_OUTPUT_DIR)
    model_name = "$(shortened_model_label(fif_paths))__$(model_output_hash(fif_paths)).jld2"
    return joinpath(output_dir, model_name)
end

function safe_isfile(path::AbstractString)
    try
        return isfile(path)
    catch err
        if err isa Base.IOError && occursin("ENAMETOOLONG", sprint(showerror, err))
            return false
        end
        rethrow()
    end
end

function existing_model_output_path(fif_paths::Tuple; output_dir::AbstractString = MODEL_OUTPUT_DIR)
    shortened_path = model_output_path(fif_paths; output_dir = output_dir)
    safe_isfile(shortened_path) && return shortened_path

    legacy_path = legacy_model_output_path(fif_paths; output_dir = output_dir)
    safe_isfile(legacy_path) && return legacy_path

    return nothing
end

function fit_and_save_group(
    fif_group::Tuple;
    output_dir::AbstractString = MODEL_OUTPUT_DIR,
    skip_existing::Bool = false,
    log_prefix::AbstractString = "",
    eye = EVENT_EYE,
    clamp_eye = CLAMP_LIMIT_EYE,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_onset = DEFAULT_EVENT_ONSET,
)
    resolved_eye = normalize_event_eye(eye)
    resolved_clamp_eye = normalize_event_eye(clamp_eye)
    resolved_solver_backend = normalize_solver_backend(solver_backend)
    resolved_formula_mode = normalize_formula_mode(formula_mode)
    resolved_event_onset = normalize_event_onset(event_onset)
    resolved_output_dir = configured_output_dir(
        output_dir;
        eye = resolved_eye,
        solver_backend = resolved_solver_backend,
        formula_mode = resolved_formula_mode,
        event_onset = resolved_event_onset,
    )

    mkpath(resolved_output_dir)
    model_file = model_output_path(fif_group; output_dir = resolved_output_dir)
    existing_model_file = skip_existing ? existing_model_output_path(fif_group; output_dir = resolved_output_dir) : nothing

    if !isnothing(existing_model_file)
        return (
            status = :skipped_existing,
            output_path = existing_model_file,
            fif_group = fif_group,
            error = "",
        )
    end

    prefix = isempty(log_prefix) ? "" : "[$(log_prefix)] "
    model = nothing

    try
        model = fit_grouped_file_model(
            fif_group;
            eye = resolved_eye,
            clamp_eye = resolved_clamp_eye,
            solver_backend = resolved_solver_backend,
            formula_mode = resolved_formula_mode,
            event_onset = resolved_event_onset,
        )
        ensure_finite_coefficients!(model)
        strip_solver_history!(model)
        Unfold.save(model_file, model; compress = false)
        return (
            status = :saved,
            output_path = model_file,
            fif_group = fif_group,
            error = "",
        )
    catch err
        error_text = sprint(showerror, err)
        println(stderr, prefix * error_text)
        println(stderr, prefix * "While processing: " * repr(fif_group))
        println(stderr, prefix * "Skipping failed group and continuing.")
        return (
            status = :failed,
            output_path = model_file,
            fif_group = fif_group,
            error = error_text,
        )
    finally
        model = nothing
        release_fit_resources!(resolved_solver_backend)
    end
end

function main(
    fif_paths::AbstractVector{<:Tuple};
    output_dir::AbstractString = MODEL_OUTPUT_DIR,
    skip_existing::Bool = false,
    log_prefix::AbstractString = "",
    eye = EVENT_EYE,
    clamp_eye = CLAMP_LIMIT_EYE,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_onset = DEFAULT_EVENT_ONSET,
)
    resolved_output_dir = configured_output_dir(
        output_dir;
        eye = eye,
        solver_backend = solver_backend,
        formula_mode = formula_mode,
        event_onset = event_onset,
    )
    mkpath(resolved_output_dir)
    results = NamedTuple[]
    total_groups = length(fif_paths)

    for (i, fif_group) in enumerate(fif_paths)
        prefix = isempty(log_prefix) ? "" : "[$(log_prefix)] "
        println(
            prefix * "[$(i)/$(total_groups)] " * fif_group_label(fif_group),
        )
        flush(stdout)
        push!(
            results,
            fit_and_save_group(
                fif_group;
                output_dir = output_dir,
                skip_existing = skip_existing,
                log_prefix = log_prefix,
                eye = eye,
                clamp_eye = clamp_eye,
                solver_backend = solver_backend,
                formula_mode = formula_mode,
                event_onset = event_onset,
            ),
        )
    end

    return results
end

function default_fif_paths()
    return [

        # list of random subjects "NDARNF525CRN","NDARNH110NV6","NDARNJ987CYD","NDARNM384EEW","NDARNM708YTF","NDARNN321YCR","NDARNN989FZ4","NDARNP381RZ4","NDARNT205ZKP","NDARNV399BV4","NDARNV983DET","NDARNX050XBN","NDARNX562GP3","NDARNZ141GNH","NDARPA750HZ1","NDARPE424GZZ","NDARPE623MBW","NDARPF325KE2","NDARPF395NV5","NDARPF460UXT","NDARPG387YY2","NDARPJ001BVE","NDARPJ151ERK","NDARPK597XH0","NDARPL651MD5","NDARPM172XGD","NDARPV423ECQ","NDARPX098XYT","NDARPX661BF1","NDARPX838GCD","NDARPX980MUN","NDARPY302MV9","NDARPY478YM0","NDARPZ720WKW"


(
    "/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRR464UTB/eeg/sub-NDARRR464UTB_task-freeView_run-3_proc-clean_raw.fif",
    "/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRR464UTB/eeg/sub-NDARRR464UTB_task-freeView_run-4_proc-clean_raw.fif",
    "/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRR464UTB/eeg/sub-NDARRR464UTB_task-freeView_run-5_proc-clean_raw.fif",
),

("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRR570MC0/eeg/sub-NDARRR570MC0_task-freeView_run-3_proc-clean_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRU820CXW/eeg/sub-NDARRU820CXW_task-freeView_run-3_proc-clean_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRV843HGK/eeg/sub-NDARRV843HGK_task-freeView_run-3_proc-clean_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRX900GP4/eeg/sub-NDARRX900GP4_task-freeView_run-3_proc-clean_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRY807MXC/eeg/sub-NDARRY807MXC_task-freeView_run-3_proc-clean_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRR464UTB/eeg/sub-NDARRR464UTB_task-freeView_run-3_proc-eyelink_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRR570MC0/eeg/sub-NDARRR570MC0_task-freeView_run-3_proc-eyelink_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRU820CXW/eeg/sub-NDARRU820CXW_task-freeView_run-3_proc-eyelink_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRV843HGK/eeg/sub-NDARRV843HGK_task-freeView_run-3_proc-eyelink_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRX900GP4/eeg/sub-NDARRX900GP4_task-freeView_run-3_proc-eyelink_raw.fif",),
("/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDARRY807MXC/eeg/sub-NDARRY807MXC_task-freeView_run-3_proc-eyelink_raw.fif",),

    ]
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(default_fif_paths())
end
