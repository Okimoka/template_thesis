using CUDA
using Serialization

const FIT_SCRIPT_PATH = joinpath(@__DIR__, "new_uf2_bids_simple_with_groups.jl")
const GROUP_DEFINITIONS_PATH = joinpath(@__DIR__, "group_definitions.jl")

const WORKER_ARG = "--worker"
const WORKERS_ARG = "--workers"
const DRY_RUN_ARG = "--dry-run"
const SKIP_EXISTING_ARG = "--skip-existing"
const CONFIG_CACHE_ARG = "--config-cache"
const GROUP_DEFINITIONS_ARG = "--group-definitions-file"
const RESULTS_FILE_ARG = "--results-file"
const SOLVER_BACKEND_ARG = "--solver-backend"
const FORMULA_ARG = "--formula"
const EVENT_EYE_ARG = "--event-eye"
const EVENT_ONSET_ARG = "--event-onset"
const PATH_PREFIX_FROM_ARG = "--path-prefix-from"
const PATH_PREFIX_TO_ARG = "--path-prefix-to"
const DEFAULT_SOLVER_BACKEND = :gpu
const DEFAULT_FORMULA_MODE = :regular
const DEFAULT_EVENT_EYE = ["L"]
const DEFAULT_EVENT_ONSET = :saccade
const VALID_EVENT_EYES = ("L", "R")

function ensure_fit_script_loaded!()
    if !isdefined(Main, :fit_and_save_group)
        Base.include(Main, FIT_SCRIPT_PATH)
    end
    return nothing
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

function normalize_event_onset(event_onset::Symbol)
    event_onset in (:saccade, :fixation) || error(
        "Unsupported event onset $(event_onset). Use :saccade or :fixation.",
    )
    return event_onset
end

normalize_event_onset(event_onset::AbstractString) = normalize_event_onset(Symbol(lowercase(strip(event_onset))))
event_onset_arg_value(event_onset) = string(normalize_event_onset(event_onset))

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

event_eye_arg_value(event_eye) = join(normalize_event_eye(event_eye), ",")

function normalize_path_prefix(prefix::AbstractString)
    normalized_prefix = strip(prefix)
    isempty(normalized_prefix) && error("Path prefixes must not be empty.")
    if normalized_prefix != "/"
        normalized_prefix = rstrip(normalized_prefix, '/')
    end
    return normalized_prefix
end

function normalize_path_remap(
    path_prefix_from::Union{Nothing,AbstractString},
    path_prefix_to::Union{Nothing,AbstractString},
)
    if isnothing(path_prefix_from) && isnothing(path_prefix_to)
        return nothing
    end

    xor(isnothing(path_prefix_from), isnothing(path_prefix_to)) && error(
        "Both $PATH_PREFIX_FROM_ARG and $PATH_PREFIX_TO_ARG must be provided together.",
    )

    return (
        from = normalize_path_prefix(path_prefix_from),
        to = normalize_path_prefix(path_prefix_to),
    )
end

function rewrite_fif_path(path::AbstractString; path_remap = nothing)
    isnothing(path_remap) && return String(path)

    if path == path_remap.from
        return path_remap.to
    end

    prefix = path_remap.from * "/"
    if startswith(path, prefix)
        relative_suffix = path[length(prefix)+1:end]
        return joinpath(path_remap.to, relative_suffix)
    end

    return String(path)
end

function discover_group_names(group_file::AbstractString)
    group_names = String[]

    for line in eachline(group_file)
        match_result = match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        isnothing(match_result) && continue
        push!(group_names, match_result.captures[1])
    end

    unique!(group_names)
    return group_names
end

function normalize_fif_group(
    group;
    path_remap = nothing,
    group_definitions_file::AbstractString = GROUP_DEFINITIONS_PATH,
)
    group isa Tuple || error("Each group definition entry must be a tuple of FIF paths, got $(typeof(group)).")
    isempty(group) && error("Encountered an empty FIF-path tuple in $(group_definitions_file).")

    normalized_paths = map(collect(group)) do path
        path isa AbstractString || error("FIF paths must be strings, got $(typeof(path)).")
        return rewrite_fif_path(String(path); path_remap = path_remap)
    end

    return Tuple(normalized_paths)
end

function fif_path_exists(
    path::AbstractString;
    directory_entries_cache::Dict{String,Union{Nothing,Set{String}}},
)
    directory = dirname(path)
    entries = get!(directory_entries_cache, directory) do
        return isdir(directory) ? Set(readdir(directory)) : nothing
    end

    return !isnothing(entries) && basename(path) in entries
end

function filter_existing_fif_paths(
    fif_group::Tuple;
    directory_entries_cache::Dict{String,Union{Nothing,Set{String}}},
    path_remap = nothing,
    group_definitions_file::AbstractString = GROUP_DEFINITIONS_PATH,
)
    existing_paths = String[]
    missing_paths = String[]

    for path in normalize_fif_group(
        fif_group;
        path_remap = path_remap,
        group_definitions_file = group_definitions_file,
    )
        if fif_path_exists(path; directory_entries_cache = directory_entries_cache)
            push!(existing_paths, path)
        else
            push!(missing_paths, path)
        end
    end

    return (
        fif_group = Tuple(existing_paths),
        missing_paths = missing_paths,
    )
end

function load_group_configs(
    ;
    selected_names::AbstractVector{<:AbstractString} = String[],
    path_prefix_from::Union{Nothing,AbstractString} = nothing,
    path_prefix_to::Union{Nothing,AbstractString} = nothing,
    group_definitions_file::AbstractString = GROUP_DEFINITIONS_PATH,
)
    resolved_group_definitions_file = abspath(group_definitions_file)
    available_names = discover_group_names(resolved_group_definitions_file)
    isempty(available_names) && error("No group definitions were found in $(resolved_group_definitions_file).")

    requested_names = isempty(selected_names) ? available_names : String.(selected_names)
    unknown_names = setdiff(requested_names, available_names)
    isempty(unknown_names) || error(
        "Unknown group definition(s): $(join(unknown_names, ", ")). Available groups: $(join(available_names, ", ")).",
    )

    path_remap = normalize_path_remap(path_prefix_from, path_prefix_to)
    group_module = Module(:UF2GroupDefinitions)
    Base.include(group_module, resolved_group_definitions_file)

    configs = NamedTuple[]
    directory_entries_cache = Dict{String,Union{Nothing,Set{String}}}()

    for group_name in requested_names
        # The bindings in group_module are created by the include above, so on
        # older Julia versions we need latest-world lookup here to avoid a
        # binding-too-new world-age error.
        fif_paths_raw = Core.eval(group_module, Symbol(group_name))
        fif_paths_raw isa AbstractVector || error("$(group_name) must be a vector of FIF-path tuples.")

        fif_paths = Tuple[]
        skipped_empty_count = 0
        groups_with_missing_paths_count = 0
        removed_missing_paths_count = 0
        skipped_missing_only_count = 0
        skipped_duplicate_count = 0
        seen_groups = Set{Tuple}()

        for fif_group in fif_paths_raw
            fif_group isa Tuple || error("$(group_name) entries must be tuples of FIF paths.")
            if isempty(fif_group)
                skipped_empty_count += 1
                continue
            end

            filtered_group = filter_existing_fif_paths(
                fif_group;
                directory_entries_cache = directory_entries_cache,
                path_remap = path_remap,
                group_definitions_file = resolved_group_definitions_file,
            )

            if !isempty(filtered_group.missing_paths)
                groups_with_missing_paths_count += 1
                removed_missing_paths_count += length(filtered_group.missing_paths)
            end

            if isempty(filtered_group.fif_group)
                skipped_missing_only_count += 1
                continue
            end

            if filtered_group.fif_group in seen_groups
                skipped_duplicate_count += 1
                continue
            end

            push!(seen_groups, filtered_group.fif_group)
            push!(fif_paths, filtered_group.fif_group)
        end

        push!(
            configs,
            (
                name = group_name,
                fif_paths = fif_paths,
                output_dir = joinpath(@__DIR__, group_name),
                skipped_empty_count = skipped_empty_count,
                groups_with_missing_paths_count = groups_with_missing_paths_count,
                removed_missing_paths_count = removed_missing_paths_count,
                skipped_missing_only_count = skipped_missing_only_count,
                skipped_duplicate_count = skipped_duplicate_count,
            ),
        )
    end

    return configs
end

function write_group_configs_cache(configs)
    cache_path, io = mktemp()

    try
        serialize(io, configs)
    finally
        close(io)
    end

    return cache_path
end

function read_group_configs_cache(cache_path::AbstractString)
    open(cache_path, "r") do io
        return deserialize(io)
    end
end

function escape_tsv_field(value)
    string_value = string(value)
    string_value = replace(string_value, '\t' => "    ")
    string_value = replace(string_value, '\n' => "\\n")
    return string_value
end

function write_results_file(results_file::AbstractString, result_rows)
    mkpath(dirname(results_file))
    open(results_file, "w") do io
        println(
            io,
            join(
                [
                    "status",
                    "group_name",
                    "group_index",
                    "group_total",
                    "output_path",
                    "error",
                    "fif_group",
                ],
                '\t',
            ),
        )

        for row in result_rows
            println(
                io,
                join(
                    [
                        escape_tsv_field(row.status),
                        escape_tsv_field(row.group_name),
                        escape_tsv_field(row.group_index),
                        escape_tsv_field(row.group_total),
                        escape_tsv_field(row.output_path),
                        escape_tsv_field(row.error),
                        escape_tsv_field(repr(row.fif_group)),
                    ],
                    '\t',
                ),
            )
        end
    end
    return results_file
end

function build_jobs(configs)
    jobs = NamedTuple[]

    for config in configs
        group_total = length(config.fif_paths)
        for (group_index, fif_group) in enumerate(config.fif_paths)
            push!(
                jobs,
                (
                    group_name = config.name,
                    output_dir = config.output_dir,
                    fif_group = fif_group,
                    group_index = group_index,
                    group_total = group_total,
                ),
            )
        end
    end

    return jobs
end

function fif_group_label_local(fif_group::Tuple)
    return join(basename.(collect(fif_group)), " + ")
end

function select_worker_jobs(
    jobs::AbstractVector;
    worker_index::Integer = 1,
    total_workers::Integer = 1,
)
    total_workers > 0 || error("total_workers must be at least 1.")
    1 <= worker_index <= total_workers || error("worker_index must be between 1 and total_workers.")

    return [job for (job_number, job) in enumerate(jobs) if mod1(job_number, total_workers) == worker_index]
end

function worker_label(
    ;
    worker_index::Integer,
    total_workers::Integer,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_eye = DEFAULT_EVENT_EYE,
    event_onset = DEFAULT_EVENT_ONSET,
)
    return string(
        uppercase(string(normalize_solver_backend(solver_backend))),
        " ",
        string(normalize_formula_mode(formula_mode)),
        " eye=",
        event_eye_arg_value(event_eye),
        " onset=",
        event_onset_arg_value(event_onset),
        " worker ",
        worker_index,
        "/",
        total_workers,
    )
end

function run_assigned_jobs(
    ;
    configs = nothing,
    selected_names::AbstractVector{<:AbstractString} = String[],
    worker_index::Integer = 1,
    total_workers::Integer = 1,
    skip_existing::Bool = false,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_eye = DEFAULT_EVENT_EYE,
    event_onset = DEFAULT_EVENT_ONSET,
    path_prefix_from::Union{Nothing,AbstractString} = nothing,
    path_prefix_to::Union{Nothing,AbstractString} = nothing,
    group_definitions_file::AbstractString = GROUP_DEFINITIONS_PATH,
    results_file::Union{Nothing,AbstractString} = nothing,
)
    resolved_solver_backend = normalize_solver_backend(solver_backend)
    resolved_formula_mode = normalize_formula_mode(formula_mode)
    resolved_event_eye = normalize_event_eye(event_eye)
    resolved_event_onset = normalize_event_onset(event_onset)
    resolved_configs = isnothing(configs) ? load_group_configs(
        ;
        selected_names = selected_names,
        path_prefix_from = path_prefix_from,
        path_prefix_to = path_prefix_to,
        group_definitions_file = group_definitions_file,
    ) : configs
    jobs = build_jobs(resolved_configs)
    assigned_jobs = select_worker_jobs(
        jobs;
        worker_index = worker_index,
        total_workers = total_workers,
    )

    label = worker_label(
        ;
        worker_index = worker_index,
        total_workers = total_workers,
        solver_backend = resolved_solver_backend,
        formula_mode = resolved_formula_mode,
        event_eye = resolved_event_eye,
        event_onset = resolved_event_onset,
    )
    try
        ensure_fit_script_loaded!()
    catch err
        println(stderr, "[$label] Failed while loading $(FIT_SCRIPT_PATH).")
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        flush(stderr)
        rethrow()
    end

    saved_count = 0
    skipped_count = 0
    failed_jobs = NamedTuple[]
    result_rows = NamedTuple[]

    println("[$label] Starting $(length(assigned_jobs)) assigned fits.")
    flush(stdout)

    for (job_position, job) in enumerate(assigned_jobs)
        println(
            "[$label] [$job_position/$(length(assigned_jobs))] $(job.group_name) $(job.group_index)/$(job.group_total): $(fif_group_label_local(job.fif_group))",
        )
        flush(stdout)

        result = Base.invokelatest(
            fit_and_save_group,
            job.fif_group;
            output_dir = job.output_dir,
            skip_existing = skip_existing,
            log_prefix = "$label $(job.group_name) $(job.group_index)/$(job.group_total)",
            eye = resolved_event_eye,
            clamp_eye = resolved_event_eye,
            solver_backend = resolved_solver_backend,
            formula_mode = resolved_formula_mode,
            event_onset = resolved_event_onset,
        )

        push!(
            result_rows,
            (
                status = String(result.status),
                group_name = job.group_name,
                group_index = job.group_index,
                group_total = job.group_total,
                output_path = result.output_path,
                error = result.error,
                fif_group = job.fif_group,
            ),
        )

        if !isnothing(results_file)
            write_results_file(results_file, result_rows)
        end

        if result.status == :saved
            saved_count += 1
        elseif result.status == :skipped_existing
            skipped_count += 1
        else
            push!(failed_jobs, (job = job, error = result.error))
        end
    end

    println(
        "[$label] Finished: saved=$(saved_count), skipped=$(skipped_count), failed=$(length(failed_jobs)).",
    )
    flush(stdout)

    return (
        worker_index = worker_index,
        total_workers = total_workers,
        saved_count = saved_count,
        skipped_count = skipped_count,
        failed_jobs = failed_jobs,
        result_rows = result_rows,
    )
end

function default_worker_count(
    total_jobs::Integer;
    solver_backend = DEFAULT_SOLVER_BACKEND,
)
    total_jobs > 0 || return 1

    env_worker_count = get(ENV, "UF2_GROUP_WORKERS", nothing)
    if !isnothing(env_worker_count)
        parsed_count = parse(Int, env_worker_count)
        parsed_count > 0 || error("UF2_GROUP_WORKERS must be a positive integer.")
        return min(parsed_count, total_jobs)
    end

    gpu_ready = try
        CUDA.functional()
    catch
        false
    end

    recommended_workers = solver_uses_gpu(solver_backend) && gpu_ready ? 2 : 1
    return min(recommended_workers, total_jobs)
end

function active_project_dir()
    active_project = Base.active_project()
    return isnothing(active_project) ? nothing : dirname(active_project)
end

function build_worker_cmd(
    ;
    worker_index::Integer,
    total_workers::Integer,
    config_cache_path::Union{Nothing,AbstractString} = nothing,
    skip_existing::Bool = false,
    selected_names::AbstractVector{<:AbstractString} = String[],
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_eye = DEFAULT_EVENT_EYE,
    event_onset = DEFAULT_EVENT_ONSET,
    path_prefix_from::Union{Nothing,AbstractString} = nothing,
    path_prefix_to::Union{Nothing,AbstractString} = nothing,
    group_definitions_file::AbstractString = GROUP_DEFINITIONS_PATH,
)
    cmd = `$(Base.julia_cmd())`
    project_dir = active_project_dir()

    if !isnothing(project_dir)
        cmd = `$cmd --project=$project_dir`
    end

    cmd = `$cmd $(abspath(@__FILE__)) $WORKER_ARG $worker_index $total_workers`
    cmd = `$cmd $SOLVER_BACKEND_ARG $(string(normalize_solver_backend(solver_backend)))`
    cmd = `$cmd $FORMULA_ARG $(string(normalize_formula_mode(formula_mode)))`
    cmd = `$cmd $EVENT_EYE_ARG $(event_eye_arg_value(event_eye))`
    cmd = `$cmd $EVENT_ONSET_ARG $(event_onset_arg_value(event_onset))`
    cmd = `$cmd $GROUP_DEFINITIONS_ARG $(abspath(group_definitions_file))`

    if !isnothing(config_cache_path)
        cmd = `$cmd $CONFIG_CACHE_ARG $config_cache_path`
    end

    if skip_existing
        cmd = `$cmd $SKIP_EXISTING_ARG`
    end

    if !isnothing(path_prefix_from)
        cmd = `$cmd $PATH_PREFIX_FROM_ARG $path_prefix_from`
    end
    if !isnothing(path_prefix_to)
        cmd = `$cmd $PATH_PREFIX_TO_ARG $path_prefix_to`
    end

    if isnothing(config_cache_path)
        for group_name in selected_names
            cmd = `$cmd $group_name`
        end
    end

    return cmd
end

function print_worker_log_tail(
    log_path::AbstractString;
    label::AbstractString,
    max_lines::Integer = 80,
)
    if !isfile(log_path)
        println(stderr, "[$label] Worker log file was not created: $(log_path)")
        flush(stderr)
        return nothing
    end

    lines = readlines(log_path)
    start_index = max(1, length(lines) - max_lines + 1)

    println(stderr, "[$label] Last $(length(lines) - start_index + 1) log line(s) from $(log_path):")
    for line in lines[start_index:end]
        println(stderr, line)
    end
    flush(stderr)

    return nothing
end

function print_summary(
    configs,
    jobs;
    worker_count::Integer,
    skip_existing::Bool = false,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_eye = DEFAULT_EVENT_EYE,
    event_onset = DEFAULT_EVENT_ONSET,
    path_prefix_from::Union{Nothing,AbstractString} = nothing,
    path_prefix_to::Union{Nothing,AbstractString} = nothing,
    group_definitions_file::AbstractString = GROUP_DEFINITIONS_PATH,
)
    println("Preparing $(length(configs)) group collections across $(length(jobs)) total fits.")
    println(
        "Using $(worker_count) worker(s). skip_existing=$(skip_existing) solver_backend=$(string(normalize_solver_backend(solver_backend))) formula=$(string(normalize_formula_mode(formula_mode))) event_eye=$(event_eye_arg_value(event_eye)) event_onset=$(event_onset_arg_value(event_onset))",
    )
    println("Group definitions: $(abspath(group_definitions_file))")
    if !isnothing(path_prefix_from)
        println("Path remap: $(path_prefix_from) -> $(path_prefix_to)")
    end
    for config in configs
        summary_bits = String[]
        config.skipped_empty_count == 0 || push!(summary_bits, "skipped_empty=$(config.skipped_empty_count)")
        config.removed_missing_paths_count == 0 || push!(
            summary_bits,
            "removed_missing_paths=$(config.removed_missing_paths_count) across $(config.groups_with_missing_paths_count) groups",
        )
        config.skipped_missing_only_count == 0 || push!(
            summary_bits,
            "skipped_all_missing=$(config.skipped_missing_only_count)",
        )
        config.skipped_duplicate_count == 0 || push!(
            summary_bits,
            "skipped_duplicates=$(config.skipped_duplicate_count)",
        )
        summary_suffix = isempty(summary_bits) ? "" : " (" * join(summary_bits, ", ") * ")"
        println("  $(config.name): $(length(config.fif_paths)) fits$(summary_suffix) -> $(config.output_dir)")
    end
    flush(stdout)
    return nothing
end

function launch_parallel_runs(
    ;
    requested_worker_count::Union{Nothing,Integer} = nothing,
    selected_names::AbstractVector{<:AbstractString} = String[],
    skip_existing::Bool = false,
    dry_run::Bool = false,
    solver_backend = DEFAULT_SOLVER_BACKEND,
    formula_mode = DEFAULT_FORMULA_MODE,
    event_eye = DEFAULT_EVENT_EYE,
    event_onset = DEFAULT_EVENT_ONSET,
    path_prefix_from::Union{Nothing,AbstractString} = nothing,
    path_prefix_to::Union{Nothing,AbstractString} = nothing,
    group_definitions_file::AbstractString = GROUP_DEFINITIONS_PATH,
    results_file::Union{Nothing,AbstractString} = nothing,
)
    resolved_solver_backend = normalize_solver_backend(solver_backend)
    resolved_formula_mode = normalize_formula_mode(formula_mode)
    resolved_event_eye = normalize_event_eye(event_eye)
    resolved_event_onset = normalize_event_onset(event_onset)
    configs = load_group_configs(
        ;
        selected_names = selected_names,
        path_prefix_from = path_prefix_from,
        path_prefix_to = path_prefix_to,
        group_definitions_file = group_definitions_file,
    )
    jobs = build_jobs(configs)
    isempty(jobs) && error("No FIF groups were found to process.")

    worker_count = isnothing(requested_worker_count) ?
        default_worker_count(length(jobs); solver_backend = resolved_solver_backend) :
        requested_worker_count
    worker_count > 0 || error("Worker count must be at least 1.")
    worker_count = min(worker_count, length(jobs))

    print_summary(
        configs,
        jobs;
        worker_count = worker_count,
        skip_existing = skip_existing,
        solver_backend = resolved_solver_backend,
        formula_mode = resolved_formula_mode,
        event_eye = resolved_event_eye,
        event_onset = resolved_event_onset,
        path_prefix_from = path_prefix_from,
        path_prefix_to = path_prefix_to,
        group_definitions_file = group_definitions_file,
    )

    dry_run && return 0

    if worker_count == 1
        result = run_assigned_jobs(
            ;
            configs = configs,
            selected_names = selected_names,
            worker_index = 1,
            total_workers = 1,
            skip_existing = skip_existing,
            solver_backend = resolved_solver_backend,
            formula_mode = resolved_formula_mode,
            event_eye = resolved_event_eye,
            event_onset = resolved_event_onset,
            path_prefix_from = path_prefix_from,
            path_prefix_to = path_prefix_to,
            group_definitions_file = group_definitions_file,
            results_file = results_file,
        )
        return isempty(result.failed_jobs) ? 0 : 1
    end

    launched_processes = NamedTuple[]
    config_cache_path = write_group_configs_cache(configs)
    worker_log_dir = mktempdir(prefix = "uf2_group_workers_")

    try
        println("Worker logs: $(worker_log_dir)")
        flush(stdout)

        for worker_index in 1:worker_count
            label = worker_label(
                ;
                worker_index = worker_index,
                total_workers = worker_count,
                solver_backend = resolved_solver_backend,
                formula_mode = resolved_formula_mode,
                event_eye = resolved_event_eye,
                event_onset = resolved_event_onset,
            )
            log_path = joinpath(worker_log_dir, "worker_$(worker_index)_of_$(worker_count).log")
            log_io = open(log_path, "w")
            println("Launching [$label]")
            flush(stdout)
            process = run(
                pipeline(
                    build_worker_cmd(
                        ;
                        worker_index = worker_index,
                        total_workers = worker_count,
                        config_cache_path = config_cache_path,
                        skip_existing = skip_existing,
                        solver_backend = resolved_solver_backend,
                        formula_mode = resolved_formula_mode,
                        event_eye = resolved_event_eye,
                        event_onset = resolved_event_onset,
                        path_prefix_from = path_prefix_from,
                        path_prefix_to = path_prefix_to,
                        group_definitions_file = group_definitions_file,
                    );
                    stdout = log_io,
                    stderr = log_io,
                );
                wait = false,
            )
            push!(launched_processes, (label = label, process = process, log_path = log_path, log_io = log_io))
        end

        failed_workers = NamedTuple[]

        for launched in launched_processes
            wait(launched.process)
            close(launched.log_io)
            if success(launched.process)
                println("[$(launched.label)] exited successfully.")
                flush(stdout)
                continue
            end

            println(
                stderr,
                "[$(launched.label)] exited with code $(launched.process.exitcode). Full log: $(launched.log_path)",
            )
            print_worker_log_tail(launched.log_path; label = launched.label)
            flush(stderr)
            push!(
                failed_workers,
                (label = launched.label, exitcode = launched.process.exitcode, log_path = launched.log_path),
            )
        end

        isempty(failed_workers) || error(
            "Parallel run finished with $(length(failed_workers)) failing worker(s).",
        )
    finally
        for launched in launched_processes
            isopen(launched.log_io) && close(launched.log_io)
        end
        rm(config_cache_path; force = true)
    end

    println("All workers finished successfully.")
    flush(stdout)
    return 0
end

function parse_args(args::AbstractVector{<:AbstractString})
    worker_index = nothing
    total_workers = nothing
    requested_worker_count = nothing
    dry_run = false
    skip_existing = false
    config_cache_path = nothing
    solver_backend = DEFAULT_SOLVER_BACKEND
    formula_mode = DEFAULT_FORMULA_MODE
    event_eye = DEFAULT_EVENT_EYE
    event_onset = DEFAULT_EVENT_ONSET
    path_prefix_from = nothing
    path_prefix_to = nothing
    group_definitions_file = GROUP_DEFINITIONS_PATH
    results_file = nothing
    selected_names = String[]

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == WORKER_ARG
            i + 2 <= length(args) || error("Usage: $WORKER_ARG <worker_index> <total_workers>")
            worker_index = parse(Int, args[i + 1])
            total_workers = parse(Int, args[i + 2])
            i += 3
        elseif arg == WORKERS_ARG
            i + 1 <= length(args) || error("Usage: $WORKERS_ARG <count>")
            requested_worker_count = parse(Int, args[i + 1])
            i += 2
        elseif arg == DRY_RUN_ARG
            dry_run = true
            i += 1
        elseif arg == SKIP_EXISTING_ARG
            skip_existing = true
            i += 1
        elseif arg == CONFIG_CACHE_ARG
            i + 1 <= length(args) || error("Usage: $CONFIG_CACHE_ARG <path>")
            config_cache_path = args[i + 1]
            i += 2
        elseif arg == GROUP_DEFINITIONS_ARG
            i + 1 <= length(args) || error("Usage: $GROUP_DEFINITIONS_ARG <path>")
            group_definitions_file = abspath(args[i + 1])
            i += 2
        elseif arg == RESULTS_FILE_ARG
            i + 1 <= length(args) || error("Usage: $RESULTS_FILE_ARG <path>")
            results_file = abspath(args[i + 1])
            i += 2
        elseif arg == SOLVER_BACKEND_ARG
            i + 1 <= length(args) || error("Usage: $SOLVER_BACKEND_ARG <cpu|gpu>")
            solver_backend = normalize_solver_backend(args[i + 1])
            i += 2
        elseif arg == FORMULA_ARG
            i + 1 <= length(args) || error("Usage: $FORMULA_ARG <regular|fixpos>")
            formula_mode = normalize_formula_mode(args[i + 1])
            i += 2
        elseif arg == EVENT_EYE_ARG
            i + 1 <= length(args) || error("Usage: $EVENT_EYE_ARG <L|R|L,R>")
            event_eye = normalize_event_eye(args[i + 1])
            i += 2
        elseif arg == EVENT_ONSET_ARG
            i + 1 <= length(args) || error("Usage: $EVENT_ONSET_ARG <saccade|fixation>")
            event_onset = normalize_event_onset(args[i + 1])
            i += 2
        elseif arg == PATH_PREFIX_FROM_ARG
            i + 1 <= length(args) || error("Usage: $PATH_PREFIX_FROM_ARG <prefix>")
            path_prefix_from = args[i + 1]
            i += 2
        elseif arg == PATH_PREFIX_TO_ARG
            i + 1 <= length(args) || error("Usage: $PATH_PREFIX_TO_ARG <prefix>")
            path_prefix_to = args[i + 1]
            i += 2
        else
            push!(selected_names, arg)
            i += 1
        end
    end

    normalize_path_remap(path_prefix_from, path_prefix_to)

    return (
        worker_index = worker_index,
        total_workers = total_workers,
        requested_worker_count = requested_worker_count,
        dry_run = dry_run,
        skip_existing = skip_existing,
        config_cache_path = config_cache_path,
        solver_backend = solver_backend,
        formula_mode = formula_mode,
        event_eye = event_eye,
        event_onset = event_onset,
        path_prefix_from = path_prefix_from,
        path_prefix_to = path_prefix_to,
        group_definitions_file = group_definitions_file,
        results_file = results_file,
        selected_names = selected_names,
    )
end

function main(args = ARGS)
    parsed_args = parse_args(args)

    if !isnothing(parsed_args.worker_index)
        cached_configs = isnothing(parsed_args.config_cache_path) ? nothing : read_group_configs_cache(parsed_args.config_cache_path)
        result = run_assigned_jobs(
            ;
            configs = cached_configs,
            selected_names = parsed_args.selected_names,
            worker_index = parsed_args.worker_index,
            total_workers = parsed_args.total_workers,
            skip_existing = parsed_args.skip_existing,
            solver_backend = parsed_args.solver_backend,
            formula_mode = parsed_args.formula_mode,
            event_eye = parsed_args.event_eye,
            event_onset = parsed_args.event_onset,
            path_prefix_from = parsed_args.path_prefix_from,
            path_prefix_to = parsed_args.path_prefix_to,
            group_definitions_file = parsed_args.group_definitions_file,
            results_file = parsed_args.results_file,
        )
        return isempty(result.failed_jobs) ? 0 : 1
    end

    return launch_parallel_runs(
        ;
        requested_worker_count = parsed_args.requested_worker_count,
        selected_names = parsed_args.selected_names,
        skip_existing = parsed_args.skip_existing,
        dry_run = parsed_args.dry_run,
        solver_backend = parsed_args.solver_backend,
        formula_mode = parsed_args.formula_mode,
        event_eye = parsed_args.event_eye,
        event_onset = parsed_args.event_onset,
        path_prefix_from = parsed_args.path_prefix_from,
        path_prefix_to = parsed_args.path_prefix_to,
        group_definitions_file = parsed_args.group_definitions_file,
        results_file = parsed_args.results_file,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
