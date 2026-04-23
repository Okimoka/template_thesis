using Printf

const LAUNCHER_SCRIPT = joinpath(@__DIR__, "run_new_uf2_group_definitions_parallel.jl")

const CHUNK_CONFIGS = [
    (
        label = "SYNCED_CLEANED_FREEVIEW_GROUPED default",
        group_name = "SYNCED_CLEANED_FREEVIEW_GROUPED",
        solver_backend = "gpu",
        formula_mode = "regular",
        event_eye = "L",
        event_onset = "saccade",
        worker_class = "gpu",
    ),
    (
        label = "SYNCED_CLEANED_FREEVIEW_GROUPED cpu",
        group_name = "SYNCED_CLEANED_FREEVIEW_GROUPED",
        solver_backend = "cpu",
        formula_mode = "regular",
        event_eye = "L",
        event_onset = "saccade",
        worker_class = "cpu",
    ),
    (
        label = "SYNCED_CLEANED_FREEVIEW_GROUPED right-eye",
        group_name = "SYNCED_CLEANED_FREEVIEW_GROUPED",
        solver_backend = "gpu",
        formula_mode = "regular",
        event_eye = "R",
        event_onset = "saccade",
        worker_class = "gpu",
    ),
    (
        label = "SYNCED_CLEANED_FREEVIEW_GROUPED fixation-formula",
        group_name = "SYNCED_CLEANED_FREEVIEW_GROUPED",
        solver_backend = "gpu",
        formula_mode = "fixpos",
        event_eye = "L",
        event_onset = "saccade",
        worker_class = "gpu",
    ),
    (
        label = "SYNCED_CLEANED_FREEVIEW_GROUPED fixation-onset",
        group_name = "SYNCED_CLEANED_FREEVIEW_GROUPED",
        solver_backend = "gpu",
        formula_mode = "regular",
        event_eye = "L",
        event_onset = "fixation",
        worker_class = "gpu",
    ),
    (
        label = "SYNCED_CLEANED_FREEVIEW_UNGROUPED default",
        group_name = "SYNCED_CLEANED_FREEVIEW_UNGROUPED",
        solver_backend = "gpu",
        formula_mode = "regular",
        event_eye = "L",
        event_onset = "saccade",
        worker_class = "gpu",
    ),
    (
        label = "SYNCED_CLEANED_ALLTASKS_GROUPED default",
        group_name = "SYNCED_CLEANED_ALLTASKS_GROUPED",
        solver_backend = "gpu",
        formula_mode = "regular",
        event_eye = "L",
        event_onset = "saccade",
        worker_class = "gpu",
    ),
    (
        label = "SYNCED_CLEANED_ALLTASKS_GROUPED_BRAINONLY default",
        group_name = "SYNCED_CLEANED_ALLTASKS_GROUPED_BRAINONLY",
        solver_backend = "gpu",
        formula_mode = "regular",
        event_eye = "L",
        event_onset = "saccade",
        worker_class = "gpu",
    ),
]

const SOLVER_FILTER = let value = lowercase(strip(get(ENV, "SOLVER_FILTER", "all")))
    value in ("all", "cpu", "gpu") || error("SOLVER_FILTER must be one of: all, cpu, gpu.")
    value
end

const FILTERED_CHUNK_CONFIGS =
    SOLVER_FILTER == "all" ? CHUNK_CONFIGS :
    filter(config -> config.solver_backend == SOLVER_FILTER, CHUNK_CONFIGS)

isempty(FILTERED_CHUNK_CONFIGS) && error("No chunk configs matched SOLVER_FILTER=$(SOLVER_FILTER).")

function usage()
    error(
        "Usage: julia build_chunked_manifest.jl <group_definitions_file> <run_dir> <chunk_size> <gpu_workers> <cpu_workers> [<path_prefix_from> <path_prefix_to>]",
    )
end

length(ARGS) in (5, 7) || usage()

group_definitions_file = abspath(ARGS[1])
run_dir = abspath(ARGS[2])
chunk_size = parse(Int, ARGS[3])
gpu_workers = parse(Int, ARGS[4])
cpu_workers = parse(Int, ARGS[5])
path_prefix_from = length(ARGS) == 7 ? ARGS[6] : nothing
path_prefix_to = length(ARGS) == 7 ? ARGS[7] : nothing

chunk_size > 0 || error("chunk_size must be positive.")
gpu_workers > 0 || error("gpu_workers must be positive.")
cpu_workers > 0 || error("cpu_workers must be positive.")
isfile(group_definitions_file) || error("Group definitions file does not exist: $(group_definitions_file)")

module ChunkManifestLauncher
end

Base.include(ChunkManifestLauncher, LAUNCHER_SCRIPT)
ChunkManifestLauncher.normalize_path_remap(path_prefix_from, path_prefix_to)

chunk_definitions_dir = joinpath(run_dir, "chunk_definitions")
results_dir = joinpath(run_dir, "results")
status_dir = joinpath(run_dir, "status")
logs_dir = joinpath(run_dir, "logs")
mkpath(chunk_definitions_dir)
mkpath(results_dir)
mkpath(status_dir)
mkpath(logs_dir)

function escape_tsv_field(value)
    string_value = string(value)
    string_value = replace(string_value, '\t' => "    ")
    string_value = replace(string_value, '\n' => "\\n")
    return string_value
end

function write_chunk_file(path::AbstractString, group_name::AbstractString, chunk_groups)
    open(path, "w") do io
        println(io, "# Auto-generated chunk definition.")
        println(io, "# Source definitions: $(group_definitions_file)")
        println(io)
        println(io, "$(group_name) = [")
        for fif_group in chunk_groups
            println(io, "    ", repr(fif_group), ",")
        end
        println(io, "]")
    end
    return path
end

function fif_group_label(fif_group)
    return join(basename.(collect(fif_group)), " + ")
end

manifest_path = joinpath(run_dir, "manifest.tsv")
planned_fits_path = joinpath(run_dir, "planned_fits.tsv")

task_id_ref = Ref(0)
available_configs = ChunkManifestLauncher.load_group_configs(
    ;
    selected_names = unique([config.group_name for config in FILTERED_CHUNK_CONFIGS]),
    path_prefix_from = path_prefix_from,
    path_prefix_to = path_prefix_to,
    group_definitions_file = group_definitions_file,
)
configs_by_name = Dict(config.name => config for config in available_configs)

open(manifest_path, "w") do manifest_io
    println(
        manifest_io,
        join(
            [
                "task_id",
                "label",
                "group_name",
                "solver_backend",
                "formula_mode",
                "event_eye",
                "event_onset",
                "workers",
                "chunk_index",
                "chunk_count",
                "start_group_index",
                "end_group_index",
                "fit_count",
                "chunk_definition_file",
                "results_file",
                "status_stem",
            ],
            '\t',
        ),
    )

    open(planned_fits_path, "w") do planned_io
        println(
            planned_io,
            join(
                    [
                        "task_id",
                        "group_name",
                        "solver_backend",
                        "formula_mode",
                        "event_eye",
                        "event_onset",
                        "chunk_index",
                        "group_index",
                    "group_label",
                    "fif_group",
                ],
                '\t',
            ),
        )

        for config in FILTERED_CHUNK_CONFIGS
            haskey(configs_by_name, config.group_name) || error("Missing group definition $(config.group_name).")
            group_values = configs_by_name[config.group_name].fif_paths
            chunk_count = cld(length(group_values), chunk_size)
            workers = config.worker_class == "cpu" ? cpu_workers : gpu_workers

            for chunk_index in 1:chunk_count
                task_id_ref[] += 1
                task_id = task_id_ref[]
                start_group_index = (chunk_index - 1) * chunk_size + 1
                end_group_index = min(chunk_index * chunk_size, length(group_values))
                chunk_groups = group_values[start_group_index:end_group_index]

                chunk_file = joinpath(
                    chunk_definitions_dir,
                    @sprintf(
                        "task_%04d__%s__chunk_%04d_of_%04d.jl",
                        task_id,
                        config.group_name,
                        chunk_index,
                        chunk_count,
                    ),
                )
                results_file = joinpath(results_dir, @sprintf("task_%04d.tsv", task_id))
                status_stem = joinpath(status_dir, @sprintf("task_%04d", task_id))

                write_chunk_file(chunk_file, config.group_name, chunk_groups)

                println(
                    manifest_io,
                    join(
                        [
                            escape_tsv_field(task_id),
                            escape_tsv_field(config.label),
                            escape_tsv_field(config.group_name),
                            escape_tsv_field(config.solver_backend),
                            escape_tsv_field(config.formula_mode),
                            escape_tsv_field(config.event_eye),
                            escape_tsv_field(config.event_onset),
                            escape_tsv_field(workers),
                            escape_tsv_field(chunk_index),
                            escape_tsv_field(chunk_count),
                            escape_tsv_field(start_group_index),
                            escape_tsv_field(end_group_index),
                            escape_tsv_field(length(chunk_groups)),
                            escape_tsv_field(chunk_file),
                            escape_tsv_field(results_file),
                            escape_tsv_field(status_stem),
                        ],
                        '\t',
                    ),
                )

                for (group_offset, fif_group) in enumerate(chunk_groups)
                    group_index = start_group_index + group_offset - 1
                    println(
                        planned_io,
                        join(
                            [
                                escape_tsv_field(task_id),
                                escape_tsv_field(config.group_name),
                                escape_tsv_field(config.solver_backend),
                                escape_tsv_field(config.formula_mode),
                                escape_tsv_field(config.event_eye),
                                escape_tsv_field(config.event_onset),
                                escape_tsv_field(chunk_index),
                                escape_tsv_field(group_index),
                                escape_tsv_field(fif_group_label(fif_group)),
                                escape_tsv_field(repr(fif_group)),
                            ],
                            '\t',
                        ),
                    )
                end
            end
        end
    end
end

println(task_id_ref[])
