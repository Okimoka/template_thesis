using Dates
using Printf

function usage()
    error("Usage: julia refresh_chunked_run_summary.jl <run_dir>")
end

length(ARGS) == 1 || usage()

run_dir = abspath(ARGS[1])
manifest_path = joinpath(run_dir, "manifest.tsv")
planned_fits_path = joinpath(run_dir, "planned_fits.tsv")

isfile(manifest_path) || error("Manifest file not found: $(manifest_path)")
isfile(planned_fits_path) || error("Planned fits file not found: $(planned_fits_path)")

function parse_tsv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && return String[], Dict{String,String}[]

    header = split(lines[1], '\t'; keepempty = true)
    rows = Dict{String,String}[]

    for line in lines[2:end]
        isempty(line) && continue
        values = split(line, '\t'; keepempty = true)
        if length(values) < length(header)
            append!(values, fill("", length(header) - length(values)))
        end
        push!(rows, Dict(header[i] => values[i] for i in eachindex(header)))
    end

    return header, rows
end

function parse_state_file(path::AbstractString)
    state = Dict{String,String}()
    isfile(path) || return state

    for line in eachline(path)
        isempty(line) && continue
        occursin('=', line) || continue
        key, value = split(line, '='; limit = 2)
        state[key] = value
    end

    return state
end

function tsv_escape(value)
    escaped = string(value)
    escaped = replace(escaped, '\t' => "    ")
    escaped = replace(escaped, '\n' => "\\n")
    return escaped
end

function write_tsv(path::AbstractString, header, rows)
    open(path, "w") do io
        println(io, join(header, '\t'))
        for row in rows
            println(io, join((tsv_escape(get(row, column, "")) for column in header), '\t'))
        end
    end
    return path
end

function parse_int_field(row::Dict{String,String}, key::AbstractString)
    raw_value = get(row, key, "")
    isempty(raw_value) && return 0
    return parse(Int, raw_value)
end

function write_id_list(path::AbstractString, task_ids)
    open(path, "w") do io
        for task_id in task_ids
            println(io, task_id)
        end
    end
    return path
end

_, manifest_rows = parse_tsv(manifest_path)
_, planned_fit_rows = parse_tsv(planned_fits_path)

result_rows_by_task = Dict{String,Dict{String,Dict{String,String}}}()
task_summary_rows = Dict{String,String}[]

task_status_counts = Dict(
    "success" => 0,
    "running" => 0,
    "partial" => 0,
    "failed" => 0,
    "pending" => 0,
)
fit_status_counts = Dict(
    "saved" => 0,
    "skipped_existing" => 0,
    "failed" => 0,
    "pending" => 0,
)

status_task_ids = Dict(
    "success" => String[],
    "running" => String[],
    "partial" => String[],
    "failed" => String[],
    "pending" => String[],
)

for manifest_row in manifest_rows
    task_id = manifest_row["task_id"]
    fit_count = parse_int_field(manifest_row, "fit_count")
    results_file = manifest_row["results_file"]
    status_file = manifest_row["status_stem"] * ".state"
    state = parse_state_file(status_file)
    state_status = get(state, "status", "")
    state_message = get(state, "message", "")
    state_updated_at = get(state, "updated_at", "")

    _, result_rows_raw = isfile(results_file) ? parse_tsv(results_file) : (String[], Dict{String,String}[])
    result_rows_lookup = Dict{String,Dict{String,String}}()
    for result_row in result_rows_raw
        result_rows_lookup[result_row["group_index"]] = result_row
    end
    result_rows_by_task[task_id] = result_rows_lookup

    saved_count = count(row -> get(row, "status", "") == "saved", result_rows_raw)
    skipped_count = count(row -> get(row, "status", "") == "skipped_existing", result_rows_raw)
    failed_count = count(row -> get(row, "status", "") == "failed", result_rows_raw)
    processed_fit_count = length(result_rows_raw)
    pending_fit_count = max(0, fit_count - processed_fit_count)
    leftover_fit_count = max(0, fit_count - saved_count - skipped_count)

    task_status = if state_status == "success"
        "success"
    elseif state_status == "running"
        "running"
    elseif saved_count + skipped_count >= fit_count && fit_count > 0
        "success"
    elseif processed_fit_count > 0
        if saved_count + skipped_count == 0 && pending_fit_count == 0
            "failed"
        else
            "partial"
        end
    elseif state_status == "failed"
        "failed"
    else
        "pending"
    end

    task_status_counts[task_status] += 1
    push!(status_task_ids[task_status], task_id)

    push!(
        task_summary_rows,
        Dict(
            "task_id" => task_id,
            "task_status" => task_status,
            "label" => manifest_row["label"],
            "group_name" => manifest_row["group_name"],
            "solver_backend" => manifest_row["solver_backend"],
            "formula_mode" => manifest_row["formula_mode"],
            "event_eye" => manifest_row["event_eye"],
            "event_onset" => get(manifest_row, "event_onset", "saccade"),
            "chunk_index" => manifest_row["chunk_index"],
            "chunk_count" => manifest_row["chunk_count"],
            "fit_count" => string(fit_count),
            "saved_count" => string(saved_count),
            "skipped_existing_count" => string(skipped_count),
            "failed_count" => string(failed_count),
            "processed_fit_count" => string(processed_fit_count),
            "pending_fit_count" => string(pending_fit_count),
            "leftover_fit_count" => string(leftover_fit_count),
            "results_file" => results_file,
            "status_file" => status_file,
            "state_status" => state_status,
            "state_updated_at" => state_updated_at,
            "state_message" => state_message,
        ),
    )
end

fit_status_rows = Dict{String,String}[]
for planned_row in planned_fit_rows
    task_id = planned_row["task_id"]
    group_index = planned_row["group_index"]
    result_row = get(get(result_rows_by_task, task_id, Dict{String,Dict{String,String}}()), group_index, nothing)

    fit_status = isnothing(result_row) ? "pending" : get(result_row, "status", "pending")
    output_path = isnothing(result_row) ? "" : get(result_row, "output_path", "")
    error_message = isnothing(result_row) ? "" : get(result_row, "error", "")

    fit_status_counts[fit_status] = get(fit_status_counts, fit_status, 0) + 1

    push!(
        fit_status_rows,
        Dict(
            "task_id" => task_id,
            "group_name" => planned_row["group_name"],
            "solver_backend" => planned_row["solver_backend"],
            "formula_mode" => planned_row["formula_mode"],
            "event_eye" => planned_row["event_eye"],
            "event_onset" => get(planned_row, "event_onset", "saccade"),
            "chunk_index" => planned_row["chunk_index"],
            "group_index" => group_index,
            "group_label" => planned_row["group_label"],
            "fif_group" => planned_row["fif_group"],
            "fit_status" => fit_status,
            "output_path" => output_path,
            "error" => error_message,
        ),
    )
end

sort!(task_summary_rows; by = row -> parse(Int, row["task_id"]))
sort!(
    fit_status_rows;
    by = row -> (
        parse(Int, row["task_id"]),
        parse(Int, row["group_index"]),
    ),
)

task_summary_path = joinpath(run_dir, "task_summary.tsv")
fit_status_path = joinpath(run_dir, "fit_status.tsv")
remaining_task_ids_path = joinpath(run_dir, "remaining_task_ids.txt")
success_task_ids_path = joinpath(run_dir, "success_task_ids.txt")
running_task_ids_path = joinpath(run_dir, "running_task_ids.txt")
partial_task_ids_path = joinpath(run_dir, "partial_task_ids.txt")
failed_task_ids_path = joinpath(run_dir, "failed_task_ids.txt")
pending_task_ids_path = joinpath(run_dir, "pending_task_ids.txt")
overview_path = joinpath(run_dir, "progress_overview.txt")

write_tsv(
    task_summary_path,
    [
        "task_id",
        "task_status",
        "label",
        "group_name",
        "solver_backend",
        "formula_mode",
        "event_eye",
        "event_onset",
        "chunk_index",
        "chunk_count",
        "fit_count",
        "saved_count",
        "skipped_existing_count",
        "failed_count",
        "processed_fit_count",
        "pending_fit_count",
        "leftover_fit_count",
        "results_file",
        "status_file",
        "state_status",
        "state_updated_at",
        "state_message",
    ],
    task_summary_rows,
)
write_tsv(
    fit_status_path,
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
        "fit_status",
        "output_path",
        "error",
    ],
    fit_status_rows,
)

write_id_list(
    remaining_task_ids_path,
    sort([
        row["task_id"] for row in task_summary_rows if row["task_status"] != "success"
    ]; by = x -> parse(Int, x)),
)
write_id_list(success_task_ids_path, sort(status_task_ids["success"]; by = x -> parse(Int, x)))
write_id_list(running_task_ids_path, sort(status_task_ids["running"]; by = x -> parse(Int, x)))
write_id_list(partial_task_ids_path, sort(status_task_ids["partial"]; by = x -> parse(Int, x)))
write_id_list(failed_task_ids_path, sort(status_task_ids["failed"]; by = x -> parse(Int, x)))
write_id_list(pending_task_ids_path, sort(status_task_ids["pending"]; by = x -> parse(Int, x)))

fits_total = length(fit_status_rows)
fits_done = get(fit_status_counts, "saved", 0) + get(fit_status_counts, "skipped_existing", 0)
fits_left = max(0, fits_total - fits_done)
generated_at = Dates.format(now(), Dates.DateFormat("yyyy-mm-dd HH:MM:SS"))

open(overview_path, "w") do io
    println(io, "Run directory: $(run_dir)")
    println(io, "Generated: $(generated_at)")
    println(io)
    println(io, "Task progress")
    println(io, "  total: $(length(task_summary_rows))")
    println(io, "  success: $(get(task_status_counts, "success", 0))")
    println(io, "  running: $(get(task_status_counts, "running", 0))")
    println(io, "  partial: $(get(task_status_counts, "partial", 0))")
    println(io, "  failed: $(get(task_status_counts, "failed", 0))")
    println(io, "  pending: $(get(task_status_counts, "pending", 0))")
    println(io)
    println(io, "Fit progress")
    println(io, "  total: $(fits_total)")
    println(io, "  saved: $(get(fit_status_counts, "saved", 0))")
    println(io, "  skipped_existing: $(get(fit_status_counts, "skipped_existing", 0))")
    println(io, "  failed: $(get(fit_status_counts, "failed", 0))")
    println(io, "  pending: $(get(fit_status_counts, "pending", 0))")
    println(io, "  done_or_already_present: $(fits_done)")
    println(io, "  left_over: $(fits_left)")
    println(io)
    println(io, "Documentation files")
    println(io, "  manifest.tsv")
    println(io, "  planned_fits.tsv")
    println(io, "  task_summary.tsv")
    println(io, "  fit_status.tsv")
    println(io, "  remaining_task_ids.txt")
end

println(
    @sprintf(
        "Refreshed summary: tasks=%d, success=%d, partial=%d, running=%d, failed=%d, pending=%d, fits_done=%d, fits_left=%d",
        length(task_summary_rows),
        get(task_status_counts, "success", 0),
        get(task_status_counts, "partial", 0),
        get(task_status_counts, "running", 0),
        get(task_status_counts, "failed", 0),
        get(task_status_counts, "pending", 0),
        fits_done,
        fits_left,
    ),
)
