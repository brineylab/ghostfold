#!/bin/bash

# A bash script to:
# 1. Generate MSAs from a FASTA file, parallelized across GPUs.
# 2. Run Colabfold for structure prediction, with optional subsampling.
# 3. Clean up and organize the final prediction folders.
#
# Features:
# - `--msa-only`: Run only the MSA generation step.
# - `--fold-only`: Run only the Colabfold prediction step on an existing MSA directory.
# - `--subsample`: Run Colabfold with multiple MSA subsampling levels.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Colabfold Cleanup Function ---
# Cleans and organizes the output files for a single subsample run.
cleanup_colabfold_outputs() {
    local subsample_dir="$1"
    local preds_dir="$subsample_dir/preds"
    local best_dir="$subsample_dir/best"

    echo "---"
    echo "🧹 Starting cleanup for: $subsample_dir"

    # Ensure the 'best' directory exists for the final PDBs.
    mkdir -p "$best_dir"

    # Loop through each individual prediction folder (e.g., preds/7092_A).
    find "$preds_dir" -mindepth 1 -maxdepth 1 -type d | while read -r pred_dir; do
        local query_name
        query_name=$(basename "$pred_dir")
        echo "   [Cleaning] $query_name"

        # 1. Create subdirectories for scores, images, and recycle files.
        mkdir -p "$pred_dir/scores" "$pred_dir/imgs" "$pred_dir/recycles"

        # 2. Move files into the new directories.
        find "$pred_dir" -maxdepth 1 -type f -name "*.json" -exec mv -t "$pred_dir/scores" {} +
        find "$pred_dir" -maxdepth 1 -type f -name "*.png" -exec mv -t "$pred_dir/imgs" {} +
        find "$pred_dir" -maxdepth 1 -type f \( -name "*.r[0-9].pdb" -o -name "*.r[0-9][0-9].pdb" \) -exec mv -t "$pred_dir/recycles" {} +

        # 3. Copy the top-ranked PDB to the 'best' directory.
        local rank_1_pdb
        rank_1_pdb=$(find "$pred_dir" -maxdepth 1 -type f -name "*rank_001*.pdb" ! -name "*.r[0-9].pdb" | head -n 1)

        if [[ -n "$rank_1_pdb" ]]; then
            local dest_pdb_path="$best_dir/${query_name}_ghostfold.pdb"
            cp "$rank_1_pdb" "$dest_pdb_path"
            echo "     -> Copied top PDB to best/$(basename "$dest_pdb_path")"
        fi

        # 4. Delete any 'done.txt' files.
        find "$pred_dir" -maxdepth 1 -type f -name "*done.txt" -delete
    done
    echo "✅ Cleanup complete for $subsample_dir."
}


# --- Colabfold Prediction Function ---
# Runs Colabfold, supporting standard and subsampling modes.
run_colabfold() {
    local project_name="$1"
    local num_gpus="$2"
    local subsample_mode="$3" # true or false
    local mask_fraction="$4"

    echo "---"
    echo "▶️ Starting Colabfold Structure Prediction..."

    if ! command -v mamba &> /dev/null; then
        echo "❌ Error: mamba command not found." >&2
        return 1
    fi

    local msa_root_dir="$project_name/msa"
    if [[ ! -d "$msa_root_dir" ]]; then
        echo "❌ Error: MSA directory not found at '$msa_root_dir'" >&2
        return 1
    fi

    local -a original_a3m_files
    while IFS= read -r -d $'\0'; do
        original_a3m_files+=("$REPLY")
    done < <(find "$msa_root_dir" -type f -name "pstMSA.a3m" -print0)

    if [[ ${#original_a3m_files[@]} -eq 0 ]]; then
        echo "⚠️ Warning: No 'pstMSA.a3m' files found in '$msa_root_dir'."
        return 0
    fi

    # --- MASKED_MSA ---
    local -a a3m_files_to_process=("${original_a3m_files[@]}")
    local -a temp_masked_files=()

    # If masking is enabled, create temporary masked files
    if [[ -n "$mask_fraction" && "$mask_fraction" != "0" ]]; then
        echo "---"
        echo "🎭 Creating temporary masked MSAs with fraction: $mask_fraction"
        a3m_files_to_process=() # Reset the array to fill with new temp files

        for msa_file in "${original_a3m_files[@]}"; do
            local temp_masked_file="${msa_file%.a3m}_masked_temp.a3m"
            echo "     [Masking] $msa_file -> $temp_masked_file"
            
            python3 mask_msa.py \
                --input_path "$msa_file" \
                --output_path "$temp_masked_file" \
                --mask_fraction "$mask_fraction"

            a3m_files_to_process+=("$temp_masked_file")
            temp_masked_files+=("$temp_masked_file")
        done
        echo "✅ Temporary masked MSAs created."
    fi

    # Define subsampling parameters
    local -a max_seq_vals
    local -a max_extra_seq_vals

    if [[ "$subsample_mode" == "true" ]]; then
        echo "🔬 Subsampling mode enabled."
        max_seq_vals=(16 32 64 128)
        max_extra_seq_vals=(32 64 128 256)
    else
        echo "🔬 Standard mode 32:64 (no subsampling)."
        max_seq_vals=(32)
        max_extra_seq_vals=(64)
    fi

    # Loop over each subsampling level
    for i in "${!max_seq_vals[@]}"; do
        local max_seq=${max_seq_vals[$i]}
        local max_extra_seq=${max_extra_seq_vals[$i]}
        local subsample_dir="$project_name/subsample_$((i + 1))"
        local preds_dir="$subsample_dir/preds"
        local zip_file="$subsample_dir.zip"

        echo "---"
        echo "▶️ Running subsample level $((i + 1)) / ${#max_seq_vals[@]}"
        echo "   Parameters: --max-seq $max_seq --max-extra-seq $max_extra_seq"
        echo "   Output Dir: $preds_dir"

        local colabfold_params="--num-recycle 10 --num-models 5 --rank ptm --recycle-early-stop-tolerance 0.5 --use-dropout --max-seq $max_seq --max-extra-seq $max_extra_seq --num-seeds 5 --save-recycles --model-type auto"
        mkdir -p "$preds_dir"

        local pids=()
        local gpu_counter=0
        for msa_file in "${a3m_files_to_process[@]}"; do
            while [[ ${#pids[@]} -ge $num_gpus ]]; do
                wait -n
                local new_pids=()
                for pid in "${pids[@]}"; do
                    if kill -0 "$pid" 2>/dev/null; then
                        new_pids+=("$pid")
                    fi
                done
                pids=("${new_pids[@]}")
            done

            local output_folder_name
            output_folder_name=$(basename "$(dirname "$msa_file")")
            local current_pred_dir="$preds_dir/$output_folder_name"
            mkdir -p "$current_pred_dir"
            local gpu_id=$(( gpu_counter % num_gpus ))
            
            echo "   [Dispatching Colabfold] Input: $output_folder_name -> GPU: $gpu_id"
            (
                export CUDA_VISIBLE_DEVICES=$gpu_id
                mamba run -n colabfold --no-capture-output colabfold_batch \
                    "$msa_file" \
                    "$current_pred_dir" \
                    $colabfold_params
            ) </dev/null & 
            pids+=($!)
            gpu_counter=$((gpu_counter + 1))
        done

        echo "⏳ Waiting for Colabfold jobs at this level to complete..."
        wait
        echo "✅ Colabfold jobs finished for subsample level $((i + 1))."

        cleanup_colabfold_outputs "$subsample_dir"

        echo "📦 Zipping results for this level..."
        (cd "$project_name" && zip -r "$(basename "$subsample_dir").zip" "$(basename "$subsample_dir")" > /dev/null)
        echo "✅ Zip file created at: $project_name/$(basename "$subsample_dir").zip"
    done
    if [[ ${#temp_masked_files[@]} -gt 0 ]]; then
        echo "---"
        echo "🧹 Cleaning up temporary masked MSA files..."
        for temp_file in "${temp_masked_files[@]}"; do
            rm -f "$temp_file"
        done
        echo "✅ Cleanup of temporary files complete."
    fi
}


# --- Main MSA Generation Function ---
# Generates MSAs and post-processes them.
run_parallel_msa() {
    local project_name="$1"
    local fasta_file="$2"
    local num_gpus="$3"
    
    local -a python_args=(
        "--config" "config.yaml" "--evolve_msa" "--no_coevolution_maps" "--num_runs" "1"
    )

    local num_seqs
    num_seqs=$(grep -c "^>" "$fasta_file")

    echo "✅ Detected $num_gpus GPUs and $num_seqs sequences for project '$project_name'."
    echo "📂 Creating main project directory: $project_name"
    mkdir -p "$project_name"

    if [[ "$num_seqs" -eq 1 ]]; then
        echo "🚀 Only one sequence found. Running on a single GPU."
        CUDA_VISIBLE_DEVICES=0 python pseudomsa.py \
            --project_name "$project_name" \
            --fasta_file "$fasta_file" \
            "${python_args[@]}"
    else
        local temp_dir
        temp_dir=$(mktemp -d -t pseudoMSA_splits_XXXXXXXX)
        echo "📂 Splitting FASTA file into temporary directory: $temp_dir"
        awk -v dir="$temp_dir" '/^>/ { if(f)close(f); f=sprintf("%s/split_seq_%04d.fasta", dir, ++n) } { print > f }' "$fasta_file"
        
        local max_jobs=$num_gpus
        [[ "$num_seqs" -lt "$num_gpus" ]] && max_jobs=$num_seqs
        echo "🚀 Starting parallel MSA generation on $max_jobs GPUs..."
        
        local pids=()
        local gpu_counter=0
        local dispatched_count=0
        local files_to_process=("$temp_dir"/split_seq_*.fasta)

        while [[ $dispatched_count -lt $num_seqs ]]; do
            if [[ ${#pids[@]} -lt $max_jobs ]]; then
                local file_path=${files_to_process[$dispatched_count]}
                local seq_num
                seq_num=$(basename "$file_path" | sed -E 's/split_seq_([0-9]+).fasta/\1/' | sed 's/^0*//')
                local gpu_id=$(( gpu_counter % max_jobs ))
                
                echo "   [Dispatching MSA] Job for sequence #$seq_num on GPU $gpu_id"
                (
                    CUDA_VISIBLE_DEVICES=$gpu_id python pseudomsa.py \
                        --project_name "$project_name" \
                        --fasta_file "$file_path" \
                        "${python_args[@]}"
                ) </dev/null &
                pids+=($!)
                dispatched_count=$((dispatched_count + 1))
                gpu_counter=$((gpu_counter + 1))
            else
                wait -n
                local new_pids=()
                for pid in "${pids[@]}"; do
                    if kill -0 "$pid" 2>/dev/null; then
                        new_pids+=("$pid")
                    fi
                done
                pids=("${new_pids[@]}")
            fi
        done
        
        echo "⏳ All MSA jobs dispatched. Waiting for completion..."
        wait
        echo "🧹 Cleaning up temporary files..."
        rm -rf "$temp_dir"
    fi
    
    echo "🔄 Starting post-processing of MSA files..."
    find "$project_name" -type f -name "pstMSA.fasta" | while read -r fasta_file_path; do
        header_id=$(basename "$(dirname "$fasta_file_path")")
        temp_file="${fasta_file_path}.tmp"
        sed "1c >${header_id}" "$fasta_file_path" > "$temp_file" && mv "$temp_file" "$fasta_file_path"
        a3m_file="${fasta_file_path%.fasta}.a3m"
        cp "$fasta_file_path" "$a3m_file"
        echo "   [Processed MSA] ${header_id}"
    done
    echo "✅ MSA generation and processing complete."
}

# --- Usage and Help Function ---
usage() {
    echo "Usage: $0 --project_name <name> [--fasta_file <file>] [options]"
    echo ""
    echo "Modes (mutually exclusive):"
    echo "  (default)          Runs the full pipeline: MSA generation -> Colabfold."
    echo "                     Requires: --project_name, --fasta_file"
    echo "  --msa-only         Only generates the MSAs."
    echo "                     Requires: --project_name, --fasta_file"
    echo "  --fold-only        Only runs Colabfold on existing MSAs."
    echo "                     Requires: --project_name"
    echo ""
    echo "Options:"
    echo "  --subsample        Enable MSA subsampling mode for Colabfold."
    echo "  --mask_msa <frac>   Mask a fraction of MSA residues (e.g., 0.15 for 15%)."
    echo "  -h, --help         Show this help message."
}

# --- Main Script Entrypoint ---
main() {
    # --- Argument Parsing ---
    local project_name=""
    local fasta_file=""
    local msa_only=false
    local fold_only=false
    local subsample=false
    local mask_fraction=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --project_name) project_name="$2"; shift 2 ;;
            --fasta_file) fasta_file="$2"; shift 2 ;;
            --msa-only) msa_only=true; shift 1 ;;
            --fold-only) fold_only=true; shift 1 ;;
            --subsample) subsample=true; shift 1 ;;
            --mask_msa) mask_fraction="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) echo "❌ Error: Unknown argument '$1'"; usage; exit 1 ;;
        esac
    done

    # --- Argument Validation ---
    if [[ "$msa_only" == true && "$fold_only" == true ]]; then
        echo "❌ Error: --msa-only and --fold-only cannot be used together." >&2; usage; exit 1
    fi
    if [[ -z "$project_name" ]]; then
        echo "❌ Error: --project_name is a required argument." >&2; usage; exit 1
    fi
    if [[ "$fold_only" == false && -z "$fasta_file" ]]; then
        echo "❌ Error: --fasta_file is required unless in --fold-only mode." >&2; usage; exit 1
    elif [[ "$fold_only" == false && ! -f "$fasta_file" ]]; then
        echo "❌ Error: FASTA file not found at '$fasta_file'" >&2; exit 1
    fi
    if [[ -n "$mask_fraction" && ! "$mask_fraction" =~ ^0\.[0-9]+$|^1\.0$|^0$ ]]; then
        echo "❌ Error: --mask_msa requires a fraction between 0.0 and 1.0 (e.g., 0.15)." >&2; usage; exit 1
    fi
    
    # --- GPU Detection ---
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ Error: nvidia-smi command not found. Cannot detect GPUs." >&2; exit 1
    fi
    local num_gpus
    num_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    if [[ -z "$num_gpus" || "$num_gpus" -eq 0 ]]; then
        echo "❌ Error: No GPUs detected by nvidia-smi." >&2; exit 1
    fi

    # --- Execution Logic ---
    if [[ "$msa_only" == true ]]; then
        echo "🚀 Running in MSA-only mode."
        run_parallel_msa "$project_name" "$fasta_file" "$num_gpus"
        echo "🎉 MSA generation finished for project '$project_name'."

    elif [[ "$fold_only" == true ]]; then
        echo "🚀 Running in Fold-only mode."
        run_colabfold "$project_name" "$num_gpus" "$subsample" "$mask_fraction"
        echo "🎉 Colabfold prediction finished for project '$project_name'."

    else # Full pipeline
        echo "🚀 Running full pipeline (MSA generation + Colabfold)."
        run_parallel_msa "$project_name" "$fasta_file" "$num_gpus"
        run_colabfold "$project_name" "$num_gpus" "$subsample" "$mask_fraction"
        echo "🎉 All tasks completed successfully for project '$project_name'!"
    fi
}

# --- Script entrypoint ---
main "$@"
