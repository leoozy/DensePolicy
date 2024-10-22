#!/bin/bash

## Check if the required arguments are provided
#if [ "$#" -ne 3 ]; then
#    echo "Usage: $0 <Meta-Llama-3-8B-Instruct> <alfworld_put> <codeact>"
#    exit 1
#fi

Model_name="$1"
Task_name="$2"
Method="$3"
Process_per_gpu="$4"
Start_test_id="$5"


# Calculate the values
Total_process=$(( Process_per_gpu * 8 ))
Max_index=$(( Total_process - 1 ))



#Launch vLLM Servers
for i in {0..7}; do
  gpu=$i
  ./downstream_test.sh "$i" "'$gpu'" "$Model_name"
done

# Run downstream tasks
for i in $(seq 0 $Max_index); do
    port=$(( 8000 +i / Process_per_gpu ))
    # when all the arguments are given, the test_id is only the agent tmux windows number
    test_id=$(( i+Start_test_id ))
    dis_id=$(( i ))
#    echo $test_id
    ./downstream_test.sh "$test_id" "'0'" "$Model_name" "$Task_name" "$Method" "$port" "$dis_id" "$Total_process"
done
