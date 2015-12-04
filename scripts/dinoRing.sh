#!
prog="./gipuma"
warping="../fusibile/fusibile"
inputdir="data/dinoRing/"
batch_name="dinoRing"
output_dir_basename="results/$batch_name"
p_folder="data/dinoRing/dinoR_par.txt"
scale=1
blocksize=11
iter=8
cost_gamma=10
cost_comb="best_n"
n_best=3
depth_max=0.8
depth_min=0.3
image_list_array=`( cd $inputdir && ls *.png) `
output_dir=${output_dir_basename}/

# fuse options
disp_thresh=0.5
normal_thresh=30
num_consistent=3
min_angle=5
max_angle=45

#warping conf
count=0
for im in $image_list_array
do
    echo $count
    img=${im%.png}
    cmd_file=${output_dir}/$img-cmd.log
    image_list=( $im )

    mkdir -p $output_dir
    for ij in $image_list_array
    do
	if [ $im != $ij ]
	then
	    image_list+=( $ij )
	fi
    done
    cmd="$prog ${image_list[@]} -images_folder $inputdir -krt_file $p_folder -output_folder $output_dir -no_display --algorithm=pm --ct_eps=2.5 --cam_scale=$scale --iterations=$iter --disp_tol=10 --norm_tol=0.2 --gtDepth_divisionFactor=1 --gtDepth_tolerance=0.1 --gtDepth_tolerance2=0.02 --blocksize=$blocksize --cost_gamma=$cost_gamma --cost_comb=best_n --n_best=$n_best --depth_max=$depth_max --depth_min=$depth_min --min_angle=$min_angle --max_angle=$max_angle"
    echo $cmd
    $cmd
    let "count += 1"
    if [ $count -eq -1 ]
    then
	    break
    fi
done
echo $warping -input_folder $output_dir -krt_file $p_folder -images_folder $inputdir --cam_scale=$scale --depth_min=$depth_min --depth_max=$depth_max --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent
$warping -input_folder $output_dir -krt_file $p_folder -images_folder $inputdir --cam_scale=$scale --depth_min=$depth_min --depth_max=$depth_max --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent -remove_black_background
