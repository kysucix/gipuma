#!
prog="./gipuma"
warping="../fusibile/fusibile"
#inputdir="/home/gsilvano/data/DTURobotImageDataSets/SampleSet1/scan1/resize50/"
inputdir="data/dtu/SampleSet/MVS Data/Rectified/scan${1}/"
batch_name="dtu_fast_${1}"
output_dir_basename="results/$batch_name"
p_folder="data/dtu/calib/"
scale=1
blocksize=15
iter=6
cost_gamma=10
cost_comb="best_n"
n_best=3
depth_max=800
depth_min=300
image_list_array=`( cd "$inputdir" && ls *_3_*) `
output_dir=${output_dir_basename}/$i/
min_angle=10
max_angle=30
max_views=9

# fuse options
disp_thresh=0.1
normal_thresh=30
num_consistent=3

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
    quotedinput=`echo $inputdir | sed "s/ /\\\\ /"`
    echo -n $quotedinput

    echo $prog ${image_list[@]} -images_folder "$inputdir" -p_folder $p_folder -output_folder $output_dir -no_display --algorithm=pm --ct_eps=2.5 --cam_scale=$scale --iterations=$iter --disp_tol=10 --norm_tol=0.2 --gtDepth_divisionFactor=1 --gtDepth_tolerance=0.1 --gtDepth_tolerance2=0.02 --blocksize=$blocksize --cost_gamma=$cost_gamma --cost_comb=best_n --n_best=$n_best --depth_max=$depth_max --depth_min=$depth_min -view_selection --min_angle=$min_angle --max_angle=$max_angle --max_views=$max_views
    $prog ${image_list[@]} -images_folder "$inputdir" -p_folder $p_folder -output_folder $output_dir -no_display --algorithm=pm --ct_eps=2.5 --cam_scale=$scale --iterations=$iter --disp_tol=10 --norm_tol=0.2 --gtDepth_divisionFactor=1 --gtDepth_tolerance=0.1 --gtDepth_tolerance2=0.02 --blocksize=$blocksize --cost_gamma=$cost_gamma --cost_comb=best_n --n_best=$n_best --depth_max=$depth_max --depth_min=$depth_min -view_selection --min_angle=$min_angle --max_angle=$max_angle --max_views=$max_views
    let "count += 1"
    if [ $count -eq -1 ]
    then
	    break
    fi
done
echo $warping -input_folder $output_dir -p_folder $p_folder -images_folder "$inputdir" --cam_scale=$scale --depth_min=$depth_min --depth_max=$depth_max --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent
$warping -input_folder $output_dir -p_folder $p_folder -images_folder "$inputdir" --cam_scale=$scale --depth_min=$depth_min --depth_max=$depth_max --disp_thresh=$disp_thresh --normal_thresh=$normal_thresh --num_consistent=$num_consistent
