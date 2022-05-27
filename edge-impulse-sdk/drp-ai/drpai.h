#ifndef DRPAI_H
#define DRPAI_H

/*****************************************
* includes
******************************************/
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <cstring>

#include <linux/drpai.h>

/*****************************************
* Static Variables for DRPAI
* Following variables need to be changed in order to custormize the AI model
*  - label_list = class labels to be classified
*  - drpai_prefix = directory name of DRP-AI Object files (DRP-AI Translator output)
******************************************/
const static std::string drpai_prefix	= "hrnet_cam";

/*****************************************
* Static Variables (No need to change)
* Following variables are the file name of each DRP-AI Object file
* drpai_file_path order must be same as the INDEX_* defined later.
******************************************/
const static std::string drpai_address_file = drpai_prefix+"/"+drpai_prefix+"_addrmap_intm.txt";
const static std::string drpai_file_path[5] =
{
	drpai_prefix+"/drp_desc.bin",
	drpai_prefix+"/"+drpai_prefix+"_drpcfg.mem",
	drpai_prefix+"/drp_param.bin",
	drpai_prefix+"/aimac_desc.bin",
	drpai_prefix+"/"+drpai_prefix+"_weight.dat",
};


/*****************************************
* Macro
******************************************/
/*Number of class to be classified (Need to change to use the customized model.)*/
#define NUM_CLASS				(1000)

/*Maximum DRP-AI Timeout threshold*/
#define DRPAI_TIMEOUT			(5)

/*Frame threshold to execute inference in every loop
 *This value must be determined by DRP-AI processing time and capture processing time.
 *For your information ResNet-50 takes around 90 msec and capture takes around 30 msec. */
#define INF_FRAME_NUM			(4)

/*Waiting Time*/
#define WAIT_TIME				(1000) /* microseconds */

/*Timer Related*/
#define AI_THREAD_TIMEOUT		(20)  /* seconds */

/*Buffer size for writing data to memory via DRP-AI Driver.*/
#define BUF_SIZE				(1024)
/*Number of bytes for single FP32 number in DRP-AI.*/
#define NUM_BYTE				(4)

/*Index to access drpai_file_path[]*/
#define INDEX_D					(0)
#define INDEX_C					(1)
#define INDEX_P					(2)
#define INDEX_A					(3)
#define INDEX_W					(4)

/*****************************************
* ERROR CODE
******************************************/
#define SUCCESS					(0x00)
#define ERR_ADDRESS_CMA			(0x01)
#define ERR_OPEN				(0x10)
#define ERR_FORMAT				(0x11)
#define ERR_READ				(0x12)
#define ERR_MMAP				(0x13)
#define ERR_MLOCK				(0x14)
#define ERR_MALLOC				(0x15)
#define ERR_GET_TIME			(0x16)
#define ERR_DRPAI_TIMEOUT		(0x20)
#define ERR_DRPAI_START			(0x21)
#define ERR_DRPAI_ASSIGN		(0x22)
#define ERR_DRPAI_WRITE			(0x23)
#define ERR_SYSTEM_ID			(-203)

/*****************************************
* Typedef
******************************************/
/* For DRP-AI Address List */
typedef struct
{
	unsigned long desc_aimac_addr;
	unsigned long desc_aimac_size;
	unsigned long desc_drp_addr;
	unsigned long desc_drp_size;
	unsigned long drp_param_addr;
	unsigned long drp_param_size;
	unsigned long data_in_addr;
	unsigned long data_in_size;
	unsigned long data_addr;
	unsigned long data_size;
	unsigned long work_addr;
	unsigned long work_size;
	unsigned long data_out_addr;
	unsigned long data_out_size;
	unsigned long drp_config_addr;
	unsigned long drp_config_size;
	unsigned long weight_addr;
	unsigned long weight_size;
} st_addr_t;

/*AI Inference for DRPAI*/
static st_addr_t drpai_address;
static std::string dir = drpai_prefix+"/";
static std::string address_file=dir+drpai_prefix+ "_addrmap_intm.txt";

static int drpai_fd = -1;
static drpai_data_t proc[DRPAI_INDEX_NUM];
//FIXME: define this per model
#define INF_OUT_SIZE 1000 * 4
// FIXME(@dasch0): currently first run is left true, as we close and reopen DRPAI on each infer
static bool first_run = true;

float drpai_output_buf[INF_OUT_SIZE];

/*****************************************
* Function Name : read_addrmap_txt
* Description	: Loads address and size of DRP-AI Object files into struct addr.
* Arguments		: addr_file = filename of addressmap file (from DRP-AI Object files)
* Return value	: 0 if succeeded
*				  not 0 otherwise
******************************************/
static int8_t read_addrmap_txt(std::string addr_file)
{
	std::ifstream ifs(addr_file);
	std::string str;
	unsigned long l_addr;
	unsigned long l_size;
	std::string element, a, s;

	ei_printf("read1");

	if (ifs.fail())
	{
		printf("[ERROR] Adddress Map List open failed : %s\n", addr_file.c_str());
		return -1;
	}

	while (getline(ifs, str))
	{
		std::istringstream iss(str);
		iss >> element >> a >> s;
		l_addr = strtol(a.c_str(), NULL, 16);
		l_size = strtol(s.c_str(), NULL, 16);

		if (element == "drp_config")
		{
			drpai_address.drp_config_addr = l_addr;
			drpai_address.drp_config_size = l_size;
		}
		else if (element == "desc_aimac")
		{
			drpai_address.desc_aimac_addr = l_addr;
			drpai_address.desc_aimac_size = l_size;
		}
		else if (element == "desc_drp")
		{
			drpai_address.desc_drp_addr = l_addr;
			drpai_address.desc_drp_size = l_size;
		}
		else if (element == "drp_param")
		{
			drpai_address.drp_param_addr = l_addr;
			drpai_address.drp_param_size = l_size;
		}
		else if (element == "weight")
		{
			drpai_address.weight_addr = l_addr;
			drpai_address.weight_size = l_size;
		}
		else if (element == "data_in")
		{
			drpai_address.data_in_addr = l_addr;
			drpai_address.data_in_size = l_size;
		}
		else if (element == "data")
		{
			drpai_address.data_addr = l_addr;
			drpai_address.data_size = l_size;
		}
		else if (element == "data_out")
		{
			drpai_address.data_out_addr = l_addr;
			drpai_address.data_out_size = l_size;
		}
		else if (element == "work")
		{
			drpai_address.work_addr = l_addr;
			drpai_address.work_size = l_size;
		}
	}

	return 0;
}

/*****************************************
* Function Name : load_data_to_mem
* Description	: Loads a file to memory via DRP-AI Driver
* Arguments		: data = filename to be written to memory
*				  drpai_fd = file descriptor of DRP-AI Driver
*				  from = memory start address where the data is written
*				  size = data size to be written
* Return value	: 0 if succeeded
*				  not 0 otherwise
******************************************/
static int8_t load_data_to_mem(std::string data, int drpai_fd,
						unsigned long from, unsigned long size)
{
	int obj_fd;
	uint8_t buf[BUF_SIZE];
	drpai_data_t drpai_data;

	printf("Loading : %s\n", data.c_str());
	obj_fd = open(data.c_str(), O_RDONLY);
	if (obj_fd < 0)
	{
		printf("[ERROR] Failed to Open : %s\n", data.c_str());
		return -1;
	}

	drpai_data.address = from;
	drpai_data.size = size;

	errno = 0;
	if ( -1 == ioctl(drpai_fd, DRPAI_ASSIGN, &drpai_data) )
	{
		printf("[ERROR] Failed to run DRPAI_ASSIGN : errno=%d\n", errno);
		return -1;
	}

	for (int i = 0 ; i<(drpai_data.size/BUF_SIZE) ; i++)
	{
		if ( 0 > read(obj_fd, buf, BUF_SIZE))
		{
			printf("[ERROR] Failed to Read : %s\n", data.c_str());
			return -1;
		}
		errno = 0;
		if ( -1 == write(drpai_fd, buf,  BUF_SIZE) )
		{
			printf("[ERROR] Failed to Write to Memory via DRP-AI Driver : errno=%d\n", errno);
			return -1;
		}
	}

	ei_printf("5\n");
	if ( 0 != (drpai_data.size % BUF_SIZE))
	{
		if ( 0 > read(obj_fd, buf, (drpai_data.size % BUF_SIZE)))
		{
			printf("[ERROR] Failed to Read : %s\n", data.c_str());
			return -1;
		}
		errno = 0;
		if ( -1 == write(drpai_fd, buf, (drpai_data.size % BUF_SIZE)) )
		{
			printf("[ERROR] Failed to Write to Memory via DRP-AI Driver  : errno=%d\n", errno);
			return -1;
		}
	}

	ei_printf("6\n");
	close(obj_fd);
	return 0;
}

/*****************************************
* Function Name :  load_drpai_data
* Description	: Loads DRP-AI Object files to memory via DRP-AI Driver.
* Arguments		: drpai_fd = file descriptor of DRP-AI Driver
* Return value	: 0 if succeeded
*				: not 0 otherwise
******************************************/
static int load_drpai_data(int drpai_fd) {
	unsigned long addr, size;
	for ( int i = 0; i < 5; i++ )
	{
		switch (i)
		{
			case (INDEX_W):
				addr = drpai_address.weight_addr;
				size = drpai_address.weight_size;
				break;
			case (INDEX_C):
				addr = drpai_address.drp_config_addr;
				size = drpai_address.drp_config_size;
				break;
			case (INDEX_P):
				addr = drpai_address.drp_param_addr;
				size = drpai_address.drp_param_size;
				break;
			case (INDEX_A):
				addr = drpai_address.desc_aimac_addr;
				size = drpai_address.desc_aimac_size;
				break;
			case (INDEX_D):
				addr = drpai_address.desc_drp_addr;
				size = drpai_address.desc_drp_size;
				break;
			default:
				break;
		}

		if (0 != load_data_to_mem(drpai_file_path[i], drpai_fd, addr, size))
		{
			printf("[ERROR] Failed to run load_data_to_mem : %s\n",drpai_file_path[i].c_str());
			return -1;
		}
	}
	return 0;
}

EI_IMPULSE_ERROR drpai_run_classifier_image_quantized(uint8_t *raw_features, bool debug) {
#if EI_CLASSIFIER_COMPILED == 1
#error "DRP-AI is not compatible with EON Compiler"
#endif
	// output data from DRPAI model
	drpai_data_t drpai_data;
	// status used to query if any internal errors occured during inferencing
	drpai_status_t drpai_status;
	// descriptor used for checking if DRPAI is done inferencing
	fd_set rfds;
	// struct used to define DRPAI timeout
	struct timespec tv;
	// retval for drpai status
	int ret_drpai;
	// retval when querying drpai status
	int inf_status = 0;

	if (first_run) {
		// Read DRP-AI Object files address and size
		if (0 != read_addrmap_txt(drpai_address_file)) {
			ei_printf("ERR: read_addrmap_txt failed : %s\n", drpai_address_file.c_str());
			return EI_IMPULSE_DRPAI_INIT_FAILED;
		}

		ei_printf("next1\n");

		//DRP-AI Driver Open
		drpai_fd = open("/dev/drpai0", O_RDWR);
		if (drpai_fd < 0) {
			ei_printf("ERR: Failed to Open DRP-AI Driver: errno=%d\n", errno);
			return EI_IMPULSE_DRPAI_INIT_FAILED;
		}

		ei_printf("next1\n");

		// Load DRP-AI Data from Filesystem to Memory via DRP-AI Driver
		ret_drpai = load_drpai_data(drpai_fd);
		if (ret_drpai != 0) {
			ei_printf("ERR: Failed to load DRPAI Data\n");
			if (0 != close(drpai_fd)) {
				ei_printf("ERR: Failed to Close DRPAI Driver: errno=%d\n", errno);
			}
			return EI_IMPULSE_DRPAI_INIT_FAILED;
		}

		ei_printf("next2\n");

		// statically store DRP object file addresses and sizes
		proc[DRPAI_INDEX_INPUT].address = drpai_address.data_in_addr;
		proc[DRPAI_INDEX_INPUT].size = drpai_address.data_in_size;
		ei_printf("DEBUG: DRP-AI input size is: %d\n", drpai_address.data_in_size);
		proc[DRPAI_INDEX_DRP_CFG].address = drpai_address.drp_config_addr;
		proc[DRPAI_INDEX_DRP_CFG].size = drpai_address.drp_config_size;
		proc[DRPAI_INDEX_DRP_PARAM].address = drpai_address.drp_param_addr;
		proc[DRPAI_INDEX_DRP_PARAM].size = drpai_address.drp_param_size;
		proc[DRPAI_INDEX_AIMAC_DESC].address = drpai_address.desc_aimac_addr;
		proc[DRPAI_INDEX_AIMAC_DESC].size = drpai_address.desc_aimac_size;
		proc[DRPAI_INDEX_DRP_DESC].address = drpai_address.desc_drp_addr;
		proc[DRPAI_INDEX_DRP_DESC].size = drpai_address.desc_drp_size;
		proc[DRPAI_INDEX_WEIGHT].address = drpai_address.weight_addr;
		proc[DRPAI_INDEX_WEIGHT].size = drpai_address.weight_size;
		proc[DRPAI_INDEX_OUTPUT].address = drpai_address.data_out_addr;
		proc[DRPAI_INDEX_OUTPUT].size = drpai_address.data_out_size;

		first_run = false;
	}

	// DRP-AI Output Memory Preparation 
	drpai_data.address = drpai_address.data_out_addr;
	drpai_data.size = drpai_address.data_out_size;
	
	// Write data input physical address & size directly to DRP-AI
	proc[DRPAI_INDEX_INPUT].address = (uintptr_t) raw_features;
	//Start DRP-AI driver
	int ioret = ioctl(drpai_fd, DRPAI_START, &proc[0]);
	if (SUCCESS != ioret) {
		if (debug) {
			ei_printf("ERR: Failed to Start DRPAI Inference: %d\n", errno);
		}
		return EI_IMPULSE_DRPAI_RUNTIME_FAILED;
	}

	// Settings For pselect - this is how DRPAI signals inferencing complete
	FD_ZERO(&rfds);
	FD_SET(drpai_fd, &rfds);
	// Define a timeout for DRP-AI to complete
	tv.tv_sec = DRPAI_TIMEOUT;
	tv.tv_nsec = 0;

	// Wait until DRP-AI ends
	ret_drpai = pselect(drpai_fd+1, &rfds, NULL, NULL, &tv, NULL);
	if(ret_drpai == 0)
	{
		if (debug) {
			ei_printf("ERR: DRPAI Inference pselect() Timeout: %d\n", errno);
		}
		return EI_IMPULSE_DRPAI_RUNTIME_FAILED;
	}
	else if(ret_drpai < 0)
	{
		if (debug) {
			ei_printf("ERR: DRPAI Inference pselect() Error: %d\n", errno);
		}
		return EI_IMPULSE_DRPAI_RUNTIME_FAILED;
	} 

	// Checks for DRPAI inference status errors
	inf_status = ioctl(drpai_fd, DRPAI_GET_STATUS, &drpai_status);
	if(inf_status != 0)
	{
		if (debug) {
			ei_printf("ERR: DRPAI Internal Error: %d\n", errno);
		}
		return EI_IMPULSE_DRPAI_RUNTIME_FAILED;
	}

	//FIXME(@dasch0): Should we instead use drpai.h::get_result here()? Implementations look to differ per demo
	// Get DRPAI Output Data
	if (ioctl(drpai_fd, DRPAI_ASSIGN, &drpai_data) != 0)
	{
		if (debug) {
			ei_printf("ERR: Failed to Assign DRPAI data: %d\n", errno);
		}
		return EI_IMPULSE_DRPAI_RUNTIME_FAILED;
	}

	if(read(drpai_fd, &drpai_output_buf[0], drpai_data.size) < 0)
	{
		if (debug) {
			ei_printf("ERR: Failed to read DRPAI output data: %d\n", errno);
		}
		return EI_IMPULSE_DRPAI_RUNTIME_FAILED;
	}
	return EI_IMPULSE_OK;
}

// close the driver
EI_IMPULSE_ERROR drpai_close(bool debug) {
	if(drpai_fd > 0) {
		if (0 != close(drpai_fd)) {
			if (debug) {
				ei_printf("ERR: Failed to Close DRP-AI Driver: errno=%d\n", errno);
			}
			return EI_IMPULSE_DRPAI_RUNTIME_FAILED;
		}
	}
	return EI_IMPULSE_OK;
}

#endif // DRPAI_H
