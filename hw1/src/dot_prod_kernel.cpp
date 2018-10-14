#include <assert.h>
#include <string.h>

extern "C" {

//void load(float* local_a,float* local_b,const float* a,const float* b,int BATCH_SIZE)
//{
 //       memcpy(local_a, a, BATCH_SIZE*8);
  //      memcpy(local_b, b, BATCH_SIZE*8);
//}

void dot_prod_kernel(const float* a, const float* b, float* c, const int num_elems) {
#pragma HLS interface m_axi port = a offset = slave bundle = gmem
#pragma HLS interface m_axi port = b offset = slave bundle = gmem
#pragma HLS interface m_axi port = c offset = slave bundle = gmem
#pragma HLS interface s_axilite port = a bundle = control
#pragma HLS interface s_axilite port = b bundle = control
#pragma HLS interface s_axilite port = c bundle = control
#pragma HLS interface s_axilite port = num_elems bundle = control
#pragma HLS interface s_axilite port = return bundle = control
assert(num_elems <= 4096);  // this helps HLS estimate the loop trip count

//My Code
int i=0;
int j=0;
int unroll_factor=8;
float local_c=0;
float local_d=0;
float local_e=0;
float local_f=0;
float temp[4096];
#pragma HLS ARRAY_PARTITION variable=temp block factor=128 dim=1
float local_a[4096];
#pragma HLS ARRAY_PARTITION variable=local_a block factor=128 dim=1
float local_b[4096];
#pragma HLS ARRAY_PARTITION variable=local_b block factor=128 dim=1
for(i=0;i<num_elems;i++){
//#pragma HLS PIPELINE II=1
	local_a[i]=a[i];
}
for(j=0;j<num_elems;j++){
//#pragma HLS PIPELINE II=1
	local_b[j]=b[j];
}

loop_a: for(i=0;i<num_elems;i++){
	//#pragma HLS UNROLL
	//#pragma HLS loop_tripcount min=0 max=4096
	#pragma HLS PIPELINE II=1
	temp[i]=local_a[i]*local_b[i];
}

loop_b: for(i=0;i<num_elems;i=i+4){

	local_c+=temp[i];
	local_d+=temp[i+1];
	local_e+=temp[i+2];
	local_f+=temp[i+3];
}


//load(local_a,local_b,a,b,num_elems);
//loop_a: for(i=0;i<num_elems;i=i+unroll_factor){
//#pragma HLS loop_tripcount min=0 max=4096
//#pragma HLS UNROLL 
//	temp=0;
//	loop_b: for(j=0;j<unroll_factor;j++){
//			#pragma HLS loop_tripcount min=0 max=unroll_factor
//			temp += local_a[i+j]*local_b[i+j];
//	}
//	local_c +=temp;	
//}
//for(i=0;i<num_elems;i++)
//{
//	#pragma HLS UNROLL factor=8
//	temp += local_a[i]*local_b[i];
//}
//for(i=1025;i<num_elems;i++)
//{
//	local_c += local_a[i]*local_b[i];
//}
local_c+=local_d;
local_e+=local_f;
*c = local_c+local_e;
}
}  // extern "C"
