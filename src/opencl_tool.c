#include "opencl_tool.h"
#include <math.h>

/**创建平台、设备、上下文、命令队列、程序对象,对大部分 OpenCL 程序相同。
 */
int CreateTool(cl_platform_id *platform,cl_device_id *device,cl_context *context,
                cl_command_queue *commandQueue,cl_program *program,const char *fileName){
    cl_int err;
    cl_uint num;
    //获得第一个可用平台
    err=clGetPlatformIDs(1, platform, &num);
    if(err!=CL_SUCCESS||num<=0||platform==NULL){
        fprintf(stderr,"no platform.");
        return -1;
    }
    //获得第一个可用设备
    err=clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, 1, device, &num);
    if(err!=CL_SUCCESS||num<=0||device==NULL){
        fprintf(stderr,"no device.");
        return -1;
    }
    //获得一个上下文
    cl_context_properties properties[]={
        CL_CONTEXT_PLATFORM,(cl_context_properties)*platform,0
    };
    *context=clCreateContextFromType(properties,CL_DEVICE_TYPE_GPU,NULL,NULL,&err);
    if(err!=CL_SUCCESS||context==NULL){
        fprintf(stderr,"no context.");
        return -1;
    }
    //通过上下文对指定设备构建命令队列
    *commandQueue=clCreateCommandQueue(*context, *device, 0, &err);
    if(err!=CL_SUCCESS||commandQueue==NULL){
        fprintf(stderr,"no commandQueue.");
        return -1;
    }
    //读取内核文件并转换为字符串
    FILE *kernelFile;
    kernelFile=fopen(fileName,"r");
    if(kernelFile==NULL){
        fprintf(stderr,"kernel file open failed.");
        return -1;
    }
    fseek(kernelFile, 0, SEEK_END);
    int fileLen = ftell(kernelFile);
    char *srcStr = (char *) malloc(sizeof(char) * fileLen);
    fseek(kernelFile, 0, SEEK_SET);
    fread(srcStr, fileLen, sizeof(char), kernelFile);
    fclose(kernelFile);
    srcStr[fileLen]='\0';
    //在上下文环境下编译指定内核文件的程序对象
    *program=clCreateProgramWithSource(*context, 1, (const char **)&srcStr, NULL, &err);
    if(err!=CL_SUCCESS||program==NULL){
        fprintf(stderr,"no program.");
        return -1;
    }
    err=clBuildProgram(*program, 0, NULL, NULL,NULL,NULL);
    if(err!=CL_SUCCESS){
        fprintf(stderr,"can not build program.\n");
        char buildLog[16384];
        clGetProgramBuildInfo(*program,*device,CL_PROGRAM_BUILD_LOG,sizeof(buildLog),
            buildLog,NULL);
        fprintf(stderr,buildLog);
        fprintf(stderr,"wrong\n");
        return -1;
    }
    return 0;
}
//资源释放
void clean(cl_context *context,cl_command_queue *commandQueue,cl_program *program,cl_kernel *kernel)
{
    if(*commandQueue!=0)
        clReleaseCommandQueue(*commandQueue);
    if(*kernel!=0)
        clReleaseKernel(*kernel);
    if(*program!=0)
        clReleaseProgram(*program);
    if(*context!=0)
        clReleaseContext(*context);
}

void setWorkItemSize(int kernel_num,size_t global_work_size[3],size_t local_work_size[3]){
    extern int CL_BLOCK;
    local_work_size[0]=CL_BLOCK;
    local_work_size[1]=1;
    local_work_size[2]=1;
    if(kernel_num>1024*1024*64){
        fprintf(stderr,"\nkernel nums too large!!!\n");
        exit(-1);
    }
    if(kernel_num>1024*1024){
        int dim0=1024/CL_BLOCK;
        int dim1=1024;
        int dim2=(kernel_num-1)/(1024*1024)+1;
        global_work_size[0]=dim0;
        global_work_size[1]=dim1;
        global_work_size[2]=dim2;
        fprintf(stderr,"\n%d,%d,%d\n",dim0,dim1,dim2);
        return;
    }
    else if(kernel_num>1024){
        int dim0=1024/CL_BLOCK;
        int dim1=(kernel_num-1)/(1024)+1;
        global_work_size[0]=dim0;
        global_work_size[1]=dim1;
        global_work_size[2]=1;
        return;
    }
    else{
        int dim0=(kernel_num-1)/CL_BLOCK+1;
        global_work_size[0]=dim0;
        global_work_size[1]=1;
        global_work_size[2]=1;
        return;
    }
}