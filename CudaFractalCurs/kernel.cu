#include <iostream>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <thrust/complex.h>
#include <cuComplex.h>
// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
float *cpu_memory;
// Window dimensions
const GLuint WIDTH = 1000, HEIGHT = 1000;
GLuint vertexShader;
GLuint fragmentShader;

GLuint shaderProgram;
// Shaders
const GLchar* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec3 color;\n"
"out vec3 ourColor;\n"
"void main()\n"
"{\n"
"gl_Position = vec4(position, 1.0);\n"
"ourColor = color;\n"
"}\0";
const GLchar* fragmentShaderSource = "#version 330 core\n"
"in vec3 ourColor;\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"color = vec4(ourColor, 1.0f);\n"
"}\n\0";

const GLuint MAX_ITERATION = 256;
const GLdouble MAX_R = 0.1, MIN_R = 0.5, MAX_I = 0.1, MIN_I = -0.1;
/* OpenGL interoperability */
dim3 blocks, threads;

//GLuint vbo; //int указатель на Vertex Buffer Object

GLuint VBO;
GLuint VAO;

struct cudaGraphicsResource *cuda_vbo_resource; //Структура предоставляющая реализацию VBO в CUDA

//Отлов ошибок CUDA
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	// Когда пользователь нажимает ESC, мы устанавливаем свойство WindowShouldClose в true, 
	// и приложение после этого закроется
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}
__device__ double mapToReal(int x, int windowWidth, double minR, double maxR)
{
	double range = maxR - minR;
	return x * (range / windowWidth) + minR;
}
__device__ double mapToImaginary(int y, int windowHeigth, double minI, double maxI)
{
	double range = maxI - minI;
	return y * (range / windowHeigth) + minI;
}
__device__ int MandelbrotFunction(double p, double q, int maxIteration)
{
	int i = 0;
	double x_t = 0.0, y_t = 0.0;
	while (i < maxIteration && x_t*x_t + y_t * y_t <= 4.0)
	{
		double temp = x_t * x_t - y_t * y_t + p;
		y_t = 2.0 * x_t * y_t + q;
		x_t = temp;
		i++;
	}

	return i;
	/*
	if (i < maxIteration)
		return 255;
	else
		return 0;*/
}
__device__ int MandelbrotFunction2(double p, double q, int maxIteration)
{
	int i = 0;
	double x_t = 0.0, y_t = 0.0;
	thrust::complex<double> c = thrust::complex<double>(p,q);
	thrust::complex<double> z(0,0);
	thrust::complex<double> r = thrust::complex<double>(2, 2);

	while ((abs(z.real()) < 2 && abs(z.imag()) < 2) && i <= maxIteration)
	{
		z = z * z + c;
		i++;
	}
	if (i < maxIteration)
		return 255;
	else
		return 0;
}
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
	unsigned int size = WIDTH * HEIGHT * sizeof(float) * 6;

	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	//	glVertexAttribPointer(1, 4, sizeof(uchar4), GL_FALSE,size, (GLvoid*)0);
	//	glEnableVertexAttribArray(0);
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
	HANDLE_ERROR(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}
__device__ void setCoord(int x, int y, int offset, float *pixels)
{
	pixels[offset * 6] = (-1.0f + 2.0f * (float)(x / (float)WIDTH));			//x
	pixels[offset * 6 + 1] = (-1.0f + 2.0f * (float)(y / (float)HEIGHT));		//y
	pixels[offset * 6 + 2] = 0.0f;	//z

}
__device__ void setColor(int offset, int ColorValue, float *pixels)
{
	float pixelColor = (float)(ColorValue % 256) / (float)256;
	float r = 0, g = 0, b = 0;
	if (pixelColor < 0.0001f)
	{
		r = 0; g = 0; b = 0;
	}
	else if (pixelColor < 0.45f)
	{
		r = (float)((int)(ColorValue * sinf(ColorValue)) % 256) / (float)256;
		g = (float)(ColorValue % 256) / (float)256;
		b = (float)((int)(ColorValue * cosf(ColorValue)) % 256) / (float)256;
	}
	else if (pixelColor < 0.60f)
	{
		r = (float)((int)(ColorValue * sinf(ColorValue)) % 256) / (float)256;
		g = (float)(ColorValue % 256) / (float)256;
		b = 1.0f;
	}
	else if (pixelColor < 0.80f)
	{
		r = (float)((int)(ColorValue * sinf(ColorValue) * sinf(ColorValue)) % 256) / (float)256;
		g = (float)(ColorValue * ColorValue % 256) / (float)256;
		b = (float)((int)(ColorValue * logf(ColorValue)) % 256) / (float)256;;
	}
	else
	{
		r = 1.0f;
		g = 0.2f;
		b = 0.66f;
	}

		pixels[offset * 6 + 3] = r;			//x
		pixels[offset * 6 + 4] = g;		//y
		pixels[offset * 6 + 5] = b;			//z
	
}
__global__ void MandelbrotKernel(float* screen, int windowHeigth, int windowWidth, double maxR, double maxI, 
	double minR, double minI, int maxIteration)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	if (x >= WIDTH || y >= HEIGHT)
		return;

	int offset = x + y * gridDim.x;
	//double p = mapToReal(x, windowWidth, minR, maxR);
	//double q = mapToImaginary(y, windowHeigth, minI, maxI);
	double p = (float)x / WIDTH - 1.5;
	double q = (float)y / HEIGHT - 0.5;;
	int MandelbrotValue = MandelbrotFunction(p, q, maxIteration);

	setCoord(x, y, offset, screen);
	setColor(offset, MandelbrotValue, screen);

}

void InitOpenGL()
{
	//Инициализация GLFW
	glfwInit();


	//Настройка GLFW
	//Задается минимальная требуемая версия OpenGL. 
	//Мажорная 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	//Минорная
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	//Установка профайла для которого создается контекст
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//Выключение возможности изменения размера окна
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
}
void initCuda(int deviceId) {
	int deviceCount = 0;
	HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

	if (deviceCount <= 0) {
		printf("No CUDA devices found\n");
		exit(-1);
	}

	HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceProperties(&properties, deviceId));

	threads.x = 32;
	threads.y = properties.maxThreadsPerBlock / threads.x - 2; // to avoid cudaErrorLaunchOutOfResources error

	blocks.x = (WIDTH + threads.x - 1) / threads.x;
	blocks.y = (HEIGHT + threads.y - 1) / threads.y;

	printf(
		"Debug: blocks(%d, %d), threads(%d, %d)\nCalculated Resolution: %d x %d\n",
		blocks.x, blocks.y, threads.x, threads.y, blocks.x * threads.x,
		blocks.y * threads.y);
}
void MaldelbrotGPU_Calculation()
{
	float *dev_screen;
	
	dim3 *blocks2 = new dim3(WIDTH,HEIGHT,1);
	cpu_memory = (float*)malloc(WIDTH * HEIGHT * sizeof(float) * 6);
	size_t size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &size, cuda_vbo_resource));
	cudaEvent_t startEvent, stopEvent;
	float elapsedTime = 0.0f;
	HANDLE_ERROR(cudaEventCreate(&startEvent));
	HANDLE_ERROR(cudaEventCreate(&stopEvent));
	HANDLE_ERROR(cudaEventRecord(startEvent, 0));

	// Render Image
	MandelbrotKernel << <*blocks2, 1 >> > (dev_screen, HEIGHT, WIDTH, MAX_R, MAX_I, MIN_R, MIN_I, MAX_ITERATION);
	HANDLE_ERROR(cudaDeviceSynchronize());

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	// Kernel Time measure
	HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
	HANDLE_ERROR(cudaEventSynchronize(stopEvent));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	std::cout << std::endl;
	printf("Время выполнения: %f ms\n", elapsedTime);
	cudaThreadSynchronize();
	cudaMemcpy(cpu_memory, dev_screen, WIDTH * HEIGHT * sizeof(float) * 6, cudaMemcpyDeviceToHost);
	int count = 0;
	/*
	for (int i = 0; i < WIDTH * HEIGHT * 6; i++)
	{
		//if (i >= WIDTH * HEIGHT * 6 - WIDTH)
		//{
			
			
			if (count == 6)
			{
				std::cout << std::endl;
				count = 0;
			}
			std::cout << " " << i << " " << cpu_memory[i] << " ";
			count++;
		//}
	}*/
}
void CreateVertexShader()
{
	GLint success;
	GLchar infoLog[512];
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
}
void CreateFragmentShader()
{
	GLint success;
	GLchar infoLog[512];
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
}
void InitShaders()
{
	GLint success;
	GLchar infoLog[512];

	CreateVertexShader();
	CreateFragmentShader();

	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}
// The MAIN function, from here we start the application and run the game loop
int main()
{

	setlocale(LC_ALL, "Russian");
	// Init GLFW
	glfwInit();
	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create a GLFWwindow object that we can use for GLFW's functions
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	// Set the required callback functions
	glfwSetKeyCallback(window, key_callback);

	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
	glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers
	glewInit();

	// Define the viewport dimensions
	glViewport(0, 0, WIDTH, HEIGHT);


	// Build and compile our shader program
	// Vertex shader
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	// Check for compile time errors
	GLint success;
	GLchar infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// Fragment shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	// Check for compile time errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// Link shaders
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	// Check for linking errors
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);


	// Set up vertex data (and buffer(s)) and attribute pointers
	GLfloat vertices[] = {
		// Positions         // Colors
		 0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // Bottom Right
		-0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // Bottom Left
		 0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f   // Top 
	};
	initCuda(0);
	
	GLuint VBO, VAO;
	createVBO(&VBO, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
	MaldelbrotGPU_Calculation();
	glGenVertexArrays(1, &VAO);
	//glGenBuffers(1, &VBO);
	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	

	

	//glBufferData(GL_ARRAY_BUFFER, sizeof(cpu_memory), cpu_memory, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0); // Unbind VAO

	GLubyte data[256];
	glGetBufferSubData(GL_ARRAY_BUFFER, 1024, 256, data);
	// Game loop
	while (!glfwWindowShouldClose(window))
	{
		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
		glfwPollEvents();

		// Render
		// Clear the colorbuffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Draw the triangle
		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, HEIGHT * WIDTH);
		glBindVertexArray(0);

		// Swap the screen buffers
		glfwSwapBuffers(window);
	}
	// Properly de-allocate all resources once they've outlived their purpose
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	// Terminate GLFW, clearing any resources allocated by GLFW.
	glfwTerminate();
	return 0;
}

