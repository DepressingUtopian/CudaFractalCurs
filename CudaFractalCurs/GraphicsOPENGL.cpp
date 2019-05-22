#include <iostream>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>
// GLFW
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
// ���������� ����
/*
const GLuint WIDTH = 800, HEIGHT = 600;

GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	// ����� ������������ �������� ESC, �� ������������� �������� WindowShouldClose � true, 
	// � ���������� ����� ����� ���������
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}

int main()
{
	
	//������������� GLFW
	glfwInit();

	
	//��������� GLFW
	//�������� ����������� ��������� ������ OpenGL. 
	//�������� 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	//��������
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	//��������� �������� ��� �������� ��������� ��������
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//���������� ����������� ��������� ������� ����
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "MaldelbrotFration", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	//������������� GLEW
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
		return -1;
	}

	//��������� ������ ���������
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	glViewport(0, 0, width, height);

	//����������� ������� ��������� ������
	glfwSetKeyCallback(window, key_callback);
	//���� ���������� ���� �� ����� �������
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glfwSwapBuffers(window);
	}

	//������� ������
	glfwTerminate();
	return 0;
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags) {
	unsigned int size = WIDTH * HEIGHT * sizeof(GLchar);

	glGenBuffers(1, vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW);

	cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res) {
	cudaGraphicsUnregisterResource(cuda_vbo_resource);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}
*/