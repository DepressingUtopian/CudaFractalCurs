#version 410 

out vec4 colorOut; 
uniform double screen_ratio; 
uniform dvec2 screen_size; 
uniform dvec2 center; 
uniform double zoom; 
uniform int itr; 

vec4 map_to_color(float t) { 
float r = 9.0 * (1.0 - t) * t * t * t; 
float g = 15.0 * (1.0 - t) * (1.0 - t) * t * t; 
float b = 8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t; 

return vec4(r, g, b, 1.0); 
} 

void main() 
{ 
dvec2 z, c; 
double temp = 0; 
int iter = 0; 
c.x = screen_ratio * (gl_FragCoord.x / screen_size.x - 0.5); 
c.y = (gl_FragCoord.y / screen_size.y - 0.5); 

c.x /= zoom; 
c.y /= zoom; 

c.x += center.x; 
c.y += center.y; 



while(iter < itr && z.x * z.x + z.y * z.y <= 4.0) 
{ 
temp = z.x * z.x - z.y * z.y + c.x; 
z.y = 2.0 * z.x * z.y + c.y; 
z.x = temp; 
iter=iter+1; 
} 
double t = double(iter) / double(itr); 

colorOut = map_to_color(float(t)); 
}