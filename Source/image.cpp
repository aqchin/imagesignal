#include "image.h"
#include "bmp.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define M_PI 3.14159

/**
 * Image
 **/
Image::Image (int width_, int height_)
{
    assert(width_ > 0);
    assert(height_ > 0);

    width           = width_;
    height          = height_;
    num_pixels      = width * height;
    pixels          = new Pixel[num_pixels];
    sampling_method = IMAGE_SAMPLING_POINT;

    assert(pixels != NULL);
}


Image::Image (const Image& src)
{
    width           = src.width;
    height          = src.height;
    num_pixels      = width * height;
    pixels          = new Pixel[num_pixels];
    sampling_method = IMAGE_SAMPLING_POINT;

    assert(pixels != NULL);
    memcpy(pixels, src.pixels, src.width * src.height * sizeof(Pixel));
}


Image::~Image ()
{
    delete [] pixels;
    pixels = NULL;
}

/*
void Image::AddNoise (double factor)
{

}
*/

void Image::Brighten (double factor)
{
  Pixel black = Pixel();

  for(int i = 0; i < num_pixels; i++)
    pixels[i] = PixelLerp(black, pixels[i], factor);
}


void Image::ChangeContrast (double factor)
{
  double c = 0.0f;

  for(int i = 0; i < num_pixels; i++)
    c += (double) pixels[i].Luminance();
  
  c /= (double) num_pixels;
  Pixel grey = Pixel((Component) c, (Component) c, (Component) c);

  for(int i = 0; i < num_pixels; i++)
    pixels[i] = PixelLerp(grey, pixels[i], factor);
}


void Image::ChangeSaturation(double factor)
{
  for(int i = 0; i < num_pixels; i++) {
    Component c = pixels[i].Luminance();
    pixels[i] = PixelLerp(Pixel(c, c, c), pixels[i], factor);
  }
}

void Image::ChangeGamma(double factor)
{
  double gamma = (double) 1 / factor;

  for(int i = 0; i < num_pixels; i++) {
    double r = pixels[i].r / 255.0f;
    double g = pixels[i].g / 255.0f;
    double b = pixels[i].b / 255.0f;
  
    pixels[i].SetClamp(
      pow(r, gamma) * 255.0f,
      pow(g, gamma) * 255.0f,
      pow(b, gamma) * 255.0f);
  }
}

Image* Image::Crop(int x, int y, int w, int h)
{
  /* Your Work Here (section 3.2.5) */
  //return NULL ;
  Pixel px;
  Image* img = new Image(w, h);

  for(int y_ = 0; y_ < h; y_++) {
    for(int x_ = 0; x_ < w; x_++) {
      px = GetPixel(x + x_, y + y_);
      img->GetPixel(x_, y_).Set(px.r, px.g, px.b, px.a);
    }
  }
  return img;
}

/*
void Image::ExtractChannel(int channel)
{
  // For extracting a channel (R,G,B) of image.  
  // Not required for the assignment
}
*/

void Image::Quantize (int nbits)
{
  int b = (int) pow(2.0f, (float) nbits);
  float x = (float) 255.0f / (b - 1);
  float y = (float) b / 256.0f;

  for(int i = 0; i < num_pixels; i++)
    pixels[i].SetClamp(
      x * ((int)(pixels[i].r * y)),
      x * ((int)(pixels[i].g * y)),
      x * ((int)(pixels[i].b * y)));
}


void Image::RandomDither (int nbits)
{
  int RNG;
  float RNGesus;
  int b = (int) pow(2.0f, (float) nbits);
  float x = (float) 255.0f / (b - 1);
  float y = (float) b / 256.0f;

  for(int i = 0; i < num_pixels; i++) {
    RNG = (rand() % 15000) - 10000;
    RNGesus = (float) RNG / 10000.0f;
    
    pixels[i].SetClamp(
      x * ((int)(pixels[i].r * y + RNGesus)),
      x * ((int)(pixels[i].g * y + RNGesus)),
      x * ((int)(pixels[i].b * y + RNGesus)));
  }
}


/* Matrix for Bayer's 4x4 pattern dither. */
/* uncomment its definition if you need it */

/*
static int Bayer4[4][4] =
{
    {15, 7, 13, 5},
    {3, 11, 1, 9},
    {12, 4, 14, 6},
    {0, 8, 2, 10}
};


void Image::OrderedDither(int nbits)
{
  // For ordered dithering
  // Not required for the assignment
}

*/

/* Error-diffusion parameters for Floyd-Steinberg*/
const double
    ALPHA = 7.0 / 16.0,
    BETA  = 3.0 / 16.0,
    GAMMA = 5.0 / 16.0,
    DELTA = 1.0 / 16.0;

void Image::FloydSteinbergDither(int nbits)
{
  Pixel old;
  int err_r, err_g, err_b;
  double * err_map_r = new double[num_pixels];
  double * err_map_g = new double[num_pixels];
  double * err_map_b = new double[num_pixels];
  int b = (int) pow(2.0f, (float) nbits);
  float x_ = (float) 255.0f / (b - 1);
  float y_ = (float) b / 256.0f;
  
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {

      GetPixel(x, y).SetClamp(
        GetPixel(x, y).r + err_map_r[y*width + x],
        GetPixel(x, y).g + err_map_g[y*width + x],
        GetPixel(x, y).b + err_map_b[y*width + x]);

      old = GetPixel(x, y);

      GetPixel(x, y).SetClamp(
        x_ * ((int)(GetPixel(x, y).r * y_)),
        x_ * ((int)(GetPixel(x, y).g * y_)),
        x_ * ((int)(GetPixel(x, y).b * y_)));

      err_r = (int) old.r - GetPixel(x, y).r;
      err_g = (int) old.g - GetPixel(x, y).g;
      err_b = (int) old.b - GetPixel(x, y).b;

      if(x != width -1) {
        // ALPHA
        err_map_r[y*width + x + 1] += ALPHA * err_r;
        err_map_g[y*width + x + 1] += ALPHA * err_g;
        err_map_b[y*width + x + 1] += ALPHA * err_b;

        if (y != height - 1) {
          // DELTA
          err_map_r[(y + 1)*width + x + 1] += DELTA * err_r;
          err_map_g[(y + 1)*width + x + 1] += DELTA * err_g;
          err_map_b[(y + 1)*width + x + 1] += DELTA * err_b;
        }
      }

      if (y != height - 1) {
        // GAMMA
        err_map_r[(y + 1)*width + x] += GAMMA * err_r;
        err_map_g[(y + 1)*width + x] += GAMMA * err_g;
        err_map_b[(y + 1)*width + x] += GAMMA * err_b;

        if(x != 0) {
          // BETA
          err_map_r[(y + 1)*width + x - 1] += BETA * err_r;
          err_map_g[(y + 1)*width + x - 1] += BETA * err_g;
          err_map_b[(y + 1)*width + x - 1] += BETA * err_b;
        }
      }
    }
  }
}

void ImageComposite(Image *bottom, Image *top, Image *result)
{
  // Extra Credit (Section 3.7).
  // This hook just takes the top image and bottom image, producing a result
  // You might want to define a series of compositing modes as OpenGL does
  // You will have to use the alpha channel here to create Mattes
  // One idea is to composite your face into a famous picture
}

int Lim(int x, int lim) {
  if (x < 0)
    return 0;
  else if (x >= lim)
    return lim - 1;
  else
    return x;
}

int LimRef(int x, int lim) {
  if (x < 0)
    return -x - 1;
  else if (x >= lim)
    return 2 * lim - x - 1;
  else
    return x;
}

void Image::Convolve(int *filter, int n, int normalization, int absval) {
  // This is my definition of an auxiliary function for image convolution 
  // with an integer filter of width n and certain normalization.
  // The absval param is to consider absolute values for edge detection.
  
  // It is helpful if you write an auxiliary convolve function.
  // But this form is just for guidance and is completely optional.
  // Your solution NEED NOT fill in this function at all
  // Or it can use an alternate form or definition
  int sumr, sumg, sumb;
  sumr = sumg = sumb = 0;

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int p = 0; p < n; p++) {
        for (int q = 0; q < n; q++) {
          int x = 0;
          int y = 0;
          int mp = 0;
          if (absval) {
            mp = 2;
            x = Lim(w - (p - mp), width);
            y = Lim(h - (q - mp), height);
            //x = w - (p - mp);
            //y = h - (q - mp); 
          }
          else {
            mp = n / 2;
            x = LimRef(w - (p - mp), width);
            y = LimRef(h - (q - mp), height);
          }
          if (!ValidCoord(x, y)) continue;
          Pixel curr = GetPixel(x, y);
          int filt = filter[q * n + p];
          sumr += (int)curr.r * filt;
          sumg += (int)curr.g * filt;
          sumb += (int)curr.b * filt;
        }
      }
      if (absval) {
        sumr = abs(sumr);
        sumg = abs(sumg);
        sumb = abs(sumb);
      }
      GetPixel(w, h).SetClamp(sumr / normalization, sumg / normalization, sumb / normalization);
      sumr = sumg = sumb = 0;
    }
  }
}

void Image::Blur(int n)
{
  /* Your Work Here (Section 3.4.1) */
  double sig = floor(n / (double) 2) / 2;
  int mp = int(n / 2);
  double* filtU = new double[n]; //f(u)
  for (int i = -mp; i <= mp; i++) {
    filtU[i + mp] = (1 / (sqrt(2 * M_PI) * sig)) * exp(-(i*i) / (2 * sig*sig));
  }

  double norm = 0.0;
  int* filt = new int[n*n]; //blur filter
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int temp = (int) ((filtU[i] * filtU[j]) / (filtU[0] * filtU[0]));
      filt[i + j*n] = temp;
      norm += temp;
    }
  }
  Convolve(filt, n, (int) norm, false);
}

void Image::Sharpen() 
{
  int* filt = new int[9]; //sharpening filter
  filt[0] = -1;
  filt[1] = -2;
  filt[2] = -1;
  filt[3] = -2;
  filt[4] = 19;
  filt[5] = -2;
  filt[6] = -1;
  filt[7] = -2;
  filt[8] = -1;
  int norm = 7;
  int n = 3;
  Convolve(filt, n, norm, false);
}

void Image::EdgeDetect(int threshold)
{
  int* filtH = new int[9];
  int* filtV = new int[9];

  filtH[0] = -1;
  filtH[1] = 0;
  filtH[2] = 1;
  filtH[3] = -2;
  filtH[4] = 0;
  filtH[5] = 2;
  filtH[6] = -1;
  filtH[7] = 0;
  filtH[8] = 1;

  filtV[0] = 1;
  filtV[1] = 2;
  filtV[2] = 1;
  filtV[3] = 0;
  filtV[4] = 0;
  filtV[5] = 0;
  filtV[6] = -1;
  filtV[7] = -2;
  filtV[8] = -1;

  Image gx = Image(*this);
  Image gy = Image(*this);
  int n = 3;
  int norm = 1;
  gx.Convolve(filtH, n, norm, true);
  gy.Convolve(filtV, n, norm, true);
  for (int i = 0; i < num_pixels; i++) {
    int gxmag = gx.pixels[i].Luminance();
    int gymag = gy.pixels[i].Luminance();
    double gmag = sqrt((double) gxmag*gxmag + gymag*gymag);
    if (gmag > threshold)
      gmag = 255;
    else
      gmag = 0;
    pixels[i].SetClamp(gmag, gmag, gmag);
  }
}


Image* Image::Scale(int sizex, int sizey)
{
  Image *img = new Image(sizex, sizey);

  for(int y = 0; y < sizey; y++)
    for(int x = 0; x < sizex; x++)
        img->GetPixel(x, y) = this->Sample(x, y, sizex / (double) width, sizey / (double) height);

  return img;
}

float hat(float x) {
  float x_ = abs(x);
  if (x_ <= 1) return 1.0f - x_;
  else return 0.0f;
}

float mitchell(float xx) {
  float x = fabs(xx);
  float fx = 0;
  if (x < 1 && x >= 0)
    fx = ((7 * x*x*x) - (12 * x*x) + (16 / 3)) / 6;
  else if (x < 2 && x >= 1)
    fx = (((-7 / 3)*x*x*x) + (12 * x*x) - (20 * x) + (32 / 3)) / 6;
  return fx;
}

void Image::Shift(double sx, double sy)
{
  for(int y = 0; y < height; y++) {
    int y_ = (0 < sy) ? height - 1 - y : y;

    for(int x = 0; x < width; x++) {
      int x_ = (0 < sx) ? width - 1 - x : x;

      if(sx == floor(sx) && sy == floor(sy))
        GetPixel(x_, y_) = (ValidCoord(x_ - (int) sx, y_ - (int) sy)) 
          ? GetPixel(x_ - (int) sx, y_ - (int) sy) : Pixel(0, 0, 0);

      else {
        if(sampling_method == IMAGE_SAMPLING_HAT) {
          if(ValidCoord(x_ - (int) sx, y_ - (int) sy)) {
            float h_r, h_g, h_b, h_all, h_tmp;
            h_r = h_g = h_b = h_all = 0.0f;
            for(int c = -1; c < 2; c++) {
              for(int r = -1; r < 2; r++) {
                if(ValidCoord(x_ - (int) sx + r, y_ - (int) sy + c)) {
                  h_tmp = hat(-((int) sx) + r + (float) sx)
                    * hat(-((int) sy) + c + (float) sy);
                  h_all += h_tmp;
                  h_r += (float) GetPixel(x_ - (int) sx + r, y_ - (int) sy + c).r * h_tmp;
                  h_g += (float) GetPixel(x_ - (int) sx + r, y_ - (int) sy + c).g * h_tmp;
                  h_b += (float) GetPixel(x_ - (int) sx + r, y_ - (int) sy + c).b * h_tmp;
                }
              }
            }
            GetPixel(x_, y_).SetClamp(h_r / h_all, h_g / h_all, h_b / h_all);
          }
          else GetPixel(x_, y_) = Pixel(0, 0, 0);
        }
        else if (sampling_method == IMAGE_SAMPLING_MITCHELL) {
          if (ValidCoord(x_ - (int) sx, y_ - (int) sy)) {
            float h_r, h_g, h_b, h_all, h_tmp;
            h_r = h_g = h_b = h_all = 0.0f;
            for(int c = -4; c < 5; c++) {
              for(int r = -4; r < 5; r++) {
                if(ValidCoord(x_ - (int) sx + r, y_ - (int) sy + c)) {
                  h_tmp = mitchell(-((int) sx) + r + (float) sx)
                    * mitchell(-((int) sy) + c + (float) sy);
                  h_all += h_tmp;
                  h_r += (float) GetPixel(x_ - (int) sx + r, y_ - (int) sy + c).r * h_tmp;
                  h_g += (float) GetPixel(x_ - (int) sx + r, y_ - (int) sy + c).g * h_tmp;
                  h_b += (float) GetPixel(x_ - (int) sx + r, y_ - (int) sy + c).b * h_tmp;
                }
              }
            }
            GetPixel(x_, y_).SetClamp(h_r / h_all, h_g / h_all, h_b / h_all);
          }
          else GetPixel(x_, y_) = Pixel(0, 0, 0);
        }
      }
    }
  }
}


/*
Image* Image::Rotate(double angle)
{
  // For rotation of the image
  // Not required in the assignment
  // But you can earn limited extra credit if you fill it in
  // (It isn't really that hard) 

    return NULL;
}
*/


void Image::Fun()
{
  float intensity = 1.15f;
  char c = 'x';
  char * color = new char[num_pixels];
  int maj = 1;
  for(int i = 0; i < num_pixels; i++) {
    if(pixels[i].g > intensity * ((int)pixels[i].r + (int)pixels[i].b)) {
      color[i] = 'g';
      if(c == 'g') maj++;
      else {
        maj--;
        if(maj < 1) {
          c = 'g';
          maj = 1;
        }
      }
    }
    else if(pixels[i].b > intensity * ((int)pixels[i].r + (int)pixels[i].g)) {
      color[i] = 'b';
      if(c == 'b') maj++;
      else {
        maj--;
        if(maj < 1) {
          c = 'b';
          maj = 1;
        }
      }
    }
    else if(pixels[i].r > intensity * ((int)pixels[i].g + (int)pixels[i].b)) {
      color[i] = 'r';
      if(c == 'r') maj++;
      else {
        maj--;
        if(maj < 1) {
          c = 'r';
          maj = 1;
        }
      }
    }
    else color[i] = 'x';
  }

  for(int i = 0; i < num_pixels; i++) {
    if(color[i] != c) {
      Component lum = pixels[i].Luminance();
      pixels[i] = Pixel(lum, lum, lum);
    }
  }
}


Image* ImageMorph (Image* I0, Image* I1, int numLines, Line* L0, Line* L1, double t)
{
  /* Your Work Here (Section 3.7) */
  // This is extra credit.
  // You can modify the function definition. 
  // This definition takes two images I0 and I1, the number of lines for 
  // morphing, and a definition of corresponding line segments L0 and L1
  // t is a parameter ranging from 0 to 1.
  // For full credit, you must write a user interface to join corresponding 
  // lines.
  // As well as prepare movies 
  // An interactive slider to look at various morph positions would be good.
  // From Beier-Neely's SIGGRAPH 92 paper

    return NULL;
}


/**
 * Image Sample
 **/
void Image::SetSamplingMethod(int method)
{
  // Sets the filter to use for Scale and Shift
  // You need to implement point sampling, hat filter and mitchell

    assert((method >= 0) && (method < IMAGE_N_SAMPLING_METHODS));
    sampling_method = method;
}

Pixel Image::Sample (double u, double v, double sx, double sy)
{
  // To sample the image in scale and shift
  // This is an auxiliary function that it is not essential you fill in or 
  // you may define it differently.
  // u and v are the floating point coords of the points to be sampled.
  // sx and sy correspond to the scale values. 
  // In the assignment, it says implement MinifyX MinifyY MagnifyX MagnifyY
  // separately.  That may be a better way to do it.
  // This hook is primarily to get you thinking about that you have to have 
  // some equivalent of this function.

  if (sampling_method == IMAGE_SAMPLING_POINT) {
    // Your work here
    return GetPixel((int) round(u / sx), (int) round(v / sy));
  }

  else if (sampling_method == IMAGE_SAMPLING_HAT) {
    int sumr, sumg, sumb;
    sumr = sumg = sumb = 0;
    float xdif = 0;
    float ydif = 0;
    if (sx < 1)
      xdif = 1 / sx;
    else
      xdif = 1;
    if (sy < 1)
      ydif = 1 / sy;
    else
      ydif = 1;
    float n = 0;
    for (int x = round(u / sx - xdif); x <= round(u / sx + xdif); x++) {
      float hu = hat((x - u / sx) / xdif);
      for (int y = round(v / sy - ydif); y <= round(v / sy + ydif); y++) {
        float hv = hat((y - v / sy) / ydif);
        float h = hu * hv;
        n += h;
        if (!ValidCoord(x, y))
          continue;
        Pixel curr = GetPixel(x, y);
        sumr += h * curr.r;
        sumg += h * curr.g;
        sumb += h * curr.b;
      }
    }
    sumr /= n;
    sumg /= n;
    sumb /= n;
    if (sumr > 255)
      sumr = 255;
    else if (sumr < 0)
      sumr = 0;
    if (sumg > 255)
      sumg = 255;
    else if (sumg < 0)
      sumg = 0;
    if (sumb > 255)
      sumb = 255;
    else if (sumb < 0)
      sumb = 0;
    return Pixel(sumr, sumg, sumb);
  }

  else if (sampling_method == IMAGE_SAMPLING_MITCHELL) {
    int sumr, sumg, sumb;
    sumr = sumg = sumb = 0;
    float xdif = 0;
    float ydif = 0;
    if (sx < 1)
      xdif = 1 / sx;
    else
      xdif = 1;
    if (sy < 1)
      ydif = 1 / sy;
    else
      ydif = 1;
    float n = 0;
    for (int x = round(u / sx - xdif); x <= round(u / sx + xdif); x++) {
      float hu = mitchell((x - u / sx) / xdif);
      for (int y = round(v / sy - ydif); y <= round(v / sy + ydif); y++) {
        float hv = mitchell((y - v / sy) / ydif);
        float h = hu * hv;
        n += h;
        if (!ValidCoord(x, y))
          continue;
        Pixel curr = GetPixel(x, y);
        sumr += h * curr.r;
        sumg += h * curr.g;
        sumb += h * curr.b;
      }
    }
    sumr /= n;
    sumg /= n;
    sumb /= n;
    if (sumr > 255)
      sumr = 255;
    else if (sumr < 0)
      sumr = 0;
    if (sumg > 255)
      sumg = 255;
    else if (sumg < 0)
      sumg = 0;
    if (sumb > 255)
      sumb = 255;
    else if (sumb < 0)
      sumb = 0;
    return Pixel(sumr, sumg, sumb);
  }

  else {
    fprintf(stderr,"I don't understand what sampling method is used\n") ;
    exit(1) ;
  }

  return Pixel() ;
}