#include "miniinfer/kernels.h"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace miniinfer {
static void same(const Tensor&a,const Tensor&b){if(a.shape()!=b.shape())throw std::invalid_argument("shape mismatch");}
Tensor matmul(const Tensor&a,const Tensor&b){if(a.shape().size()!=2||b.shape().size()!=2||a.dim(1)!=b.dim(0))throw std::invalid_argument("matmul shape"); Tensor c({a.dim(0),b.dim(1)}); for(size_t i=0;i<c.dim(0);++i)for(size_t j=0;j<c.dim(1);++j)for(size_t k=0;k<a.dim(1);++k)c[i*c.dim(1)+j]+=a[i*a.dim(1)+k]*b[k*b.dim(1)+j]; return c;}
Tensor matvec(const Tensor&a,const Tensor&x){if(a.shape().size()!=2||x.shape().size()!=1||a.dim(1)!=x.dim(0))throw std::invalid_argument("matvec shape"); Tensor y({a.dim(0)}); for(size_t i=0;i<a.dim(0);++i)for(size_t k=0;k<a.dim(1);++k)y[i]+=a[i*a.dim(1)+k]*x[k]; return y;}
Tensor add(const Tensor&a,const Tensor&b){same(a,b);Tensor c(a.shape());for(size_t i=0;i<c.size();++i)c[i]=a[i]+b[i];return c;}
Tensor mul(const Tensor&a,const Tensor&b){same(a,b);Tensor c(a.shape());for(size_t i=0;i<c.size();++i)c[i]=a[i]*b[i];return c;}
Tensor rmsnorm(const Tensor&x,const Tensor&w,float eps){same(x,w);float ss=0;for(float v:x.values())ss+=v*v;float inv=1/std::sqrt(ss/x.size()+eps);Tensor y(x.shape());for(size_t i=0;i<x.size();++i)y[i]=x[i]*inv*w[i];return y;}
Tensor softmax(const Tensor&x){Tensor y(x.shape());float m=*std::max_element(x.values().begin(),x.values().end());float s=0;for(size_t i=0;i<x.size();++i){y[i]=std::exp(x[i]-m);s+=y[i];}for(float&v:y.values())v/=s;return y;}
Tensor silu(const Tensor&x){Tensor y(x.shape());for(size_t i=0;i<x.size();++i)y[i]=x[i]/(1+std::exp(-x[i]));return y;}
void rope(Tensor&x,size_t pos,float theta){if(x.shape().size()!=1||x.size()%2)throw std::invalid_argument("rope dimension");const size_t half=x.size()/2;for(size_t i=0;i<half;++i){float f=std::pow(theta,-static_cast<float>(i)/half);float a=pos*f,c=std::cos(a),s=std::sin(a),u=x[i],v=x[i+half];x[i]=u*c-v*s;x[i+half]=v*c+u*s;}}
}
