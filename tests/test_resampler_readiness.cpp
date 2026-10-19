#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "pb_resampler.h"
int main(){
 int checks=0;
 for(double profile:{.92,.936})for(int taps:{16,64})for(int chunk:{1,37,256}){
  pbshift::SincResampler res(2,taps);res.setInterpolationCutoff(profile);
  std::vector<float> x(769),y(769);for(int i=0;i<769;++i){x[i]=float(.3*std::sin(.127*i));y[i]=float(.2*std::cos(.043*i));}
  int fed=0,iteration=0;
  auto check=[&](double ratio){
   auto reference=res;
   const int predicted=res.readyOutputCount(ratio);
   std::vector<float> a(16384),b(16384);float* op[]={a.data(),b.data()};
   const int actual=reference.process(op,16384,ratio);
   if(predicted!=actual){std::printf("FAIL taps=%d chunk=%d fed=%d ratio=%.17g predicted=%d actual=%d\n",taps,chunk,fed,ratio,predicted,actual);return false;}
   const int n=res.process(op,16384,ratio);if(n!=actual||res.readyOutputCount(ratio)!=0)return false;
   ++checks;return true;
  };
  const double ratios[]={.25,.5,std::pow(2.,-7./12),1,1.001,1.5,2,4};
  while(fed<int(x.size())){
   const int n=std::min(chunk,int(x.size())-fed);const float* in[]={x.data()+fed,y.data()+fed};res.feed(in,n);fed+=n;
   if(!check(ratios[iteration++%8]))return 1;
  }
  res.finish();for(double ratio:ratios)if(!check(ratio))return 1;
 }
 // Changing the ratio must preserve the next physical source position.
 for(int taps:{16,64}){
  pbshift::SincResampler res(1,taps);
  std::vector<float> x(4096,.1f),y(128);const float* ip[]={x.data()};float* op[]={y.data()};res.feed(ip,4096);
  double position=0;
  for(double ratio:{.25,.5,1.5,4.,.7,2.,1.,.25}){
   if(std::abs(res.nextSourcePos(ratio)-position)>1e-10)return 2;
   if(res.process(op,17,ratio)!=17)return 3;
   position+=17/ratio;
   if(std::abs(res.nextSourcePos(ratio)-position)>1e-10)return 4;
   for(double next:{.25,1.,4.}){
    if(std::abs(res.nextSourcePos(next)-position)>1e-10)return 5;
    int n=res.outputsBefore(next,1024);
    if(n<0 || (n>0 && position+(n-1)/next+taps>=1024))return 6;
    ++checks;
   }
  }
 }
 // The profile is fixed after input is buffered, and can be chosen anew on reset.
 {
  std::vector<float> x(4096),a(8192),b(8192),c(8192);for(int i=0;i<4096;++i)x[i]=float(.3*std::sin(2*3.141592653589793*.46*i));
  const float* ip[]={x.data()};float* ap[]={a.data()};float* bp[]={b.data()};float* cp[]={c.data()};
  pbshift::SincResampler wide(1,64,12),control(1,64,12),legacy(1,64,12);
  wide.setInterpolationCutoff(.936);control.setInterpolationCutoff(.936);wide.feed(ip,4096);control.feed(ip,4096);legacy.feed(ip,4096);
  wide.setInterpolationCutoff(.92);wide.finish();control.finish();legacy.finish();int n=wide.process(ap,8192,2);
  if(n!=control.process(bp,8192,2) || n!=legacy.process(cp,8192,2) || a!=b || a==c)return 7;
  wide.reset();wide.setInterpolationCutoff(.92);wide.feed(ip,4096);wide.finish();std::fill(a.begin(),a.end(),0);
  if(wide.process(ap,8192,2)!=n || a!=c)return 8;
  checks+=3;
 }
 std::printf("PASS resampler readiness, ratio continuity and profile lifecycle: %d checks\n",checks);
}
