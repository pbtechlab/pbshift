#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include "pb_multires.h"
#include "pbshift/pbshift.h"
using Audio=std::vector<std::vector<float>>;
static Audio input(){
 Audio x(2,std::vector<float>(16000,0));
 for(int start:{2000,8000})for(int i=0;i<64;++i){
  float v=i==0?1.0f:float(.35*std::cos(.31*i)*std::exp(-i/12.0));
  x[0][start+i]=v;x[1][start+i+3]=-.71f*v;
 }
 return x;
}
static Audio expected(const Audio& x,double ratio){
 Audio y(2,std::vector<float>(size_t(std::llround(x[0].size()*ratio)),0));
 for(int start:{2000,8000}){long long at=std::llround(start*ratio);for(int c=0;c<2;++c)std::copy(x[c].begin()+start,x[c].begin()+start+67,y[c].begin()+at);}
 return y;
}
static bool render(pbshift::Stretcher& st,const Audio& x,double ratio,int feedChunk,int readChunk,Audio& y){
 st.setTimeStretch(ratio);std::vector<const float*> in(2);for(size_t i=0;i<x[0].size();i+=feedChunk){int n=std::min<size_t>(feedChunk,x[0].size()-i);for(int c=0;c<2;++c)in[c]=x[c].data()+i;st.feed(in.data(),n);if(st.available()!=0)return false;}
 st.finish();y.assign(2,{});Audio block(2,std::vector<float>(readChunk));float* out[]={block[0].data(),block[1].data()};
 while(st.available()>0){int n=st.read(out,readChunk);if(n<=0)return false;for(int c=0;c<2;++c)y[c].insert(y[c].end(),block[c].begin(),block[c].begin()+n);}
 return true;
}
int main(){int checks=0,fail=0;auto check=[&](bool ok,const char* what){++checks;if(!ok){++fail;std::printf("FAIL %s\n",what);}};Audio x=input(),y;
 for(double ratio:{.25,.5,.75,.9,1.0,1.1,1.25,2.0,4.0}){
  auto target=expected(x,ratio);check(pbshift::MultiResStretch::isolatedTransientStretch(x,ratio,8000,y)&&y==target,"sample-exact stereo event transport");
  pbshift::Config cfg;cfg.sampleRate=8000;cfg.channels=2;cfg.mode=pbshift::Config::Mode::Music;cfg.tier=pbshift::Config::Tier::Offline;
  pbshift::Stretcher st;st.configure(cfg);Audio a,b;check(render(st,x,ratio,1,37,a)&&a==target,"Offline Music dispatch with one-sample feeds");st.reset();check(render(st,x,ratio,8192,8192,b)&&b==target,"reset, large feed/read and exact timing");
 }
 auto reject=[&](Audio a,double ratio,const char* why){Audio sentinel{{123.f}};bool admitted=pbshift::MultiResStretch::isolatedTransientStretch(a,ratio,8000,sentinel);check(!admitted&&sentinel==Audio{{123.f}},why);};
 auto background=x;for(auto& c:background)for(auto& v:c)v+=.000001f;reject(background,2,"continuous floor retained by fallback");
 Audio longEvent(1,std::vector<float>(16000));for(int i=2000;i<2800;++i)longEvent[0][i]=.1f;reject(longEvent,2,"long event rejected");
 Audio slow(1,std::vector<float>(16000));for(int i=0;i<240;++i)slow[0][2000+i]=float((i<180?i+1:240-i)/180.0);reject(slow,2,"slow attack rejected");
 Audio sustained(1,std::vector<float>(16000));for(int i=2000;i<2320;++i)sustained[0][i]=.1f;reject(sustained,2,"sustained short note rejected");
 Audio collision(1,std::vector<float>(16000));for(int start:{2000,2170})for(int i=0;i<64;++i)collision[0][start+i]=i==0?1.f:.01f;reject(collision,.25,"overlapping target events fallback");
 Audio edge(1,std::vector<float>(16000));for(int i=0;i<64;++i)edge[0][i]=i==60?1.f:.01f;reject(edge,.25,"leading packet not truncated");
 reject(Audio(2,std::vector<float>(16000)),2,"silence has no event profile");reject(x,0,"invalid ratio rejected");
 auto nonfinite=x;nonfinite[1][2000]=std::numeric_limits<float>::quiet_NaN();reject(nonfinite,2,"nonfinite channel rejected");
 std::printf("isolated transients: %d checks, %d failures\n",checks,fail);return fail?1:0;
}
