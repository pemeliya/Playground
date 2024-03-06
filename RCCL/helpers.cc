
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <numeric>
#include <random>

#include "test_main.h"
#include "qccl_lib.h"

  //! https://visjs.github.io/vis-network/examples/network/edgeStyles/smoothWorldCup.html
  //! neato topology.dot -Tpng > topology.png 
void output_dot(const Matrix<Node>& stageA, 
                     const Matrix<Node>& stageB) {
    static const char *colors[] = {
        "aquamarine", "cornflowerblue", "chocolate1", "darkgreen",
        "darkorchid", "deeppink", "fuchsia", "goldenrod2"};
    uint32_t ncolors = sizeof(colors)/sizeof(colors[0]);

    auto& m = stageA;
    std::ofstream ofs("topology.dot");
    ofs << "digraph G {\nsplines=true;\noverlap=scale\n";

    float R = 5, x0 = 10, y0 = 10, da = -M_PI*2.0f/m.numRows();
    for(uint32_t i = 0; i < m.numRows(); i++) {
      float xf = x0 + R*std::cos(da*i + M_PI/2),
            yf = y0 + R*std::sin(da*i + M_PI/2);
      auto C = colors[i % ncolors];
      ofs << i << " [pos=\"" << xf << ',' << yf << 
            "!\", style=filled,color=\"" << C << "\",shape=circle];\n";
    }
    for(uint32_t i = 0; i < m.numRows(); i++) {
      for(uint32_t j = 0; j < m.numCols(); j++) {
        ofs << i << " -> " << m[i][j].out;
        // check all GPUs and find the one having GPU i in (j+1)th position
        ofs << " [color=" << colors[i % ncolors];
        if(j == 0) {
          ofs << ",penwidth=3.0];\n";
        } else
          ofs << ",penwidth=1.5];\n";
      }
    }

    auto& m2 = stageB;
    for(uint32_t i = 0; i < m2.numRows(); i++) {
      for(uint32_t j = 0; j < m2.numCols(); j++) {
        // how the data comes to node i at position j ??
        uint32_t t = s_bogus;
        for(uint32_t k = 0; k < m.numRows(); k++) {
          if(k != i && m[k][j+1].out == i) {
            t = k; break;
          }
        }
        auto C = t == s_bogus ? "black" : colors[t % ncolors];
        ofs << i << " -> " << m2[i][j].out;
        ofs << " [color=" << C <<
        ",style=\"dashed\",penwidth=1.5];\n";
      }
    }
    ofs << '}';
  }

std::vector< uint32_t > permute_op(uint32_t nGpus) 
{
    std::vector< uint32_t > permute(nGpus);
    for(uint32_t i = 0; i < nGpus; i++) {
      // this is cyclic neighbor exchange OP
      permute[i] = (i + 1) % nGpus; // defines to which node GPU[i] should send its data
    }
#if 0
    std::random_device rd;
    std::mt19937 g(rd());
    using Distr = std::uniform_int_distribution<uint32_t>;
    using PP = Distr::param_type;
    Distr D;
    for (uint32_t i = nGpus - 1; (int)i > 0; i--)
    while(1) {
      uint32_t idx = D(g, PP(0, i));
      if(permute[idx] != i) {
        std::swap(permute[i], permute[idx]);
        break;
      }
    }
#endif
    return permute;
}
