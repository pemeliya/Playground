
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <numeric>
#include <random>

#include "test_main.h"
#include "qccl_lib.h"

std::vector< uint32_t > TestFramework::permute_op() 
{
  std::vector< uint32_t > permute(m_nGpus);
  for(uint32_t i = 0; i < m_nGpus; i++) {
    // this is cyclic neighbor exchange OP
    permute[i] = (i + 1) % m_nGpus; // defines to which node GPU[i] should send its data
  }
#if 0
  std::random_device rd;
  std::mt19937 g(441); //g(rd());
  using Distr = std::uniform_int_distribution<uint32_t>;
  using PP = Distr::param_type;
  Distr D;
  for (uint32_t i = m_nGpus - 1; (int)i > 0; i--) {
    while(1) {
      uint32_t idx = D(g, PP(0, i));
      if(permute[idx] != i) {
        std::swap(permute[i], permute[idx]);
        break;
      }
    }
  } // for
#endif
  return permute;
}

void TestFramework::init_extra_peers() {

  auto permute = permute_op();
  for(uint32_t i = 0; i < m_nGpus; i++) {
    //VLOG(0) << i << " permute: " << permute[i];
    m_commGraph[i][0].out = permute[i];  // gpu i sends to gpu permute[i]
    m_commGraph[permute[i]][0].in = i;   // gpu permute[i] receives from gpu i
  }

  // the # of incoming links and outgoing links (the target link is already counted)
  std::vector< uint32_t > numLinks(m_nGpus, 1);
  for(uint32_t i = 0; i < m_nGpus; i++) {

    auto t = m_commGraph[i][0].out; // target node for GPU i
    // iterate until all outgoing links for GPU i are filled
    // VLOG(0) << "Examining " << i << " -> " << t;
#if 1
    for(uint32_t jc = i + 1, n = 1; 
                         jc < 500 && n <= m_nExtraPeers; jc++) {
      uint32_t j = jc % m_nGpus; // sometimes we need 2 rounds
      // skip self, the target node, and nodes with too many extra peers
      if(i == j || t == j || numLinks[j] > m_nExtraPeers) { 
        continue;
      }
      // graph[j][z] can include (i,t) if:
      // 1. there is no (i,t) in graph[j] - in that row
      // 2. i != j and t != j 

      // NOTE: for N extra peers we need N rounds ??/
      // check if this slot is empty
      if(m_commGraph[j][n].in != s_bogus)
        continue;
      
      // VLOG(0) << "Found " << n << "th gateway " << i << "," << j << "," << t; 
      // increase the number of nodes processed
      // use node j as a gateway to send data from i to t
      auto z = n++;
      numLinks[j]++;
      m_commGraph[j][z].in = i;  // node j receives z-th piece from node i
      m_commGraph[j][z].out = t; // node j forwards z-th piece to node t
    }
#else
    // alternative approach: use gateways living on target target nodes
    // so that we have one direct write and one direct read kernel
    if(m_nExtraPeers != 1) {
      throw std::runtime_error("This approach works only for one peer!");
    }
    int z = 1;
    m_commGraph[t][z].in = i;   // node t receives z-th piece from node i
    m_commGraph[t][z].out = t;  // node t saves z-th piece to itself
#endif
  } // for i m_nGpus
  VLOG(0) << "Legend: GPU x send: (i,j): gpu[x] receives from gpu[i] and sends to gpu[j]";
  for(uint32_t i = 0; i < m_nGpus; i++) { 
    VLOG(0) << "GPU " << i << " send: " << m_commGraph.printRow(i);
  }
  for(const auto& a : m_commGraph) {
    if(a.in == s_bogus || a.out == s_bogus) {
      ThrowError<>("Uninitialized node for stageA!");
    }
  }
  // for(const auto& b : m_stageB) {
  //   if(b.in == s_bogus || b.out == s_bogus) {
  //     ThrowError<>("Uninitialized node for stageB!");
  //   }
  // }
  output_dot();
}

//! https://visjs.github.io/vis-network/examples/network/edgeStyles/smoothWorldCup.html
//! neato topology.dot -Tpng > topology.png 
void TestFramework::output_dot() {
  static const char *colors[] = {
        "aquamarine", "cornflowerblue", "chocolate1", "darkgreen",
        "darkorchid", "deeppink", "fuchsia", "goldenrod2"};
  uint32_t ncolors = sizeof(colors)/sizeof(colors[0]);

  auto& m = m_commGraph;
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
  // for(uint32_t i = 0; i < m.numRows(); i++) {
  //   ofs << i << " -> " << m[i][0].out;
  //   ofs << " [color=" << colors[i % ncolors];
  //   ofs << ",penwidth=3.0];\n";
  // }
  for(uint32_t i = 0; i < m.numRows(); i++) {
    
    ofs << i << " -> " << m[i][0].out << " [color=" << 
          colors[i % ncolors] << ",penwidth=3.0];\n";

    for(uint32_t j = 1; j < m.numCols(); j++) {
      // this is a gateway link: color is determined by the source node
      auto C = colors[m[i][j].in % ncolors];

      ofs << i << " -> " << m[i][j].out << " [color="
          << C << ",style=\"dashed\",penwidth=1.5];\n";

      ofs << m[i][j].in << " -> " << i << " [color=" 
          << C << ",style=\"dashed\",penwidth=1.5];\n";
    }
  }
#if 0
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
#endif    
  ofs << '}';
}
