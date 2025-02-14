/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto
    Copyright (C) 2018-2019 SAI Team

    SAI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SAI is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef UCTSEARCH_H_INCLUDED
#define UCTSEARCH_H_INCLUDED

#include <list>
#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <future>

#include "ThreadPool.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GameState.h"
#include "UCTNode.h"
#include "Utils.h"
#include "Network.h"


class SearchResult {
public:
    SearchResult() = default;
    bool is_sai_head() const { return m_value_head_sai; }
    bool valid() const { return m_valid;  }
    float eval() const { return m_value;  }
    float get_alpkt() const { return m_alpkt; }
    float get_beta() const { return m_beta; }
    float get_beta2() const { return m_beta; }
    float eval_with_bonus(float bonus, float base) const;
    bool is_forced() const { return m_forced; }
    void set_forced() { m_forced = true; }
    static SearchResult from_eval(float value, float alpkt, float beta, float beta2, bool sai_head = true) {
        return SearchResult(value, alpkt, beta, beta2, sai_head);
    }
    static SearchResult from_node(const UCTNode* node, bool sai_head = true);
    static SearchResult from_score(float board_score, bool sai_head = true) {
        return SearchResult(Utils::winner(board_score), board_score, 10.0f, 10.0f, sai_head);
    }
private:
    explicit SearchResult(float value, float alpkt, float beta, float beta2, bool sai_head = true)
        : m_valid(true), m_value(value), m_alpkt(alpkt),
          m_beta(beta), m_beta2(beta2), m_value_head_sai(sai_head) {}
    bool m_valid{false};
    float m_value{0.5f};
    float m_alpkt{0.0f};
    float m_beta{1.0f};
    float m_beta2{-1.0f};
    bool m_value_head_sai{true};
    bool m_forced{false};
};

namespace TimeManagement {
    enum enabled_t {
        AUTO = -1, OFF = 0, ON = 1, FAST = 2, NO_PRUNING = 3
    };
};

class UCTSearch {
public:
    /*
        Depending on rule set and state of the game, we might
        prefer to pass, or we might prefer not to pass unless
        it's the last resort. Same for resigning.
    */
    using passflag_t = int;
    static constexpr passflag_t NORMAL   = 0;
    static constexpr passflag_t NOPASS   = 1 << 0;
    static constexpr passflag_t NORESIGN = 1 << 1;

    /*
        Default memory limit in bytes.
        ~1.6GiB on 32-bits and about 5.2GiB on 64-bits.
    */
    static constexpr size_t DEFAULT_MAX_MEMORY =
        (sizeof(void*) == 4 ? 1'600'000'000 : 5'200'000'000);

    /*
        Minimum allowed size for maximum tree size.
    */
    static constexpr size_t MIN_TREE_SPACE = 100'000'000;

    /*
        Value representing unlimited visits or playouts. Due to
        concurrent updates while multithreading, we need some
        headroom within the native type.
    */
    static constexpr auto UNLIMITED_PLAYOUTS =
        std::numeric_limits<int>::max() / 2;

    static constexpr auto FAST_ROLL_OUT_VISITS = 20;
    static constexpr auto EXPLORE_MOVE_VISITS = 30;

    UCTSearch(GameState& g, Network & network);
    int think(int color, passflag_t passflag = NORMAL);
#ifdef USE_EVALCMD
    void set_firstmove(int move);
    int get_firstmove(int id) const;
    void set_firstmove_blackeval(float eval);
    float get_firstmove_blackeval(int id) const;
    Network::Netresult dump_evals(int req_playouts, std::string & dump_str,
                                  std::string & sgf_str);
    void dump_evals_recursion(std::string & dump_str, UCTNode* const node,
                              int father_progid, int color, std::string & sgf_str,
                              std::vector<float> & value_vec,
                              std::vector<float> & alpkt_vec,
                              std::vector<float> & beta_vec);
#endif
    void set_playout_limit(int playouts);
    void set_visit_limit(int visits);
    void ponder();
    bool is_running() const;
    void increment_playouts();
    float final_japscore();
    void tree_stats();
    std::string explain_last_think() const;
    SearchResult play_simulation(GameState& currstate, UCTNode* const node);
    AgentEval get_root_agent_eval() const;
    void prepare_root_node();

private:
    float get_min_psa_ratio() const;
    void dump_stats(FastState& state, UCTNode& parent, const std::map<int,int> & initial_visits);
    void print_move_choices_by_policy(KoState& state, UCTNode& parent,
                                      int at_least_as_many, float probab_threash);
    void tree_stats(const UCTNode& node);
    std::string get_pv(FastState& state, UCTNode& parent);
    std::string get_analysis(int playouts);
    bool should_resign(passflag_t passflag, float besteval);
    bool have_alternate_moves(int elapsed_centis, int time_for_move);
    int est_playouts_left(int elapsed_centis, int time_for_move) const;
    size_t prune_noncontenders(int color, int elapsed_centis = 0, int time_for_move = 0,
                               bool prune = true);
    bool stop_thinking(int elapsed_centis = 0, int time_for_move = 0) const;
    int get_best_move(passflag_t passflag);
    void update_root(bool is_evaluating = false);
    bool advance_to_new_rootstate();
    void select_playable_dame(FullBoard *board);
    void select_dame_sequence(FullBoard *board);
    bool is_stopping (int move) const;
    bool is_better_move(int move1, int move2, float & estimated_score);
    void explore_move(int move);
    void explore_root_nopass();
    void fast_roll_out();
    void output_analysis(FastState & state, UCTNode & parent);

    GameState & m_rootstate;
    std::unique_ptr<GameState> m_last_rootstate;
    std::unique_ptr<UCTNode> m_root;
    std::atomic<int> m_nodes{0};
    std::atomic<int> m_playouts{0};
    std::atomic<bool> m_run{false};
    int m_maxplayouts;
    int m_maxvisits;
    std::string m_think_output;

#ifdef USE_EVALCMD
    int m_nodecounter=0;
    bool m_evaluating=false;
    std::vector<int> m_1st_move;
    std::vector<float> m_1st_move_blackeval;
#endif
#ifndef NDEBUG
    struct sim_node_info {
        std::string movestr = "na";
        std::string leafstr = "na";
        int visits = -1;
        float score = 1000.0f;
        float eval = -1.0f;
        float avg = -1.0f;
    };

    std::vector<sim_node_info> m_info;
#endif

    // Advanced search parameters
    bool m_chn_scoring = true;

    // Max number of visits per node: nodes with this or more visits
    // are never selected. Acts on first generation children of root
    // node, since the deeper generations always have fewer visits.
    // If equal to 0 it is ignored.
    int m_per_node_maxvisits = 0;

    // List of moves allowed as first generation choices during the
    // search. Only applies to the first move in the simulation.
    // If empty it is ignored.
    std::vector<int> m_allowed_root_children = {};

    // If, during the search, any of these vertexes is the move of a
    // node with at least m_stopping_visits, the flag is set to
    // true.  If the vector is empty or the visits are 0 it is
    // ignored.
    std::vector<int> m_stopping_moves = {};
    int m_stopping_visits = 0;
    bool m_stopping_flag = false;
    bool m_nopass = false;
    int m_last_resign_request = -1;

    int m_bestmove = FastBoard::PASS;

    std::list<Utils::ThreadGroup> m_delete_futures;

    Network & m_network;
};

class UCTWorker {
public:
    UCTWorker(GameState & state, UCTSearch * search, UCTNode * root)
      : m_rootstate(state), m_search(search), m_root(root) {}
    void operator()();
private:
    GameState & m_rootstate;
    UCTSearch * m_search;
    UCTNode * m_root;
};

#endif
