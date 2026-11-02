#pragma once
// Вкладка Network: топология, связи, узлы, рёбра и окна сети.
#include "gui_common.h"

namespace ucuda::ui_detail {
static const char* const kNetTopologyNames[] = {
    "Custom", "Chain", "Ring", "Star", "Grid", "King graph", "All-to-all", "Small world"
};


double net_value_at(const NetworkResult& r, int p, int node, int var);


int net_degree(const NetworkConfig& c, int node);


void net_mark_custom(NetworkConfig& c);


std::string net_validate_laws(const NetworkSession& s, const NetworkConfig& c);


void draw_network_controls(AppModel& model, SystemLibrary& /*lib*/);


void draw_network_graph_window(AppModel& model);


void draw_network_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view);


void draw_network_raster_window(AppModel& model, PlotRenderer& renderer, HeatmapView& view);


void draw_network_topology_block(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_coupling_block(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_nodes_table(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_edges_table(AppModel& model, NetworkSession& s, NetworkConfig& c) ;
}
