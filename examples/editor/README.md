# SonataEdgeEditor in airavata-cerebrum

Cerebrum library includes a simple functionality to select and replace edges of 
a given edge type in a edge SONTA file.

The SonataEdgeEditor class is generic and can accommodate multiple edge type ids
at the same time.  A user can generate their own set of edges based on their own
requirements,  format it to a polars data frame matching the input file edge 
table and use it to replace the edges. The class handles attributes of the edge
table. Also, the class rebuilds the edges indices if the input file had any 
(*Note: According to the SONATA standard, edge indices are optional*).

SONTA edge file from the the Blue Brain project has an index that is not part 
of the standard.
The index appears under the 'edge_type_to_index' H5 group, and index tables 
are named 'node_id_to_range' and 'range_to_edge_id'. Even though the table is 
called 'node_id_to_range', it actually has the edge ids. The editor handles this
custom index along with the SONATA standard indices 'source_to_target'  and
'target_to_source' 
