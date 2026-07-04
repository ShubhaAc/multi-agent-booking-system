from graph import build_graph

graph = build_graph()

png_bytes = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved to graph.png")