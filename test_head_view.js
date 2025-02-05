/**
 * @fileoverview Transformer Visualization D3 javascript code.
 *
 *
 *  Based on: https://github.com/jessevig/bertviz/blob/master/bertviz/head_view.js
 *
 **/

require.config({
    paths: {
        d3: 'https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min',
      jquery: 'https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.0/jquery.min',
    }
  });
  
  requirejs(['jquery', 'd3'], function ($, d3) {
  
      const params = PYTHON_PARAMS; // HACK: PYTHON_PARAMS is a template marker that is replaced by actual params.
      const TEXT_SIZE = 15;
      const BOXWIDTH = 20;
      const BOXHEIGHT = 50;
      const MATRIX_WIDTH = 20;
      const CHECKBOX_SIZE = 20;
      const TEXT_TOP = 40;
  
      console.log("d3 version", d3.version)
      let headColors;
      try {
          headColors = d3.scaleOrdinal(d3.schemeCategory10);
      } catch (err) {
          console.log('Older d3 version')
          headColors = d3.scale.category10();
      }
      let config = {};
      initialize();
      renderVis();
  
      function initialize() {
          config.attention = params['attention'];
          config.filter = params['default_filter'];
          config.rootDivId = params['root_div_id'];
          config.nLayers = config.attention[config.filter]['attn'].length;
          config.nHeads = config.attention[config.filter]['attn'][0].length;
          config.layers = params['include_layers']
  
          if (params['heads']) {
              config.headVis = new Array(config.nHeads).fill(false);
              params['heads'].forEach(x => config.headVis[x] = true);
          } else {
              config.headVis = new Array(config.nHeads).fill(true);
          }
          config.initialTextLength = config.attention[config.filter].right_text.length;
          config.layer_seq = (params['layer'] == null ? 0 : config.layers.findIndex(layer => params['layer'] === layer));
          config.layer = config.layers[config.layer_seq]
  
          let layerEl = $(`#${config.rootDivId} #layer`);
          for (const layer of config.layers) {
              layerEl.append($("<option />").val(layer).text(layer));
          }
          layerEl.val(config.layer).change();
          layerEl.on('change', function (e) {
              config.layer = +e.currentTarget.value;
              config.layer_seq = config.layers.findIndex(layer => config.layer === layer);
              renderVis();
          });
  
          $(`#${config.rootDivId} #filter`).on('change', function (e) {
              config.filter = e.currentTarget.value;
              renderVis();
          });
      }
  
      function renderVis() {
  
          // Load parameters
          const attnData = config.attention[config.filter];
          const topText = attnData.left_text;
          const bottomText = attnData.right_text;
  
          // Select attention for given layer
          const layerAttention = attnData.attn[config.layer_seq];
  
          // Clear vis
          $(`#${config.rootDivId} #vis`).empty();
  
          // Determine size of visualization
          const width = Math.max(topText.length, bottomText.length) * BOXWIDTH + TEXT_TOP;
          const height = MATRIX_WIDTH + (2*BOXHEIGHT); // Adjust for spacing

          const svg = d3.select(`#${config.rootDivId} #vis`)
              .append('svg')
              .attr("width", "100%")
              .attr("height", height + "px");
  
          // Display tokens on top and bottom of visualization
          renderText(svg, topText, true, layerAttention, TEXT_TOP);
          renderText(svg, bottomText, false, layerAttention, BOXWIDTH+MATRIX_WIDTH/2);
  
          // Render attention arcs
          renderAttention(svg, layerAttention);
  
          // Draw squares at top of visualization, one for each head
          drawCheckboxes(0, svg, layerAttention);
      }
  
      function renderText(svg, text, isTop, attention, topPos) {
  
          const textContainer = svg.append("svg:g")
                .attr("id", isTop ? "top" : "bottom"); // If isTop is true, it assigns id="top". Otherwise, it assigns id="bottom".
  
          // Add attention highlights superimposed over words
          textContainer.append("g")
              .classed("attentionBoxes", true)
              .selectAll("g")
              .data(attention)
              .enter()
              .append("g")
              .attr("head-index", (d, i) => i)
              .selectAll("rect")
              .data(d => isTop ? d : transpose(d)) // if right text, transpose attention to get right-to-left weights
              .enter()
              .append("rect")
              .attr("x", function () {
                  var headIndex = +this.parentNode.getAttribute("head-index");
                  return topPos + boxOffsets(headIndex);
              })
              .attr("y", (+1) * BOXHEIGHT)
              .attr("width", BOXWIDTH)
              .attr("height", BOXHEIGHT / activeHeads())
              .attr("fill", function () {
                  return headColors(+this.parentNode.getAttribute("head-index"))
              })
              .style("opacity", 0.0);
  
          const tokenContainer = textContainer.append("g").selectAll("g")
              .data(text)
              .enter()
              .append("g");
  
          // Add gray background that appears when hovering over text
          tokenContainer.append("rect")
              .classed("background", true)
              .style("opacity", 0.0)
              .attr("fill", "lightgray")
              .attr("x", (d, i) => TEXT_TOP + (i+1/2) * BOXWIDTH)
              .attr("y", (d, i) => isTop ? topPos - BOXHEIGHT / 2 : topPos + BOXHEIGHT/2 + TEXT_SIZE/2)
              .attr("width", BOXWIDTH)
              .attr("height", BOXHEIGHT);
  
          // Add token text
          const textEl = tokenContainer.append("text")
              .text(d => d)
              .attr("font-size", TEXT_SIZE + "px")
              .style("cursor", "default")
              .style("-webkit-user-select", "none")
              .attr("x", (d, i) => TEXT_TOP + i * BOXWIDTH)
              .attr("y", (d, i) => isTop ? topPos - TEXT_SIZE : topPos + BOXHEIGHT + TEXT_SIZE)
              .style("text-anchor", "middle");
  
          if (isTop) {
              textEl.style("text-anchor", "middle") // Center text
                    .attr("dx", 0)  // No horizontal shift
                    .attr("dy", TEXT_SIZE + 5); // Shift text downward slightly
          } else {
              textEl.style("text-anchor", "middle") // Center text
                    .attr("dx", 0)  // No horizontal shift
                    .attr("dy", -5); // Shift text upward slightly
          }
            
  
          tokenContainer.on("mouseover", function (d, index) {
  
              // Show gray background for moused-over token
              textContainer.selectAll(".background")
                  .style("opacity", (d, i) => i === index ? 1.0 : 0.0)
  
              // Reset visibility attribute for any previously highlighted attention arcs
              svg.select("#attention")
                  .selectAll("line[visibility='visible']")
                  .attr("visibility", null)
  
              // Hide group containing attention arcs
              svg.select("#attention").attr("visibility", "hidden");
  
              // Set to visible appropriate attention arcs to be highlighted
              if (isTop) {
                  svg.select("#attention").selectAll("line[top-token-index='" + index + "']").attr("visibility", "visible");
              } else {
                  svg.select("#attention").selectAll("line[bottom-token-index='" + index + "']").attr("visibility", "visible");
              }
  
              // Update color boxes superimposed over tokens
              const id = isTop ? "bottom" : "top";
              const topPos = isTop ? MATRIX_WIDTH + BOXWIDTH : 0;
              svg.select("#" + id)
                  .selectAll(".attentionBoxes")
                  .selectAll("g")
                  .attr("head-index", (d, i) => i)
                  .selectAll("rect")
                  .attr("x", function () {
                      const headIndex = +this.parentNode.getAttribute("head-index");
                      return topPos + boxOffsets(headIndex);
                  })
                  .attr("x", (d, i) => TEXT_TOP + i * BOXWIDTH)  // Align horizontally
                  .attr("y", function () {  // Adjust for top-bottom layout
                      const headIndex = +this.parentNode.getAttribute("head-index");
                      return topPos + boxOffsets(headIndex);
                  })
                  .attr("width", BOXWIDTH)
                  .attr("height", BOXHEIGHT / activeHeads())
                  .style("opacity", function (d) {
                      const headIndex = +this.parentNode.getAttribute("head-index");
                      if (config.headVis[headIndex])
                          if (d) {
                              return d[index];
                          } else {
                              return 0.0;
                          }
                      else
                          return 0.0;
                  });
          });
  
          textContainer.on("mouseleave", function () {
  
              // Unhighlight selected token
              d3.select(this).selectAll(".background")
                  .style("opacity", 0.0);
  
              // Reset visibility attributes for previously selected lines
              svg.select("#attention")
                  .selectAll("line[visibility='visible']")
                  .attr("visibility", null) ;
              svg.select("#attention").attr("visibility", "visible");
  
              // Reset highlights superimposed over tokens
              svg.selectAll(".attentionBoxes")
                  .selectAll("g")
                  .selectAll("rect")
                  .style("opacity", 0.0);
          });
      }
  
      function renderAttention(svg, attention) {
  
          // Remove previous dom elements
          svg.select("#attention").remove();
  
          // Add new elements
          svg.append("g")
              .attr("id", "attention") // Container for all attention arcs
              .selectAll(".headAttention")
              .data(attention)
              .enter()
              .append("g")
              .classed("headAttention", true) // Group attention arcs by head
              .attr("head-index", (d, i) => i)
              .selectAll(".tokenAttention")
              .data(d => d)
              .enter()
              .append("g")
              .classed("tokenAttention", true) // Group attention arcs by left token
              .attr("top-token-index", (d, i) => i)
              .selectAll("line")
              .data(d => d)
              .enter()
              .append("line")
              .attr("x1", (d, topTokenIndex) => TEXT_TOP + topTokenIndex * BOXWIDTH + (BOXWIDTH / 2))  // Center X for top
              .attr("y1", BOXHEIGHT)  // Start from top tokens
              .attr("x2", (d, bottomTokenIndex) => TEXT_TOP + bottomTokenIndex * BOXWIDTH + (BOXWIDTH / 2))  // Align with bottom
              .attr("y2", MATRIX_WIDTH + BOXHEIGHT)  // Move arcs to bottom tokens
              .attr("stroke-width", 2)
              .attr("stroke", function () {
                  const headIndex = +this.parentNode.parentNode.getAttribute("head-index");
                  return headColors(headIndex)
              })
              .attr("top-token-index", function () {
                  return +this.parentNode.getAttribute("top-token-index")
              })
              .attr("bottom-token-index", (d, i) => i)
          ;
          updateAttention(svg)
      }
  
      function updateAttention(svg) {
          svg.select("#attention")
              .selectAll("line")
              .attr("stroke-opacity", function (d) {
                  const headIndex = +this.parentNode.parentNode.getAttribute("head-index");
                  // If head is selected
                  if (config.headVis[headIndex]) {
                      // Set opacity to attention weight divided by number of active heads
                      return d / activeHeads()
                  } else {
                      return 0.0;
                  }
              })
      }
  
      function boxOffsets(i) {
          const numHeadsAbove = config.headVis.reduce(
              function (acc, val, cur) {
                  return val && cur < i ? acc + 1 : acc;
              }, 0);
          return numHeadsAbove * (BOXWIDTH / activeHeads());
      }
  
      function activeHeads() {
          return config.headVis.reduce(function (acc, val) {
              return val ? acc + 1 : acc;
          }, 0);
      }
  
      function drawCheckboxes(top, svg) {
          const checkboxContainer = svg.append("g");
          const checkbox = checkboxContainer.selectAll("rect")
              .data(config.headVis)
              .enter()
              .append("rect")
              .attr("fill", (d, i) => headColors(i))
              .attr("x", (d, i) => i * CHECKBOX_SIZE)
              .attr("y", top)
              .attr("width", CHECKBOX_SIZE)
              .attr("height", CHECKBOX_SIZE);
  
          function updateCheckboxes() {
              checkboxContainer.selectAll("rect")
                  .data(config.headVis)
                  .attr("fill", (d, i) => d ? headColors(i): lighten(headColors(i)));
          }
  
          updateCheckboxes();
  
          checkbox.on("click", function (d, i) {
              if (config.headVis[i] && activeHeads() === 1) return;
              config.headVis[i] = !config.headVis[i];
              updateCheckboxes();
              updateAttention(svg);
          });
  
          checkbox.on("dblclick", function (d, i) {
              // If we double click on the only active head then reset
              if (config.headVis[i] && activeHeads() === 1) {
                  config.headVis = new Array(config.nHeads).fill(true);
              } else {
                  config.headVis = new Array(config.nHeads).fill(false);
                  config.headVis[i] = true;
              }
              updateCheckboxes();
              updateAttention(svg);
          });
      }
  
      function lighten(color) {
          const c = d3.hsl(color);
          const increment = (1 - c.l) * 0.6;
          c.l += increment;
          c.s -= increment;
          return c;
      }
  
      function transpose(mat) {
          return mat[0].map(function (col, i) {
              return mat.map(function (row) {
                  return row[i];
              });
          });
      }
  
  });