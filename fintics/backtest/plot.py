"""Plotting utilities for visualizing backtest results."""

import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

class BacktestPlot:
    """Provide plotting helpers for :class:`Backtest`."""

    _price_indicators = []
    _indicators = []

    def set_indicators(self, price_indicators: list = [], indicators: list = []) -> None:
        """Manually set indicator lists for plotting."""

        self._price_indicators = price_indicators
        self._indicators = indicators

    def auto_set_indicators(self, price_prefix: str = 'P_', indicator_prefix: str = 'X_') -> None:
        """Automatically detect indicators based on column prefixes."""

        _p = self._df.filter(like=price_prefix).columns.tolist()
        _x = self._df.filter(like=indicator_prefix).columns.tolist()
        if 'y' in self._df.columns:
            if len(_x) > 0:
                _x = [['y'], _x]
            else:
                _x = ['y']

        if len(_p) > 0 or len(_x) > 0:
            self.set_indicators(price_indicators=_p, indicators=_x)

    def plot(self, light_mode: bool = True, plot_drawdown: bool = False, plot_growth: bool = True, plot_reinvest: bool = True, browser_render:bool=False, return_fig:bool=False):
        """Render interactive chart of prices, indicators, and performance."""

        if browser_render:
            pio.renderers.default = "browser"

        try:
            df = self._get_backtest_df()
            performance = self.get_performance()
        except Exception as e:
            print(f"Error getting backtest data: {e}")
            print("Make sure to run a backtest first using backtest.run() or manually set _backtest_df")
            return

        plot_items=['chart', 'entrypoint', 'exitpoint', 'pl']
        if plot_drawdown:
            plot_items.append('drawdown')
        if plot_growth:
            plot_items.append('growth')

        try:
            has_trade = df.TradesNum() > 0
        except Exception:
            # If TradesNum() fails, assume no trades
            has_trade = False
            
        base_row_len = 2 if has_trade else 1

        if len(self._indicators) > 0 and isinstance(self._indicators[0], list):
            rows_len = base_row_len + len(self._indicators)
        elif isinstance(self._indicators, list) and len(self._indicators) > 0:
            rows_len = base_row_len + 1
        else:
            rows_len = base_row_len

        _price_indicators = self._price_indicators
        _indicators = self._indicators

        try:
            fig = make_subplots(
                rows=rows_len, cols=2,
                column_widths=[0.7, 0.3],
                row_heights=[0.5]*(rows_len),
                specs=[
                    [{"rowspan": 1}, {"rowspan": rows_len, "type": "table"}],
                    *[[{"rowspan": 1}, None] for _ in range(rows_len-1)]
                ],
                shared_xaxes=True,
                vertical_spacing=0.03,
                horizontal_spacing=0.03
            )
        except Exception as e:
            print(f"Error creating subplots: {e}")
            return

        # add backtest info and results table
        try:
            fig.add_trace(go.Table(
                header={'values': ['name', 'results'], 'fill_color': '#2a3139', 'font': dict(color='white')},
                cells={'values': [performance.index, performance.to_list()], 'align': ['left','right'], 'fill_color': 'rgba(30, 34, 40, 0.8)', 'font': dict(color='#d2d2d2', size=12)}), 
                row=1,
                col=2
            )
        except Exception as e:
            print(f"Error adding performance table: {e}")

        # Get price_type safely
        price_type = getattr(df, '_info', {}).get('price_type', 'Close')
        
        # add price and price indicators traces
        if 'y' in df.columns:
            signal = df['y'].diff() if price_type == 'Close' else df['y'].diff().shift()
            df['up'] = np.where(signal>0, df[price_type], np.nan)
            df['down'] = np.where(signal<0, df[price_type], np.nan)
            marker_up, marker_down = df['up'].dropna(), df['down'].dropna()
        else:
            marker_up, marker_down = pd.Series(dtype=float), pd.Series(dtype=float)
            
        marker_size = 10

        _Price_Trace = {
            'candlechart': go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'),
            'linechart': go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', marker={'color': 'blue'}, line={'width': 1}),
        }
        
        try:
            pl_hist = df.PlHist() if hasattr(df, 'PlHist') else pd.Series([0]*len(df), index=df.index)
            pl_growth = df.PlGrowth() if hasattr(df, 'PlGrowth') else pd.Series([0]*len(df), index=df.index)
        except Exception:
            pl_hist = pd.Series([0]*len(df), index=df.index)
            pl_growth = pd.Series([0]*len(df), index=df.index)
            
        _Traces = {
            'chart': _Price_Trace['linechart'] if light_mode else _Price_Trace['candlechart'],
            'entrypoint': go.Scatter(x=marker_up.index, y=marker_up, mode='markers', name='Entry', marker_symbol='triangle-up', marker_color='#13cd6e', marker_size=marker_size),
            'exitpoint': go.Scatter(x=marker_down.index, y=marker_down, mode='markers', name='Exit', marker_symbol='triangle-down', marker_color='#ef0d4b', marker_size=marker_size),
            'pl': go.Scatter(x=df.index, y=pl_hist, mode='lines', name='PL', line={'color': 'rgb(143, 0, 255)', 'width': 1}),
            'drawdown': go.Scatter(x=df.index, y=pl_hist.cummax(), name='Drawdown', fill='tonexty', fillcolor="rgba(143, 0, 255, 0.2)", line={'color': 'rgb(143, 0, 255)', 'width': 0}),
            'growth': go.Scatter(x=df.index, y=pl_growth, mode='lines', name='Growth', line={'color': 'rgb(51, 204, 255)', 'width': 1})
        }

        price_traces = [_Traces['chart']]
        if has_trade and len(marker_up) > 0:
            price_traces.append(_Traces['entrypoint'])
        if has_trade and len(marker_down) > 0:
            price_traces.append(_Traces['exitpoint'])
        
        for indicator in _price_indicators:
            if indicator in df.columns:
                price_traces.append(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line={'width': 1}))

        try:
            for trace in price_traces:
                fig.add_trace(trace, row=1, col=1)
        except Exception as e:
            print(f"Error adding price traces: {e}")
            return
        

        # add indicators traces
        for i, indicators in enumerate(_indicators):
            try:
                if isinstance(indicators, list):
                    for indicator in indicators:
                        if indicator in df.columns:
                            indicator_trace = go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line={'width': 1})
                            fig.add_trace(indicator_trace, row=2+i, col=1)
                else:
                    if indicators in df.columns:
                        indicator_trace = go.Scatter(x=df.index, y=df[indicators], mode='lines', name=indicators, line={'width': 1})
                        fig.add_trace(indicator_trace, row=2, col=1)
            except Exception as e:
                print(f"Error adding indicator traces: {e}")

        # add PL traces
        if has_trade:
            try:
                fig.add_trace(_Traces['pl'], row=rows_len, col=1)
                if plot_drawdown:
                    fig.add_trace(_Traces['drawdown'], row=rows_len, col=1)
                if plot_growth:
                    fig.add_trace(_Traces['growth'], row=rows_len, col=1)
            except Exception as e:
                print(f"Error adding PL traces: {e}")

        # setting titles and labels
        try:
            fig.update_xaxes(rangeslider_visible=False)
            fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05, row=rows_len, col=1)
            fig.update_yaxes(title_text='Price', row=1, col=1)
            fig.update_yaxes(title_text='P/L', row=rows_len, col=1)
            fig.update_layout(
                title='Backtest',
                template='plotly_dark',
                legend={'x': 0.0, 'y': 1.0, 'xanchor': 'left', 'yanchor': 'bottom', 'orientation': 'h'},
                autosize=True,
                xaxis_rangeslider_yaxis_rangemode="auto"
            )
            fig.update_layout(height=840)
        except Exception as e:
            print(f"Error updating layout: {e}")

        plot_id = 'backtest-dashboard'
        js = f"""
            document.body.style.margin = '0';
            document.body.style.backgroundColor = 'rgb(17, 17, 17)';

            const gd = document.getElementById('{plot_id}');

            if (gd) {{
                gd.on('plotly_relayout', (eventData) => {{
                    console.log('Relayout event:', eventData);
                    
                    // Skip if autorange is being set
                    if (eventData && eventData['xaxis.autorange']) {{
                        const updates = {{}};
                        
                        Object.keys(gd.layout).filter(k => k.startsWith('yaxis')).forEach(axisKey => {{
                            updates[axisKey + '.autorange'] = true;
                        }});
                        Plotly.relayout(gd, updates);
                        return;
                    }}

                    // Only proceed if we have x-axis range changes
                    if (!eventData || (!eventData['xaxis.range[0]'] && !eventData['xaxis.range'])) {{
                        return;
                    }}
                    
                    console.log('Processing x-axis range change');
                    
                    const updates = {{}};
                    let xRange;
                    
                    // Get x-axis range from different possible sources
                    if (eventData['xaxis.range']) {{
                        xRange = eventData['xaxis.range'];
                    }} else if (gd.layout.xaxis && gd.layout.xaxis.range) {{
                        xRange = gd.layout.xaxis.range;
                    }} else {{
                        console.log('Could not determine x-axis range');
                        return;
                    }}
                    
                    const xRangeStart = new Date(xRange[0]).getTime();
                    const xRangeEnd = new Date(xRange[1]).getTime();
                    
                    console.log(`X-range: ${{new Date(xRangeStart).toISOString()}} to ${{new Date(xRangeEnd).toISOString()}}`);
                    
                    // Get all y-axes from layout
                    const yAxes = Object.keys(gd.layout).filter(k => k.startsWith('yaxis'));
                    console.log('Available y-axes:', yAxes);
                    
                    // For each y-axis, collect data from traces that belong to it
                    yAxes.forEach(yAxisKey => {{
                        const yAxisData = [];
                        const yAxisId = yAxisKey === 'yaxis' ? 'y' : yAxisKey.replace('yaxis', 'y');
                        
                        console.log(`Processing ${{yAxisKey}} (ID: ${{yAxisId}})`);
                        
                        gd.data.forEach((trace, traceIndex) => {{
                            if (trace.type === 'table' || !trace.x || !trace.y) return;
                            
                            // Check if this trace belongs to the current y-axis
                            let belongsToThisAxis = false;
                            
                            if (trace.yaxis) {{
                                // Trace has explicit yaxis reference
                                belongsToThisAxis = (trace.yaxis === yAxisId);
                            }} else if (trace.xaxis) {{
                                // Map xaxis to yaxis (x->y, x2->y2, etc.)
                                const expectedYAxis = trace.xaxis.replace('x', 'y') || 'y';
                                belongsToThisAxis = (expectedYAxis === yAxisId);
                            }} else {{
                                // Default: first traces go to first y-axis, etc.
                                belongsToThisAxis = (yAxisId === 'y');
                            }}
                            
                            if (belongsToThisAxis) {{
                                console.log(`Trace ${{traceIndex}} (${{trace.name}}) belongs to ${{yAxisId}}`);
                                
                                // Collect y-values within x-range
                                for (let i = 0; i < trace.x.length; i++) {{
                                    const xTimestamp = new Date(trace.x[i]).getTime();
                                    if (xTimestamp >= xRangeStart && xTimestamp <= xRangeEnd) {{
                                        if (trace.type === 'candlestick') {{
                                            // For candlestick, include OHLC values
                                            if (trace.open && !isNaN(trace.open[i]) && isFinite(trace.open[i])) {{
                                                yAxisData.push(trace.open[i]);
                                            }}
                                            if (trace.high && !isNaN(trace.high[i]) && isFinite(trace.high[i])) {{
                                                yAxisData.push(trace.high[i]);
                                            }}
                                            if (trace.low && !isNaN(trace.low[i]) && isFinite(trace.low[i])) {{
                                                yAxisData.push(trace.low[i]);
                                            }}
                                            if (trace.close && !isNaN(trace.close[i]) && isFinite(trace.close[i])) {{
                                                yAxisData.push(trace.close[i]);
                                            }}
                                        }} else if (trace.y[i] !== null && trace.y[i] !== undefined && !isNaN(trace.y[i]) && isFinite(trace.y[i])) {{
                                            yAxisData.push(trace.y[i]);
                                        }}
                                    }}
                                }}
                            }}
                        }});
                        
                        // Update y-axis range if we have data
                        if (yAxisData.length > 0) {{
                            const yMin = Math.min(...yAxisData);
                            const yMax = Math.max(...yAxisData);
                            
                            // Calculate margin
                            let margin;
                            if (yMax !== yMin) {{
                                margin = (yMax - yMin) * 0.05;
                            }} else {{
                                margin = Math.abs(yMax) * 0.01 || 0.1;
                            }}
                            
                            const newRange = [yMin - margin, yMax + margin];
                            updates[yAxisKey + '.range'] = newRange;
                            
                            console.log(`${{yAxisKey}}: ${{yAxisData.length}} points, range [${{yMin.toFixed(2)}}, ${{yMax.toFixed(2)}}] -> [${{newRange[0].toFixed(2)}}, ${{newRange[1].toFixed(2)}}]`);
                        }} else {{
                            console.log(`No data found for ${{yAxisKey}}`);
                        }}
                    }});
                    
                    // Apply updates
                    if (Object.keys(updates).length > 0) {{
                        console.log('Applying updates:', updates);
                        Plotly.relayout(gd, updates);
                    }} else {{
                        console.log('No updates to apply');
                    }}
                }});
            }}
            """

        try:
            # Try different display methods based on environment
            try:
                if return_fig:
                    return fig
                # First try the standard show method
                fig.show(post_script=[js])
                
            except Exception as show_error:
                print(f"Standard show failed: {show_error}")
                
                # Try alternative renderers
                alternative_renderers = ["browser", "notebook", "json"]
                for renderer in alternative_renderers:
                    try:
                        print(f"Trying renderer: {renderer}")
                        pio.renderers.default = renderer
                        fig.show(post_script=[js])
                        print(f"Plot displayed successfully with {renderer} renderer")
                        break
                    except Exception as renderer_error:
                        print(f"{renderer} renderer failed: {renderer_error}")
                        continue
                else:
                    # If all show methods fail, save as HTML file
                    print("All display methods failed, saving as HTML file...")
                    html_file = "/tmp/fintics_backtest_plot.html"
                    fig.write_html(html_file)
                    print(f"Plot saved as HTML: {html_file}")
                    print("Open this file in your browser to view the plot")
                    
                    # Try to open in browser
                    try:
                        import webbrowser
                        webbrowser.open(f"file://{html_file}")
                        print("Attempting to open in browser...")
                    except Exception as browser_error:
                        print(f"Could not open browser: {browser_error}")
                        
        except Exception as e:
            print(f"Error displaying plot: {e}")
            print("Plot creation completed but display failed.")
            print("The plot object is valid - display issues are environment-related.")
