import pandas as pd
import time
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, Plot, CustomJS, HoverTool, FuncTickFormatter, SingleIntervalTicker, ColumnDataSource, LabelSet
from bokeh.models.widgets import DataTable, TableColumn, Panel, Tabs, Div
from bokeh.layouts import layout, row
from flask import Flask, render_template

from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8


def graph_bokeh(timestamps):  # data and temporal
    a, b = timestamps

    for each in range(len(list(a))):
        dataSeconds_df = a[each]
        df = b[each]

        S1_count = dataSeconds_df['Student1_UtteranceList'].tolist()
        S2_count = dataSeconds_df['Student2_UtteranceList'].tolist()

        difference = dataSeconds_df['Difference'].tolist()
        Length = [x for x in range(len(S1_count))]
        newLength = [time.strftime('%M:%S', time.gmtime(x)) for x in Length]

        dictionary = dict(zip(Length, newLength))

        Text = df['Text'].tolist()
        S1_words_Count = df['S1_words_Count'].tolist()
        S2_words_Count = df['S2_words_Count'].tolist()

        S1_Total = (pd.to_numeric(df.S1_words_Count, errors='coerce').fillna(0).astype(np.int64)).sum()
        S2_Total = (pd.to_numeric(df.S2_words_Count, errors='coerce').fillna(0).astype(np.int64)).sum()

        # df['S1_Q_Count'] = pd.to_numeric(df['S1_Q_Count'], errors='coerce')
        # df['S2_Q_Count'] = pd.to_numeric(df['S2_Q_Count'], errors='coerce')

        S1_Q_Count = (pd.to_numeric(df['S1_Q_Count'], errors='coerce')).tolist()
        S2_Q_Count = (pd.to_numeric(df['S2_Q_Count'], errors='coerce')).tolist()

        S1Q_Total = (pd.to_numeric(df.S1_Q_Count, errors='coerce').fillna(0).astype(np.int64)).sum()
        S2Q_Total = (pd.to_numeric(df.S2_Q_Count, errors='coerce').fillna(0).astype(np.int64)).sum()

        S1_Q1_Count = (pd.to_numeric(df['S1_Q1_Count'], errors='coerce')).tolist()
        S1_Q0_Count = (pd.to_numeric(df['S1_Q0_Count'], errors='coerce')).tolist()

        S1Q1_Total = (pd.to_numeric(df.S1_Q1_Count, errors='coerce').fillna(0).astype(np.int64)).sum()
        S1Q0_Total = (pd.to_numeric(df.S1_Q0_Count, errors='coerce').fillna(0).astype(np.int64)).sum()


        GName = df['GroupName'][0]

        TotalSecond = df['TotalSecond'].tolist()
        Timestamp = df['Timestamp'].tolist()
        Role = df['Role'].tolist()

        Speaker = (['You' if x == "S1" else 'Partner' if x == "S2" else "Other" for x in df['Speaker'].tolist()])

        # Speaker = (['You' if x == "S1" else 'Partner' if x == "S2" else "Other" for x in df['Speaker'].tolist()])


        source = ColumnDataSource(
            data=dict(x=Length, y0=S1_count, y1=S2_count, difference=difference, Text=Text, Speaker=Speaker,
                      TotalSecond=TotalSecond, Role=Role, Timestamp=Timestamp, S1_words_Count=S1_words_Count,
                      S2_words_Count=S2_words_Count, S1_Q_Count=S1_Q_Count, S2_Q_Count=S2_Q_Count,
                      S1_Q1_Count=S1_Q1_Count, S1_Q0_Count=S1_Q0_Count))
        s1 = source

        s2 = ColumnDataSource(data=dict(
            Time=["00:01", "00:12", "00:24", "00:46", "01:02", "01:05", "01:12", "01:24", "01:46", "02:02", "02:05",
                  "02:15", "02:25", "02:35"],
            Speaker=["You", "Partner", "Partner", "You", "Partner", "Partner", "You",
                     "Partner", "Partner", "You", "Partner", "You", "Partner", "You"],
            Role=["D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N"],
            Sentence=[" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
            S1_Count=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], S2_Count=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        s02 = ColumnDataSource(data=dict(x=[]))

        rate = S1_Total / (S1_Total + S2_Total)
        end1 = 6.283185307179586 * rate

        rate1 = S1Q_Total / (S1Q_Total + S2Q_Total)
        end3 = 6.283185307179586 * rate1

        rate2 = S1Q0_Total / (S1Q0_Total + S1Q1_Total)
        end_middle = end3 * rate2
        end6 = end3

        s002 = ColumnDataSource(
            data=dict(start1=[0], end1=[end1], start2=[end1], end2=[6.283185307179586], S1_Total=[S1_Total],
                      S2_Total=[S2_Total], start3=[0], end3=[end3], start4=[end3], end4=[6.283185307179586],
                      S1Q_Total=[S1Q_Total], S2Q_Total=[S2Q_Total], start5=[0], end5=[end_middle], start6=[end_middle],
                      end6=[end6], S1Q1_Total=[S1Q1_Total], S1Q0_Total=[S1Q0_Total], S1Q1_Total_rate=[4.5], S1Q0_Total_rate=[4.5], y_empty=[5]))

        columns = [
            TableColumn(field="Time", title="Time", width=23),
            TableColumn(field="Role", title="Role", width=13),
            TableColumn(field="Speaker", title="Speaker", width=42),
            TableColumn(field="Sentence", title="Sentence", width=602),
            # TableColumn(field="S1_Count", title="S1_Count", width=10),
            # TableColumn(field="S2_Count", title="S2_Count", width=10),
            # TableColumn(field="S1Q_Count", title="S1Q_Count", width=10),
            # TableColumn(field="S2Q_Count", title="S2Q_Count", width=10)
        ]

        data_table = DataTable(source=s2, columns=columns, width=685, height=350, header_row=True, index_position=None,
                               reorderable=False)

        p = figure(tools='reset', title="Number of Words", plot_width=200, plot_height=325)


        # p = figure(tools='reset, hover', title="Number of Words", plot_width=200, plot_height=325,
        #            tooltips="S1: @S1_Total, S2: @S2_Total")

        p.toolbar.logo = None
        p.toolbar_location = None
        p.xaxis.axis_label = "Total Words"
        p.xaxis.axis_label_text_color = "whitesmoke"
        p.yaxis.visible = False

        p.yaxis.axis_line_width = 0.1
        p.yaxis.axis_line_alpha = 0
        p.xaxis.axis_line_color = None

        p.xaxis.major_label_text_color = None
        p.xaxis.major_label_text_alpha = 0

        p.xaxis.minor_tick_line_color = None
        p.xaxis.minor_tick_line_alpha = 0
        p.xaxis.major_tick_line_color = None
        p.xaxis.major_tick_line_alpha = 0

        p.grid.grid_line_color = None
        p.background_fill_color = "#2C3A45"
        # p.legend.location = "bottom_center"
        # p.legend.label_text_color = "#edefa7"
        # p.legend.background_fill_color = "#475c6d"
        p.border_fill_color = "#2C3A45"
        p.title.text_color = "whitesmoke"
        p.outline_line_width = 2
        p.outline_line_alpha = 0.8

        # p.wedge(x=0, y=0, start_angle='start1', end_angle='end1', radius=0.8, color='orange', source=s002)
        # p.wedge(x=0, y=0, start_angle='start2', end_angle='end2', radius=0.8, color='#3D53DA', source=s002)

        tooltips01 = [("You", "@S1_Total")]
        tooltips02 = [("Partner", "@S2_Total")]

        diamonds01 = p.wedge(x=0, y=0, start_angle='start1', end_angle='end1', radius=0.8, color='#F8BE6F', source=s002)
        diamonds02 = p.wedge(x=0, y=0, start_angle='start2', end_angle='end2', radius=0.8, color='#3D53DA', source=s002)

        p.add_tools(HoverTool(tooltips=tooltips01, renderers=[diamonds01]))
        p.add_tools(HoverTool(tooltips=tooltips02, renderers=[diamonds02]))


        # p1 = figure(x_range=(-10,10), y_range=(-10,10), tools='reset, hover', title="Number of Questions", plot_width=400, plot_height=325,
        #             tooltips="S1: @S1Q_Total, S2: @S2Q_Total")

        p1 = figure(x_range=(-10, 10), y_range=(-10, 10), title="      Number of Questions                    Question Types",
                    plot_width=400, plot_height=325, tooltips=None)

        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.axis_label = "Total Questions"
        p1.xaxis.axis_label_text_color = "whitesmoke"
        # p.xaxis.axis_label_text_alpha = 0
        p1.yaxis.visible = False

        p1.yaxis.axis_line_width = 0.1
        p1.yaxis.axis_line_alpha = 0
        p1.xaxis.axis_line_color = None

        p1.xaxis.major_label_text_color = None
        p1.xaxis.major_label_text_alpha = 0

        p1.xaxis.minor_tick_line_color = None
        p1.xaxis.minor_tick_line_alpha = 0
        p1.xaxis.major_tick_line_color = None
        p1.xaxis.major_tick_line_alpha = 0

        p1.grid.grid_line_color = None
        p1.background_fill_color = "#2C3A45"

        p1.border_fill_color = "#2C3A45"
        p1.title.text_color = "whitesmoke"
        p1.outline_line_width = 2
        p1.outline_line_alpha = 0.8

        tooltips3 = [("You", "@S1Q_Total")]
        tooltips4 = [("Partner", "@S2Q_Total")]

        diamonds3 = p1.wedge(x=-5, y=0, start_angle='start3', end_angle='end3', radius=3.9, color='#F8BE6F', source=s002)
        diamonds4 = p1.wedge(x=-5, y=0, start_angle='start4', end_angle='end4', radius=3.9, color='#3D53DA', source=s002)

        p1.add_tools(HoverTool(tooltips=tooltips3, renderers=[diamonds3]))
        p1.add_tools(HoverTool(tooltips=tooltips4, renderers=[diamonds4]))

        p1.hbar_stack(["S1Q1_Total_rate", "S1Q0_Total_rate"], y='y_empty', height=2, color=[ "#cf8f5f","#f5d584"], source=s002)

        labels = LabelSet(x=0.22, y=4.3, text='S1Q1_Total', x_offset=4, y_offset=3, source=s002, render_mode='canvas',text_font_style = "bold", text_font_size="9pt")
        labText = LabelSet(x=-0.3, y=7, text=["Closed"], x_offset=4, y_offset=3, render_mode='canvas', text_font_size="9pt", text_font_style = "bold", text_color="whitesmoke")
        labText_ = LabelSet(x=-0.3, y=6, text=["Questions"], x_offset=4, y_offset=3, render_mode='canvas', text_font_size="9pt",text_font_style = "bold", text_color="whitesmoke")

        p1.add_layout(labText_)
        p1.add_layout(labText)
        p1.add_layout(labels)


        labels2 = LabelSet(x=7.7, y=4.3, text='S1Q0_Total', x_offset=4, y_offset=3, source=s002, render_mode='canvas',text_font_style = "bold", text_font_size="9pt")
        labText2 = LabelSet(x=7.2, y=7, text=["Open"], x_offset=4, y_offset=3, render_mode='canvas', text_font_size="9pt",text_font_style = "bold", text_color ="whitesmoke")
        labText2_ = LabelSet(x=5.8, y=6, text=["Questions"], x_offset=4, y_offset=3, render_mode='canvas', text_font_size="9pt",text_font_style = "bold", text_color="whitesmoke")

        p1.add_layout(labels2)
        p1.add_layout(labText2)
        p1.add_layout(labText2_)


        fg = figure(tools='xbox_select, reset', active_drag="xbox_select", title="Number of Word Spoken Over Time",
                    plot_width=690, plot_height=325)
        fg.line('x', 'y0', legend="You", source=s1, line_width=3, line_color="#F8BE6F")
        fg.circle('x', 'y0', source=source, color="#F8BE6F", selection_color="#F8BE6F", size=0.1)
        fg.circle('x', 'y1', source=source, color="#9ebced", selection_color="#F8BE6F", size=0.1)
        fg.line('x', 'y1', legend="Your Partner", source=source, line_width=3, line_color="#9ebced")
        # fg.circle('x', 'difference', source=source, color="#9ebced", selection_color="#F8BE6F", size=0.1)
        # fg.line('x', 'difference', legend="DIFFERENCE", source=source, line_width=2.5, line_color="#FFD7D7")


        fg.xaxis.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];          
        """ % dictionary)

        fg.xaxis.ticker = SingleIntervalTicker(interval=300)
        fg.xaxis.bounds = (0, max(Length))

        fg.background_fill_color = "#2C3A45"
        # fg.grid.grid_line_color = "yellow"
        fg.grid.grid_line_alpha = 0.1
        fg.border_fill_color = "#2C3A45"
        fg.title.text_color = "whitesmoke"
        # fg.min_border_left = 80
        fg.outline_line_width = 2
        fg.outline_line_alpha = 0.8
        fg.outline_line_color = "#314150"

        fg.legend.location = "top_left"
        fg.outline_line_color = "whitesmoke"

        fg.xaxis.major_label_text_color = "whitesmoke"
        fg.yaxis.major_label_text_color = "whitesmoke"

        fg.xaxis.axis_label_text_color = "whitesmoke"
        fg.yaxis.axis_label_text_color = "whitesmoke"

        fg.xaxis.axis_label = "Time (min:sec)"
        fg.yaxis.axis_label = "Number of Words "

        fg.xaxis.axis_line_width = 2
        fg.xaxis.axis_line_alpha = 0.6
        fg.xaxis.axis_line_color = "whitesmoke"

        fg.yaxis.axis_line_width = 2
        fg.yaxis.axis_line_alpha = 0.6
        fg.yaxis.axis_line_color = "whitesmoke"

        fg.xaxis.minor_tick_line_color = "whitesmoke"
        fg.xaxis.minor_tick_line_alpha = 0.6

        fg.yaxis.minor_tick_line_color = "whitesmoke"
        fg.yaxis.minor_tick_line_alpha = 0.6

        fg.legend.background_fill_color = "#475c6d"
        fg.legend.label_text_color = "#edefa7"

        # tab1 = Panel(child=p, title="wordC")
        # tab2 = Panel(child=p1, title="questionQ")
        # tabs = Tabs(tabs=[tab1, tab2])

        l2 = layout([
            # [fg, p1],
            [fg, p, p1],
            [data_table]
        ])

        s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s02=s02, s002=s002, s2=s2), code="""
                        var inds = cb_obj.indices;
                        var d1 = s1.data;
                        var d02 = s02.data;
                        var d002 = s002.data;
                        var d2 = s2.data;

                        d02['x'] = []
                        d2['Time'] = []
                        d2['Speaker'] = []
                        d2['Sentence'] = []
                        d2['Role'] = []
                        d2['S1_Count']=[]
                        d2['S2_Count']=[]
                        d2['S1Q_Count']=[]
                        d2['S2Q_Count']=[]
                        d2['S1Q1_Count']=[]
                        d2['S1Q0_Count']=[]

                        d002['S1_Total']=[]
                        d002['S2_Total']=[]

                        d002['S1Q_Total']=[]
                        d002['S2Q_Total']=[]

                        d002['S1Q1_Total']=[]
                        d002['S1Q0_Total']=[]
                        
                        d002['S1Q1_Total_rate']=[]
                        d002['S1Q0_Total_rate']=[]
                        

                        d002['start1']=[]
                        d002['end1']=[]
                        d002['start2']=[]
                        d002['end2']=[]

                        d002['start3']=[]
                        d002['end3']=[]
                        d002['start4']=[]
                        d002['end4']=[]


                        var S1_Total = 0;
                        var S2_Total = 0;

                        var S1Q_Total = 0;
                        var S2Q_Total = 0;

                        var S1Q1_Total = 0;
                        var S2Q0_Total = 0;



                        for (var i = 0; i < inds.length; i++) {
                                d02['x'].push(parseInt(d1['x'][inds[i]]))

                        }
                        var h = d02['x'].sort(function(a, b){return b - a});
                        var myMin=h[0];
                        var myMax=h[h.length - 1];

                        for (index = 0; index<d1['TotalSecond'].length; index++){
                            for (k=0; k<h.length; k++){
                                if (h[k]==d1['TotalSecond'][index]){
                                    d2['Time'].push(d1['Timestamp'][index]);
                                    d2['Role'].push(d1['Role'][index]);
                                    d2['Speaker'].push(d1['Speaker'][index]);
                                    d2['Sentence'].push(d1['Text'][index]);

                                    d2['S1_Count'].push(d1['S1_words_Count'][index]);
                                    d2['S2_Count'].push(d1['S2_words_Count'][index]);

                                    d2['S1Q_Count'].push(d1['S1_Q_Count'][index]);
                                    d2['S2Q_Count'].push(d1['S2_Q_Count'][index]);

                                    d2['S1Q1_Count'].push(d1['S1_Q1_Count'][index]);
                                    d2['S1Q0_Count'].push(d1['S1_Q0_Count'][index]);

                                    S1_Total = S1_Total+d1['S1_words_Count'][index]+",";
                                    S2_Total = S2_Total+d1['S2_words_Count'][index]+",";   

                                    S1Q_Total = S1Q_Total+d1['S1_Q_Count'][index]+",";
                                    S2Q_Total = S2Q_Total+d1['S2_Q_Count'][index]+","; 

                                    S1Q1_Total = S1Q1_Total+d1['S1_Q1_Count'][index]+",";
                                    S1Q0_Total = S1Q0_Total+d1['S1_Q0_Count'][index]+","; 

                        }
                        }
                        }

                        S1_Total = S1_Total.split(',').map(function(el){ return +el;});
                        S2_Total = S2_Total.split(',').map(function(el){ return +el;});

                        S1_Total = S1_Total.filter(Boolean);
                        S2_Total = S2_Total.filter(Boolean);

                        
                        var S1_Total = S1_Total.reduce(function(a, b){return a+b;});
                        var S2_Total = S2_Total.reduce(function(a, b){return a+b;});
                                               

                        d002['S1_Total'].push(S1_Total);
                        d002['S2_Total'].push(S2_Total);

                        var rate = S1_Total/(S1_Total+S2_Total)*2;

                        var start1=[0];
                        var end1=[rate];
                        var start2=[rate];
                        var end2=[2];

                        var end1 = end1.map(function(n){return (n* 22/7);});
                        var end2 = end2.map(function(n){return (n* 22/7);});
                        var start2 = start2.map(function(n){return (n* 22/7);});

                        d002['start1'].push(start1);
                        d002['start2'].push(start2);
                        d002['end1'].push(end1);
                        d002['end2'].push(end2);

                        S1Q_Total = S1Q_Total.split(',').map(function(el){ return +el;});
                        S2Q_Total = S2Q_Total.split(',').map(function(el){ return +el;});

                        S1Q_Total = S1Q_Total.filter(Boolean);
                        S2Q_Total = S2Q_Total.filter(Boolean);



                        // ***If you want to make the rate 0 uncomment the following two lines and comment if else part.
                        //var S1Q_Total = S1Q_Total.reduce(function(a, b){return a+b;});
                        //var S2Q_Total = S2Q_Total.reduce(function(a, b){return a+b;});
                        
                        
                        if (S1Q_Total === undefined || S1Q_Total.length == 0) {
                            var S1Q_Total = 0;}
                        else {
                            var S1Q_Total = S1Q_Total.reduce(function(a, b){return a+b;});
                        }
                        
                        if (S2Q_Total === undefined || S2Q_Total.length == 0) {
                            var S2Q_Total = 0;}
                        else {
                            var S2Q_Total = S2Q_Total.reduce(function(a, b){return a+b;});
                        }
   

                        d002['S1Q_Total'].push(S1Q_Total);
                        d002['S2Q_Total'].push(S2Q_Total);
                        

                        S1Q1_Total = S1Q1_Total.split(',').map(function(el){ return +el;});
                        S1Q0_Total = S1Q0_Total.split(',').map(function(el){ return +el;});

                        S1Q1_Total = S1Q1_Total.filter(Boolean);
                        S1Q0_Total = S1Q0_Total.filter(Boolean);

              
                        if (S1Q1_Total === undefined || S1Q1_Total.length == 0) {
                            var S1Q1_Total = 0;}
                        else {
                            var S1Q1_Total = S1Q1_Total.reduce(function(a, b){return a+b;});
                        }
                        
                        if (S1Q0_Total === undefined || S1Q0_Total.length == 0) {
                            var S1Q0_Total = 0;}
                        else {
                            var S1Q0_Total = S1Q0_Total.reduce(function(a, b){return a+b;});
                        }


                        d002['S1Q1_Total'].push(S1Q1_Total);
                        d002['S1Q0_Total'].push(S1Q0_Total);
                        
                        var S1Q0_Total_rate = (S1Q0_Total/(S1Q0_Total+S1Q1_Total))*9;
                        var S1Q1_Total_rate = (S1Q1_Total/(S1Q0_Total+S1Q1_Total))*9;

                        
                        d002['S1Q0_Total_rate'].push(S1Q0_Total_rate);
                        d002['S1Q1_Total_rate'].push(S1Q1_Total_rate);
                        




                        function printFollowers(S1Q_Total, S1Q1_Total, S1Q0_Total) {

                        //var myString="Out of "+ S1Q_Total+ " questions,</br>" + " you asked: </br> </br> <b>" + S1Q0_Total + "</b> open questions &</br>" + S1Q1_Total + " closed questions";                        
                        
                        var myString="</br> </br></br><b>" + S1Q0_Total + "</b> open questions &</br> <b>" + S1Q1_Total + "</b> closed questions";                        

                          document.getElementById('printFollower').innerHTML = myString;
                          console.log(myString); 
                        }

                        //printFollowers(S1Q_Total, S1Q1_Total, S1Q0_Total);   
                        
                        var myString="<P><b>Open Questions:</b></br> Searching for deeper responses. </b> <i>How (Why) do you create a variable?</i><P><b>Closed Questions:</b> </br> Simple factual or recall questions. </b> <i>What is this? Where is forever block?</i></b>";                        
                          
                        document.getElementById('printFollower').innerHTML = myString;
                          //console.log(myString); 
                  


                        var rate1 = S1Q_Total/(S1Q_Total+S2Q_Total)*2;

                        var start3=[0];
                        var end3=[rate1];
                        var start4=[rate1];
                        var end4=[2];

                        var end3 = end3.map(function(n){return (n* 22/7);});
                        var end4 = end4.map(function(n){return (n* 22/7);});
                        var start4 = start4.map(function(n){return (n* 22/7);});

                        d002['start3'].push(start3);
                        d002['start4'].push(start4);
                        d002['end3'].push(end3);
                        d002['end4'].push(end4);


                        var myvideo = document.getElementById('myvideo');
                       // var myvideo1 = document.getElementById('myvideo1');

                        myvideo.play();
                        //myvideo.setAttribute('height', '300');
                        //myvideo.setAttribute('width', '700');


                        var aa = d2['Time'][0];
                        var aa1 = parseInt(aa.substring(0, 2))*60;
                        var aa2 = parseInt(aa.substring(3, 5));               
                        var aaa = aa1+aa2;

                        console.log(aaa);

                        myvideo.currentTime = aaa ;
                        myvideo.play();

                        //myvideo1.currentTime = aaa ;
                        //myvideo1.play();
                        

                        console.log(d2);
                        console.log(typeof d2);
                        var nm= JSON.stringify(d2.Sentence);
                     // alert(nm);
                        console.log(nm);
                        console.log(typeof nm);


                        $(function(){

                        //alert(nm);
                        var html='';
                        $.ajax({
                        type: "POST",
                        contentType: "application/json;charset=utf-8",
                        url: "/second",
                        traditional: "true",
                        data: nm,
                        dataType: "json",
                        success: function(response){

                        for(var key in response) {
                        var value = response[key];

                        var num = parseInt(key) +1 ; // the array begins from 0 so I added 1.

                            html+='<div>'+ '&emsp;'+ num+': ' + value+'</div>'
                        }
                        //$("#myText").append(html); //this appends the data at the end.

                        document.getElementById("myText").innerHTML = html

                        console.log(response);
                        //console.log(typeof response);
                        },     
                         error: function() {
                         alert('error loading orders');
                         }
                        });
                        });




                        s2.change.emit();
                        s002.change.emit();
                        console.log(d002);

                        var aa = 0

                    """)
                                 )

        # output_file('Graphs/' + GName + '.html')
        # show(l2)

        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        # render template
        script, div = components(l2)

        html = render_template(
            'index.html',
            plot_script=script,
            plot_div=div,

            js_resources=js_resources,
            css_resources=css_resources,

        )
        return encode_utf8(html)
