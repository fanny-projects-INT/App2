import ast
import unittest
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from dashboard.behavior_overview import recent_sessions, summary_markup, timeline_markup, compact_progression_figure


class BehaviorOverviewTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'Date':pd.to_datetime(['2026-10-01','2026-10-02','2026-10-02']),
                             'Version':[1,2,10], 'Protocol':[1,2,3]})

    def test_recent_sessions_order_versions_numerically_and_keep_distinct_rows(self):
        sessions=recent_sessions(self.frame())
        self.assertEqual([s['version'] for s in sessions],['10','2','1'])
        self.assertEqual([s['label'] for s in sessions],['Task','Training 2','Training 1'])
        self.assertEqual(len(recent_sessions(pd.concat([self.frame(),self.frame()]))),6)

    def test_rolling_30_days_cross_month_and_filter_boundaries(self):
        frame=pd.DataFrame({'Date':pd.to_datetime(['2026-09-08','2026-09-09','2026-10-08','2026-10-09']),
                            'Version':[1,2,3,4], 'Protocol':[1]*4})
        timeline=timeline_markup(recent_sessions(frame, limit=None),'2026-10-08')
        self.assertEqual(timeline.count('data-date='),30)
        self.assertEqual(timeline.count('class="behavior-session-dot"'),2)
        self.assertIn('data-date="2026-09-09"',timeline)
        self.assertIn('data-date="2026-10-08"',timeline)
        self.assertNotIn('v1',timeline)
        self.assertNotIn('v4',timeline)

    def test_leap_year_and_empty_window(self):
        timeline=timeline_markup([], '2024-03-01')
        self.assertEqual(timeline.count('data-date='),30)
        self.assertIn('data-date="2024-02-01"',timeline)
        self.assertIn('data-date="2024-02-29"',timeline)
        self.assertIn('data-date="2024-03-01"',timeline)
        self.assertNotIn('class="behavior-session-dot"',timeline)

    def test_simple_cards_and_horizontal_sessions_markup(self):
        cards=summary_markup('<img src=x>',self.frame())
        self.assertEqual(cards.count('class="behavior-overview-card"'),5)
        self.assertIn('&lt;img src=x&gt;',cards)
        self.assertNotIn('<img src=x>',cards)
        self.assertNotIn('<svg',cards)
        self.assertNotIn('SESSION MIX',cards)
        timeline=timeline_markup(recent_sessions(self.frame()), '2026-10-08')
        self.assertEqual(timeline.count('class="behavior-session-dot"'),3)
        self.assertEqual(timeline.count('data-date='),30)
        self.assertIn('oldest to newest',timeline)
        self.assertIn('v10',timeline)

    def test_overview_has_no_alyx_dependency_and_uses_native_html(self):
        source=(Path(__file__).resolve().parents[1]/'dashboard'/'behavior_overview.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        imports=[ast.unparse(node) for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('alyx' in item for item in imports))
        self.assertNotIn('get_mouse_info',source)
        self.assertNotIn('<script>',source)
        self.assertIn('overflow-x:auto',source)
        self.assertNotIn('st.button(',source)
        self.assertNotIn('st.session_state',source)

    def test_overview_renders_from_local_metadata(self):
        app=AppTest.from_string('''
import pandas as pd
from dashboard.behavior_overview import render_behavior_overview
today=pd.Timestamp.now().normalize()
frame=pd.DataFrame({'Date':[today-pd.Timedelta(days=5),today], 'Version':[1,2], 'Protocol':[1,3]})
render_behavior_overview('VF101',frame)
''')
        app.run()
        self.assertFalse(app.exception)
        timeline=next(e.proto.body for e in app.get('html') if '<ol class="behavior-session-track"' in e.proto.body)
        self.assertEqual(timeline.count('class="behavior-session-dot"'),2)
        self.assertEqual(len(app.button_group),0)
        self.assertEqual(len(app.button),0)

    def test_same_day_versions_remain_distinct_and_tooltips_are_escaped(self):
        sessions=recent_sessions(self.frame())
        sessions[0]['label']='<img src=x>'
        timeline=timeline_markup(sessions,'2026-10-08')
        self.assertEqual(timeline.count('class="behavior-session-dot"'),3)
        self.assertIn('v2',timeline)
        self.assertIn('v10',timeline)
        self.assertIn('&lt;img src=x&gt;',timeline)
        self.assertNotIn('<img src=x>',timeline)
        self.assertIn('title="02 Oct 2026 · Training 2 · v2"',timeline)

    def test_day_numbers_and_weekday_names_cross_year_boundary(self):
        timeline=timeline_markup([], '2026-01-08')
        self.assertEqual(timeline.count('class="behavior-session-number"'),30)
        self.assertEqual(timeline.count('class="behavior-session-day week-start"'),4)
        self.assertEqual(timeline.count('class="behavior-session-weekday"'),30)
        self.assertIn('title="Monday">M</div>',timeline)
        self.assertIn('title="Sunday">S</div>',timeline)
        self.assertNotIn('W1</div>',timeline)
        self.assertIn('data-date="2025-12-31"',timeline)
        self.assertIn('data-date="2026-01-01"',timeline)

    def test_type_counts_include_all_history_and_unknown_protocols(self):
        frame=pd.DataFrame({'Date':['2020-01-01',None,'2026-10-08','2026-10-08','2026-10-08'],
                            'Version':[1]*5, 'Protocol':['1',1,3,99,None]})
        cards=summary_markup('VF101',frame)
        self.assertIn('SESSION TYPES · ALL HISTORY',cards)
        self.assertIn('<strong>2</strong>Training 1',cards)
        self.assertIn('<strong>0</strong>Training 2',cards)
        self.assertIn('<strong>1</strong>Task',cards)
        self.assertIn('<strong>0</strong>Flexifliping',cards)
        self.assertIn('<strong>2</strong>Other',cards)

    def test_compact_preview_preserves_data_axes_and_original_figures(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        training=make_subplots(rows=1,cols=2)
        training.add_trace(go.Scatter(x=[1,2],y=[10,20],marker={'size':9},
                                     customdata=['first','second'],hovertemplate='%{customdata}'),row=1,col=1)
        training.add_trace(go.Scatter(x=[1,2],y=[3,4]),row=1,col=2)
        regression=make_subplots(rows=1,cols=3)
        for column in range(1,4):
            regression.add_trace(go.Scatter(x=[1,2],y=[column,column+1]),row=1,col=column)
        regression.update_xaxes(tickmode='array',tickvals=list(range(10)),ticktext=[str(i) for i in range(10)])
        before=training.to_json()
        preview=compact_progression_figure({'training_outcomes':training,'regression':regression})
        self.assertEqual(len(preview.data),5)
        self.assertEqual(list(preview.data[0].y),[10,20])
        self.assertEqual(list(preview.data[1].y),[3,4])
        self.assertEqual(preview.data[1].xaxis,'x4')
        self.assertEqual(preview.data[2].xaxis,'x5')
        self.assertEqual(preview.data[4].xaxis,'x3')
        self.assertEqual(list(preview.data[0].customdata),['first','second'])
        self.assertEqual(preview.data[0].hovertemplate,'%{customdata}')
        self.assertEqual(training.to_json(),before)
        self.assertEqual(preview.layout.height,500)
        title=next(a for a in preview.layout.annotations if a.text=='Valid bouts')
        self.assertEqual(title.xanchor,'left')
        self.assertEqual(title.x,preview.layout.xaxis.domain[0])
        self.assertEqual(title.xshift,-preview.layout.margin.l)
        self.assertEqual(title.yshift,18)
        self.assertGreater(preview.layout.xaxis.domain[1]-preview.layout.xaxis.domain[0],
                           preview.layout.xaxis5.domain[1]-preview.layout.xaxis5.domain[0])
        self.assertEqual(len(preview.layout.xaxis5.tickvals),4)
        self.assertFalse(preview.layout.xaxis6.showticklabels)
        self.assertFalse(preview.layout.xaxis3.showticklabels)
        self.assertEqual(preview.layout.margin.b,40)
        self.assertTrue(preview.layout.xaxis4.automargin)
        self.assertNotIn('Failures · global dashed',[a.text for a in preview.layout.annotations])
        self.assertFalse(preview.layout.xaxis2.visible)
        self.assertEqual(preview.data[0].marker.size,10)
        self.assertEqual(training.data[0].marker.size,9)

    def test_compact_preview_renders_as_one_chart(self):
        app=AppTest.from_string('''
from dashboard.behavior_overview import render_progression_preview
render_progression_preview({})
''').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('plotly_chart')),1)
        self.assertEqual(len(compact_progression_figure({}).layout.annotations),0)

    def test_global_failure_curve_overlays_sessions_without_mutating_cache(self):
        import plotly.graph_objects as go
        from dashboard.behavior_overview import combined_failure_figure
        sessions=go.Figure(go.Scatter(x=[1,2],y=[.2,.3],name='2026-10-08'))
        sessions.update_xaxes(range=[0,30])
        distribution=go.Figure([
            go.Bar(x=[1,2],y=[.1,.4]),
            go.Scatter(x=[1,2],y=[.15,.35],fill='tozeroy')])
        distribution.update_xaxes(range=[.5,50.5])
        before=(sessions.to_json(),distribution.to_json())
        merged=combined_failure_figure({'kde_failures_by_session':sessions,
                                       'histogram_kde_failures':distribution})
        self.assertEqual(len(merged.data),2)
        self.assertEqual(merged.data[-1].line.color,'#111111')
        self.assertEqual(merged.data[-1].line.dash,'dash')
        self.assertIsNone(merged.data[-1].fill)
        self.assertEqual(list(merged.data[-1].y),[.15,.35])
        self.assertEqual(tuple(merged.layout.xaxis.range),(0,30))
        self.assertEqual((sessions.to_json(),distribution.to_json()),before)
        preview=compact_progression_figure({'kde_failures_by_session':sessions,
                                            'histogram_kde_failures':distribution})
        annotations=[annotation.text for annotation in preview.layout.annotations]
        self.assertIn('Consecutive failure distribution',annotations)
        self.assertTrue(any('Earlier' in text and 'Latest' in text and 'All sessions' in text
                            for text in annotations))
        chronology=next(annotation for annotation in preview.layout.annotations
                        if 'Earlier' in annotation.text)
        self.assertEqual(chronology.y,.82)
        self.assertEqual(chronology.bgcolor,'rgba(0,0,0,0)')
        self.assertNotIn('●',chronology.text)
        fallback=combined_failure_figure({'histogram_kde_failures':distribution})
        self.assertEqual(len(fallback.data),1)
        self.assertEqual(tuple(fallback.layout.xaxis.range),(0,30))

    def test_session_info_uses_the_same_compact_cards_as_overview(self):
        app=AppTest.from_string('''
from dashboard.behavior_overview import render_session_info
render_session_info([('PROTOCOL','Task'),('PROBABILITIES','90/30'),('DURATION','10 min')])
''').run()
        self.assertFalse(app.exception)
        cards=next(e.proto.body for e in app.get('html') if '<div class="behavior-overview-grid">' in e.proto.body)
        self.assertEqual(cards.count('class="behavior-overview-card"'),3)
        self.assertIn('90/30',cards)
        self.assertIn('10 min',cards)
