import json
import os
import uuid
from IPython.core.display import display, HTML, Javascript
from view_util import *

# Adapted from https://github.com/jessevig/bertviz/tree/master
def head_view(attention, tokens=None, layer=None, heads=None, include_layers=None, html_action='view'):

    attn_data = []
    if tokens is None: raise ValueError("'tokens' is required")
    if include_layers is None: include_layers = list(range(len(attention))) # include all layers
    attention = format_attention(attention, include_layers)
    attn_data.append(
        {
            'name': None,
            'attn': attention.tolist(),
            'left_text': tokens,
            'right_text': tokens
        }
    )

    if layer is not None and layer not in include_layers:
        raise ValueError(f"Layer {layer} is not in include_layers: {include_layers}")

    # Generate unique div id to enable multiple visualizations in one notebook
    vis_id = 'bertviz-%s'%(uuid.uuid4().hex)

    # Compose html
    if len(attn_data) > 1:
        options = '\n'.join(
            f'<option value="{i}">{attn_data[i]["name"]}</option>'
            for i, d in enumerate(attn_data)
        )
        select_html = f'Attention: <select id="filter">{options}</select>'
    else:
        select_html = ""
    
    vis_html = f"""      
        <div id="{vis_id}" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                Layer: <select id="layer"></select>
                {select_html}
            </span>
            <div id='vis'></div>
        </div>
    """

    for d in attn_data:
        attn_seq_len_left = len(d['attn'][0][0])
        if attn_seq_len_left != len(d['left_text']):
            raise ValueError(
                f"Attention has {attn_seq_len_left} positions, while number of tokens is {len(d['left_text'])} "
                f"for tokens: {' '.join(d['left_text'])}"
            )

    params = {
        'attention': attn_data,
        'default_filter': "0",
        'root_div_id': vis_id,
        'layer': layer,
        'heads': heads,
        'include_layers': include_layers
    }

    # require.js must be imported for Colab or JupyterLab:
    if html_action == 'view':
        display(HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'))
        display(HTML(vis_html))
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        vis_js = open(os.path.join(__location__, 'head_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))
        display(Javascript(vis_js))

    elif html_action == 'return':
        html1 = HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>')

        html2 = HTML(vis_html)

        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        vis_js = open(os.path.join(__location__, 'head_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))
        html3 = Javascript(vis_js)
        script = '\n<script type="text/javascript">\n' + html3.data + '\n</script>\n'

        head_html = HTML(html1.data + html2.data + script)
        return head_html

    else:
        raise ValueError("'html_action' parameter must be 'view' or 'return")


def get_head_view(model, train_results, config, trunc=30):
    sampler = train_results['sampler']
    batch = sampler.generate()[0][0][:1]
    _, attn_map = model(batch, get_attn=True)

    trunc = trunc
    attn_tensors = [torch.zeros((1,config.num_heads[l],config.seq_len,config.seq_len)) for l in range(config.num_layers)]
    for l, attn in attn_map.items():
        attn = attn.unsqueeze(0)  # Number of heads in this layer
        attn_tensors[l][:1, :config.num_heads[l], :, :] = attn  # Fill attention tensor

    trunc_attn = [attn_tensors[l][:1, :config.num_heads[l], trunc:, trunc:] for l in range(config.num_layers)]
    trunc_attn = [trunc_attn[l]/trunc_attn[l].sum(dim=-1, keepdims=True) for l in range(config.num_layers)]

    head_view(trunc_attn, batch[0].tolist()[trunc:], html_action='view')