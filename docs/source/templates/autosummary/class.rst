{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}
{% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
       :toctree:

    {% for item in methods %}
       {{ name }}.{{ item }}
    {%- endfor %}

    {% for item in all_methods %}
       {%- if item in ['__call__', '__getitem__', '__iter__'] %}
       {{ name }}.{{ item }}
       {%- endif -%}
    {%- endfor %}

{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
       :toctree:

    {% for item in attributes %}
       {{ name }}.{{ item }}
    {%- endfor %}

{% endif %}
{% endblock %}

{% block references %}
    .. footbibliography::
{% endblock %}
