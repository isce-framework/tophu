{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{% block references %}
    .. footbibliography::
{% endblock %}
