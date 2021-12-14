# This file is part of CairoSVG
# Copyright © 2010-2015 Kozea
#
# This library is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CairoSVG.  If not, see <http://www.gnu.org/licenses/>.

"""
Images manager.

"""

import os.path
from io import BytesIO

from PIL import Image

from .helpers import node_format, preserve_ratio, preserved_ratio, size
from .parser import Tree
from .surface import cairo
from .url import parse_url


def image(surface, node):
    """Draw an image ``node``."""
    base_url = node.get('{http://www.w3.org/XML/1998/namespace}base')
    if not base_url and node.url:
        base_url = os.path.dirname(node.url) + '/'
    href = node.get('{http://www.w3.org/1999/xlink}href') or node.get('href')
    if not href:
        raise ValueError('Image with empty href')
    url = parse_url(href, base_url)
    image_bytes = node.fetch_url(url, 'image/*')

    if len(image_bytes) < 5:
        return

    x, y = size(surface, node.get('x'), 'x'), size(surface, node.get('y'), 'y')
    width = size(surface, node.get('width'), 'x')
    height = size(surface, node.get('height'), 'y')

    if image_bytes[:4] == b'\x89PNG':
        png_file = BytesIO(image_bytes)
    elif (image_bytes[:5] in (b'<svg ', b'<?xml', b'<!DOC') or
            image_bytes[:2] == b'\x1f\x8b'):
        if 'x' in node:
            del node['x']
        if 'y' in node:
            del node['y']
        tree = Tree(
            url=url.geturl(), url_fetcher=node.url_fetcher,
            bytestring=image_bytes, tree_cache=surface.tree_cache,
            unsafe=node.unsafe)
        tree_width, tree_height, viewbox = node_format(
            surface, tree, reference=False)
        if not viewbox:
            tree_width = tree['width'] = width
            tree_height = tree['height'] = height
        node.image_width = tree_width or width
        node.image_height = tree_height or height
        scale_x, scale_y, translate_x, translate_y = preserve_ratio(
            surface, node)

        # Clip image region
        surface.context.rectangle(x, y, width, height)
        surface.context.clip()

        # Draw image
        surface.context.save()
        surface.context.translate(x, y)
        surface.set_context_size(
            *node_format(surface, tree, reference=False), scale=1,
            preserved_ratio=preserved_ratio(tree))
        surface.context.translate(*surface.context.get_current_point())
        surface.context.scale(scale_x, scale_y)
        surface.context.translate(translate_x, translate_y)
        surface.draw(tree)
        surface.context.restore()
        return
    else:
        png_file = BytesIO()
        Image.open(BytesIO(image_bytes)).save(png_file, 'PNG')
        png_file.seek(0)

    image_surface = cairo.ImageSurface.create_from_png(png_file)

    node.image_width = image_surface.get_width()
    node.image_height = image_surface.get_height()
    scale_x, scale_y, translate_x, translate_y = preserve_ratio(
        surface, node)

    # Paint raster image
    surface.context.save()
    surface.context.translate(x, y)
    surface.context.scale(scale_x, scale_y)
    surface.context.translate(translate_x, translate_y)
    if not surface.context.bitmap_context is None:
        a, b, c, d, e, f = surface.context.get_matrix().as_tuple()
        print("Cairo item size: " + str(node.image_width) + ", " + str(node.image_height))
        print("Cairo bitmap matrix: " + str((a,b,c,d,e,f)))
        corners = ((0,0), (node.image_width, 0), (0,node.image_height), (node.image_width,node.image_height))
        for corner in corners:
            corner_x = a * corner[0] + c * corner[1] + e
            corner_y = b * corner[0] + d * corner[1] + f
            print("Corner: " + str((corner_x, corner_y)))
            if surface.bitmap_min_x is None or surface.bitmap_min_x > corner_x:
                surface.bitmap_min_x = corner_x
            if surface.bitmap_min_y is None or surface.bitmap_min_y > corner_y:
                surface.bitmap_min_y = corner_y
            if surface.bitmap_max_x is None or surface.bitmap_max_x < corner_x:
                surface.bitmap_max_x = corner_x
            if surface.bitmap_max_y is None or surface.bitmap_max_y < corner_y:
                surface.bitmap_max_y = corner_y
        surface.context.bitmap_context.set_source_surface(image_surface)
        surface.context.bitmap_context.paint()
    # Setting this bitmap_available allows FLUX Studio frontend to read bitmap layer
    print("Calculated min: " + str((surface.bitmap_min_x, surface.bitmap_min_y)))
    print("Calculated max: " + str((surface.bitmap_max_x, surface.bitmap_max_y)))
    surface.bitmap_available = True
    surface.context.restore()
    
    # Clip image region (if necessary)
    if not (translate_x == 0 and
            translate_y == 0 and
            width == scale_x * node.image_width and
            height == scale_y * node.image_height):
        surface.context.rectangle(x, y, width, height)
        surface.context.clip()
