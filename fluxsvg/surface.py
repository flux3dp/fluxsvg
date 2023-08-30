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
Cairo surface creators.

"""

import io
import types
import logging
import traceback

logger = logging.getLogger("FLUXSVG.surface")
logger.info('Importing cairocffi')

import cairocffi as cairo

from .colors import color
from .defs import (
    apply_filter_after_painting, apply_filter_before_painting, clip_path,
    filter_, gradient_or_pattern, linear_gradient, marker, mask, paint_mask,
    parse_all_defs, pattern, prepare_filter, radial_gradient, use)
from .helpers import (
    UNITS, PointError, apply_matrix_transform, clip_rect, node_format,
    normalize, paint, preserved_ratio, size, transform, get_layer_name)
from .image import image
from .parser import Tree
from .path import draw_markers, path
from .shapes import circle, ellipse, line, polygon, polyline, rect
from .svg import svg
from .text import text
from .url import parse_url
import sys
import beamify.context as beamify
from PIL import Image, ImageColor, ImageOps

Image.MAX_IMAGE_PIXELS = 1e10

SHAPE_ANTIALIAS = {
    'optimizeSpeed': cairo.ANTIALIAS_FAST,
    'crispEdges': cairo.ANTIALIAS_NONE,
    'geometricPrecision': cairo.ANTIALIAS_BEST,
}

TEXT_ANTIALIAS = {
    'optimizeSpeed': cairo.ANTIALIAS_FAST,
    'optimizeLegibility': cairo.ANTIALIAS_GOOD,
    'geometricPrecision': cairo.ANTIALIAS_BEST,
}

TEXT_HINT_STYLE = {
    'geometricPrecision': cairo.HINT_STYLE_NONE,
    'optimizeLegibility': cairo.HINT_STYLE_FULL,
}

TEXT_HINT_METRICS = {
    'geometricPrecision': cairo.HINT_METRICS_OFF,
    'optimizeLegibility': cairo.HINT_METRICS_ON,
}

TAGS = {
    'a': text,
    'circle': circle,
    'clipPath': clip_path,
    'ellipse': ellipse,
    'filter': filter_,
    'image': image,
    'line': line,
    'linearGradient': linear_gradient,
    'marker': marker,
    'mask': mask,
    'path': path,
    'pattern': pattern,
    'polyline': polyline,
    'polygon': polygon,
    'radialGradient': radial_gradient,
    'rect': rect,
    'svg': svg,
    'text': text,
    'textPath': text,
    'tspan': text,
    'use': use,
}

NON_FILLABLE_TAGS = [
    'line',
]

PATH_TAGS = frozenset((
    'circle', 'ellipse', 'line', 'path', 'polygon', 'polyline', 'rect'))

INVISIBLE_TAGS = frozenset((
    'clipPath', 'filter', 'linearGradient', 'marker', 'mask', 'pattern',
    'radialGradient', 'symbol'))


def create_function(name):

    def y(self, *args, **kwargs):
        name = sys._getframe().f_code.co_name
        if not self.bitmap_context is None:
            getattr(self.bitmap_context, name)(*args, **kwargs)
        getattr(self.fill_context, name)(*args, **kwargs)
        return getattr(self.path_context, name)(*args, **kwargs)

    if hasattr(y.__code__, 'co_posonlyargcount'):
        y_code = types.CodeType(y.__code__.co_argcount,
                                y.__code__.co_posonlyargcount,
                                y.__code__.co_kwonlyargcount,
                                y.__code__.co_nlocals,
                                y.__code__.co_stacksize,
                                y.__code__.co_flags,
                                y.__code__.co_code,
                                y.__code__.co_consts,
                                y.__code__.co_names,
                                y.__code__.co_varnames,
                                y.__code__.co_filename,
                                name,
                                y.__code__.co_firstlineno,
                                y.__code__.co_lnotab)
    else:
        y_code = types.CodeType(y.__code__.co_argcount,
                                y.__code__.co_kwonlyargcount,
                                y.__code__.co_nlocals,
                                y.__code__.co_stacksize,
                                y.__code__.co_flags,
                                y.__code__.co_code,
                                y.__code__.co_consts,
                                y.__code__.co_names,
                                y.__code__.co_varnames,
                                y.__code__.co_filename,
                                name,
                                y.__code__.co_firstlineno,
                                y.__code__.co_lnotab)
    # print("Cloning %s" % name, file=sys.stderr)
    return types.FunctionType(y_code, y.__globals__, name)


class SuperContext():
    def __init__(self, cairo_path, cairo_bitmap, cairo_fill):
        self.path_context = cairo.Context(cairo_path)
        if not cairo_bitmap is None:
            self.bitmap_context = cairo.Context(cairo_bitmap)
        else:
            self.bitmap_context = None
        self.fill_context = cairo.Context(cairo_fill)
        # We just clone all context's functions to supercontext
        method_list = [func for func in dir(self.path_context) if callable(getattr(self.path_context, func))]
        for method in method_list:
            if method.startswith("_"):
                continue
            setattr(SuperContext, method, create_function(method))
            pass


class Surface(object):
    """Abstract base class for CairoSVG surfaces.

    The ``width`` and ``height`` attributes are in device units (pixels for
    PNG, else points).

    The ``context_width`` and ``context_height`` attributes are in user units
    (i.e. in pixels), they represent the size of the active viewport.

    """

    # Subclasses must either define this or override _create_surface()
    surface_class = None

    @classmethod
    def convert(cls, bytestring=None, **kwargs):
        """Convert a SVG document to the format for this class.

        Specify the input by passing one of these:

        :param bytestring: The SVG source as a byte-string.
        :param file_obj: A file-like object.
        :param url: A filename.

        And the output with:

        :param write_to: The filename of file-like object where to write the
                         output. If None or not provided, return a byte string.

        Only ``bytestring`` can be passed as a positional argument, other
        parameters are keyword-only.

        """
        dpi = kwargs.pop('dpi', 72)
        parent_width = kwargs.pop('parent_width', None)
        parent_height = kwargs.pop('parent_height', None)
        scale = kwargs.pop('scale', 1)
        loop_compensation = kwargs.pop('loop_compensation', 0)
        write_to = kwargs.pop('write_to', None)
        kwargs['bytestring'] = bytestring
        tree = Tree(**kwargs)
        output = write_to or io.BytesIO()
        clip_rect = kwargs.pop('clip_rect', None)
        if clip_rect:
            tree['clip'] = 'rect({})'.format(','.join(map(str, clip_rect)))
        instance = cls(
            tree, [output, io.BytesIO(), None], dpi, None, parent_width, parent_height, scale, mode="fluxclient-parse", loop_compensation=loop_compensation)
        instance.finish()
        # if write_to is None:
        #     return output.getvalue()
        instance.bcontext.sort()
        # instance.bcontext.output()

        return instance.bcontext

    @classmethod
    def divide(cls, bytestring=None, params={}, dpi=72, loop_compensation=0):
        """Divide SVG into layers by colors and bitmap"""
        parent_width = None
        parent_height = None
        scale = params.get('scale', 254 / 72)  # Scaling from inch to pixel
        kwargs = {}
        kwargs['bytestring'] = bytestring
        tree = Tree(**kwargs)
        # The first one should be svg for strokes and fills, second one should be svg for bitmap and gradient, and the third one should be colored bitmap svg
        output = [io.BytesIO(), io.BytesIO(), io.BytesIO()]
        instance = cls(tree, output, dpi, None, parent_width, parent_height, scale,
                       mode="beamstudio", loop_compensation=loop_compensation)
        instance.finish()

        if not instance.bitmap_available:
            # Remove bitmap result if no bitmap are drawn
            output[1] = None

        result = {
            'strokes': output[0],
            'bitmap': output[1],
            'colors': output[2],
            'bitmap_offset': (instance.bitmap_min_x, instance.bitmap_min_y)
        }
        return result

    @classmethod
    def divide_by_layer(cls, bytestring=None, params={}, dpi=72, loop_compensation=0):
        """Divide SVG into layers by colors and bitmap"""
        parent_width = None
        parent_height = None
        scale = params.get('scale', 254 / 72)  # Scaling from inch to pixel
        kwargs = {}
        kwargs['bytestring'] = bytestring
        tree = Tree(**kwargs)
        output = {'nolayer': io.BytesIO(), 'bitmap': io.BytesIO()}
        instance = cls(tree, output, dpi, None, parent_width, parent_height, scale,
                       mode="beamstudio-by-layer", loop_compensation=loop_compensation)
        instance.finish()

        if not instance.bitmap_available:
            # Remove bitmap result if no bitmap are drawn
            output['bitmap'] = None
        output['bitmap_offset'] = (instance.bitmap_min_x, instance.bitmap_min_y)
        return output

    @classmethod
    def divide_path_and_fill(cls, bytestring=None, dpi=72, loop_compensation=0):
        """Divide SVG into layers by colors and path"""
        parent_width = None
        parent_height = None
        scale = 1
        kwargs = {}
        kwargs['bytestring'] = bytestring
        tree = Tree(**kwargs)
        output = [io.BytesIO(), None, io.BytesIO()]
        instance = cls(tree, output, dpi, None, parent_width, parent_height, scale,
                       mode="fluxclient-divide", loop_compensation=loop_compensation)
        instance.finish()
        fill_images = [output[2]] if instance.fill_available else []

        return (output[0], fill_images)

    def __init__(self, tree, outputs, dpi, parent_surface=None,
                 parent_width=None, parent_height=None, scale=1, mode="default", loop_compensation=0):
        """Create the surface from a filename or a file-like object.

        The rendered content is written to ``output`` which can be a filename,
        a file-like object, ``None`` (render in memory but do not write
        anything) or the built-in ``bytes`` as a marker.

        Call the ``.finish()`` method to make sure that the output is
        actually written.

        """
        self.cairo = None
        self.bitmap_available = False
        self.bitmap_min_x = None
        self.bitmap_min_y = None
        self.bitmap_max_x = None
        self.bitmap_max_y = None
        self.fill_available = False
        self.context_width, self.context_height = parent_width, parent_height
        self.cursor_position = [0, 0]
        self.cursor_d_position = [0, 0]
        self.text_path_width = 0
        self.tree_cache = {(tree.url, tree.get('id')): tree}
        self.root_node = tree
        if parent_surface:
            self.markers = parent_surface.markers
            self.gradients = parent_surface.gradients
            self.patterns = parent_surface.patterns
            self.masks = parent_surface.masks
            self.paths = parent_surface.paths
            self.filters = parent_surface.filters
        else:
            self.markers = {}
            self.gradients = {}
            self.patterns = {}
            self.masks = {}
            self.paths = {}
            self.filters = {}
        self._old_parent_node = self.parent_node = None
        self.is_by_layer = mode in ['beamstudio-by-layer']
        self.output = outputs[0] if not self.is_by_layer else outputs['nolayer']
        self.outputs = outputs
        self.dpi = dpi
        self.mode = mode
        self.font_size = size(self, '12pt')
        self.stroke_and_fill = True
        # Avoid recursive href
        self.ref_set = set()
        width, height, viewbox = node_format(self, tree)
        width = width or (6000 / scale)
        height = height or (3750 / scale)
        viewbox = viewbox or (0, 0, width, height)

        print('Cairo start: ' + str(mode), file=sys.stderr)
        print('Cairo Size: ' + str(width) + ' ' + str(height), file=sys.stderr)
        print('Cairo loop compensation: ' + str(loop_compensation), file=sys.stderr)
        width *= scale
        height *= scale
        self.root_width, self.root_height, self.root_scale, self.root_viewbox = width, height, scale, viewbox
        # Actual surface dimensions: may be rounded on raster surfaces types
        self.cairo, self.width, self.height = self._create_surface(self.output,
                                                                   width * self.device_units_per_user_units,
                                                                   height * self.device_units_per_user_units)

        if self.mode.startswith('beamstudio') or self.mode == 'fluxclient-layer-preview':
            self.cairo_bitmap = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(
                width * self.device_units_per_user_units), int(height * self.device_units_per_user_units))
        else:
            self.cairo_bitmap = None

        if self.mode.startswith('fluxclient'):
            self.cairo_fill = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(
                width * self.device_units_per_user_units), int(height * self.device_units_per_user_units))
        else:
            if self.is_by_layer:
                self.cairo_fill = self.cairo
            else:
                self.cairo_fill = cairo.SVGSurface(outputs[2], int(
                    width * self.device_units_per_user_units), int(height * self.device_units_per_user_units))
        self.context = SuperContext(self.cairo, self.cairo_bitmap, self.cairo_fill)
        self.bcontext = beamify.Context()
        self.bcontext.set_compensation_length(loop_compensation)
        # We must scale the context as the surface size is using physical units
        print('The units scale is %f' % self.device_units_per_user_units, file=sys.stderr)
        self.context.scale(self.device_units_per_user_units, self.device_units_per_user_units)
        # self.bitmap_context.scale(self.device_units_per_user_units, self.device_units_per_user_units)
        # self.bcontext.scale(self.device_units_per_user_units, self.device_units_per_user_units)
        # Initial, non-rounded dimensions
        self.set_context_size(
            width, height, viewbox, scale, preserved_ratio(tree))
        self.context.move_to(0, 0)
        self.bcontext.move_to(0, 0)
        self.draw(tree)

    @property
    def points_per_pixel(self):
        """Surface resolution."""
        return 1 / (self.dpi * UNITS['pt'])

    @property
    def device_units_per_user_units(self):
        """Ratio between Cairo device units and user units.

        Device units are points for everything but PNG, and pixels for
        PNG. User units are pixels.

        """
        return self.points_per_pixel

    def _create_surface(self, output, width, height):
        """Create and return ``(cairo_surface, width, height)``."""
        cairo_surface = self.surface_class(output, width, height)
        return cairo_surface, width, height

    def start_layer_surface(self, layer_name):
        if not layer_name:
            return
        self.outputs[layer_name] = io.BytesIO()
        self.layer_surface, _, _ = self._create_surface(self.outputs[layer_name],
                                                        self.root_width * self.device_units_per_user_units,
                                                        self.root_height * self.device_units_per_user_units)
        layer_context = SuperContext(self.layer_surface, self.cairo_bitmap, self.layer_surface)
        self.context = layer_context
        return layer_context

    def end_layer_surface(self):
        self.layer_surface.finish()
        return

    def set_context_size(self, width, height, viewbox, scale, preserved_ratio):
        """Set the Cairo context size, set the SVG viewport size."""
        if viewbox:
            x, y, x_size, y_size = viewbox
            self.context_width, self.context_height = x_size, y_size
            x_ratio, y_ratio = width / x_size, height / y_size
            matrix = cairo.Matrix()
            bmatrix = beamify.Matrix()
            if preserved_ratio and x_ratio > y_ratio:
                matrix.translate((width - x_size * y_ratio) / 2, 0)
                matrix.scale(y_ratio, y_ratio)
                matrix.translate(-x / x_ratio * y_ratio, -y)
                bmatrix.translate((width - x_size * y_ratio) / 2, 0)
                bmatrix.scale(y_ratio, y_ratio)
                bmatrix.translate(-x / x_ratio * y_ratio, -y)
            elif preserved_ratio and x_ratio < y_ratio:
                matrix.translate(0, (height - y_size * x_ratio) / 2)
                matrix.scale(x_ratio, x_ratio)
                matrix.translate(-x, -y / y_ratio * x_ratio)
                bmatrix.translate(0, (height - y_size * x_ratio) / 2)
                bmatrix.scale(x_ratio, x_ratio)
                bmatrix.translate(-x, -y / y_ratio * x_ratio)
            else:
                matrix.scale(x_ratio, y_ratio)
                matrix.translate(-x, -y)
                bmatrix.scale(x_ratio, y_ratio)
                bmatrix.translate(-x, -y)

            apply_matrix_transform(self, matrix)
        else:
            self.context_width, self.context_height = width, height
            if scale != 1:
                matrix = cairo.Matrix()
                matrix.scale(scale, scale)
                bmatrix = beamify.Matrix()
                bmatrix.scale(scale, scale)
                apply_matrix_transform(self, matrix)
    
    def set_dashes(self, dashes, offset):
        dashes = [dash * self.root_scale for dash in dashes]
        offset *= self.root_scale
        self.context.set_dash(dashes, offset)
        self.bcontext.set_dash(dashes, offset)

    def finish(self):
        """Read the surface content."""
        self.cairo.finish()
        if self.cairo_bitmap is not None:
            bitmapIO = self.outputs['bitmap'] if self.is_by_layer else self.outputs[1]
            if self.bitmap_min_x is not None:
                image_data = self.cairo_bitmap.write_to_png()
                image = Image.open(io.BytesIO(image_data))
                image = image.crop((self.bitmap_min_x, self.bitmap_min_y, self.bitmap_max_x, self.bitmap_max_y))
                image.save(bitmapIO, format="PNG")
            else:
                self.cairo_bitmap.write_to_png(bitmapIO)
        if self.mode == "fluxclient-divide":
            self.cairo_fill.write_to_png(self.outputs[2])

    def draw(self, node):
        # print("Drawing ", node.tag, node , file=sys.stderr)
        """Draw ``node`` and its children."""
        # print(node.tag)
        # Do not draw defs
        if node.tag == 'defs':
            parse_all_defs(self, node)
            return
        # Do not draw elements with width or height of 0
        if (('width' in node and size(self, node['width']) == 0) or
           ('height' in node and size(self, node['height']) == 0)):
            return

        # Save context and related attributes
        old_parent_node = self.parent_node
        old_font_size = self.font_size
        old_context_size = self.context_width, self.context_height
        self.parent_node = node
        self.font_size = size(self, node.get('font-size', '12pt'))
        self.context.save()
        self.bcontext.save()

        # Apply transformations
        transform(self, node.get('transform'))

        # Find and prepare opacity, masks and filters
        mask = parse_url(node.get('mask')).fragment
        filter_ = parse_url(node.get('filter')).fragment
        try:
            opacity = float(node.get('opacity', 1))
        except:
            opacity = 1

        if filter_:
            prepare_filter(self, node, filter_)

        if filter_ or mask or (opacity < 1 and node.children):
            self.context.push_group()
            # self.bcontext.push_group()

        # Move to (node.x, node.y)
        self.context.move_to(
            size(self, node.get('x'), 'x'),
            size(self, node.get('y'), 'y'))

        self.bcontext.move_to(
            size(self, node.get('x'), 'x'),
            size(self, node.get('y'), 'y'))

        # print("Move ", size(self, node.get('x'), 'x'), size(self, node.get('y'), 'y'), file=sys.stderr)

        # Set node's drawing informations if the ``node.tag`` method exists
        line_cap = node.get('stroke-linecap')
        if line_cap == 'square':
            self.context.set_line_cap(cairo.LINE_CAP_SQUARE)
        if line_cap == 'round':
            self.context.set_line_cap(cairo.LINE_CAP_ROUND)

        join_cap = node.get('stroke-linejoin')
        if join_cap == 'round':
            self.context.set_line_join(cairo.LINE_JOIN_ROUND)
        if join_cap == 'bevel':
            self.context.set_line_join(cairo.LINE_JOIN_BEVEL)

        dash_array = normalize(node.get('stroke-dasharray', '')).split()
        if dash_array:
            dashes = [size(self, dash) for dash in dash_array]
            if sum(dashes):
                offset = size(self, node.get('stroke-dashoffset'))
                self.set_dashes(dashes, offset)

        miter_limit = float(node.get('stroke-miterlimit', 4))
        self.context.set_miter_limit(miter_limit)

        # Clip
        rect_values = clip_rect(node.get('clip'))
        if len(rect_values) == 4:
            top = size(self, rect_values[0], 'y')
            right = size(self, rect_values[1], 'x')
            bottom = size(self, rect_values[2], 'y')
            left = size(self, rect_values[3], 'x')
            x = size(self, node.get('x'), 'x')
            y = size(self, node.get('y'), 'y')
            width = size(self, node.get('width'), 'x')
            height = size(self, node.get('height'), 'y')
            self.context.save()
            self.context.translate(x, y)
            self.context.rectangle(
                left, top, width - left - right, height - top - bottom)
            self.context.restore()
            self.context.clip()

            self.bcontext.save()
            self.bcontext.translate(x, y)
            self.bcontext.rectangle(
                left, top, width - left - right, height - top - bottom)
            self.bcontext.restore()
            self.bcontext.clip()

        clip_path = parse_url(node.get('clip-path')).fragment
        if clip_path:
            path = self.paths.get(clip_path)
            if path:
                self.context.save()
                self.bcontext.save()
                if path.get('clipPathUnits') == 'objectBoundingBox':
                    x = size(self, node.get('x'), 'x')
                    y = size(self, node.get('y'), 'y')
                    width = size(self, node.get('width'), 'x')
                    height = size(self, node.get('height'), 'y')
                    self.context.translate(x, y)
                    self.context.scale(width, height)
                    self.bcontext.translate(x, y)
                    self.bcontext.scale(width, height)
                path.tag = 'g'
                self.stroke_and_fill = False
                self.draw(path)
                self.stroke_and_fill = True
                self.context.restore()
                self.bcontext.restore()
                # TODO: fill rules are not handled by cairo for clips
                # if node.get('clip-rule') == 'evenodd':
                #     self.context.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
                self.context.clip()
                self.context.set_fill_rule(cairo.FILL_RULE_WINDING)
                self.bcontext.clip()
                # self.bcontext.set_fill_rule(cairo.FILL_RULE_WINDING)

        # print("Parsing ", node.tag, node, file=sys.stderr)
        # Only draw known tags
        if node.tag in TAGS:
            try:
                TAGS[node.tag](self, node)
            except PointError:
                # Error in point parsing, do nothing
                pass
            except Exception:
                logger.error('When drawing {}, {}, Exception Occured:\n{}'.format(
                    node.tag, node, traceback.format_exc()))
                logger.error('Ignore this node.')

        # Get stroke and fill opacity
        stroke_opacity = node.get('stroke-opacity', 1)
        try:
            stroke_opacity = float(stroke_opacity)
        except:
            stroke_opacity = 1
        fill_opacity = node.get('fill-opacity', 1)
        try:
            fill_opacity = float(fill_opacity)
        except:
            fill_opacity = 1

        if opacity < 1 and not node.children:
            stroke_opacity *= opacity
            fill_opacity *= opacity

        # Manage display and visibility
        display = node.get('display', 'inline') != 'none'
        visible = display and (node.get('visibility', 'visible') != 'hidden')

        # Set font rendering properties
        self.context.set_antialias(SHAPE_ANTIALIAS.get(
            node.get('shape-rendering'), cairo.ANTIALIAS_DEFAULT))

        font_options = self.context.get_font_options()
        font_options.set_antialias(TEXT_ANTIALIAS.get(
            node.get('text-rendering'), cairo.ANTIALIAS_DEFAULT))
        font_options.set_hint_style(TEXT_HINT_STYLE.get(
            node.get('text-rendering'), cairo.HINT_STYLE_DEFAULT))
        font_options.set_hint_metrics(TEXT_HINT_METRICS.get(
            node.get('text-rendering'), cairo.HINT_METRICS_DEFAULT))
        self.context.set_font_options(font_options)

        # Fill and stroke
        if self.stroke_and_fill and visible and node.tag in TAGS:
            # Fill
            node_filled = False
            is_gradient_or_pattern = False
            self.context.save()
            self.bcontext.save()
            fill_name = node.get('fill', 'black')
            if fill_name == 'none' or fill_name == '#FFF' or fill_name == '#FFFFFF' or node.tag in NON_FILLABLE_TAGS:
                fill_opacity = 0
            paint_source, paint_color = paint(fill_name)
            fill_paint_color = paint_color
            if not gradient_or_pattern(self, node, paint_source):
                if node.get('fill-rule') == 'evenodd':
                    self.context.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
                self.context.set_source_rgba(*color(paint_color, fill_opacity))
                self.bcontext.set_source_rgba(*color(paint_color, fill_opacity))
            else:
                is_gradient_or_pattern = True
            # self.context.context.fill_preserve()
            if not is_gradient_or_pattern and (color(paint_color, fill_opacity)[3] == 0 or fill_opacity == 0):
                pass
            else:
                if not node.tag in ['svg', 'clipPath']:
                    self.fill_available = True
                node_filled = True
                self.context.fill_context.fill_preserve()
            self.context.restore()
            self.bcontext.restore()

            # Stroke
            self.context.save()
            self.bcontext.save()
            line_width = float(size(self, node.get('stroke-width', '1')))
            paint_source, paint_color = paint(node.get('stroke'))

            if not gradient_or_pattern(self, node, paint_source):
                self.context.set_source_rgba(*color(paint_color, stroke_opacity))

            if self.mode.startswith("fluxclient"):
                # if the element is fill only, no strokes:
                if color(paint_color, stroke_opacity)[3] == 0:
                    r, g, b, a = color(fill_paint_color, 0)
                    # set context stroke color = opacity
                    self.context.set_source_rgba(r, g, b, 0)
                    self.bcontext.hide_path()
                # if the element is filled by linewidth
                elif line_width > 1:
                    r, g, b, a = color(fill_paint_color, 1)
                    # set context stroke color = fill color, and reproduce line
                    self.context.set_source_rgba(r, g, b, 1)
                    self.bcontext.hide_path()
                # Stroke only when not filled
                if not node_filled:
                    self.context.path_context.set_line_width(line_width)
                else:
                    self.context.path_context.set_line_width(0)

                # write path to main context, and don't writes to fill_context
                self.context.fill_context.set_line_width(0)
                self.context.path_context.stroke()
                self.context.fill_context.stroke()

                # write path to laser cutting canvas
                self.bcontext.stroke()
            else:
                self.context.set_line_width(line_width)
                if self.is_by_layer:
                    self.context.set_line_width(1)
                if not fill_paint_color == paint_color:
                    # do not stroke fill context
                    self.context.path_context.stroke()
                    self.context.fill_context.set_line_width(0)
                    self.context.fill_context.stroke()
                else:
                    self.context.path_context.stroke()
                    self.context.fill_context.stroke()

            self.context.restore()
            self.bcontext.restore()

        elif not visible:
            self.context.new_path()
            self.bcontext.new_path()

        # Draw path markers
        draw_markers(self, node)

        # Draw children
        if display and node.tag not in INVISIBLE_TAGS and node.tag not in PATH_TAGS:
            for child in node.children:
                layer_name = None
                root_context = None
                if self.is_by_layer and node == self.root_node and child.tag == 'g':
                    layer_name = get_layer_name(child)
                    if layer_name in self.outputs:
                        i = 2
                        while '{}_{}'.format(layer_name, i) in self.outputs:
                            i += 1
                if layer_name and layer_name != 'nolayer':
                    logger.info('Create new surface & context for {}'.format(layer_name))
                    root_context = self.context
                    layer_context = self.start_layer_surface(layer_name)
                    self.context = layer_context
                    self.context.scale(self.device_units_per_user_units, self.device_units_per_user_units)
                    self.set_context_size(
                        self.root_width, self.root_height, self.root_viewbox, self.root_scale, preserved_ratio(self.root_node))
                    self.context.move_to(0, 0)
                self.draw(child)
                if layer_name and layer_name != 'nolayer':
                    logger.info('{} Ended'.format(layer_name))
                    self.end_layer_surface()
                    self.context = root_context

        # Apply filter, mask and opacity
        if filter_ or mask or (opacity < 1 and node.children):
            self.context.pop_group_to_source()
            if filter_:
                apply_filter_before_painting(self, node, filter_)
            if mask in self.masks:
                paint_mask(self, node, mask, opacity)
            else:
                self.context.paint_with_alpha(opacity)
            if filter_:
                apply_filter_after_painting(self, node, filter_)

        # Clean cursor's position after 'text' tags
        if node.tag == 'text':
            self.cursor_position = [0, 0]
            self.cursor_d_position = [0, 0]
            self.text_path_width = 0

        self.context.restore()
        self.bcontext.restore()
        self.parent_node = old_parent_node
        self.font_size = old_font_size
        self.context_width, self.context_height = old_context_size


class PDFSurface(Surface):
    """A surface that writes in PDF format."""
    surface_class = cairo.PDFSurface


class PSSurface(Surface):
    """A surface that writes in PostScript format."""
    surface_class = cairo.PSSurface


class PNGSurface(Surface):
    """A surface that writes in PNG format."""

    def _create_surface(self, output, width, height):
        """Create and return ``(cairo_surface, width, height)``."""
        width = int(width)
        height = int(height)
        cairo_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        return cairo_surface, width, height

    def finish(self):
        """Read the PNG surface content."""
        if self.output is not None:
            self.cairo.write_to_png(self.outputs[0])
        return super().finish()


class SVGSurface(Surface):
    """A surface that writes in SVG format.

    It may seem pointless to render SVG to SVG, but this can be used
    with ``output=None`` to get a vector-based single page cairo surface.

    """
    surface_class = cairo.SVGSurface


class ImageSurface(Surface):
    # Surface to generate image for printing, Could be write in PNGSurface if it is not used by other functions
    surface_class = cairo.ImageSurface

    def _create_surface(self, output, width, height):
        width, height= int(width), int(height)
        cairo_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        return cairo_surface, width, height
    
    @classmethod
    def divide_path_and_fill(cls, bytestring=None, dpi=72, loop_compensation=0):
        kwargs = {}
        kwargs['bytestring'] = bytestring
        tree = Tree(**kwargs)
        output = [None, None, None]
        instance = cls(tree, output, dpi, None, parent_width=None, parent_height=None, scale=1,
                       mode='fluxclient-divide', loop_compensation=loop_compensation)
        data = instance.finish()
        return None, data
    
    @classmethod
    def generate_layer_preview(cls, bytestring=None, dpi=72, layer_color='#333333'):
        kwargs = {}
        kwargs['bytestring'] = bytestring
        tree = Tree(**kwargs)
        output = [None, None, None]
        instance = cls(tree, output, dpi, None, parent_width=None, parent_height=None, scale=1,
                       mode='fluxclient-layer-preview', loop_compensation=0)
        return instance.layer_preview_finish(layer_color)

    # ref: https://stackoverflow.com/questions/29332424/changing-colour-of-an-image
    # adjust a little bit to match the color in beam studio
    def tint_image(self, img, color='#ffffff'):
        tr, tg, tb = ImageColor.getrgb(color)
        # create look-up tables to map luminosity to adjusted tint
        # (using floating-point math only to compute table)
        luts = (tuple(map(lambda lr: lr + tr, range(256))) +
                tuple(map(lambda lg: lg + tg, range(256))) +
                tuple(map(lambda lb: lb + tb, range(256))))
        l = ImageOps.grayscale(img)  # 8-bit luminosity version of whole image
        if Image.getmodebands(img.mode) < 4:
            merge_args = (img.mode, (l, l, l))  # for RGB verion of grayscale
        else:  # include copy of src image's alpha layer
            a = Image.new('L', img.size)
            a.putdata(img.getdata(3))
            merge_args = (img.mode, (l, l, l, a))  # for RGBA verion of grayscale
            luts += tuple(range(256))  # for 1:1 mapping of copied alpha values
        return Image.merge(*merge_args).point(luts)

    def layer_preview_finish(self, layer_color):
        path_data = io.BytesIO()
        self.cairo.write_to_png(path_data)
        base_image = Image.open(path_data)
        if self.fill_available:
            fill_data = io.BytesIO()
            self.cairo_fill.write_to_png(fill_data)
            fill_image = Image.open(fill_data)
            base_image = Image.alpha_composite(base_image, fill_image)
        if self.cairo_bitmap is not None and self.bitmap_available:
            bitmap_data = io.BytesIO()
            if self.bitmap_min_x is not None:
                image_data = self.cairo_bitmap.write_to_png()
                bitmap_image = Image.open(io.BytesIO(image_data))
                bitmap_image = bitmap_image.crop((self.bitmap_min_x, self.bitmap_min_y, self.bitmap_max_x, self.bitmap_max_y))
                bitmap_image.save(bitmap_data, format='PNG')
            else:
                self.cairo_bitmap.write_to_png(bitmap_data)
            bitmap_image = Image.open(bitmap_data)
            bitmap_image = self.tint_image(bitmap_image, layer_color)
            x = round(self.bitmap_min_x) if self.bitmap_min_x is not None else 0
            y = round(self.bitmap_min_y) if self.bitmap_min_y is not None else 0
            base_image.paste(bitmap_image, (x, y), bitmap_image)
        width, height = base_image.size
        MAX_WIDTH = 500
        if width > MAX_WIDTH:
            ratio = MAX_WIDTH / width
            new_height = int(height * ratio)
            base_image = base_image.resize((MAX_WIDTH, new_height), Image.LANCZOS)
        output = io.BytesIO()
        base_image.save(output, format='PNG')
        return output

    
    def finish(self):
        outputs = [io.BytesIO()]
        self.cairo.write_to_png(outputs[0])
        if self.fill_available:
            output = io.BytesIO()
            self.cairo_fill.write_to_png(output)
            outputs.append(output)
        return outputs
