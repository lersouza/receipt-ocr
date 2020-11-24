import argparse
import json
import os
import string
import torch

from functools import partial
from itertools import product, repeat, chain
from pathlib import Path
from random import choice, choices, randint, random
from typing import List, Tuple

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from multiprocessing import Pool

BACK_COLORS = ['#FFFFFF', '#F3C76E', '#B6BAB6', '#E7DEBE', '#ECE3D6']
FONT_COLORS = ['#000000', '#070138']

KNOWN_CHARACTERS = string.digits + string.ascii_letters
CHAR_TABLE = {c: i + 1 for i, c in enumerate(KNOWN_CHARACTERS)}


def list_available_fonts(fonts_path: str,
                         sizes_range: range = None,
                         return_type: str = 'image_font'):
    """
    Lists a combination of (font, color) for all available fonts in `font_path`
    and sizes in `sizes_range`.

    If `sizes_range` is not specified, we assume a range from 16 to 22.

    It is also possible to control how the data is returned:
    - If return_type is equals to `'image_font'` (default), the resulting list will
      contain objects of type `ImageFont`.

    - Otherwise, it will be a list of strings in the format
      `'font name, font size'` will be returned.
    """
    available_fonts = list(Path(fonts_path).glob('**/*.ttf'))
    available_sizes = range(16, 22) if sizes_range is None else sizes_range

    all_fonts = product(available_fonts, available_sizes)

    def create_font(font_name: str, font_size: int):
        if return_type == 'image_font':
            return ImageFont.truetype(str(font_name), int(font_size))

        return f'{font_name}, {font_size}'

    return [create_font(f, s) for f, s in all_fonts]


def fit_text(tokens: List[str],
             draw: ImageDraw.Draw,
             font: ImageFont,
             max_width: int,
             max_height: int,
             offset_x: int = 0,
             offset_y: int = 0,
             line_spacing: int = 0,
             max_height_threshold: int = 20):
    """
    Fits the given `tokens` array into many lines based on
    `draw`, `font` and desired `max_width` and `max_height`.

    This function returns a tuple:
    - `[0]`: Is a list of lines in the images. For each line, a list of tokens \
             is provided (as a result of `line.split()`).
    - `[1]`: A list of width for each line.
    - `[2]`: The initial position where the tokens do not fit anymore.

    A `offset_x` and `offset_y` can be provided.
    Also, an additional space between lines can be provided in `line_spacing`.

    PS: The calculations are based on estimations returned by `draw.textsize()`.
    """
    lines = [[]]
    sizes = [[offset_x, 0]]
    unfit = None

    total_height = offset_y

    for i, token in enumerate(tokens):
        # we include spaces for size calculation, since we are
        # running through tokens
        text_to_draw = token + ' '
        w, h = draw.textsize(text_to_draw, font=font)

        estimated_height = (total_height + h + line_spacing + \
                            max_height_threshold)

        if sizes[-1][0] + w > max_width:
            lines.append([])
            sizes.append([offset_x, h])

            total_height = sum([i[1] for i in sizes]) + offset_y

        if estimated_height >= max_height:  # put some bottom margin
            unfit = i
            break

        lines[-1].append(token)
        sizes[-1][0] += w
        sizes[-1][1] = max(h, sizes[-1][1])

    return lines, sizes, unfit


def generate_random_string(min_size: int = 3, max_size: int = 10):
    """
    Generates a random string of size between `min_size` and `max_size`.
    """
    return ''.join(choices(KNOWN_CHARACTERS, k=randint(min_size, max_size)))


def create_image(size: Tuple[int, int] = (300, 600)) -> Image:
    """
    Creates an image of of size `size`, in RGB scheme.
    The image's background color is chosen at random from `BACK_COLORS`.
    """
    backround_color = choice(BACK_COLORS)
    image = Image.new('RGB', size, color=backround_color)

    return image


def replaceit_or_leaveit(original):
    """
    Replaces an `original` token:
    - 95% of the time with itself
    - 5% of the time with a random string.
    """
    if random() > 0.95:
        return generate_random_string()

    return original


def generate_image_from_text(size: Tuple[int, int], fonts_dir: str,
                             output_dir: str, image_data: Tuple[int, str]):
    """
    Generate an image from a piece of text in `image_data`.
    - `size`: A tuple indicating the size of target image.
    - `fonts_dir`: A directory with `.ttf` font files.
    - `output_dir`: The directory where to output files.
    - `image_data[0]`: the index associated with the sample in the dataset.
    - `image_data[1]`: the text to be used in the target image.

    Images are generated in 300x600, RGB format.
    An arbitrary font (located in ./fonts dir) will be choosen for the image.

    Image will be output to ./output.
    """
    size = size or (300, 600)
    fonts_dir = fonts_dir or './fonts'
    output_dir = output_dir or './output'

    image_index, text = image_data

    fonts = list_available_fonts(fonts_dir)
    image_name = os.path.join(output_dir, f'{image_index}.png')
    image = create_image(size)
    draw = ImageDraw.Draw(image)
    font = choice(fonts)
    forecolor = choice(FONT_COLORS)
    line_spacing = randint(0, 10)

    offset_x = randint(0, size[0] // 2)
    offset_y = randint(0, size[1] // 2)

    original = text.split()
    tokens = [replaceit_or_leaveit(t) for t in original]

    lines, sizes, unfit = fit_text(tokens,
                                   draw,
                                   font,
                                   size[0],
                                   size[1],
                                   offset_x=offset_x,
                                   offset_y=offset_y,
                                   line_spacing=line_spacing)

    original, tokens = original[:unfit], tokens[:unfit]
    y = offset_y

    mask = torch.zeros(*size)

    for line, text_size in zip(lines, sizes):
        x = offset_x
        full_text = ' '.join(line)

        for char in full_text:
            # Here, we generate the CharGrid Mask for the file
            char_size = draw.textsize(char, font)
            mask[x:x + char_size[0],
                 y:y + char_size[1]] = CHAR_TABLE.get(char, 0)
            x += char_size[0]

        # Then we generate the image itself, line by line
        draw.text((offset_x, y), full_text, font=font, fill=forecolor)
        y += text_size[1] + line_spacing

    image.save(image_name)
    torch.save(mask, image_name + '.mask')

    with open(f'{image_name}.json', 'w+') as labels:
        json.dump({
            'original': ' '.join(original),
            'image': ' '.join(tokens)
        }, labels)


def main(args):
    """ Main routine to generate all images. """

    os.makedirs(args.output_dir, exist_ok=True)

    wiki = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    wiki = wiki.filter(lambda example: len(example['text']) >= 128)
    wiki_data = wiki[:args.wiki_samples]['text']

    target_size = (args.width, args.height)

    generator = partial(generate_image_from_text, target_size, args.fonts_dir,
                        args.output_dir)

    with Pool(processes=os.cpu_count()) as p:
        text_collection = chain.from_iterable(repeat(wiki_data, args.scale))
        it = p.imap_unordered(generator, enumerate(text_collection))

        with tqdm(total=len(wiki_data) * args.scale) as pbar:
            for _ in it:
                pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir',
                        type=str,
                        default='./output',
                        help='Output dir for generating images')

    parser.add_argument('--fonts_dir',
                        type=str,
                        default='./fonts',
                        help='Directory with .ttf font files')

    parser.add_argument('--width',
                        type=int,
                        default=300,
                        help='Width of images to be generated')

    parser.add_argument('--height',
                        type=int,
                        default=600,
                        help='Height of images to be generated')

    parser.add_argument('--wiki_samples',
                        type=int,
                        default=32_000,
                        help='Number os samples to use from WikiText')

    parser.add_argument(
        '--scale',
        type=int,
        default=3,
        help='Number of times to use each sample from WikiText')

    args = parser.parse_args()
    main(args)
