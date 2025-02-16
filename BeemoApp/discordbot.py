import logging
import discord
from discord.ext import commands
import asyncio
from tensorflow.keras.models import load_model
from BNBpredictor import collect_new_data_and_labels as collect_bnb, retrain_model as retrain_bnb_model, save_results_to_google_sheets as save_bnb_results
from QNQpredictor import collect_new_data_and_labels as collect_qnq, retrain_qnq_model, QNQ_save_results_to_google_sheets as save_qnq_results
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Update the image path to be relative
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SavedIMG', 'last_processed_image.png')

# Get configuration from environment variables
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', '0'))  # Default to 0 if not set

# Initialize Discord intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Explicitly enable message content intent

# Initialize bot with intents
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    
@bot.command(name='hello')
async def hello_beemo(ctx):
    await ctx.send('Hello! I am Beemo, your friendly bee monitoring bot! 🐝')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    await bot.process_commands(message)

@bot.event
async def on_ready():
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send("Beemo is now online! 🐝")

@bot.command(name='bnbincorrect')
async def bnbincorrect(ctx):
    await trigger_retraining('bnb', ctx)
    await ctx.send('BNB model retraining triggered!')

@bot.command(name='qnqincorrect')
async def qnqincorrect(ctx):
    await trigger_retraining('qnq', ctx)
    await ctx.send('QNQ model retraining triggered!')

async def trigger_retraining(model_type, ctx):
    try:
        if model_type == 'bnb':
            # Collect new data and retrain BNB model
            new_data = collect_bnb()
            if new_data:
                retrain_bnb_model()
                save_bnb_results()
                await ctx.send('BNB model has been retrained successfully!')
            else:
                await ctx.send('No new data available for BNB model retraining.')
        elif model_type == 'qnq':
            # Collect new data and retrain QNQ model
            new_data = collect_qnq()
            if new_data:
                retrain_qnq_model()
                save_qnq_results()
                await ctx.send('QNQ model has been retrained successfully!')
            else:
                await ctx.send('No new data available for QNQ model retraining.')
    except Exception as e:
        await ctx.send(f'An error occurred during {model_type.upper()} model retraining: {str(e)}')

async def ask_true_label(ctx):
    try:
        # Send message asking for true label
        await ctx.send("What is the true label for this prediction? Please respond with 'yes' or 'no'.")
        
        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel

        # Wait for a response
        msg = await bot.wait_for('message', check=check, timeout=30.0)
        
        return msg.content.lower()
    except asyncio.TimeoutError:
        await ctx.send('No response received within 30 seconds.')
        return None

# Function to send a message to a Discord channel
async def send_discord_message(message, image_path=None):
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        if image_path and os.path.exists(image_path):
            await channel.send(message, file=discord.File(image_path))
        else:
            await channel.send(message)

def run_discord_bot():
    bot.run(TOKEN)

if __name__ == "__main__":
    asyncio.run(run_discord_bot())
