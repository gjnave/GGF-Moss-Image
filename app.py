"""GGF Image Maker: purpose-built MOSS workflow, app-owned one-shot GPU workers."""
from __future__ import annotations
import argparse
import atexit
import base64
import html
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
import gradio as gr
from access import MODES, load as load_access, save as save_access, verify as verify_access
from model_store import APP, MODEL_DIR, VARIANTS, verify, download, selected_model, select_model, model_files
from validation import validate
from caption_output import display_caption
from dimensions import SHAPES, DEFAULT_SHAPE, DEFAULT_SIZE, MIN_EDGE, MAX_EDGE, preview_dimensions, validate_dimensions
from prompt_example import DEFAULT_PROMPT, layout_prompt

JOBS = APP/'jobs'
OUTPUTS = APP/'outputs'
WORKSPACES = JOBS/'workspaces'
LOCK = threading.RLock()
FILE_LOCK = threading.RLock()
ACTIVE = None
PROCESS = None
SERVERS = []
STOPPING = False
LOCAL_URL = ''
PHONE_LINK = 'Off. Enable phone access in Settings and restart.'
DOWNLOAD = None
DOWNLOAD_NOTE = ''
CONTROL_NAMES = ['prompt','width','height','negative','steps','cfg','seed','enhance','slg',
                 'enhance_seed','lora','lora_strength','count']
DEFAULTS = [DEFAULT_PROMPT,DEFAULT_SIZE,DEFAULT_SIZE,'worst quality, bad anatomy, watermark, jpeg artifact',25,2.5,0,True,False,0,'None',1.0,1]
CSS = (APP/'ui.css').read_text(encoding='utf-8') + '''
#ggf-hero img {float:left;width:68px;margin:0 18px 10px 0;}
#ggf-status {min-height:54px;} #ggf-status progress {width:180px;}
#ggf-footer {padding:12px;} .gradio-container {color-scheme:dark;}
@media(max-width:600px){#ggf-hero img{width:42px;margin-right:12px;}}
'''
SHORTCUT = '''(token)=>{
 if(!window.__ggfMossShortcut){window.__ggfMossShortcut=true;
 document.addEventListener('keydown',e=>{if(!e.ctrlKey||e.key!=='Enter'||e.repeat||e.isComposing||document.hidden)return;
 const b=document.querySelector('#ggf-generate button,button#ggf-generate');
 if(b&&!b.disabled&&b.offsetParent!==null){e.preventDefault();b.click();}});}return token;}'''


def token_path(token):
    if not isinstance(token,str) or not re.fullmatch('[0-9a-f]{32}',token):
        raise ValueError('Refresh this page to initialize its workspace.')
    return WORKSPACES/token/'state.json'


def load_state(token):
    try:
        with FILE_LOCK:
            return json.loads(token_path(token).read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return {}


def save_state(token, **fields):
    with FILE_LOCK:
        path=token_path(token)
        path.parent.mkdir(parents=True,exist_ok=True)
        data=load_state(token)
        data.update(fields)
        stage=path.with_name('state-'+uuid.uuid4().hex+'.tmp')
        stage.write_text(json.dumps(data,indent=2),encoding='utf-8')
        os.replace(stage,path)


def saved_results(data):
    return [p for p in data.get('results',[]) if Path(p).is_file()
            and Path(p).resolve().is_relative_to(OUTPUTS.resolve())]


def save_controls(token,*values):
    save_state(token,controls=dict(zip(CONTROL_NAMES,values)))


def model_note():
    ready,note=verify()
    return ('✓ ' if ready else 'Open Settings: ')+note


def model_panel():
    with LOCK:
        downloading = DOWNLOAD is not None
        busy = ACTIVE is not None
        message = DOWNLOAD_NOTE
    variant = selected_model()
    if downloading:
        body = '<div class="model-missing"><progress></progress> ' + html.escape(message) + '</div>'
    else:
        ready, note = verify(variant=variant)
        css = 'model-ready' if ready else 'model-missing'
        label = '✓ Installed' if ready else 'Download a model before generating'
        body = f'<div class="{css}">{label} · {html.escape(VARIANTS[variant]["label"])}<br>{html.escape(note)}</div>'
        if message:
            body += '<p>' + html.escape(message) + '</p>'
    return body, gr.update(interactive=not downloading and not busy), gr.update(interactive=downloading)


def change_model(variant):
    global DOWNLOAD_NOTE
    with LOCK:
        if ACTIVE or DOWNLOAD:
            return gr.update(value=selected_model()), 'Stop the current generation or download before changing models.'
        select_model(variant)
        DOWNLOAD_NOTE = ''
    ready, _ = verify()
    return gr.update(value=variant), ('Selected model is installed and ready.' if ready else
        'Now click Install / resume below. The app will not substitute another model.')


def install_models():
    global DOWNLOAD, DOWNLOAD_NOTE
    with LOCK:
        if ACTIVE or DOWNLOAD or STOPPING:
            return model_panel()
        variant = selected_model()
        job = {'variant':variant, 'cancel':threading.Event()}
        DOWNLOAD = job
        DOWNLOAD_NOTE = 'Starting download / checking existing files…'
    def update(message):
        global DOWNLOAD_NOTE
        with LOCK:
            DOWNLOAD_NOTE = message
    def run():
        global DOWNLOAD
        try:
            download(variant, progress=update, cancel=job['cancel'])
            update('✓ Installed. Your selected model is ready — open Create to make an image.')
        except Exception as error:
            update(str(error))
        finally:
            with LOCK:
                DOWNLOAD = None
    threading.Thread(target=run, daemon=True).start()
    return model_panel()


def stop_download():
    with LOCK:
        if DOWNLOAD:
            DOWNLOAD['cancel'].set()
    return 'Stopping download. Partial files are kept so you can resume.'


def initial_tab():
    return 'create' if verify()[0] else 'settings'


def apply_shape(shape):
    return SHAPES.get(shape) or (gr.update(),gr.update())


def show_saved(token):
    with LOCK:
        if ACTIVE:
            return refresh(token)
        images=sorted(OUTPUTS.glob('*.png'),key=lambda p:p.stat().st_mtime,reverse=True)[:50]
        expanded=''
        if images:
            from PIL import Image
            try:
                with Image.open(images[0]) as picture:
                    expanded=display_caption(json.loads(picture.info.get('GGF_MOSS','{}')).get('expanded_prompt',''))
            except (OSError,ValueError):
                pass
        save_state(token,results=[str(p) for p in images],expanded=expanded,running=False,
                   status=f'Showing {len(images)} saved image(s). All files remain in the outputs folder.')
    return refresh(token)


def select_saved(token, event: gr.SelectData):
    images=saved_results(load_state(token))
    if not isinstance(event.index,int) or not 0 <= event.index < len(images):
        return gr.update()
    from PIL import Image
    try:
        with Image.open(images[event.index]) as picture:
            caption=display_caption(json.loads(picture.info.get('GGF_MOSS','{}')).get('expanded_prompt',''))
        save_state(token,expanded=caption)
        return caption
    except (OSError,ValueError):
        return gr.update()


def status_html(data,busy=False):
    note=html.escape(data.get('status','Ready.'))
    if busy:
        elapsed=int(max(0,time.time()-data.get('started',time.time())))
        if data.get('step'):
            bar=f"<progress value='{int(data['step'])}' max='{int(data['total'])}'></progress>"
        else:
            bar='<progress></progress>'
        return f'{bar} {note} · {elapsed}s elapsed'
    return note


def refresh(token):
    data=load_state(token)
    with LOCK:
        busy=ACTIVE is not None
        own=busy and ACTIVE['token']==token
    display=data if own or not busy else {'status':'Another browser is generating on this PC.'}
    caption=display_caption(data.get('expanded',''))
    if not busy and data.get('expanded') and not caption:
        display={**display,'status':'Previous result used an invalid caption-model reply. Ready for a new request.'}
    if not busy and data.get('running'):
        display={**data,'status':'Previous run interrupted. Saved images remain available.'}
    with LOCK:
        downloading = DOWNLOAD is not None
    ready = False if downloading else verify()[0]
    label='Generating…' if busy else 'Generate image · Ctrl+Enter'
    return (saved_results(data),status_html(display,busy),caption,
            gr.update(interactive=not busy and ready,value=label),gr.update(interactive=not busy and ready),
            f'PC: {LOCAL_URL} · Phone: {PHONE_LINK}')


def restore(token):
    if not isinstance(token,str) or not re.fullmatch('[0-9a-f]{32}',token):
        token=uuid.uuid4().hex
    data=load_state(token)
    values=[data.get('controls',{}).get(n,d) for n,d in zip(CONTROL_NAMES,DEFAULTS)]
    if values[10] not in loras():
        values[10]='None'
    try:
        validate_dimensions(values[1],values[2])
    except (ValueError,TypeError,OverflowError):
        values[1:3]=[DEFAULT_SIZE,DEFAULT_SIZE]
    restored_shape=next((name for name,size in SHAPES.items() if size == tuple(values[1:3])), 'Custom')
    return (token,*values,*refresh(token),gr.update(selected=initial_tab()),restored_shape)


def run_job(job):
    global ACTIVE,PROCESS
    token=job['token']
    try:
        for index in range(job['count']):
            if job['cancel'].is_set():
                break
            request=dict(job['request'])
            request['seed']=(request['seed']+index) % (2**53)
            result=OUTPUTS/f'GGF-Image-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:10]}.png'
            request['output']=str(result)
            request_path=job['folder']/f'request-{index+1}.json'
            request_path.write_text(json.dumps(request,indent=2),encoding='utf-8')
            save_state(token,status=f'Image {index+1} of {job["count"]}: loading models…',step=0)
            with (job['folder']/f'worker-{index+1}.log').open('w',encoding='utf-8') as log:
                with LOCK:
                    if job['cancel'].is_set():
                        break
                    process=subprocess.Popen([sys.executable,'-u',str(APP/'worker.py'),str(request_path)],
                        cwd=APP,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',
                        errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    PROCESS=process
                error=None
                complete=False
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        event=json.loads(line)
                    except ValueError:
                        continue
                    fields={}
                    if 'status' in event:
                        fields['status']=f'Image {index+1} of {job["count"]}: '+event['status']
                    if 'prompt' in event:
                        fields['expanded']=display_caption(event['prompt'])
                    if 'step' in event:
                        fields.update(step=event['step'],total=event['total'])
                    if 'error' in event:
                        error=event['error']
                        fields['expanded']=''
                    if 'complete' in event:
                        complete=result.is_file()
                        if complete:
                            fields['results']=saved_results(load_state(token))+[str(result)]
                    if fields:
                        save_state(token,**fields)
                process.wait()
                with LOCK:
                    if PROCESS is process:
                        PROCESS=None
                if job['cancel'].is_set():
                    break
                if process.returncode or not complete:
                    raise RuntimeError(error or f'Worker stopped. See {job["folder"].name}/worker-{index+1}.log')
        results=saved_results(load_state(token))
        note=('Stopped. ' if job['cancel'].is_set() else 'Complete. ')+f'{len(results)} image(s) saved. GPU worker released.'
        save_state(token,status=note,running=False,step=0)
    except Exception as error:
        save_state(token,status='Generation failed: '+str(error),running=False,step=0)
    finally:
        with LOCK:
            if PROCESS is not None and PROCESS.poll() is None:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(PROCESS.pid),'/T','/F'],
                        capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                else:
                    PROCESS.terminate()
            ACTIVE=None
            PROCESS=None


def submit(token,*args,preview=False):
    global ACTIVE
    try:
        values=dict(zip(CONTROL_NAMES,args))
        request=validate(values)
        request['model']=selected_model()
        count=int(values['count'])
        if not 1 <= count <= 16 or count != float(values['count']):
            raise ValueError('Choose 1–16 images. Each is generated sequentially.')
        if request['lora'] not in loras():
            raise ValueError('Choose a LoRA from this app or None.')
        if preview:
            request['layout_source']=[request['width'],request['height']]
            request['width'],request['height']=preview_dimensions(request['width'],request['height'])
            count=1
        layout_prompt(request['prompt'],request['width'],request['height'],request.get('layout_source'))
        ready,note=verify(variant=request['model'])
        if not ready:
            raise ValueError(note+' Open Settings and install the selected model.')
        with LOCK:
            if STOPPING:
                raise RuntimeError('App is stopping. Restart RUN.bat on the PC.')
            if ACTIVE is not None:
                raise RuntimeError('This PC already has an active generation.')
            if DOWNLOAD is not None:
                raise RuntimeError('Wait for the model download to finish in Settings.')
            if selected_model() != request['model']:
                raise RuntimeError('Model selection changed. Please generate again.')
            folder=JOBS/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:10])
            folder.mkdir(parents=True)
            save_state(token,controls=values,results=[],expanded='',running=True,
                started=time.time(),status='Starting: checking models and loading…',step=0)
            job={'token':token,'request':request,'count':count,'folder':folder,'cancel':threading.Event()}
            ACTIVE=job
            threading.Thread(target=run_job,args=(job,),daemon=True).start()
        return refresh(token)
    except Exception as error:
        with LOCK:
            if ACTIVE is None:
                try:
                    save_state(token,status=str(error),running=False)
                except ValueError:
                    pass
        values=refresh(token)
        return (values[0],html.escape(str(error)),*values[2:])


def stop_job():
    with LOCK:
        if ACTIVE is None:
            return 'No generation is running.'
        ACTIVE['cancel'].set()
        process=PROCESS
        if process is not None and process.poll() is None:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            else:
                process.terminate()
    return 'Stopping this app’s worker. Saved images remain on disk.'


def release():
    with LOCK:
        return ('Stop the current generation first.' if ACTIVE else
                'GPU memory is free: the app closes its model worker after each image.')


def stop_server():
    global STOPPING
    with LOCK:
        if ACTIVE or DOWNLOAD:
            return 'Stop the current generation or model download before shutting down.'
        STOPPING=True
    def close():
        time.sleep(1.5)
        for server in SERVERS:
            server.close()
        os._exit(0)
    threading.Thread(target=close,daemon=True).start()
    return 'App stopping. Restart RUN.bat on this PC to use it again.'


def loras():
    folder=MODEL_DIR/'loras'
    return ['None']+sorted(str(p.relative_to(folder)).replace('\\','/') for p in folder.rglob('*.safetensors'))


def theme():
    palette=dict(body_background_fill='#07101f',body_text_color='#e8eef9',
        block_background_fill='#111c2f',block_label_background_fill='#18283f',
        block_label_text_color='#e8eef9',input_background_fill='#091425',
        button_primary_background_fill='#f5b942',
        button_primary_text_color='#091425',button_secondary_background_fill='#192d47',
        button_secondary_text_color='#e8eef9')
    return gr.themes.Base(primary_hue='amber',neutral_hue='slate').set(
        **palette,**{k+'_dark':v for k,v in palette.items()})


def build_ui():
    settings=load_access()
    logo=base64.b64encode((APP/'assets/ggf-brain-logo.png').read_bytes()).decode()
    with gr.Blocks(title='GGF Image Maker') as demo:
        token=gr.BrowserState('',storage_key='ggf-moss-workspace-v1',secret='ggf-moss-local-workspace-v1')
        with gr.Column(elem_id='ggf-shell'):
            gr.HTML(f"<div id='ggf-hero'><img src='data:image/png;base64,{logo}' alt='Get Going Fast'>"
                "<div class='kicker'>GET GOING FAST · LOCAL AI</div><h1>GGF Image Maker</h1>"
                "<p>Turn your ideas into pictures, posters and illustrated text.</p>"
                "<div class='links'><a href='https://getgoingfast.pro' target='_blank'>GetGoingFast.pro ↗</a>"
                "<a href='https://youtube.com/@TheAIHobbyGuy' target='_blank'>The AI Hobby Guy ↗</a></div></div>")
            with gr.Tabs(selected=initial_tab()) as tabs:
                with gr.Tab('Create',id='create'):
                    gr.Markdown('Describe your picture, choose its shape, then generate. **Use it from your phone:** enable phone access in Settings. Your PC runs generation and must stay on.')
                    prompt=gr.Textbox(value=DEFAULT_PROMPT,label='What would you like to create?',lines=8,
                        placeholder='A watercolor poster of a mountain bookshop, warm light, with the title "THE LAST CHAPTER".')
                    gr.Markdown('Try the Get Going Fast JSON layout below, or replace it with an ordinary description. JSON layouts go straight to MOSS; caption expansion applies only to ordinary descriptions.')
                    example=gr.Button('Load Get Going Fast JSON example')
                    with gr.Row():
                        shape=gr.Dropdown(list(SHAPES),value=DEFAULT_SHAPE,label='Picture size')
                        width=gr.Slider(MIN_EDGE,MAX_EDGE,value=DEFAULT_SIZE,step=16,label='Width · pixels')
                        height=gr.Slider(MIN_EDGE,MAX_EDGE,value=DEFAULT_SIZE,step=16,label='Height · pixels')
                    size_note=gr.Markdown('768 × 768 · lower-resolution default. Choose a preset or adjust the sliders.')
                    gr.Markdown('Sizes use multiples of 16, within 0.25–4 megapixels. Invalid combinations are rejected before models load. Large images need more memory; 1088 is used instead of unsupported 1080.')
                    enhance=gr.Checkbox(value=True,label='Expand my idea into a detailed caption locally')
                    with gr.Accordion('More control',open=False):
                        negative=gr.Textbox(value=DEFAULTS[3],label='Things to avoid')
                        with gr.Row():
                            steps=gr.Slider(1,100,value=25,step=1,label='Generation steps')
                            cfg=gr.Slider(0,15,value=2.5,step=0.1,label='Prompt guidance')
                        with gr.Row():
                            seed=gr.Number(value=0,precision=0,label='Image seed · -1 for random')
                            enhance_seed=gr.Number(value=0,precision=0,label='Caption seed · -1 for random')
                            count=gr.Number(value=1,minimum=1,maximum=16,precision=0,label='Images · sequential')
                        slg=gr.Checkbox(value=False,label='Layer 22 guidance · first half of sampling')
                        gr.Markdown('Layer guidance is off in your supplied workflow. Sampler: ER-SDE · schedule: simple.')
                        with gr.Row():
                            lora=gr.Dropdown(loras(),value='None',label='Optional LoRA')
                            strength=gr.Slider(-2,2,value=1,step=0.05,label='LoRA strength')
                        gr.Markdown('Place compatible MOSS LoRAs in this app’s models/loras folder, then refresh the list.')
                        scan=gr.Button('Refresh LoRA list')
                    preview=gr.Button('Turbo preview · half size',variant='secondary')
                    generate=gr.Button('Generate image · Ctrl+Enter',variant='primary',elem_id='ggf-generate')
                    stop=gr.Button('Stop generation · keep saved images')
                    status=gr.HTML('Ready.',elem_id='ggf-status')
                    gallery=gr.Gallery(label='Your images',columns=2,height=580,object_fit='contain',interactive=False)
                    saved=gr.Button('Show saved images on this PC')
                    expanded=gr.Textbox(label='Prompt sent to the image model',lines=4,interactive=False)
                    gr.Markdown('Images and settings are saved in outputs. Turbo preview reduces dimensions by half, stopping at MOSS’s supported minimum pixel count; sampling settings stay the same.')
                with gr.Tab('Settings',id='settings'):
                    with gr.Column(elem_id='ggf-liz-guide'):
                        with gr.Row():
                            guide_clip=APP/'assets/liz-moss-setup.mp4'
                            if guide_clip.is_file():
                                gr.Video(str(guide_clip),label='L/iZ · getting started',interactive=False,
                                         autoplay=False,height=225,buttons=[])
                            else:
                                gr.Image(str(APP/'assets/liz-guide.png'),label='L/iZ · getting started',interactive=False,height=225,buttons=[])
                            gr.Markdown('### L/iZ’s quick start\n**Before you can create pictures, download a model below.** '
                                'Choose Smaller for less disk space and memory, or Larger for full precision. '
                                'Click Install / resume and wait for the green **Installed** message. Then open Create. '
                                'Once your selected model is installed, the app opens to Create next time.')
                    gr.Markdown('### Choose and download your model')
                    model_choice=gr.Dropdown([(v['label'],k) for k,v in VARIANTS.items()],
                        value=selected_model(),label='Image model')
                    gr.Markdown('Both choices use MOSS-SFT’s 5B image model: INT8 is smaller; BF16 keeps full precision and needs more memory. '
                        'The text encoder and VAE are downloaded once inside this app. Total downloads: about **13.0 GB** for Smaller or **17.8 GB** for Larger. '
                        'Installing both keeps both image files; switching does not delete anything.')
                    selection_note=gr.Markdown()
                    model_status=gr.HTML(value=model_panel()[0],elem_id='ggf-model-installed')
                    install=gr.Button('Install / resume selected model and support files',variant='primary')
                    cancel_download=gr.Button('Stop download · keep partial files',interactive=False)
                    download_stop_note=gr.Markdown()
                    check=gr.Button('Refresh installed status')
                    to_create=gr.Button('Open Create')
                    gr.Markdown('### Use it from your phone')
                    mode=gr.Dropdown(list(MODES),value=next(k for k,v in MODES.items() if v==settings['mode']),label='Phone access')
                    username=gr.Textbox(value=settings['username'],label='Remote username · optional')
                    password=gr.Textbox(type='password',label='Remote password · blank disables remote login')
                    save=gr.Button('Save phone settings')
                    access_note=gr.Markdown()
                    gr.Markdown('Restart RUN.bat after changing phone settings. The local PC page always opens without a login. Wi-Fi mode uses your network; a temporary link provides access elsewhere.')
                    gr.Markdown('### App controls')
                    gr.Markdown('For app updates, close this server and run **UPDATE.bat** beside your member launcher. '
                                'Manual installs: run `.venv\\Scripts\\python.exe update.py`, then reinstall requirements-lock.txt. '
                                'Models, saved images and settings stay on this PC.')
                    free=gr.Button('Release models / free GPU memory')
                    free_note=gr.Markdown()
                    shutdown=gr.Button('Stop server')
                    confirm=gr.Button('Confirm stop · restart on this PC required',visible=False)
                    shutdown_note=gr.Markdown()
            phone=gr.Markdown(elem_id='ggf-phone-status')
            gr.HTML("<div id='ggf-footer'>Powered by MOSS-Image · OpenMOSS · local Comfy inference modules.<br>Get Going Fast</div>")
        controls=[prompt,width,height,negative,steps,cfg,seed,enhance,slg,enhance_seed,lora,strength,count]
        results=[gallery,status,expanded,generate,preview,phone]
        generate.click(submit,[token,*controls],results,queue=False,show_progress='hidden')
        preview.click(lambda *args:submit(*args,preview=True),[token,*controls],results,queue=False,show_progress='hidden')
        stop.click(stop_job,outputs=status,queue=False)
        saved.click(show_saved,token,results,queue=False)
        gallery.select(select_saved,token,expanded,queue=False)
        example.click(lambda:DEFAULT_PROMPT,outputs=prompt,queue=False).then(save_controls,[token,*controls],queue=False)
        shape.change(apply_shape,shape,[width,height],queue=False).then(
            save_controls,[token,*controls],queue=False)
        def size_status(w,h):
            try:
                w,h=validate_dimensions(w,h)
                return f'{w} × {h} · {w*h/1e6:.2f} megapixels'
            except (ValueError,TypeError,OverflowError) as error:
                return '**Choose another size:** '+str(error)
        gr.on([width.change,height.change],size_status,[width,height],size_note,queue=False)
        gr.on([width.input,height.input],lambda:gr.update(value='Custom'),outputs=shape,queue=False)
        gr.on([c.input for c in controls],save_controls,[token,*controls],queue=False)
        scan.click(lambda:gr.update(choices=loras(),value='None'),outputs=lora,queue=False)
        model_outputs=[model_status,install,cancel_download]
        model_choice.change(change_model,model_choice,[model_choice,selection_note],queue=False)
        install.click(install_models,outputs=model_outputs,queue=False)
        cancel_download.click(stop_download,outputs=download_stop_note,queue=False)
        check.click(model_panel,outputs=model_outputs,queue=False)
        to_create.click(lambda:gr.update(selected='create'),outputs=tabs,queue=False)
        save.click(save_access,[mode,username,password],access_note,queue=False)
        free.click(release,outputs=free_note,queue=False)
        shutdown.click(lambda:gr.update(visible=True),outputs=confirm,queue=False)
        confirm.click(stop_server,outputs=shutdown_note,queue=False)
        timer=gr.Timer(1)
        timer.tick(refresh,token,results,queue=False,show_progress='hidden')
        timer.tick(model_panel,outputs=model_outputs,queue=False,show_progress='hidden')
        demo.load(restore,token,[token,*controls,*results,tabs,shape],queue=False,js=SHORTCUT)
    return demo


def lan_url(url):
    import psutil
    from urllib.parse import urlparse
    candidates=[]
    stats=psutil.net_if_stats()
    for name,addresses in psutil.net_if_addrs().items():
        if name not in stats or not stats[name].isup or any(n in name.lower() for n in ['vpn','virtual','wsl','docker','loopback','vmware','tailscale']):
            continue
        for entry in addresses:
            if entry.family==socket.AF_INET and entry.netmask:
                ip=ipaddress.ip_address(entry.address)
                if ip.is_private and not ip.is_loopback and not ip.is_link_local:
                    candidates.append(entry.address)
    return f'http://{candidates[0]}:{urlparse(url).port}' if candidates else 'No Wi-Fi/Ethernet address found.'


def launch(demo,port,**kwargs):
    for candidate in range(port,port+20):
        try:
            result=demo.launch(server_port=candidate,prevent_thread_lock=True,
                allowed_paths=[str(OUTPUTS)],blocked_paths=[str(MODEL_DIR),str(APP/'access.json')],
                css=CSS,theme=theme(),show_error=True,**kwargs)
            SERVERS.append(demo)
            return result
        except OSError as error:
            if 'Cannot find empty port' not in str(error) and getattr(error,'winerror',None)!=10048:
                raise
    raise RuntimeError('No free app port.')


def main():
    global LOCAL_URL,PHONE_LINK
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--port',type=int,default=7891)
    args=parser.parse_args()
    for directory in [JOBS,OUTPUTS,WORKSPACES]:
        directory.mkdir(parents=True,exist_ok=True)
    # One server owns this app's GPU lock, even if RUN.bat is double-clicked twice.
    (APP/'runtime').mkdir(exist_ok=True)
    guard=(APP/'runtime/app.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        if guard.seek(0,2)==0:
            guard.write(b'0');guard.flush()
        guard.seek(0)
        try:
            msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise RuntimeError('This app is already running. Use its existing browser page.')
    atexit.register(stop_job)
    local=build_ui()
    _,LOCAL_URL,_=launch(local,args.port,server_name='127.0.0.1',share=False,
        auth=None,inbrowser=not args.no_browser)
    print('LOCAL_URL='+LOCAL_URL,flush=True)
    settings=load_access()
    if settings['mode']!='local':
        try:
            remote=build_ui()
            auth=(lambda u,p:verify_access(u,p,settings)) if settings['hash'] else None
            _,url,public=launch(remote,args.port+1000,
                server_name='0.0.0.0' if settings['mode']=='lan' else '127.0.0.1',
                share=settings['mode']=='public',auth=auth,inbrowser=False)
            PHONE_LINK=public or lan_url(url)
        except Exception as error:
            PHONE_LINK='Could not start phone access: '+str(error)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_job()
        for demo in SERVERS:
            demo.close()


if __name__=='__main__':
    main()
