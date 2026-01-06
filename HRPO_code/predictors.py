from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template
import utils
import task


class GPT4Predictor(ABC):
    def __init__(self, args, model, tokenizer):
        self.args = args  # config
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def inference_new(self, ex, prompt):
        pass


class BBQTernaryPredictor(GPT4Predictor):
    categories = ['(A)', '(B)', '(C)']

    def safe_render(self, prompt, text):
        safe_text = text.replace("}", "\\}").replace("{{", "{ {")
        return Template(prompt).render(text=safe_text)

    def safe_render_sd(self, prompt, response,text):
        safe_text = text.replace("}", "\\}").replace("{{", "{ {")
        return Template(prompt).render(answer=response,text=safe_text)

    def inference_new(self, exs, prompt, device, prompt_sd=None):
        batch_data = []
        for ex in exs:
            if self.args['pa']:
                if self.args['bias_type'] == 'Gender':
                    if 'F' in ex['stereo_group']:
                        batch_data.append((ex['image'], Template(prompt).render(persona='female', text=ex['text'])))
                    elif 'M' in ex['stereo_group']:
                        batch_data.append((ex['image'], Template(prompt).render(persona='male', text=ex['text'])))
                    elif 'trans' in ex['stereo_group']:
                        batch_data.append((ex['image'], Template(prompt).render(persona='transgender', text=ex['text'])))
                elif (self.args['bias_type'] == 'Race' or self.args['bias_type'] == 'Religion'
                      or self.args['bias_type'] == 'Disability' or self.args['bias_type'] == 'Nationality'):
                    batch_data.append((ex['image'], Template(prompt).render(persona=ex['stereo_group'][0], text=ex['text'])))
                elif self.args['bias_type'] == 'Age':
                    if ex['stereo_group'] == ["old"]:
                        batch_data.append((ex['image'], Template(prompt).render(persona='old people', text=ex['text'])))
                    elif ex['stereo_group'] == ["nonOld"]:
                        batch_data.append((ex['image'], Template(prompt).render(persona='people who are not old', text=ex['text'])))
                elif self.args['bias_type'] == 'SES':
                    if ex['stereo_group'] == ["low SES"]:
                        batch_data.append((ex['image'], Template(prompt).render(persona='low socioeconomic status', text=ex['text'])))
                    elif ex['stereo_group'] == ["high SES"]:
                        batch_data.append((ex['image'], Template(prompt).render(persona='high socioeconomic status', text=ex['text'])))
                elif self.args['bias_type'] == 'Sexual':
                    if len(ex['stereo_group']) == 2:
                        batch_data.append((ex['image'], Template(prompt).render(persona='{} or {}'.format(ex['stereo_group'][0], ex['stereo_group'][1]), text=ex['text'])))
                    elif len(ex['stereo_group']) == 1:
                        batch_data.append((ex['image'], Template(prompt).render(persona=ex['stereo_group'][0], text=ex['text'])))
                elif self.args['bias_type'] == 'Physical':
                    if ex['stereo_group'] == 'negDress':
                        batch_data.append((ex['image'], Template(prompt).render(persona='people who dress negatively', text=ex['text'])))
                    elif ex['stereo_group'] == 'visibleDifference':
                        batch_data.append((ex['image'], Template(prompt).render(persona='people who visible difference', text=ex['text'])))
                    else:
                        batch_data.append((ex['image'], Template(prompt).render(persona=ex['stereo_group'][0], text=ex['text'])))
            else:
                batch_data.append((ex['image'], self.safe_render(prompt, ex['text'])))
        response = utils.ds2(self.args, device, self.model, self.tokenizer, batch_data, max_tokens=6, n=1, temperature=self.args['temperature'], batch_size=self.args['batch_size'])  
        if self.args['sd']:
            for i in range(len(response)):
                batch_data[i] = (exs[i]['image'], self.safe_render_sd(prompt_sd, response[i], exs[i]['text']))
            response = utils.ds2(self.args, device, self.model, self.tokenizer, batch_data, max_tokens=6, n=1,
                                 temperature=self.args['temperature'],
                                 batch_size=self.args['batch_size'])
        preds = []
        for i in range(len(response)):
            pred = -1
            response[i] = self.filter_output(response[i])
            if ('A' in response[i].strip() or exs[i]['ans0_info'].lower() in response[i].strip().lower()) and '(B)' not in response[i].strip() and '(C)' not in response[i].strip():
                pred = 0
            elif ('B' in response[i].strip() or exs[i]['ans1_info'].lower() in response[i].strip().lower()) and '(A)' not in response[i].strip() and '(C)' not in response[i].strip():
                pred = 1
            elif ('C' in response[i].strip() or exs[i]['ans2_info'].lower() in response[i].strip().lower()) and '(A)' not in response[i].strip() and '(B)' not in response[i].strip():
                pred = 2
            preds.append(pred)
        return preds

    def filter_output(self, output):
        words = ['Answer', 'answer', 'Example', 'example', 'Explanation', 'explanation', 'Prediction', 'prediction',
                 'Choose', 'choose', 'Based', 'based', 'Information', 'information', 'Analysis', 'analysis', 'Ana',
                 'ana', 'Label', 'label', 'Reason', 'reason', 'Context', 'context', 'CONTEXT']
        for i in range(len(words)):
            output = output.replace(words[i], '')
        return output


class VLBiasTernaryPredictor(GPT4Predictor):
    categories = ['(A)', '(B)', '(C)']

    def safe_render(self, prompt, text):
        safe_text = text.replace("}", "\\}").replace("{{", "{ {")
        return Template(prompt).render(text=safe_text)

    def safe_render_sd(self, prompt, response,text):
        safe_text = text.replace("}", "\\}").replace("{{", "{ {")
        return Template(prompt).render(answer=response,text=safe_text)

    def inference_new(self, exs, prompt, device, prompt_sd=None):
        batch_data = []
        for ex in exs:
            if self.args['pa']:
                if self.args['bias_type'] == 'Race_x_gender':
                    if 'F' in ex['stereo_group']:
                        ex['stereo_group'] = ex['stereo_group'].replace('F', 'Female')
                    elif 'M' in ex['stereo_group']:
                        ex['stereo_group'] = ex['stereo_group'].replace('M', 'Male')
                    batch_data.append((ex['image'], Template(prompt).render(persona=ex['stereo_group'], text=ex['text'])))
                elif self.args['bias_type'] == 'Race_x_SES':
                    if 'lowSES' in ex['stereo_group']:
                        ex['stereo_group'] = ex['stereo_group'].replace('lowSES', 'low socioeconomic status')
                    elif 'highSES' in ex['stereo_group']:
                        ex['stereo_group'] = ex['stereo_group'].replace('highSES', 'high socioeconomic status')
                    batch_data.append((ex['image'], Template(prompt).render(persona=ex['stereo_group'], text=ex['text'])))
                elif self.args['bias_type'] == 'SES':
                    if 'lowSES' in ex['stereo_group']:
                        batch_data.append((ex['image'], Template(prompt).render(persona='low socioeconomic status', text=ex['text'])))
                    elif 'highSES' in ex['stereo_group']:
                        batch_data.append((ex['image'], Template(prompt).render(persona='high socioeconomic status', text=ex['text'])))
                    else:
                        batch_data.append((ex['image'], Template(prompt).render(persona=ex['stereo_group'], text=ex['text'])))
                else:
                    batch_data.append((ex['image'], Template(prompt).render(persona=ex['stereo_group'], text=ex['text'])))

            else:
                batch_data.append((ex['image'], self.safe_render(prompt, ex['text'])))
        response = utils.ds2(self.args, device, self.model, self.tokenizer, batch_data, max_tokens=30, n=1, temperature=self.args['temperature'], batch_size=self.args['batch_size'])  
        if self.args['sd']:
            for i in range(len(response)):
                batch_data[i] = (exs[i]['image'], self.safe_render_sd(prompt_sd, response[i], exs[i]['text']))
            response = utils.ds2(self.args, device, self.model, self.tokenizer, batch_data, max_tokens=30, n=1,
                                 temperature=self.args['temperature'],
                                 batch_size=self.args['batch_size']) 
        preds = []
        for i in range(len(response)):
            pred = -1
            response[i] = self.filter_output(response[i])
            if 'ans2_info' in exs[i].keys():
                if ('A' in response[i].strip() or exs[i]['ans0_info'].lower() in response[i].strip().lower()) and '(B)' not in response[i].strip() and '(C)' not in response[i].strip():
                    pred = 0
                elif ('B' in response[i].strip() or exs[i]['ans1_info'].lower() in response[i].strip().lower()) and '(A)' not in response[i].strip() and '(C)' not in response[i].strip():
                    pred = 1
                elif ('C' in response[i].strip() or exs[i]['ans2_info'].lower() in response[i].strip().lower()) and '(A)' not in response[i].strip() and '(B)' not in response[i].strip():
                    pred = 2
            else:
                if ('A' in response[i].strip() or exs[i]['ans0_info'].lower() in response[i].strip().lower()) and '(B)' not in response[i].strip():
                    pred = 0
                elif ('B' in response[i].strip() or exs[i]['ans1_info'].lower() in response[i].strip().lower()) and '(A)' not in response[i].strip():
                    pred = 1
            preds.append(pred)
        return preds

    def filter_output(self, output):
        words = ['Answer', 'answer', 'Example', 'example', 'Explanation', 'explanation', 'Prediction', 'prediction',
                 'Choose', 'choose', 'Based', 'based', 'Information', 'information', 'Analysis', 'analysis', 'Ana',
                 'ana', 'Label', 'label', 'Reason', 'reason', 'Context', 'context', 'CONTEXT']
        for i in range(len(words)):
            output = output.replace(words[i], '')
        return output


class VLBiasOpenTask(GPT4Predictor):
    def safe_render(self, prompt, text):
        safe_text = text.replace("}", "\\}").replace("{{", "{ {")
        return Template(prompt).render(text=safe_text)

    def safe_render_sd(self, prompt, response,text):
        safe_text = text.replace("}", "\\}").replace("{{", "{ {")
        return Template(prompt).render(answer=response,text=safe_text)

    def inference_new(self, exs, prompt, device, prompt_sd=None):
        batch_data = []
        for ex in exs:
            if self.args['pa']:
                batch_data.append((ex['image'], Template(prompt).render(persona=ex['stereo_group'], text=ex['text'])))

            else:
                batch_data.append((ex['image'], self.safe_render(prompt, ex['text'])))
        response = utils.ds2(self.args, device, self.model, self.tokenizer, batch_data, max_tokens=512, n=1, temperature=self.args['temperature'], batch_size=self.args['batch_size'])  
        if self.args['sd']:
            for i in range(len(response)):
                batch_data[i] = (exs[i]['image'], self.safe_render_sd(prompt_sd, response[i], exs[i]['text']))
            response = utils.ds2(self.args, device, self.model, self.tokenizer, batch_data, max_tokens=512, n=1,
                                 temperature=self.args['temperature'],
                                 batch_size=self.args['batch_size'])  
        return response
