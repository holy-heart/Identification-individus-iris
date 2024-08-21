from django import forms
from .models import Oeil
class OeilForm(forms.ModelForm):
    
    class Meta:
        model = Oeil
        fields = ("image",)
        labels ={
            "image" : '',
        }
        widgets = {
            'image': forms.FileInput(attrs={'class': 'custom-file-input'}),
        }
