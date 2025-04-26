package com.densebrain.rif.client.service.types;

import javax.xml.namespace.QName;
import javax.xml.stream.XMLStreamReader;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Invoke_getPullParser_6_1_Test {

    @Test
    public void testGetPullParser() throws Exception {
        Invoke invoke = new Invoke();
        QName qName = new QName("", "ns1");
        String className = "className";
        String methodName = "methodName";
        String serializedParams = "serializedParams";
        invoke.setClassName(className);
        invoke.setMethodName(methodName);
        invoke.setSerializedParams(serializedParams);
        XMLStreamReader xmlStreamReader = Mockito.mock(XMLStreamReader.class);
        when(xmlStreamReader.getLocalName()).thenReturn("className");
        when(xmlStreamReader.getNamespaceURI()).thenReturn("");
        when(xmlStreamReader.getElementText()).thenReturn(className);
        when(xmlStreamReader.getAttributeValue("", "")).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.START_ELEMENT);
        when(xmlStreamReader.getElementText()).thenReturn(methodName);
        when(xmlStreamReader.getAttributeValue("", "")).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.START_ELEMENT);
        when(xmlStreamReader.getElementText()).thenReturn(serializedParams);
        when(xmlStreamReader.getAttributeValue("", "")).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("serializedParams");
        when(xmlStreamReader.getElementText()).thenReturn(serializedParams);
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("");
        when(xmlStreamReader.getElementText()).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.START_ELEMENT);
        when(xmlStreamReader.getElementText()).thenReturn(serializedParams);
        when(xmlStreamReader.getAttributeValue("", "")).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("");
        when(xmlStreamReader.getElementText()).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("");
        when(xmlStreamReader.getElementText()).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("");
        when(xmlStreamReader.getElementText()).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("");
        when(xmlStreamReader.getElementText()).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("");
        when(xmlStreamReader.getElementText()).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("");
        when(xmlStreamReader.getElementText()).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("");
        when(xmlStreamReader.getElementText()).thenReturn("");
        when(xmlStreamReader.nextTag()).thenReturn(XMLStreamReader.END_ELEMENT);
        when(xmlStreamReader.getLocalName()).thenReturn("");
    }
}
