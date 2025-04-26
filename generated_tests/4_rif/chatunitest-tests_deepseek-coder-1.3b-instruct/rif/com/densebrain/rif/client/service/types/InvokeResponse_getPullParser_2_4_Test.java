package com.densebrain.rif.client.service.types;

import javax.xml.namespace.QName;
import javax.xml.stream.XMLStreamReader;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class InvokeResponse_getPullParser_2_4_Test {

    @Test
    public void testGetPullParser() throws Exception {
        InvokeResponse invokeResponse = new InvokeResponse();
        invokeResponse.set_return("test");
        QName qName = new QName("", "test");
        XMLStreamReader xmlStreamReaderMock = Mockito.mock(XMLStreamReader.class);
        when(xmlStreamReaderMock.getLocalName()).thenReturn(qName.getLocalPart());
        when(xmlStreamReaderMock.getElementText()).thenReturn("test");
        List<Object> elementList = new ArrayList<>();
        elementList.add(qName);
        elementList.add("test");
        List<Object> attribList = new ArrayList<>();
        XMLStreamReader xmlStreamReader = new org.apache.axis2.databinding.utils.reader.ADBXMLStreamReaderImpl(qName, elementList.toArray(new Object[0]), attribList.toArray(new Object[0]));
        XMLStreamReader result = invokeResponse.getPullParser(qName);
        assertEquals(xmlStreamReaderMock.getLocalName(), result.getLocalName());
        assertEquals(xmlStreamReaderMock.getElementText(), result.getElementText());
    }
}
