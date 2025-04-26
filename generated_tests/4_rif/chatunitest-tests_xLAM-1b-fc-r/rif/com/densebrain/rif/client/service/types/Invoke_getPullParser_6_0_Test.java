package com.densebrain.rif.client.service.types;

import javax.xml.namespace.QName;
import javax.xml.stream.XMLStreamReader;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Invoke_getPullParser_6_0_Test {

    @Test
    public void testGetPullParser() throws Exception {
        // Arrange
        Invoke invoke = new Invoke();
        QName qName = new QName("", "test");
        String className = "testClass";
        String methodName = "testMethod";
        String serializedParams = "testParams";
        // Set the mocked values
        invoke.setClassName(className);
        invoke.setMethodName(methodName);
        invoke.setSerializedParams(serializedParams);
        // Create a mock for XMLStreamReader
        XMLStreamReader mockReader = Mockito.mock(XMLStreamReader.class);
        // Configure the mock to return the expected values
        when(mockReader.getLocalName()).thenReturn(qName.getLocalPart());
        when(mockReader.getNamespaceURI()).thenReturn(qName.getNamespaceURI());
        // Act
        XMLStreamReader result = invoke.getPullParser(qName);
        // Assert
        assertEquals(qName.getLocalPart(), result.getLocalName());
        assertEquals(qName.getNamespaceURI(), result.getNamespaceURI());
    }
}
