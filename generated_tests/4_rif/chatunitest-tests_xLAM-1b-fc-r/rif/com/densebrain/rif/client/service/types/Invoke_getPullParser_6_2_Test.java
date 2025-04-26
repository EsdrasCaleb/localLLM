package com.densebrain.rif.client.service.types;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Invoke_getPullParser_6_2_Test {

    @Test
    public void testGetPullParser() {
        // Arrange
        Invoke invoke = new Invoke();
        invoke.setClassName("testClass");
        invoke.setMethodName("testMethod");
        invoke.setSerializedParams("testParams");
        javax.xml.namespace.QName qName = new javax.xml.namespace.QName("http://densebrain.com/rif/client/service/types", "invoke", "ns1");
        // Act
        javax.xml.stream.XMLStreamReader result = invoke.getPullParser(qName);
        // Assert
        assertEquals("testClass", result.getLocalName());
        assertEquals("testMethod", result.getLocalName());
        assertEquals("testParams", result.getLocalName());
    }
}
