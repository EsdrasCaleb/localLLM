package com.densebrain.rif.client.service.types;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class InvokeResponse_getPullParser_2_0_Test {

    @Test
    public void testGetPullParser() {
        InvokeResponse invokeResponse = mock(InvokeResponse.class);
        when(invokeResponse.get_return()).thenReturn("test_return");
        javax.xml.namespace.QName qName = new javax.xml.namespace.QName("", "return");
        javax.xml.stream.XMLStreamReader xmlStreamReader = invokeResponse.getPullParser(qName);
        assertNotNull(xmlStreamReader);
        // Asserting expected behavior
        // ...
    }
}
