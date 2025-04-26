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

class InvokeResponse_getPullParser_2_1_Test {

    @Test
    public void testGetPullParser() throws Exception {
        // Create a mock instance of the InvokeResponse class
        InvokeResponse invokeResponse = Mockito.mock(InvokeResponse.class);
        // Define the input parameters
        QName qName = new QName("http://densebrain.com/rif/client/service/types", "invokeResponse", "ns1");
        // Call the method under test
        XMLStreamReader xmlStreamReader = invokeResponse.getPullParser(qName);
        // Verify the behavior of the method
        // For simplicity, we assume that the method returns a non-null XMLStreamReader
        assertNotNull(xmlStreamReader);
    }
}
