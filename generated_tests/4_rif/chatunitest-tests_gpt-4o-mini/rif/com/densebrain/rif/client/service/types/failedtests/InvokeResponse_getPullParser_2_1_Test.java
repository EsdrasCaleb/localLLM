package com.densebrain.rif.client.service.types;

import javax.xml.namespace.QName;
import javax.xml.stream.XMLStreamReader;
import org.apache.axis2.databinding.utils.reader.ADBXMLStreamReaderImpl;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class InvokeResponse_getPullParser_2_1_Test {

    private InvokeResponse invokeResponse;

    @BeforeEach
    public void setUp() {
        invokeResponse = new InvokeResponse();
    }

    @Test
    public void testGetPullParser_WithValidQName_ReturnsXMLStreamReader() {
        // Arrange
        QName qName = new QName("http://densebrain.com/rif/client/service/types", "invokeResponse", "ns1");
        invokeResponse.set_return("testValue");
        // Act
        XMLStreamReader xmlStreamReader = invokeResponse.getPullParser(qName);
        // Assert
        assertNotNull(xmlStreamReader);
        assertEquals(ADBXMLStreamReaderImpl.class, xmlStreamReader.getClass());
    }

    @Test
    public void testGetPullParser_WithNullReturnValue_ReturnsXMLStreamReader() {
        // Arrange
        QName qName = new QName("http://densebrain.com/rif/client/service/types", "invokeResponse", "ns1");
        invokeResponse.set_return(null);
        // Act
        XMLStreamReader xmlStreamReader = invokeResponse.getPullParser(qName);
        // Assert
        assertNotNull(xmlStreamReader);
    }
}
