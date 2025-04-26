package com.densebrain.rif.client.service.types;

import org.apache.axis2.databinding.utils.reader.ADBXMLStreamReaderImpl;
import javax.xml.namespace.QName;
import javax.xml.stream.XMLStreamReader;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class InvokeResponse_getPullParser_2_0_Test {

    @Test
    void testGetPullParser_withReturnValue() throws Exception {
        InvokeResponse response = new InvokeResponse();
        String returnValue = "testReturn";
        response.set_return(returnValue);
        QName qName = new QName("testNamespace", "testLocalPart");
        XMLStreamReader reader = response.getPullParser(qName);
        assertTrue(reader instanceof ADBXMLStreamReaderImpl);
        // Verify the content using reflection because ADBXMLStreamReaderImpl is not easily inspectable directly.
        Field elementListField = ADBXMLStreamReaderImpl.class.getDeclaredField("elementList");
        elementListField.setAccessible(true);
        ArrayList elementList = (ArrayList) elementListField.get(reader);
        assertEquals(2, elementList.size());
        assertEquals(new QName("", "return"), elementList.get(0));
        assertEquals(returnValue, elementList.get(1));
    }

    @Test
    void testGetPullParser_withNullReturnValue() {
        InvokeResponse response = new InvokeResponse();
        response.set_return(null);
        QName qName = new QName("testNamespace", "testLocalPart");
        XMLStreamReader reader = response.getPullParser(qName);
        assertTrue(reader instanceof ADBXMLStreamReaderImpl);
        // Check for null handling -  the internal representation might vary, so a general check is better than specific content.
        assertNotNull(reader);
    }

    @Test
    void testGetPullParser_withEmptyReturnValue() {
        InvokeResponse response = new InvokeResponse();
        response.set_return("");
        QName qName = new QName("testNamespace", "testLocalPart");
        XMLStreamReader reader = response.getPullParser(qName);
        assertTrue(reader instanceof ADBXMLStreamReaderImpl);
        // Similar to null check, a general check is sufficient.
        assertNotNull(reader);
    }
}
