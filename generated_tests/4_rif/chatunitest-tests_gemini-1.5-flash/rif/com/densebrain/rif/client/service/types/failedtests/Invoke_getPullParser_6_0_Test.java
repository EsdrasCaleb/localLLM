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

public class Invoke_getPullParser_6_0_Test {

    @Test
    void testGetPullParser_withValidQName() throws Exception {
        Invoke invoke = new Invoke();
        invoke.setClassName("MyClass");
        invoke.setMethodName("myMethod");
        invoke.setSerializedParams("myParams");
        QName qName = new QName("testNamespace", "testLocalPart");
        XMLStreamReader reader = invoke.getPullParser(qName);
        assertTrue(reader instanceof ADBXMLStreamReaderImpl);
        // Verify the content of the XMLStreamReader (This part is complex and requires more advanced XMLStreamReader assertions, which are beyond the scope of a simple unit test.  A more robust approach would involve using a library like Woodstox to parse and validate the XML stream.)
        // Instead, we'll verify the internal state of the ADBXMLStreamReaderImpl indirectly.
        Field elementListField = ADBXMLStreamReaderImpl.class.getDeclaredField("elementList");
        elementListField.setAccessible(true);
        ArrayList elementList = (ArrayList) elementListField.get(reader);
        // Check the number of elements
        assertEquals(6, elementList.size());
        assertEquals("className", ((QName) elementList.get(0)).getLocalPart());
        assertEquals("MyClass", elementList.get(1));
        assertEquals("methodName", ((QName) elementList.get(2)).getLocalPart());
        assertEquals("myMethod", elementList.get(3));
        assertEquals("serializedParams", ((QName) elementList.get(4)).getLocalPart());
        assertEquals("myParams", elementList.get(5));
    }

    @Test
    void testGetPullParser_withNullQName() {
        Invoke invoke = new Invoke();
        assertThrows(NullPointerException.class, () -> invoke.getPullParser(null));
    }

    @Test
    void testGetPullParser_withEmptyData() {
        Invoke invoke = new Invoke();
        QName qName = new QName("testNamespace", "testLocalPart");
        XMLStreamReader reader = invoke.getPullParser(qName);
        assertTrue(reader instanceof ADBXMLStreamReaderImpl);
        // Verify the content of the XMLStreamReader (similar to the previous test, simplified verification)
        try {
            Field elementListField = ADBXMLStreamReaderImpl.class.getDeclaredField("elementList");
            elementListField.setAccessible(true);
            ArrayList elementList = (ArrayList) elementListField.get(reader);
            assertEquals(6, elementList.size());
            assertEquals("", elementList.get(1));
            assertEquals("", elementList.get(3));
            assertEquals("", elementList.get(5));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access internal state of ADBXMLStreamReaderImpl");
        }
    }
}
