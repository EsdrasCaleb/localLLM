package com.densebrain.rif.client.service.types;

import java.util.ArrayList;
import java.util.List;
import javax.xml.namespace.QName;
import javax.xml.stream.XMLStreamReader;
import org.apache.axis2.databinding.utils.reader.ADBXMLStreamReaderImpl;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Invoke_getPullParser_6_0_Test {

    private Invoke invoke;

    @BeforeEach
    void setUp() {
        invoke = new Invoke();
        invoke.setClassName("TestClass");
        invoke.setMethodName("testMethod");
        invoke.setSerializedParams("serializedParams");
    }

    @Test
    void testGetPullParser() throws Exception {
        QName qName = new QName("http://densebrain.com/rif/client/service/types", "invoke", "ns1");
        XMLStreamReader xmlStreamReader = invoke.getPullParser(qName);
        assertNotNull(xmlStreamReader);
        assertEquals(qName, xmlStreamReader.getName());
        List<QName> expectedElementList = new ArrayList<>();
        expectedElementList.add(new QName("", "className"));
        expectedElementList.add(new QName("", "TestClass"));
        expectedElementList.add(new QName("", "methodName"));
        expectedElementList.add(new QName("", "testMethod"));
        expectedElementList.add(new QName("", "serializedParams"));
        expectedElementList.add(new QName("", "serializedParams"));
        List<QName> actualElementList = new ArrayList<>();
        while (xmlStreamReader.hasNext()) {
            xmlStreamReader.next();
            if (xmlStreamReader.isStartElement()) {
                actualElementList.add(xmlStreamReader.getName());
            }
        }
        assertEquals(expectedElementList.size(), actualElementList.size());
        for (int i = 0; i < expectedElementList.size(); i++) {
            assertEquals(expectedElementList.get(i), actualElementList.get(i));
        }
    }
}
