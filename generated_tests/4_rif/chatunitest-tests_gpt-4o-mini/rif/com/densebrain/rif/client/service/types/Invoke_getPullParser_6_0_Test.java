package com.densebrain.rif.client.service.types;

import javax.xml.namespace.QName;
import org.apache.axis2.databinding.utils.reader.ADBXMLStreamReaderImpl;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Invoke_getPullParser_6_0_Test {

    private Invoke invoke;

    @BeforeEach
    public void setUp() {
        invoke = new Invoke();
        invoke.setClassName("TestClass");
        invoke.setMethodName("testMethod");
        invoke.setSerializedParams("param1,param2");
    }

    @Test
    public void testGetPullParser() {
        QName qName = new QName("http://densebrain.com/rif/client/service/types", "invoke", "ns1");
        ADBXMLStreamReaderImpl result = (ADBXMLStreamReaderImpl) invoke.getPullParser(qName);
        assertNotNull(result);
    }
}
