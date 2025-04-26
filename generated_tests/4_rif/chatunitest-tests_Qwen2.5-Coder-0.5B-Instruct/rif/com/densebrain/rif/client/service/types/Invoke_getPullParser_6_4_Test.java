package com.densebrain.rif.client.service.types;

import javax.xml.namespace.QName;
import javax.xml.stream.XMLStreamReader;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Invoke_getPullParser_6_4_Test {

    private Invoke invoke;

    @BeforeEach
    public void setUp() {
        invoke = Mockito.mock(Invoke.class);
    }

    @Test
    public void testGetPullParser() throws Exception {
        QName qName = new QName("http://densebrain.com/rif/client/service/types", "invoke", "ns1");
        XMLStreamReader xmlStreamReader = invoke.getPullParser(qName);
        // Add assertions here to verify that the xmlStreamReader is of the expected type
    }
}
