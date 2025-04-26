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
    void testGetPullParser_withNullQName() {
        Invoke invoke = new Invoke();
        assertThrows(NullPointerException.class, () -> invoke.getPullParser(null));
    }
}
