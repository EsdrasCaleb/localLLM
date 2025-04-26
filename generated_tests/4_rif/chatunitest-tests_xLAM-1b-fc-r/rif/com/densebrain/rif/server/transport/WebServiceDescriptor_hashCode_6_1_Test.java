package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_hashCode_6_1_Test {

    @Test
    public void testHashCode() {
        WebServiceDescriptor wsd = new WebServiceDescriptor(String.class, "targetNamespace", "typesNamespace");
        assertEquals(wsd.hashCode(), wsd.getServiceClazz().hashCode());
    }
}
