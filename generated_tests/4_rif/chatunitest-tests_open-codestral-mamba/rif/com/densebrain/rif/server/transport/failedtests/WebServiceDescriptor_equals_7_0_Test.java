package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_equals_7_0_Test {

    @Test
    public void testEquals_sameObject_returnsTrue() {
        WebServiceDescriptor wsd = new WebServiceDescriptor(null, null, null);
        assertTrue(wsd.equals(wsd));
    }

    @Test
    public void testEquals_nullObject_returnsFalse() {
        WebServiceDescriptor wsd = new WebServiceDescriptor(null, null, null);
        assertFalse(wsd.equals(null));
    }

    @Test
    public void testEquals_differentClass_returnsFalse() {
        WebServiceDescriptor wsd = new WebServiceDescriptor(null, null, null);
        assertFalse(wsd.equals(new Object()));
    }

    @Test
    public void testEquals_sameServiceClass_returnsTrue() {
        Class serviceClazz = mock(Class.class);
        WebServiceDescriptor wsd1 = new WebServiceDescriptor(serviceClazz, null, null);
        WebServiceDescriptor wsd2 = new WebServiceDescriptor(serviceClazz, null, null);
        assertTrue(wsd1.equals(wsd2));
    }

    @Test
    public void testEquals_differentServiceClass_returnsFalse() {
        Class serviceClazz1 = mock(Class.class);
        Class serviceClazz2 = mock(Class.class);
        WebServiceDescriptor wsd1 = new WebServiceDescriptor(serviceClazz1, null, null);
        WebServiceDescriptor wsd2 = new WebServiceDescriptor(serviceClazz2, null, null);
        assertFalse(wsd1.equals(wsd2));
    }
}
